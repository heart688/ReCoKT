"""Full-parameter KT baseline primitives; all data access is caller supplied.

Clients are ``{school_id: window_records}``, with one context and <=50 targets
per window. No function loads data/checkpoints or chooses a budget on evaluation
data. Controllers own phases, storage, external seeds, and source school subsets.
Results/checkpoints contain CPU tensors and ordinary Python metadata, suitable
for torch 2.2 checkpoint serialization. ``fit`` mutates its model and optionally
appends to caller-owned history, cumulative cost, and checkpoint dictionaries.
"""

from copy import deepcopy
import json
import math
import time

import numpy as np
import torch
from torch import nn

from . import federated as fed
from .data import make_loader
from .metrics import binary_metrics, summarize_predictions
from .model import DKT, ModelConfig
from .runtime import (choose_device, fork_device_rng, move_batch, parameter_bytes,
                      state_copy, synchronize)
from .sakt import SAKT, SAKTConfig
from .withdrawal_posthoc import calibrate_delta


DEFAULT_SETTINGS = dict(rounds=20, local_epochs=1, batch_size=32, lr=.002,
                        clip_norm=5., optimizer="adam")
RECOVERY_BUDGETS = (1, 5, 10, 20)
CALIBRATION_FRACTIONS = (.1, .5)
_BILL_FIELDS = ("optimizer_steps", "targets_seen", "communication_bytes", "seconds")


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def make_model(model_name, n_skills, seed, **params):
    """Use manifest['n_skills']; never infer a catalog from retained records.

    Model dimensions are keyword parameters, saved by the existing metadata()
    methods. AKT is imported lazily from the independently maintained wrapper.
    Initialization is seeded independently of the caller's torch RNG state.
    """
    _integer(n_skills, "n_skills")
    _integer(seed, "seed", 0)
    constructors = {"dkt": (DKT, ModelConfig), "dkt-lstm": (DKT, ModelConfig),
                    "sakt": (SAKT, SAKTConfig)}
    key = model_name.lower()
    if key == "akt":
        from .akt import AKT, AKTConfig
        constructors[key] = (AKT, AKTConfig)
    if key not in constructors:
        raise ValueError(f"Unsupported model: {model_name}; available: DKT, SAKT, AKT")
    cls, config_cls = constructors[key]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return cls(config_cls(n_skills=n_skills, **params))


model_factory = make_model
build_model = make_model


def model_from_metadata(metadata, seed, state_dict=None):
    """Rebuild an existing model.metadata() mapping; optionally restore state."""
    params = dict(metadata)
    name, n_skills = params.pop("architecture"), params.pop("n_skills")
    model = make_model(name, n_skills, seed, **params)
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=True)
    return model


def window_forward(model, batch):
    """Return next-target logits, labels and mask, without detaching gradients."""
    return model.next_response(batch)


def prediction_features(model, batch):
    """Capture actual final-head inputs in the SAME next_response forward.

    DKT's head sees [B,L,H], including the unused final event; SAKT's head
    already sees [B,L-1,D]. Features stay connected to the encoder, allowing
    a caller's added readout loss to backpropagate through it. No model source
    changes or persistent hooks are needed. AKT exposes a native single-pass
    feature function aligned to raw positions 1:, which takes precedence.
    The exposed head must otherwise be Linear.
    """
    native = getattr(model, "prediction_features", None)
    if callable(native):
        features, logits, targets, mask = native(batch)
        return dict(features=features, logits=logits, targets=targets, mask=mask)
    if not isinstance(model.head, nn.Linear):
        raise ValueError("prediction_features requires model.head to be nn.Linear")
    captured = []
    hook = model.head.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
    try:
        logits, targets, mask = window_forward(model, batch)
    finally:
        hook.remove()
    if len(captured) != 1:
        raise ValueError("Expected exactly one final-head invocation")
    features = captured[0]
    if features.ndim != 3 or features.shape[0] != logits.shape[0]:
        raise ValueError("Expected final-head inputs shaped [B,L,D]")
    if features.shape[1] == logits.shape[1] + 1:
        features = features[:, 1:] if model.metadata()["architecture"].upper() == "AKT" else features[:, :-1]
    if features.shape[:2] != logits.shape:
        raise ValueError("Final-head inputs do not align with next-target logits")
    return dict(features=features, logits=logits, targets=targets, mask=mask)


def _settings(settings):
    recipe = dict(DEFAULT_SETTINGS, **(settings or {}))
    for name in ("rounds", "local_epochs", "batch_size"):
        _integer(recipe[name], name)
    for name in ("lr", "clip_norm"):
        if not math.isfinite(recipe[name]) or recipe[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if recipe["optimizer"].lower() not in {"adam", "sgd", "adagrad"}:
        raise ValueError("Unknown optimizer")
    if recipe.get("aggregation", "target_counts") not in {"target_counts", "target_count", "equal_school"}:
        raise ValueError("aggregation must be target_counts or equal_school")
    return recipe


def _clients(clients):
    if not clients:
        raise ValueError("At least one client is required")
    weights = {}
    for school, records in clients.items():
        if not isinstance(school, str) or not school:
            raise ValueError("School IDs must be nonempty strings")
        for record in records:
            n = len(record["skills"])
            if not 2 <= n <= 51:
                raise ValueError("Windows require one context and 1..50 targets; do not truncate sequences")
            if record["school_id"] != school:
                raise ValueError("Record school_id must match its client")
        weights[school] = fed.count_targets(records)
        if weights[school] <= 0:
            raise ValueError(f"Client {school} has no next-target training data")
    return weights


def _cost(cost=None):
    if cost is None:
        cost = {}
    for key in _BILL_FIELDS:
        cost.setdefault(key, 0. if key == "seconds" else 0)
    cost.setdefault("logs", [])
    return cost


def _bill(cost):
    return {key: cost[key] for key in _BILL_FIELDS}


def _snapshot(model, round_number, cost):
    return dict(round=round_number, model=model.metadata(), state_dict=state_copy(model),
                cost=_bill(cost))


def _local(model, records, settings, device, seed):
    # The loader has its own generator; also seed dropout for exact round resume.
    with fork_device_rng(seed, device):
        return fed.local_update(model, records, settings, device, seed)


def fit(model, clients, settings, device, seed, history=None, *, rounds=None,
        round_offset=0, checkpoints=None, cost=None, on_round=None):
    """Full-client target-count FedAvg, with fresh optimizer per client/round.

    ``rounds`` is this call's number of rounds (default settings['rounds']).
    ``round_offset`` is the count already completed; it controls deterministic
    client seeds as well as round numbering. Pass the existing history AND cost
    to append on resume. History rounds are zero based; logs/checkpoints use
    completed-round counts. Every archived round has real per-client deltas,
    full-data target weights and the post-aggregation global state_dict.

    ``checkpoints={1: None, 5: None, ...}`` requests cumulative round snapshots;
    snapshots contain CPU state, model metadata, and cumulative training cost.
    Communication counts simulated model broadcast + update bytes, excluding
    controller file I/O. Logs contain per-round and per-client training costs.
    After archiving/billing, on_round(model, history, cost) can persist progress;
    callback time is excluded from seconds and belongs to controller I/O cost.
    """
    recipe, device = _settings(settings), choose_device(device)
    _integer(seed, "seed", 0)
    _integer(round_offset, "round_offset", 0)
    rounds = recipe["rounds"] if rounds is None else _integer(rounds, "rounds", 0)
    if history and history[-1]["round"] != round_offset - 1:
        raise ValueError("history must end immediately before round_offset")
    target_counts, cost = _clients(clients), _cost(cost)
    weights = ({s: 1 for s in clients} if recipe.get("aggregation") == "equal_school"
               else dict(target_counts))
    model.to(device).requires_grad_(True)
    synchronize(device)
    started, previous_seconds = time.perf_counter(), cost["seconds"]
    callback_seconds = 0.
    for index in range(round_offset, round_offset + rounds):
        deltas, local_logs = {}, {}
        for school in sorted(clients):
            local_seed = fed.client_seed(seed, index, school)
            deltas[school], info = _local(model, clients[school], recipe, device, local_seed)
            local_logs[school] = dict(info, seed=local_seed, weight=weights[school],
                                     train_targets=target_counts[school])
            cost["optimizer_steps"] += info["optimizer_steps"]
            cost["targets_seen"] += info["targets_seen"]
        fed.apply_update(model, fed.aggregate(deltas, weights))
        cost["communication_bytes"] += 2 * len(clients) * parameter_bytes(model)
        synchronize(device)
        cost["seconds"] = previous_seconds + time.perf_counter() - started - callback_seconds
        cost["logs"].append(dict(round=index + 1, clients=local_logs,
            mean_client_train_bce=sum(v["loss"] for v in local_logs.values()) / len(local_logs),
            cumulative_cost=_bill(cost)))
        if history is not None:
            history.append(dict(round=index, deltas=deltas, weights=dict(weights),
                                target_counts=dict(target_counts), state_dict=state_copy(model)))
        if checkpoints is not None and index + 1 in checkpoints:
            checkpoints[index + 1] = _snapshot(model, index + 1, cost)
        if on_round is not None:
            callback_started = time.perf_counter()
            on_round(model, history, cost)
            callback_seconds += time.perf_counter() - callback_started
    synchronize(device)
    cost["seconds"] = previous_seconds + time.perf_counter() - started - callback_seconds
    return cost


def _result(model, initial_state, history, cost, checkpoints, settings, seed, method, **extra):
    # No live module/optimizer in results: controllers can serialize directly.
    return dict(model=model.metadata(), initial_state=initial_state,
                state_dict=state_copy(model), history=history, cost=cost,
                checkpoints=checkpoints, settings=dict(settings), seed=seed,
                method=method, **extra)


def prepare(model, clients, settings, device="cpu", seed=1):
    """Train Original and retain its identical init plus EVERY source round."""
    recipe = _settings(settings)
    initial_state, history = state_copy(model), []
    cost = fit(model, clients, recipe, device, seed, history)
    return _result(model, initial_state, history, cost, {}, recipe, seed, "Original")


def retrain(prepared, clients, settings=None, device="cpu", seed=None):
    """Global retrain on caller-supplied retained clients, from saved source init."""
    recipe = _settings(prepared["settings"] if settings is None else settings)
    seed = prepared["seed"] if seed is None else seed
    model = model_from_metadata(prepared["model"], seed, prepared["initial_state"])
    initial_state, history = state_copy(model), []
    cost = fit(model, clients, recipe, device, seed, history)
    return _result(model, initial_state, history, cost, {}, recipe, seed, "Retrain")


def _budgets(budgets):
    values = sorted({_integer(v, "budget") for v in budgets})
    if not values:
        raise ValueError("At least one positive round budget is required")
    return values


def recover(prepared, clients, settings=None, device="cpu", seed=None, *,
            method="RetainFT", budgets=RECOVERY_BUDGETS):
    """One shared recovery trajectory; snapshot cumulative 1/5/10/20 rounds.

    NoT negates exactly the FIRST named parameter tensor once, before recovery.
    Both methods recover with ordinary full-parameter retained-school BCE.
    Custom budgets (including the earlier 1/3/6 recipe) use the same mechanism.
    """
    if method not in {"RetainFT", "NoT"}:
        raise ValueError("method must be RetainFT or NoT")
    recipe = _settings(prepared["settings"] if settings is None else settings)
    seed = prepared["seed"] if seed is None else seed
    model = model_from_metadata(prepared["model"], seed, prepared["state_dict"])
    negated_parameter = None
    if method == "NoT":
        negated_parameter, parameter = next(model.named_parameters())
        with torch.no_grad():
            parameter.neg_()
    initial_state, history = state_copy(model), []
    checkpoints = dict.fromkeys(_budgets(budgets))
    cost = fit(model, clients, recipe, device, seed, history,
               rounds=max(checkpoints), checkpoints=checkpoints)
    return _result(model, initial_state, history, cost, checkpoints, recipe, seed,
                   method, negated_parameter=negated_parameter)


def fed_eraser(prepared, clients, settings=None, device="cpu", seed=None, *,
               calibration_fraction=.1, budgets=None):
    """Replay retained historical updates, calibrating direction from R only.

    Predeclared fractions .1 and .5 are separate runs, not validation-selected.
    Each later source round calibrates a fixed-seed ceil(fraction*N_windows)
    sample per school using the configured local passes, with minimum one
    window. Norms use the whole client delta; weights remain FULL train target
    counts. Round zero reuses cached retained deltas from the identical init,
    with zero new optimizer/communication cost. Historical preparation/storage
    costs belong to the source run and controller, not deletion training.

    Optional budgets count REPLAYED source rounds, sharing a single trajectory;
    None replays the entire source history. Early snapshots are partial replays.
    Neither withdrawn raw records nor withdrawn historical deltas are accessed.
    """
    if calibration_fraction not in CALIBRATION_FRACTIONS:
        raise ValueError("calibration_fraction must be predeclared .1 or .5")
    recipe = _settings(prepared["settings"] if settings is None else settings)
    seed = prepared["seed"] if seed is None else seed
    source_history = prepared["history"]
    checkpoints = dict.fromkeys(_budgets([len(source_history)] if budgets is None else budgets))
    if max(checkpoints) > len(source_history):
        raise ValueError("Replay budget exceeds the available source history")
    target_counts = _clients(clients)
    device = choose_device(device)
    model = model_from_metadata(prepared["model"], seed, prepared["initial_state"]).to(device)
    initial_state, history, cost = state_copy(model), [], _cost()
    synchronize(device)
    started = time.perf_counter()
    for index in range(max(checkpoints)):
        original = source_history[index]
        if original["round"] != index:
            raise ValueError("FedEraser requires consecutive full source history from round zero")
        weights = {s: original["weights"][s] for s in clients}
        deltas, local_logs = {}, {}
        for school in sorted(clients):
            if original.get("target_counts", original["weights"])[school] != target_counts[school]:
                raise ValueError("Retained clients must contain their complete original training windows")
            old = original["deltas"][school]
            if index == 0:
                deltas[school] = {name: value.detach().cpu().clone() for name, value in old.items()}
                local_logs[school] = dict(cached=True, optimizer_steps=0, targets_seen=0,
                                         calibration_windows=0, weight=weights[school])
                continue
            local_seed = fed.client_seed(seed, index, school)
            records = clients[school]
            count = max(1, math.ceil(calibration_fraction * len(records)))
            ids = np.random.default_rng(local_seed).permutation(len(records))[:count]
            sample = [records[int(i)] for i in ids]
            fresh, info = _local(model, sample, recipe, device, local_seed)
            deltas[school] = calibrate_delta(old, fresh)
            local_logs[school] = dict(info, cached=False, calibration_windows=count,
                                     seed=local_seed, weight=weights[school])
            cost["optimizer_steps"] += info["optimizer_steps"]
            cost["targets_seen"] += info["targets_seen"]
            cost["communication_bytes"] += 2 * parameter_bytes(model)
        fed.apply_update(model, fed.aggregate(deltas, weights))
        synchronize(device)
        cost["seconds"] = time.perf_counter() - started
        cost["logs"].append(dict(round=index + 1, clients=local_logs,
                                 calibration_fraction=calibration_fraction, cumulative_cost=_bill(cost)))
        history.append(dict(round=index, deltas=deltas, weights=dict(weights),
                            target_counts=dict(target_counts), state_dict=state_copy(model)))
        if index + 1 in checkpoints:
            checkpoints[index + 1] = _snapshot(model, index + 1, cost)
    cost["seconds"] = time.perf_counter() - started
    return _result(model, initial_state, history, cost, checkpoints, recipe, seed,
                   "FedEraser", calibration_fraction=calibration_fraction)


@torch.no_grad()
def predict_rows(model, clients, device="cpu", batch_size=64):
    """Per-target rows in loader order, with stable identity and raw logits."""
    device = choose_device(device)
    previous_mode = model.training
    model.to(device).eval()
    rows, seen = [], set()
    try:
        for school, records in sorted(clients.items()):
            if records:
                _clients({school: records})
            for batch in make_loader(records, batch_size, False, 0):
                logits, targets, mask = window_forward(model, move_batch(batch, device))
                logits, targets, mask = logits.cpu().double(), targets.cpu(), mask.cpu()
                probabilities = logits.sigmoid()
                if not bool(torch.isfinite(logits[mask]).all()):
                    raise FloatingPointError("Nonfinite prediction logits")
                for b, t in mask.nonzero().tolist():
                    sample_id = batch["sample_ids"][b][t + 1]
                    if not sample_id or sample_id in seen:
                        raise ValueError("Prediction target sample IDs must be nonempty and unique")
                    seen.add(sample_id)
                    rows.append(dict(sample_id=sample_id, school_id=school,
                        student_id=batch["student_ids"][b], skill_id=int(batch["skills"][b, t + 1]),
                        history=batch["histories"][b][t + 1], target=int(targets[b, t]),
                        probability=float(probabilities[b, t]), logit=float(logits[b, t])))
    finally:
        model.train(previous_mode)
    return rows


pred_rows = predict_rows


def evaluate_rows(rows, *, expected_schools=None, min_group_targets=20, min_group_students=5):
    """Macro school AUC; school -> student -> target balanced BCE and Brier.

    AUC averages schools with both classes; coverage explicitly reports how
    many qualify. target_weighted_bce pools all original targets equally;
    mean_school_bce is the previous school-macro, within-school-target mean.
    Both are separate from student_school_balanced_bce. Empty schools remain
    in per_school with None metrics and do not invent observations.
    """
    rows = list(rows)
    report = summarize_predictions(rows, min_group_targets=min_group_targets,
                                   min_group_students=min_group_students)
    observed = set(report["per_school"])
    schools = sorted(observed if expected_schools is None else set(expected_schools))
    if not observed.issubset(schools):
        raise ValueError("Observed school outside expected_schools")
    groups = {}
    for row in rows:
        groups.setdefault(row["school_id"], {}).setdefault(row["student_id"], []).append(row)
    per_school = {}
    for school in schools:
        students = groups.get(school, {})
        summary = deepcopy(report["per_school"].get(school, binary_metrics([], [])))
        student_metrics = [binary_metrics([r["target"] for r in rs], [r["probability"] for r in rs])
                           for rs in students.values()]
        for metric in ("bce", "brier"):
            summary["student_balanced_" + metric] = (
                math.fsum(m[metric] for m in student_metrics) / len(student_metrics)
                if student_metrics else None)
        summary["n_students"] = len(students)
        per_school[school] = summary

    def mean(field):
        values = [m[field] for m in per_school.values() if m[field] is not None]
        return math.fsum(values) / len(values) if values else None

    coverage = dict(report["coverage"], n_expected_schools=len(schools),
        n_observed_schools=len(observed), n_auc_schools=sum(m["auc"] is not None for m in per_school.values()),
        n_targets=len(rows), n_students=sum(m["n_students"] for m in per_school.values()),
        school_fraction=len(observed) / len(schools) if schools else 0.)
    coverage["auc_school_fraction"] = coverage["n_auc_schools"] / len(schools) if schools else 0.
    return dict(macro_auc=mean("auc"), student_school_balanced_bce=mean("student_balanced_bce"),
        student_school_balanced_brier=mean("student_balanced_brier"), mean_school_bce=mean("bce"),
        target_weighted_bce=report["overall"]["bce"], target_weighted_brier=report["overall"]["brier"],
        per_school=per_school, coverage=coverage)


def prediction_arrays(rows, *, metadata=None):
    """Return aligned non-object arrays for caller-owned np.savez_compressed.

    IDs, grouping labels, targets and probabilities accompany every prediction;
    optional run metadata is a scalar JSON string (loadable with allow_pickle=False).
    Raw logits are included when present on all rows. This function performs no I/O.
    """
    rows = list(rows)
    arrays = {}
    for key in ("sample_id", "school_id", "student_id", "history"):
        arrays[key] = np.asarray([r[key] for r in rows], dtype=str)
    for key in ("skill_id", "target"):
        arrays[key] = np.asarray([r[key] for r in rows], dtype=np.int64)
    arrays["probability"] = np.asarray([r["probability"] for r in rows], dtype=np.float64)
    if all("logit" in row for row in rows):
        arrays["logit"] = np.asarray([r["logit"] for r in rows], dtype=np.float64)
    arrays["metadata_json"] = np.asarray(json.dumps(metadata or {}, sort_keys=True, allow_nan=False))
    return arrays
