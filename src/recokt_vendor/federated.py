from copy import deepcopy
import math
import time

import torch

from .data import make_loader
from .runtime import move_batch, parameter_bytes, state_copy, synchronize


def count_targets(records):
    return sum(len(record["skills"]) - 1 for record in records)


def client_seed(seed, round_index, school):
    # Stable across processes; unlike Python's randomized hash().
    import hashlib
    token = f"{seed}:{round_index}:{school}".encode()
    return int(hashlib.sha256(token).hexdigest()[:8], 16) % (2**31)


def local_update(model, records, settings, device, seed):
    if not records or count_targets(records) == 0:
        raise ValueError("A training client has no next-response targets")
    local = deepcopy(model).to(device)
    local.train()
    optimizer_name = settings.get("optimizer", "adam").lower()
    optimizers = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD,
                  "adagrad": torch.optim.Adagrad}
    if optimizer_name not in optimizers:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    optimizer = optimizers[optimizer_name](local.parameters(), lr=settings["lr"])
    total_loss, targets, batches = 0.0, 0, 0
    loader = make_loader(records, settings["batch_size"], True, seed)
    # Optimizer state resets for each client/communication round, explicitly.
    for _ in range(settings["local_epochs"]):
        for batch in loader:
            batch = move_batch(batch, device)
            losses, mask = local.losses(batch)
            n = int(mask.sum())
            if n == 0:
                continue
            loss = losses[mask].mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(local.parameters(),
                                                  settings.get("clip_norm", 5.0))
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("Nonfinite gradient norm")
            optimizer.step()
            total_loss += float(loss.detach()) * n
            targets += n
            batches += 1
    before, after = state_copy(model), state_copy(local)
    delta = {name: after[name] - before[name] for name in before}
    return delta, {"loss": total_loss / max(targets, 1),
                   "targets_seen": targets, "optimizer_steps": batches,
                   "compute_device": str(device),
                   "parameter_devices": sorted({str(p.device) for p in local.parameters()}),
                   "gradient_devices": sorted({str(p.grad.device) for p in local.parameters()
                                                if p.grad is not None})}


def aggregate(deltas, weights):
    if set(deltas) != set(weights) or not deltas:
        raise ValueError("Nonempty client deltas and weights must have identical keys")
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Aggregation weights must be positive")
    first = next(iter(deltas.values()))
    return {name: sum(deltas[school][name] * (weights[school] / total)
                      for school in deltas) for name in first}


def apply_update(model, delta):
    state = state_copy(model)
    for name in state:
        if delta[name].shape != state[name].shape:
            raise ValueError(f"Update shape mismatch: {name}")
        state[name] += delta[name]
        if not bool(torch.isfinite(state[name]).all()):
            raise FloatingPointError(f"Nonfinite updated parameter: {name}")
    model.load_state_dict(state)


def train_federated(model, clients, settings, device, seed, store_interval=None,
                    callback=None):
    """Single-process FL simulation: records never enter aggregate().

    Each school independently trains from the broadcast model; the server
    receives model deltas and target counts. This is NOT a secured FL deployment.
    """
    rounds = settings["rounds"]
    weights = {school: count_targets(records) for school, records in clients.items()}
    history, trace = [], []
    cost = {"optimizer_steps": 0, "targets_seen": 0, "communication_bytes": 0,
            "probe_backward_calls": 0, "validation_seconds": 0.0}
    synchronize(device)
    started = time.perf_counter()
    for index in range(rounds):
        deltas, losses = {}, []
        for school in sorted(clients):
            delta, info = local_update(model, clients[school], settings, device,
                                       client_seed(seed, index, school))
            deltas[school] = delta
            losses.append(info["loss"])
            cost["optimizer_steps"] += info["optimizer_steps"]
            cost["targets_seen"] += info["targets_seen"]
        if store_interval is not None and index % store_interval == 0:
            history.append({"round": index, "deltas": deltas,
                            "weights": dict(weights)})
        apply_update(model, aggregate(deltas, weights))
        cost["communication_bytes"] += 2 * len(clients) * parameter_bytes(model)
        row = {"round": index + 1, "mean_client_train_bce": sum(losses) / len(losses)}
        trace.append(row)
        if callback:
            validation_started = time.perf_counter()
            callback(model, row)
            synchronize(device)
            cost["validation_seconds"] += time.perf_counter() - validation_started
    synchronize(device)
    cost["seconds"] = time.perf_counter() - started
    cost["training_seconds_excluding_validation"] = cost["seconds"] - cost["validation_seconds"]
    return history, trace, cost


@torch.no_grad()
def predict(model, clients, device, batch_size=64):
    rows = []
    model.eval()
    for school, records in sorted(clients.items()):
        for batch in make_loader(records, batch_size, False, 0):
            tensors = move_batch(batch, device)
            logits, targets, mask = model.next_response(tensors)
            probabilities = torch.sigmoid(logits).cpu()
            targets, mask = targets.cpu(), mask.cpu()
            skills = batch["skills"]
            for b, t in mask.nonzero().tolist():
                rows.append({"sample_id": batch["sample_ids"][b][t + 1],
                             "school_id": str(school),
                             "student_id": batch["student_ids"][b],
                             "skill_id": int(skills[b, t + 1]),
                             "history": batch["histories"][b][t + 1],
                             "target": int(targets[b, t]),
                             "probability": float(probabilities[b, t])})
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicated prediction targets: check chunk boundaries")
    if any(not math.isfinite(r["probability"]) for r in rows):
        raise FloatingPointError("Nonfinite prediction")
    return rows
