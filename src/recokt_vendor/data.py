"""Auditable student-disjoint KT slices; importing this module does not import torch.

Each window contains one context event followed by at most ``max_steps`` targets.
Its first sample ID is empty: an overlapping context is never a second target.
History labels use only earlier same-KC responses *inside that window*.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import tempfile
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path


SPLITS = ("train", "val", "test")
HISTORY_LABELS = frozenset(("short", "00", "01", "10", "11"))


def _digest(*parts):
    payload = json.dumps(parts, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _positive_int(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _integer(value, minimum=0):
    try:
        number = Decimal(str(value).strip())
        if number.is_finite() and number == number.to_integral_value() and number >= minimum:
            return int(number)
    except (InvalidOperation, ValueError, TypeError):
        pass
    return None


def _binary(value):
    parsed = _integer(value)
    return parsed if parsed in (0, 1) else None


def _kc_tags(value):
    """Canonicalize the complete tag, including duplicated multi-KC rows."""
    tokens = re.split(r"[,;_]+", str(value or "").strip())
    parsed = [_integer(token) for token in tokens]
    if not parsed or any(token is None for token in parsed):
        return None
    return set(parsed)


def _split_students(students, seed, school):
    # Hash ordering avoids dependence on CSV order and consumes no response labels.
    students = sorted(students, key=lambda user: _digest("split", seed, school, user))
    n = len(students)
    if n < 3:
        return None
    n_train = min(max(1, math.floor(0.8 * n)), n - 2)
    n_val = min(max(1, math.floor(0.1 * n)), n - n_train - 1)
    return {
        "train": students[:n_train],
        "val": students[n_train:n_train + n_val],
        "test": students[n_train + n_val:],
    }


def _windows(events, school, student, max_steps):
    """events are (encoded skill, binary response, stable event identifier)."""
    records = []
    for start in range(0, len(events) - 1, max_steps):
        window = events[start:start + max_steps + 1]
        previous = defaultdict(list)
        histories = []
        for skill, response, _ in window:
            past = previous[skill]
            histories.append("".join(map(str, past[-2:])) if len(past) >= 2 else "short")
            past.append(response)
        records.append({
            "school_id": school,
            "student_id": student,
            "skills": [event[0] for event in window],
            "responses": [event[1] for event in window],
            "history": histories,
            "sample_ids": [""] + [event[2] for event in window[1:]],
        })
    return records


def _base_manifest(kind, schools, target, vocab, seed, max_steps):
    return {
        "format_version": 1,
        "data_kind": kind,
        "n_skills": len(vocab),
        "schools": list(schools),
        "target_school": target,
        "skill_vocab": dict(vocab),
        "vocab_sha256": _digest("vocabulary", sorted(vocab.items())),
        "split_seed": seed,
        "split_unit": "student",
        "split_fractions": {"train": 0.8, "val": 0.1, "test": 0.1},
        "split_rounding": "floor train/val, remainder test; at least one student per split",
        "student_id_scheme": "domain-separated SHA256 pseudonym; not a privacy guarantee",
        "max_steps": max_steps,
        "max_events_per_window": max_steps + 1,
        "window_overlap_events": 1,
        "history_scope": "preceding same-KC responses within current window only",
        "sample_id_policy": "empty at context position zero; unique stable ID per target",
    }


def make_synthetic(seed=42, n_schools=3, n_students=24, n_skills=6, length=24,
                   max_steps=50):
    """Return plumbing-only data; ``n_students`` is the count PER SCHOOL."""
    for name, value, minimum in (("n_schools", n_schools, 1), ("n_students", n_students, 3),
                                 ("n_skills", n_skills, 1), ("length", length, 2),
                                 ("max_steps", max_steps, 1)):
        _positive_int(value, name, minimum)
    schools = [f"synthetic-school-{i + 1:03d}" for i in range(n_schools)]
    vocab = {str(i): i for i in range(n_skills)}
    manifest = _base_manifest("synthetic", schools, schools[len(schools) // 2],
                              vocab, seed, max_steps)
    manifest.update(plumbing_only=True, vocab_source="fixed_synthetic_schema",
                    n_students_per_school=n_students, synthetic_length=length)
    clients = {}
    for school in schools:
        clients[school] = {split: [] for split in SPLITS}
        partition = _split_students(list(range(n_students)), seed, school)
        for split, users in partition.items():
            for user in users:
                student = _digest("synthetic-student", seed, school, user)
                rng = random.Random(_digest("synthetic-events", seed, school, user))
                events = [(rng.randrange(n_skills), rng.randrange(2),
                           _digest("synthetic-event", seed, school, user, step))
                          for step in range(length)]
                clients[school][split].extend(_windows(events, school, student, max_steps))
    bundle = {"manifest": manifest, "clients": clients}
    _validate_bundle(bundle)
    return bundle


def _read_assist(csv_path, namespace="assist2009"):
    """Read locally, retaining identifiers only in memory; return aggregate audit counts."""
    counts = Counter()
    ownership = defaultdict(set)
    events = {}
    with Path(csv_path).open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        names = {name.strip().lower(): name for name in reader.fieldnames}
        required = {"school_id", "user_id", "skill_id", "correct", "order_id"}
        if not required.issubset(names):
            raise ValueError("CSV requires school_id, user_id, skill_id, correct, order_id")
        has_original = "original" in names
        for row in reader:
            counts["input_rows"] += 1
            get = lambda key: row.get(names.get(key, ""), "")
            school = _integer(get("school_id"), minimum=1)
            user = _integer(get("user_id"), minimum=1)
            if school is None:
                counts["rejected_missing_or_nonpositive_school_rows"] += 1
                continue
            if user is None:
                counts["rejected_missing_or_nonpositive_student_rows"] += 1
                continue
            school, user = str(school), str(user)
            # Detect cross-school users BEFORE filtering labels, scaffolds or skills.
            ownership[user].add(school)
            if has_original and _binary(get("original")) != 1:
                counts["rejected_non_main_problem_rows"] += 1
                continue
            order = _integer(get("order_id"))
            if order is None:
                counts["rejected_invalid_order_rows"] += 1
                continue
            response, tags = _binary(get("correct")), _kc_tags(get("skill_id"))
            if response is None:
                counts["rejected_nonbinary_rows"] += 1
            if tags is None:
                counts["rejected_missing_or_invalid_skill_rows"] += 1
            problem = str(get("problem_id")).strip() or None
            key = (user, order)
            invalid = response is None or tags is None
            if key in events:
                counts["duplicate_event_rows_collapsed"] += 1
                event = events[key]
                conflict = (event["response"] != response or event["school"] != school
                            or (event["problem"] is not None and problem is not None
                                and event["problem"] != problem))
                event["invalid"] |= invalid or conflict
                event["conflict"] |= conflict
                event["tags"].update(tags or ())
                if event["problem"] is None:
                    event["problem"] = problem
            else:
                events[key] = {"school": school, "response": response,
                               "tags": tags or set(), "problem": problem,
                               "invalid": invalid, "conflict": False}
    cross_school = {user for user, schools in ownership.items() if len(schools) > 1}
    counts["quarantined_cross_school_students"] = len(cross_school)
    histories = defaultdict(lambda: defaultdict(list))
    for (user, order), event in events.items():
        if user in cross_school:
            counts["quarantined_cross_school_events"] += 1
            continue
        if event["invalid"]:
            counts["quarantined_invalid_events"] += 1
            counts["quarantined_conflicting_events"] += int(event["conflict"])
            continue
        tag = "|".join(str(kc) for kc in sorted(event["tags"]))
        if len(event["tags"]) > 1:
            counts["multi_kc_atomic_events"] += 1
        event_id = _digest(f"{namespace}-event", user, order)
        histories[event["school"]][user].append((order, tag, event["response"], event_id))
    for users in histories.values():
        for user in users:
            users[user].sort(key=lambda event: event[0])
    counts["observed_students_with_valid_school"] = len(ownership)
    counts["observed_positive_schools"] = len({s for schools in ownership.values() for s in schools})
    counts["usable_schools"] = len(histories)
    counts["usable_students"] = sum(len(users) for users in histories.values())
    counts["usable_events"] = sum(len(events) for users in histories.values() for events in users.values())
    return histories, dict(counts), has_original


def prepare_assist(csv_path, output_dir, split_seed=42, max_steps=50, n_schools=3,
                   target_school=None, max_students_per_school=120,
                   min_student_events=2, min_school_students=3,
                   min_school_train_targets=1, dataset_kind="assist2009_corrected",
                   source_release=None):
    """Prepare and save an explicit adaptation of corrected ASSIST2009 CSV data.

    Candidate clients need >=3 eligible students and at least one training target.
    Unlike FDKT, no 1,000-event threshold is imposed for this bounded pilot. Rank
    clients by TRAIN target support after a label-independent student cap. All
    counts are computed after main-problem/binary/identity/duplicate repairs.
    Unknown evaluation KCs break sequences; no transition bridges a dropped event.
    Returns a full bundle. No records or raw student IDs are logged.
    """
    _positive_int(max_steps, "max_steps")
    _positive_int(n_schools, "n_schools")
    _positive_int(min_student_events, "min_student_events", 2)
    _positive_int(min_school_students, "min_school_students", 3)
    _positive_int(min_school_train_targets, "min_school_train_targets")
    if max_students_per_school is not None:
        _positive_int(max_students_per_school, "max_students_per_school", 3)
    namespaces = {"assist2009_corrected": "assist2009", "assist2012_school": "assist2012"}
    if dataset_kind not in namespaces:
        raise ValueError("Unsupported ASSIST source kind")
    namespace = namespaces[dataset_kind]
    histories, audit, has_original = _read_assist(csv_path, namespace=namespace)
    partitions, support = {}, {}
    for school, users in histories.items():
        eligible = [user for user, events in users.items() if len(events) >= min_student_events]
        audit["excluded_short_students"] = audit.get("excluded_short_students", 0) + len(users) - len(eligible)
        eligible.sort(key=lambda user: _digest("student-cap", split_seed, school, user))
        if max_students_per_school is not None:
            audit["students_excluded_by_cap"] = audit.get("students_excluded_by_cap", 0) + max(0, len(eligible) - max_students_per_school)
            eligible = eligible[:max_students_per_school]
        partition = (_split_students(eligible, split_seed, school)
                     if len(eligible) >= min_school_students else None)
        if partition is not None:
            n_targets = sum(len(users[user]) - 1 for user in partition["train"])
            if n_targets >= min_school_train_targets:
                partitions[school] = partition
                support[school] = n_targets
    ranked = sorted(partitions, key=lambda school: (-support[school], int(school)))
    audit["candidate_schools_with_three_students"] = len(ranked)
    if len(ranked) < n_schools:
        raise ValueError(f"Need {n_schools} eligible real schools; found {len(ranked)}")
    requested_target = None
    if target_school is not None:
        parsed_target = _integer(target_school, minimum=1)
        requested_target = str(parsed_target) if parsed_target is not None else None
        if requested_target not in partitions:
            raise ValueError("target_school must be an eligible observed real school")
    selected = ranked[:n_schools]
    if requested_target is not None and requested_target not in selected:
        selected = [requested_target] + ranked[:n_schools - 1]
    selected.sort(key=lambda school: (-support[school], int(school)))
    target = requested_target or selected[len(selected) // 2]
    tags = sorted({event[1] for school in selected
                   for user in partitions[school]["train"]
                   for event in histories[school][user]})
    vocab = {tag: index for index, tag in enumerate(tags)}
    manifest = _base_manifest(dataset_kind, selected, target, vocab,
                              split_seed, max_steps)
    with Path(csv_path).open("rb") as source:
        source_hash = hashlib.file_digest(source, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else None
    if source_hash is None:
        digest = hashlib.sha256()
        with Path(csv_path).open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        source_hash = digest.hexdigest()
    manifest.update({
        "plumbing_only": False,
        "source_file_sha256": source_hash,
        "source_release": source_release or "caller-supplied corrected ASSISTments 2009-2010 CSV; variant not inferred",
        "csv_encoding": "utf-8-sig with replacement of invalid bytes; numeric IDs/tags validated",
        "correctness_policy": "observed binary correct column (first-attempt outcome); never filter attempt_count==1",
        "main_problem_policy": "original==1" if has_original else "original field absent; not filtered",
        "event_identity": "(user_id, order_id); merge KC tags; quarantine conflicting outcomes/problem IDs",
        "multi_kc_policy": "complete sorted KC tag set is one atomic skill; never expand one event",
        "vocab_source": "selected_train_only",
        "unknown_eval_policy": "drop unknown events and break sequences; drop resulting singleton segments",
        "client_selection": {"rank_by": "training_next_response_target_count",
                             "training_target_support": {s: support[s] for s in selected},
                             "max_students_per_school": max_students_per_school,
                             "student_cap_policy": ("no student cap" if max_students_per_school is None
                                                    else "deterministic identifier hash, before split"),
                             "minimum_students_per_school": min_school_students,
                             "minimum_events_per_student": min_student_events,
                             "minimum_training_targets_per_school": min_school_train_targets,
                             "minimum_event_threshold": None,
                             "target_forced": requested_target is not None},
        "adaptations": ["student-disjoint 80/10/10, including validation",
                        "complete multi-KC sets treated as atomic skills",
                        "main problems only when original is available",
                        "cross-school students quarantined globally",
                        "train-only vocabulary; unknown evaluation events break sequences",
                        ("no student cap; training-support client selection" if max_students_per_school is None
                         else "bounded student cap and training-support client selection; no 1000-event cutoff"),
                        "50 targets per window with one overlapping context event"],
        "audit": audit,
    })
    clients = {}
    evaluations = {}
    student_counts = {}
    for school in selected:
        clients[school] = {split: [] for split in SPLITS}
        evaluations[school] = {}
        student_counts[school] = {}
        for split in SPLITS:
            counts = Counter(students_before_encoding=len(partitions[school][split]))
            for user in partitions[school][split]:
                raw_events = histories[school][user]
                student = _digest(f"{namespace}-student", user)
                segments, current = [], []
                for _, tag, response, event_id in raw_events:
                    if tag not in vocab:
                        counts["unknown_events_dropped"] += 1
                        if current:
                            segments.append(current)
                            current = []
                    else:
                        current.append((vocab[tag], response, event_id))
                if current:
                    segments.append(current)
                targets = sum(max(0, len(segment) - 1) for segment in segments)
                counts["original_targets"] += len(raw_events) - 1
                counts["emitted_targets"] += targets
                counts["dropped_targets"] += len(raw_events) - 1 - targets
                counts["known_singleton_segments_dropped"] += sum(len(segment) == 1 for segment in segments)
                counts["students_with_targets"] += int(targets > 0)
                for segment in segments:
                    clients[school][split].extend(_windows(segment, school, student, max_steps))
            student_counts[school][split] = len(partitions[school][split])
            evaluations[school][split] = {key: counts[key] for key in (
                "students_before_encoding", "students_with_targets", "unknown_events_dropped",
                "original_targets", "emitted_targets", "dropped_targets", "known_singleton_segments_dropped")}
    manifest["student_counts"] = student_counts
    manifest["encoding_counts"] = evaluations
    bundle = {"manifest": manifest, "clients": clients}
    save_bundle(bundle, output_dir)
    return bundle


def _validate_record(record, school, n_skills):
    if record.get("school_id") != school:
        raise ValueError("Record/client school mismatch")
    if not re.fullmatch(r"[0-9a-f]{64}", record.get("student_id", "")):
        raise ValueError("student_id must be a SHA256 pseudonym")
    fields = ("skills", "responses", "history", "sample_ids")
    if any(not isinstance(record.get(key), list) for key in fields):
        raise ValueError("Record sequences must be lists")
    length = len(record["skills"])
    if length < 2 or any(len(record[key]) != length for key in fields):
        raise ValueError("Record sequences must align and contain context plus a target")
    previous = defaultdict(list)
    for skill, response, history in zip(record["skills"], record["responses"], record["history"]):
        if type(skill) is not int or not 0 <= skill < n_skills:
            raise ValueError("Skill outside immutable vocabulary")
        if type(response) is not int or response not in (0, 1):
            raise ValueError("Responses must be observed binary integers")
        past = previous[skill]
        expected = "".join(map(str, past[-2:])) if len(past) >= 2 else "short"
        if history != expected:
            raise ValueError("History must use only preceding same-KC responses in this window")
        past.append(response)
    ids = record["sample_ids"]
    if ids[0] != "" or any(not isinstance(value, str) or not value for value in ids[1:]):
        raise ValueError("Only context position zero may have an empty sample ID")


def _validate_bundle(bundle, require_all=True):
    manifest, clients = bundle["manifest"], bundle["clients"]
    if manifest.get("format_version") != 1 or manifest.get("data_kind") not in ("synthetic", "assist2009_corrected", "assist2012_school"):
        raise ValueError("Unsupported data manifest")
    _positive_int(manifest.get("n_skills"), "n_skills")
    schools = manifest.get("schools")
    if not isinstance(schools, list) or not all(isinstance(s, str) and s for s in schools) or len(set(schools)) != len(schools):
        raise ValueError("Manifest schools must be unique nonempty strings")
    if manifest.get("target_school") not in schools:
        raise ValueError("target_school must belong to the source manifest")
    if (require_all and set(clients) != set(schools)) or not set(clients).issubset(schools):
        raise ValueError("Manifest/client mismatch")
    if "skill_vocab" in manifest:
        vocab = manifest["skill_vocab"]
        if sorted(vocab.values()) != list(range(manifest["n_skills"])):
            raise ValueError("Vocabulary must be a contiguous fixed mapping")
        if manifest.get("vocab_sha256", _digest("vocabulary", sorted(vocab.items()))) != _digest("vocabulary", sorted(vocab.items())):
            raise ValueError("Vocabulary fingerprint mismatch")
    owners, targets = {}, set()
    for school, splits in clients.items():
        if set(splits) != set(SPLITS):
            raise ValueError("Every loaded client requires train, val, test")
        for split, records in splits.items():
            for record in records:
                _validate_record(record, school, manifest["n_skills"])
                if "max_steps" in manifest and len(record["skills"]) > manifest["max_steps"] + 1:
                    raise ValueError("Window exceeds max_steps targets")
                student = record["student_id"]
                if owners.setdefault(student, (school, split)) != (school, split):
                    raise ValueError("Student leakage across schools or splits")
                for sample_id in record["sample_ids"][1:]:
                    if sample_id in targets:
                        raise ValueError("Duplicate next-response target")
                    targets.add(sample_id)


def _atomic_text(path, write):
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=".write-", delete=False) as handle:
            name = Path(handle.name)
            write(handle)
        name.replace(path)
    finally:
        if name is not None and name.exists():
            name.unlink()


def save_bundle(bundle, path):
    """Write manifest.json and independent clients/<hash>/<split>.jsonl files.

    Partial bundles may be loaded for retention-only work but cannot overwrite a
    complete dataset: save requires all manifest schools. This never fits a vocab.
    """
    _validate_bundle(bundle)
    root = Path(path)
    if (root / "manifest.json").exists():
        raise FileExistsError("Prepared split already exists; use a new output directory")
    files = {}
    for school in bundle["manifest"]["schools"]:
        files[school] = {}
        for split in SPLITS:
            relative = Path("clients") / _digest("school-path", school) / f"{split}.jsonl"
            files[school][split] = relative.as_posix()
            def write_rows(handle, records=bundle["clients"][school][split]):
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n")
            _atomic_text(root / relative, write_rows)
    manifest = dict(bundle["manifest"], client_files=files)
    _atomic_text(root / "manifest.json", lambda handle: json.dump(manifest, handle, indent=2, sort_keys=True))


def load_bundle(path, schools=None):
    """Read ONLY requested clients; the source manifest/vocabulary stay unchanged.

    ``schools=[]`` reads only manifest.json. No glob, stat, existence check, or open
    is performed on excluded client files. Unknown requested schools are errors.
    """
    root = Path(path).resolve()
    with (root / "manifest.json").open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    selected = list(manifest["schools"] if schools is None else schools)
    if len(set(selected)) != len(selected) or any(s not in manifest["schools"] for s in selected):
        raise ValueError("Requested schools must be unique source school strings")
    clients = {}
    for school in selected:
        clients[school] = {}
        for split in SPLITS:
            relative = Path(manifest["client_files"][school][split])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Client file path must stay inside the bundle")
            file_path = (root / relative).resolve()
            if not file_path.is_relative_to(root):
                raise ValueError("Client file path must stay inside the bundle")
            with file_path.open(encoding="utf-8") as handle:
                clients[school][split] = [json.loads(line) for line in handle if line.strip()]
    bundle = {"manifest": manifest, "clients": clients}
    _validate_bundle(bundle, require_all=schools is None)
    return bundle


def _collate_records(records):
    import torch

    length = max(len(record["skills"]) for record in records)
    shape = (len(records), length)
    batch = {"skills": torch.zeros(shape, dtype=torch.long),
             "responses": torch.zeros(shape, dtype=torch.float32),
             "mask": torch.zeros(shape, dtype=torch.bool),
             "school_ids": [], "student_ids": [], "histories": [], "sample_ids": []}
    for index, record in enumerate(records):
        n = len(record["skills"])
        batch["skills"][index, :n] = torch.tensor(record["skills"], dtype=torch.long)
        batch["responses"][index, :n] = torch.tensor(record["responses"], dtype=torch.float32)
        batch["mask"][index, :n] = True
        batch["school_ids"].append(record["school_id"])
        batch["student_ids"].append(record["student_id"])
        batch["histories"].append(record["history"] + [""] * (length - n))
        batch["sample_ids"].append(record["sample_ids"] + [""] * (length - n))
    return batch


def make_loader(records, batch_size, shuffle, seed):
    """Return a reproducibly shuffled CPU torch DataLoader (zero workers).

    mask marks real EVENTS, including context. Targets use positions 1: and their
    nonempty sample IDs. Response tensors are float32, skills int64, mask bool.
    """
    import torch
    from torch.utils.data import DataLoader

    _positive_int(batch_size, "batch_size")
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(records, batch_size=batch_size, shuffle=bool(shuffle and len(records)),
                      generator=generator, collate_fn=_collate_records, num_workers=0)
