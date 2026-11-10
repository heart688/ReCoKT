"""Binary prediction metrics and paired comparisons."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from numbers import Integral, Real
from typing import Any, NamedTuple


__all__ = ["binary_metrics", "summarize_predictions"]

_BCE_EPSILON = 1e-15
_HISTORIES = frozenset({"00", "01", "10", "11", "short"})
_REQUIRED_FIELDS = (
    "sample_id", "school_id", "student_id", "skill_id", "history",
    "target", "probability",
)


class _Prediction(NamedTuple):
    sample_id: str
    school_id: str
    student_id: str
    skill_id: int
    history: str
    target: int
    probability: float


def _target(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value not in (0, 1):
        raise ValueError(f"{context} must be an integer target in {{0, 1}}")
    return int(value)


def _probability(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{context} must be a finite real probability in [0, 1]")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{context} must be a finite real probability in [0, 1]") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{context} must be a finite real probability in [0, 1]")
    return result


def _as_list(values: Iterable[Any], context: str) -> list[Any]:
    try:
        return list(values)
    except TypeError as exc:
        raise ValueError(f"{context} must be iterable") from exc


def _mean_bce(targets: Sequence[int], probabilities: Sequence[float]) -> float:
    # Floor the probability assigned to the observed outcome, preserving zero
    # loss for correct predictions at 0/1 and finite loss for incorrect ones.
    def loss(target: int, probability: float) -> float:
        if target:
            return -math.log(max(probability, _BCE_EPSILON))
        if 1.0 - probability < _BCE_EPSILON:
            return -math.log(_BCE_EPSILON)
        return -math.log1p(-probability)

    return math.fsum(loss(y, p) for y, p in zip(targets, probabilities)) / len(targets)


def _metrics(targets: Sequence[int], probabilities: Sequence[float]) -> dict[str, Any]:
    n = len(targets)
    if not n:
        return {"n": 0, "auc": None, "bce": None, "brier": None, "accuracy": None}

    positives = sum(targets)
    negatives = n - positives
    auc = None
    if positives and negatives:
        # Count positive/negative concordant pairs exactly, with half credit
        # for equal scores. Integer pair counts avoid rank-sum cancellation.
        ordered = sorted(zip(probabilities, targets))
        negatives_before = 0
        twice_concordant = 0
        start = 0
        while start < n:
            end = start + 1
            while end < n and ordered[end][0] == ordered[start][0]:
                end += 1
            tied_positives = sum(ordered[i][1] for i in range(start, end))
            tied_negatives = end - start - tied_positives
            twice_concordant += tied_positives * (2 * negatives_before + tied_negatives)
            negatives_before += tied_negatives
            start = end
        auc = twice_concordant / (2 * positives * negatives)

    return {
        "n": n,
        "auc": auc,
        "bce": _mean_bce(targets, probabilities),
        "brier": math.fsum((p - y) ** 2 for y, p in zip(targets, probabilities)) / n,
        "accuracy": sum((p >= 0.5) == y for y, p in zip(targets, probabilities)) / n,
    }


def binary_metrics(
    targets: Iterable[int], probabilities: Iterable[float]
) -> dict[str, int | float | None]:
    """Return ``n``, ``auc``, ``bce``, ``brier``, and ``accuracy``.

    Inputs must have equal lengths. Targets are integer 0/1; probabilities
    are finite real numbers in [0, 1]. Booleans and numeric strings are rejected.
    Invalid input raises ValueError, including NaN and infinity.

    Empty inputs have n=0 and all four metrics None. AUC is also None for a
    single class, and gives tied positive/negative scores half credit. BCE is
    the mean natural-log loss, flooring observed-outcome likelihoods at 1e-15
    so boundary probabilities give finite loss. Brier and accuracy use the
    original probabilities; accuracy predicts 1 at probability >= 0.5.
    """
    labels = _as_list(targets, "targets")
    scores = _as_list(probabilities, "probabilities")
    if len(labels) != len(scores):
        raise ValueError("targets and probabilities must have the same length")
    validated_targets = [_target(value, f"targets[{i}]") for i, value in enumerate(labels)]
    validated_probabilities = [
        _probability(value, f"probabilities[{i}]") for i, value in enumerate(scores)
    ]
    return _metrics(validated_targets, validated_probabilities)


def _validate_rows(rows: Iterable[Mapping[str, Any]], context: str) -> list[_Prediction]:
    validated = []
    seen = set()
    for index, row in enumerate(_as_list(rows, context)):
        location = f"{context}[{index}]"
        if not isinstance(row, Mapping):
            raise ValueError(f"{location} must be a mapping")
        missing = [field for field in _REQUIRED_FIELDS if field not in row]
        if missing:
            raise ValueError(f"{location} is missing required fields: {', '.join(missing)}")
        for field in ("sample_id", "school_id", "student_id"):
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f"{location}.{field} must be a nonempty string")
        if row["sample_id"] in seen:
            raise ValueError(f"{context} contains a duplicate sample_id")
        seen.add(row["sample_id"])
        skill = row["skill_id"]
        if isinstance(skill, bool) or not isinstance(skill, Integral):
            raise ValueError(f"{location}.skill_id must be an integer")
        history = row["history"]
        if not isinstance(history, str) or history not in _HISTORIES:
            raise ValueError(f"{location}.history must be '00', '01', '10', '11', or 'short'")
        validated.append(_Prediction(
            sample_id=row["sample_id"],
            school_id=row["school_id"],
            student_id=row["student_id"],
            skill_id=int(skill),
            history=history,
            target=_target(row["target"], f"{location}.target"),
            probability=_probability(row["probability"], f"{location}.probability"),
        ))
    return validated


def _students(rows: Sequence[_Prediction]) -> set[tuple[str, str]]:
    return {(row.school_id, row.student_id) for row in rows}


def _describe(
    rows: Sequence[_Prediction],
    reference: Mapping[str, _Prediction] | None,
    min_targets: int,
    min_students: int,
) -> dict[str, Any]:
    targets = [row.target for row in rows]
    probabilities = [row.probability for row in rows]
    result = _metrics(targets, probabilities)
    n_students = len(_students(rows))
    result["support"] = {
        "n_targets": len(rows),
        "n_students": n_students,
        "meets_minimum": len(rows) >= min_targets and n_students >= min_students,
    }
    if reference is not None:
        reference_probabilities = [reference[row.sample_id].probability for row in rows]
        result["bce_excess"] = (
            result["bce"] - _mean_bce(targets, reference_probabilities) if rows else None
        )
        result["probability_mae"] = (
            math.fsum(abs(p - q) for p, q in zip(probabilities, reference_probabilities))
            / len(rows) if rows else None
        )
    return result


def summarize_predictions(
    rows: Iterable[Mapping[str, Any]],
    reference_rows: Iterable[Mapping[str, Any]] | None = None,
    min_group_targets: int = 20,
    min_group_students: int = 5,
) -> dict[str, Any]:
    """Summarize predictions, observed groups, and support coverage.

    Every row requires sample_id, school_id, student_id (nonempty strings),
    skill_id (integer), history ('00', '01', '10', '11', or 'short'), target
    (integer 0/1), and probability (finite real in [0, 1]). Sample IDs must be
    unique. Inputs are never modified. Student identities are counted by
    (school_id, student_id), allowing school-local anonymized identifiers.

    The result contains:
      * overall: binary_metrics plus support.
      * per_school: school_id -> metrics plus support.
      * per_history: history -> metrics plus support.
      * per_skill_history: (skill_id, history) -> metrics plus support.
      * per_school_skill_history: (school_id, skill_id, history) -> metrics
        plus support, preserving within-school, within-skill heterogeneity.
      * coverage: each of the four group-family names maps to n_groups,
        n_supported_groups, n_targets, n_supported_targets, target_fraction,
        n_students, n_supported_students, and student_fraction.

    Support contains n_targets, n_students, and meets_minimum. Minimums must
    be nonnegative integers; both must be met. All observed groups retain
    descriptive metrics even below these minimums. Coverage fractions count
    targets or distinct students in supported groups over all targets or
    students. Students appearing in several supported groups count once.
    Empty denominators give 0.0; absent groups are omitted.

    A reference must have exactly the same unique sample IDs, labels, and
    group metadata. Order may differ. Each metrics record then also contains
    bce_excess (current mean BCE minus reference mean BCE) and probability_mae
    (mean absolute paired probability difference), both None for empty input.
    Aggregates always weight individual targets equally, including targets
    in groups below the minimums. No statistical significance is inferred.
    """
    for name, value in (
        ("min_group_targets", min_group_targets),
        ("min_group_students", min_group_students),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")

    current = _validate_rows(rows, "rows")
    reference = None
    if reference_rows is not None:
        reference = {row.sample_id: row for row in _validate_rows(reference_rows, "reference_rows")}
        if {row.sample_id for row in current} != reference.keys():
            raise ValueError("rows and reference_rows must have exactly the same sample_ids")
        for row in current:
            # All fields except probability must agree for the same sample.
            if row[:-1] != reference[row.sample_id][:-1]:
                raise ValueError("paired sample_id must have matching target and group metadata")

    families: dict[str, dict[Any, list[_Prediction]]] = {
        "per_school": {}, "per_history": {}, "per_skill_history": {},
        "per_school_skill_history": {},
    }
    for row in current:
        for family, key in (
            ("per_school", row.school_id),
            ("per_history", row.history),
            ("per_skill_history", (row.skill_id, row.history)),
            ("per_school_skill_history", (row.school_id, row.skill_id, row.history)),
        ):
            families[family].setdefault(key, []).append(row)

    result = {
        "overall": _describe(current, reference, min_group_targets, min_group_students),
        "coverage": {},
    }
    total_students = len(_students(current))
    for family, groups in families.items():
        summaries = {}
        supported_targets = 0
        supported_students: set[tuple[str, str]] = set()
        supported_groups = 0
        for key in sorted(groups):
            members = groups[key]
            summary = _describe(members, reference, min_group_targets, min_group_students)
            summaries[key] = summary
            if summary["support"]["meets_minimum"]:
                supported_groups += 1
                supported_targets += len(members)
                supported_students.update(_students(members))
        result[family] = summaries
        result["coverage"][family] = {
            "n_groups": len(groups),
            "n_supported_groups": supported_groups,
            "n_targets": len(current),
            "n_supported_targets": supported_targets,
            "target_fraction": supported_targets / len(current) if current else 0.0,
            "n_students": total_students,
            "n_supported_students": len(supported_students),
            "student_fraction": len(supported_students) / total_students if total_students else 0.0,
        }
    return result
