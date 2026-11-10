"""Paper-aligned case metrics and five-withdrawal/five-seed aggregation.

Within each school, student accuracies and Brier scores have equal weight.
Schools have equal weight; RMSE is the square root of the case-level Brier,
before averaging withdrawals and seeds. No PyTorch dependency is needed.
"""
import argparse
from collections import defaultdict
import json
import math
from numbers import Integral, Real
from pathlib import Path
import statistics
import string

from recokt_vendor.metrics import binary_metrics


def school_metrics(rows):
    """Return existing target-weighted metrics and student-balanced metrics."""
    rows = list(rows)
    if not rows:
        raise ValueError('each school needs at least one prediction')
    students = defaultdict(list)
    for row in rows:
        if len(row) != 3:
            raise ValueError('expected student, target and probability')
        student, target, probability = row
        if not isinstance(student, str) or not student.strip():
            raise ValueError('student ID must be a nonempty string')
        students[student].append((target, probability))
    # binary_metrics checks targets, probabilities, finiteness and lengths.
    overall = binary_metrics([y for _, y, _ in rows], [p for _, _, p in rows])
    student_results = [binary_metrics([y for y, _ in values], [p for _, p in values])
                       for values in students.values()]
    overall['student_balanced_acc'] = sum(x['accuracy'] for x in student_results) / len(student_results)
    overall['student_balanced_brier'] = sum(x['brier'] for x in student_results) / len(student_results)
    return overall


def case_metrics(byschool):
    if not byschool:
        raise ValueError('no schools to evaluate')
    per_school = {school: school_metrics(rows) for school, rows in byschool.items()}
    aucs = [v['auc'] for v in per_school.values()]
    brier = sum(v['student_balanced_brier'] for v in per_school.values()) / len(per_school)
    return {'auc': None if None in aucs else sum(aucs) / len(aucs),
            'acc': sum(v['student_balanced_acc'] for v in per_school.values()) / len(per_school),
            'student_balanced_brier': brier, 'rmse': math.sqrt(brier), 'per_school': per_school}


def _number(case, key):
    value = case.get(key)
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f'{key} must be a finite number')
    value = float(value)
    if not 0 <= value <= 1:
        raise ValueError(f'{key} must be within [0, 1]')
    return value


def aggregate_cases(cases):
    """Aggregate a complete, matched grid of 5 seeds x 5 withdrawal cases."""
    cases = list(cases)
    if len(cases) != 25:
        raise ValueError('expected 25 cases: five seeds and five withdrawals')
    groups = {}
    context = None
    universe = None
    bank_by_seed, registry_by_withdrawal = {}, {}
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError('case must be a mapping')
        ds, arch, split = (case.get(k) for k in ('dataset', 'architecture', 'split'))
        if any(not isinstance(x, str) or not x.strip() for x in (ds, arch, split)) or split not in ('val', 'test'):
            raise ValueError('dataset, architecture and valid split are required')
        if context is None:
            context = (ds, arch, split)
        elif context != (ds, arch, split):
            raise ValueError('cases must share dataset, architecture and split')
        seed, withdrawn = case.get('seed'), case.get('withdrawn')
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 1:
            raise ValueError('seed must be a positive integer')
        if not isinstance(withdrawn, str) or not withdrawn.strip():
            raise ValueError('withdrawn school must be a nonempty string')
        for field in ('bank_sha256', 'registry_sha256'):
            digest = case.get(field)
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in string.hexdigits for c in digest):
                raise ValueError(f'{field} must be a SHA-256 digest')
        bank = case['bank_sha256'].lower()
        registry = case['registry_sha256'].lower()
        if seed in bank_by_seed and bank_by_seed[seed] != bank:
            raise ValueError('all withdrawals for a seed must use the same bank')
        if withdrawn in registry_by_withdrawal and registry_by_withdrawal[withdrawn] != registry:
            raise ValueError('each withdrawal must use the same registry for all seeds')
        bank_by_seed[seed] = bank
        registry_by_withdrawal[withdrawn] = registry
        per_school = case.get('per_school')
        if not isinstance(per_school, dict) or len(per_school) != 19 or case.get('schools') != 19:
            raise ValueError('each case must contain the same 19 retained schools')
        if any(not isinstance(s, str) or not s.strip() for s in per_school) or withdrawn in per_school:
            raise ValueError('invalid retained-school identifiers')
        current_universe = frozenset(per_school) | {withdrawn}
        if universe is None:
            universe = current_universe
        elif universe != current_universe:
            raise ValueError('cases must use the same 20-school federation')
        auc, acc, brier, rmse = (_number(case, k) for k in ('auc', 'acc', 'student_balanced_brier', 'rmse'))
        for school in per_school.values():
            if not isinstance(school, dict):
                raise ValueError('per-school metrics must be mappings')
        for top, field in ((auc, 'auc'), (acc, 'student_balanced_acc'),
                           (brier, 'student_balanced_brier')):
            expected = statistics.mean(_number(school, field) for school in per_school.values())
            if not math.isclose(top, expected, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f'case metric does not match per-school {field}')
        if not math.isclose(rmse, math.sqrt(brier), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('case RMSE must be the square root of case Brier')
        key = (int(seed), withdrawn)
        if key in groups:
            raise ValueError('duplicate seed and withdrawal')
        groups[key] = {'auc': auc, 'acc': acc, 'rmse': rmse}
    seeds = sorted({seed for seed, _ in groups})
    withdrawals = sorted({w for _, w in groups})
    if len(seeds) != 5 or len(withdrawals) != 5 or len(groups) != len(seeds) * len(withdrawals):
        raise ValueError('expected the same five withdrawals for each of five seeds')
    seed_values = [{'seed': seed, **{metric: statistics.mean(groups[seed, w][metric] for w in withdrawals)
                                    for metric in ('auc', 'acc', 'rmse')}} for seed in seeds]
    result = {'dataset': context[0], 'architecture': context[1], 'split': context[2],
              'withdrawals': withdrawals, 'cases': len(cases), 'seed_values': seed_values}
    for metric in ('auc', 'acc', 'rmse'):
        values = [x[metric] for x in seed_values]
        result[metric + '_mean'] = statistics.mean(values)
        result[metric + '_sample_sd'] = statistics.stdev(values)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, nargs='+', required=True,
                        help='25 JSON outputs from recokt-bank evaluate')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cases = [json.loads(path.read_text()) for path in args.cases]
    result = aggregate_cases(cases)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    main()
