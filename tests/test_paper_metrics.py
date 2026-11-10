import copy
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import unittest

from recokt.paper_metrics import aggregate_cases, case_metrics, school_metrics


def make_cases():
    schools = [str(x) for x in range(20)]
    cases = []
    for seed in range(1, 6):
        for exit_index in range(5):
            # Unequal case errors distinguish mean(sqrt(error)) from sqrt(mean(error)).
            error = (.1 + .05 * exit_index) ** 2
            per = {s: {'auc': .7, 'student_balanced_acc': .6,
                       'student_balanced_brier': error}
                   for s in schools if s != schools[exit_index]}
            cases.append({'dataset': 'as09', 'architecture': 'dkt', 'split': 'test',
                          'seed': seed, 'withdrawn': schools[exit_index], 'schools': 19,
                          'bank_sha256': str(seed)*64, 'registry_sha256': str(exit_index)*64,
                          'auc': .7, 'acc': .6, 'student_balanced_brier': error,
                          'rmse': math.sqrt(error), 'per_school': per})
    return cases


class PaperMetricsTests(unittest.TestCase):
    def test_balances_students_and_schools(self):
        byschool = {
            'A': [('a', 1, .9), ('a', 0, .1), ('a', 1, .8), ('b', 1, .1)],
            'B': [('c', 1, .8), ('d', 0, .9)],
        }
        result = case_metrics(byschool)
        self.assertAlmostEqual(result['per_school']['A']['accuracy'], .75)
        self.assertAlmostEqual(result['per_school']['A']['student_balanced_acc'], .5)
        self.assertAlmostEqual(result['acc'], .5)
        self.assertAlmostEqual(result['student_balanced_brier'], .42)
        self.assertAlmostEqual(result['rmse'], math.sqrt(.42))
        self.assertIsNotNone(result['auc'])

    def test_aggregate_averages_case_rmse_before_seeds(self):
        result = aggregate_cases(make_cases())
        self.assertAlmostEqual(result['rmse_mean'], statistics.mean([.1, .15, .2, .25, .3]))
        self.assertNotAlmostEqual(result['rmse_mean'], math.sqrt(statistics.mean(x*x for x in [.1, .15, .2, .25, .3])))
        self.assertAlmostEqual(result['auc_mean'], .7)
        self.assertEqual(result['cases'], 25)
        self.assertEqual(len(result['seed_values']), 5)

    def test_rejects_invalid_predictions(self):
        for rows in ([], [('x', 1, float('nan'))], [('x', 2, .4)], [('', 0, .2)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                school_metrics(rows)

    def test_cli_writes_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = []
            for index, case in enumerate(make_cases()):
                path = root / f'case_{index}.json'
                path.write_text(json.dumps(case))
                inputs.append(str(path))
            output = root / 'summary.json'
            subprocess.run([sys.executable, '-m', 'recokt.paper_metrics', '--cases',
                            *inputs, '--output', str(output)], check=True)
            self.assertAlmostEqual(json.loads(output.read_text())['rmse_mean'], .2)

    def test_rejects_missing_duplicate_and_inconsistent_cases(self):
        cases = make_cases()
        bad_sets = [cases[:-1], cases[:-1] + [cases[0]]]
        for key, value in [('split', 'val'), ('dataset', 'as12'), ('rmse', .9), ('auc', None)]:
            bad = copy.deepcopy(cases)
            bad[0][key] = value
            bad_sets.append(bad)
        for key, value in [('bank_sha256', 'f'*64), ('registry_sha256', 'a'*64)]:
            bad = copy.deepcopy(cases)
            bad[0][key] = value
            bad_sets.append(bad)
        bad = copy.deepcopy(cases)
        bad[0]['per_school'].pop(next(iter(bad[0]['per_school'])))
        bad_sets.append(bad)
        bad = copy.deepcopy(cases)
        bad[0]['per_school'][next(iter(bad[0]['per_school']))]['student_balanced_acc'] = .9
        bad_sets.append(bad)
        for bad in bad_sets:
            with self.subTest(bad=bad[0]), self.assertRaises(ValueError):
                aggregate_cases(bad)


if __name__ == '__main__':
    unittest.main()
