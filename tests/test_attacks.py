import math
import unittest

from recokt.attacks.design import balanced_masks
from recokt.attacks.features import trajectory_features
from recokt.attacks.school import mean_response_loglik, paired_five_seed, school_feature
from recokt.attacks.student import auc, kde, normal, values


class AttackTests(unittest.TestCase):
    def test_balanced_design(self):
        cal = [f'c{i}' for i in range(64)]
        ev = [f'e{i}' for i in range(64)]
        masks = balanced_masks('as09', cal, ev)
        self.assertEqual(masks, balanced_masks('as09', cal, ev))
        for lo, hi in ((0, 64), (64, 72), (72, 88)):
            for student in cal + ev:
                self.assertEqual(sum(m[student] for m in masks[lo:hi]), (hi-lo)//2)
            for mask in masks[lo:hi]:
                self.assertEqual(sum(mask[x] for x in cal), 32)
                self.assertEqual(sum(mask[x] for x in ev), 32)
        with self.assertRaises(ValueError):
            balanced_masks('as09', cal, cal)

    def test_features_follow_true_label(self):
        value = trajectory_features([.8, .2], [1, 0])
        self.assertAlmostEqual(value['mean_loglik'], math.log(.8))
        self.assertAlmostEqual(value['mean_true_label_logodds'], math.log(.8+1e-12)-math.log(.2+1e-12))
        self.assertAlmostEqual(mean_response_loglik([.8, .2], [1, 0]), math.log(.8))
        self.assertEqual(school_feature([-1, -3]), -2)
        with self.assertRaises(ValueError):
            trajectory_features([1.1], [1])

    def test_school_paired_folds_and_ties(self):
        result = paired_five_seed([2]*5, [-2]*5)
        self.assertEqual([x['paired_auc'] for x in result['folds']], [1]*5)
        self.assertEqual(paired_five_seed([0]*5, [0]*5)['mean_paired_auc'], .5)
        with self.assertRaises(ValueError):
            paired_five_seed([1], [0])

    def test_student_scoring_basics(self):
        self.assertEqual(auc([(0, 0), (1, 0)]), .5)
        self.assertEqual(auc([(0, 0), (1, 1)]), 1)
        self.assertGreater(normal(0, 0, 1), normal(2, 0, 1))
        self.assertTrue(math.isfinite(kde(1, [1, 1])))
        self.assertEqual(values([(0, 0), (1, 1)], .5)['tpr'], 1)


if __name__ == '__main__':
    unittest.main()
