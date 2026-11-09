"""Frozen whole-student balanced IN/OUT assignment used by the DKT audit."""
import hashlib
import random


def balanced_masks(dataset, calibration, evaluation):
    """Return 88 masks for ordered, disjoint sets of 64 + 64 student IDs.

    Order of the students is part of the frozen input. System groups are
    reference 0:64, calibration 64:72 and evaluation 72:88.
    """
    if dataset not in ('as09', 'as12'):
        raise ValueError('unknown dataset')
    if len(calibration) != 64 or len(evaluation) != 64:
        raise ValueError('expected 64 calibration and 64 evaluation students')
    students = list(calibration) + list(evaluation)
    if any(not isinstance(s, str) for s in students) or len(set(students)) != 128:
        raise ValueError('student IDs must be 128 distinct strings')
    masks = [{} for _ in range(88)]
    for group, lo, hi in (('reference', 0, 64), ('calibration', 64, 72), ('evaluation', 72, 88)):
        count = hi-lo
        for label, subset in (('cal', calibration), ('eval', evaluation)):
            seed = int(hashlib.sha256(f'balanced-v1:{dataset}:{group}:{label}'.encode()).hexdigest(), 16)
            rng = random.Random(seed)
            order = list(subset)
            rng.shuffle(order)
            for pos in range(0, 64, 2):
                a, b = order[pos:pos+2]
                ones = set(rng.sample(range(count), count//2))
                for k in range(count):
                    masks[lo+k][a] = int(k in ones)
                    masks[lo+k][b] = int(k not in ones)
    return masks
