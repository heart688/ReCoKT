"""Five-seed paired Gaussian school-model audit from aggregate features.

This is a portable version of the main-table scoring rule in
``privacy_score_group_v1.py``. Predictions and school membership labels must
be prepared independently; this module does not train shadow models.
"""
import math
import statistics


def mean_response_loglik(probabilities, responses, epsilon=1e-12):
    """Mean true-response log likelihood for one student's fixed targets."""
    if not probabilities or len(probabilities) != len(responses):
        raise ValueError('expected nonempty aligned predictions and responses')
    values = []
    for p, y in zip(probabilities, responses):
        if y not in (0, 1) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError('invalid prediction or binary response')
        p = max(epsilon, min(1-epsilon, p))
        values.append(math.log(p if y else 1-p))
    return statistics.mean(values)


def school_feature(student_scores):
    """Equal-weight mean of per-student mean response log likelihoods."""
    values = list(student_scores)
    if not values or any(not math.isfinite(x) for x in values):
        raise ValueError('expected finite student scores')
    return statistics.mean(values)


def _fit(positive, negative):
    if len(positive) < 2 or len(positive) != len(negative):
        raise ValueError('need paired training examples in both classes')
    m1, m0 = statistics.mean(positive), statistics.mean(negative)
    v1, v0 = statistics.variance(positive), statistics.variance(negative)
    pooled = (v1 + v0) / 2
    return m1, m0, max((v1 + pooled)/2, 1e-12), max((v0 + pooled)/2, 1e-12)


def _llr(x, pars):
    m1, m0, v1, v0 = pars
    return -.5*(math.log(v1)+(x-m1)**2/v1) + .5*(math.log(v0)+(x-m0)**2/v0)


def paired_five_seed(positive, clean):
    """Leave-one-seed-out scores and paired AUC for one fixed withdrawal.

    `positive` is Pre or Post; `clean` is the corresponding independent Clean.
    Each has five school-mean log-likelihood features in the same seed order.
    """
    if len(positive) != 5 or len(clean) != 5:
        raise ValueError('expected five paired seeds')
    if any(not math.isfinite(x) for x in [*positive, *clean]):
        raise ValueError('expected finite features')
    folds = []
    for i in range(5):
        params = _fit([x for k,x in enumerate(positive) if k != i],
                      [x for k,x in enumerate(clean) if k != i])
        p, n = _llr(positive[i], params), _llr(clean[i], params)
        folds.append({'positive_score': p, 'clean_score': n,
                      'paired_auc': float(p > n) + .5 * (p == n)})
    return {'folds': folds, 'mean_paired_auc': statistics.mean(x['paired_auc'] for x in folds)}
