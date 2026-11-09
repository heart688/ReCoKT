"""Offline per-student features from a fixed causal prediction trajectory."""
import math
import statistics


def trajectory_features(probabilities, responses, epsilon=1e-12):
    """Return mean log likelihood and mean true-label log odds.

    Each probability is for the response at the same target position, using
    only the preceding history. ReCoKT route outputs must already be fused.
    """
    if not probabilities or len(probabilities) != len(responses):
        raise ValueError('expected nonempty aligned predictions and responses')
    loglik, logodds = [], []
    for p, y in zip(probabilities, responses):
        if y not in (0, 1) or not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError('invalid prediction or binary response')
        p = max(epsilon, min(1-epsilon, p))
        q = p if y else 1-p
        loglik.append(math.log(q))
        logodds.append(math.log(q+epsilon)-math.log(1-q+epsilon))
    return {'mean_loglik': statistics.mean(loglik),
            'mean_true_label_logodds': statistics.mean(logodds)}


def query_route(components, school, trajectories, withdrawn=()):
    """Score a caller-supplied cohort using the fixed ReCoKT service route.

    `trajectories` maps IDs to dictionaries containing full `skills` and
    `responses` lists. IDs and individual scores can be sensitive. This
    function does not persist its input or output.
    """
    from recokt.predict import predict_route

    result = {}
    for student, trajectory in trajectories.items():
        skills, responses = trajectory['skills'], trajectory['responses']
        predictions = predict_route(components, school, skills, responses, withdrawn)
        result[student] = trajectory_features(predictions, responses[1:])
    return result
