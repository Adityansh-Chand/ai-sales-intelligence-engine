"""Per-feature attribution for a single scored account.

For a logistic regression on standardised inputs, each feature's additive
contribution to the log-odds is exactly `coefficient * standardised_value`.
That is what this returns -- a real decomposition of the model's own output,
not a hand-authored importance table. The contributions plus the intercept
reconstruct the logit exactly, which `tests/test_basic.py` asserts.
"""
from models.scoring import load_artifacts
from pipeline.features import FEATURE_ORDER


def _parts():
    pipeline, _ = load_artifacts()
    scaler = pipeline.named_steps["scaler"]
    logistic = pipeline.named_steps["model"]
    return scaler, logistic


def feature_importance():
    """Global importance: absolute fitted coefficients, normalised to sum to 1.

    Standardised inputs make these comparable across features of different units.
    """
    _, logistic = _parts()
    magnitudes = {
        name: abs(float(value))
        for name, value in zip(FEATURE_ORDER, logistic.coef_[0])
    }
    total = sum(magnitudes.values()) or 1.0
    return {name: round(value / total, 4) for name, value in magnitudes.items()}


def signed_weights():
    """Fitted coefficients with their sign, in standardised feature space."""
    _, logistic = _parts()
    return {
        name: round(float(value), 4)
        for name, value in zip(FEATURE_ORDER, logistic.coef_[0])
    }


def explain(features):
    """Rank this account's features by their contribution to the log-odds.

    Returns [(feature_name, contribution), ...] sorted by absolute contribution.
    A positive contribution pushed the score up, a negative one pulled it down.
    """
    scaler, logistic = _parts()
    values = list(features)

    contributions = []
    for index, name in enumerate(FEATURE_ORDER):
        standardised = (values[index] - scaler.mean_[index]) / scaler.scale_[index]
        contributions.append(
            (name, round(float(standardised * logistic.coef_[0][index]), 4))
        )

    return sorted(contributions, key=lambda item: abs(item[1]), reverse=True)


def explain_detail(features):
    """Explanation with the intercept and reconstructed logit included.

    Useful for verifying the decomposition is complete rather than indicative.
    """
    _, logistic = _parts()
    ranked = explain(features)
    intercept = float(logistic.intercept_[0])
    return {
        "intercept": round(intercept, 4),
        "contributions": ranked,
        "logit": round(intercept + sum(value for _, value in ranked), 4),
    }
