import math


def _clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def normalized_features(features):
    visits, spend, account_age_days, usage_frequency, support_tickets, renewal_days = features
    return {
        "engagement_score": _clamp(visits / 20),
        "spend_score": _clamp(spend / 50000),
        "maturity_score": _clamp(account_age_days / 730),
        "usage_score": _clamp(usage_frequency / 100),
        "support_penalty": _clamp(support_tickets / 10),
        "renewal_urgency": _clamp(1 - (renewal_days / 365)),
    }


def predict(features):
    values = normalized_features(features)
    logit = (
        -1.25
        + 1.35 * values["engagement_score"]
        + 1.20 * values["spend_score"]
        + 0.65 * values["maturity_score"]
        + 1.10 * values["usage_score"]
        + 0.55 * values["renewal_urgency"]
        - 1.00 * values["support_penalty"]
    )
    return 1 / (1 + math.exp(-logit))


def segment(score):
    if score >= 0.70:
        return "high_propensity"
    if score >= 0.45:
        return "medium_propensity"
    return "low_propensity"
