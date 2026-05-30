from models.scoring import normalized_features


def feature_importance():
    return {
        "engagement_score": 0.24,
        "spend_score": 0.21,
        "usage_score": 0.20,
        "maturity_score": 0.12,
        "renewal_urgency": 0.10,
        "support_penalty": -0.13,
    }


def explain(features):
    values = normalized_features(features)
    weights = feature_importance()
    contributions = {
        name: round(values[name] * weight, 4)
        for name, weight in weights.items()
    }

    return sorted(
        contributions.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
