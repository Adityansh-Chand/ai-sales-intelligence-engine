from models.explainability import explain
from models.scoring import predict, segment
from pipeline.features import build_features


def test_high_engagement_account_scores_high():
    features = build_features({
        "visits": 18,
        "spend": 42000,
        "account_age_days": 640,
        "usage_frequency": 88,
        "support_tickets": 1,
        "renewal_days": 30,
    })

    score = predict(features)

    assert score >= 0.70
    assert segment(score) == "high_propensity"


def test_low_activity_account_scores_low():
    features = build_features({
        "visits": 2,
        "spend": 1000,
        "account_age_days": 20,
        "usage_frequency": 5,
        "support_tickets": 6,
        "renewal_days": 300,
    })

    assert predict(features) < 0.45


def test_explanation_returns_ranked_contributions():
    features = build_features({
        "visits": 20,
        "spend": 50000,
        "account_age_days": 730,
        "usage_frequency": 100,
        "support_tickets": 0,
        "renewal_days": 10,
    })

    contributions = explain(features)

    assert contributions[0][0] in {"engagement_score", "spend_score", "usage_score"}
