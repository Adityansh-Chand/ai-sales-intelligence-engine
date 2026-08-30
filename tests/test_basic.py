"""Behavioural tests for the fitted propensity model.

These assert properties that must hold for any correct model rather than exact
score values, so retraining on a new seed does not spuriously break them.
Numeric quality bars live in test_model_quality.py.
"""
import pytest

from models.explainability import explain, explain_detail, feature_importance, signed_weights
from models.scoring import load_artifacts, predict, segment
from pipeline.features import FEATURE_ORDER, build_features

STRONG_ACCOUNT = {
    "visits": 18,
    "spend": 42000,
    "account_age_days": 640,
    "usage_frequency": 88,
    "support_tickets": 1,
    "renewal_days": 30,
}
WEAK_ACCOUNT = {
    "visits": 2,
    "spend": 1000,
    "account_age_days": 20,
    "usage_frequency": 5,
    "support_tickets": 6,
    "renewal_days": 300,
}


def test_scores_are_probabilities():
    for account in (STRONG_ACCOUNT, WEAK_ACCOUNT):
        score = predict(build_features(account))
        assert 0.0 <= score <= 1.0


def test_strong_account_outranks_weak_account():
    """The core ordering property. Independent of any threshold."""
    strong = predict(build_features(STRONG_ACCOUNT))
    weak = predict(build_features(WEAK_ACCOUNT))
    assert strong > weak


def test_segments_follow_fitted_thresholds():
    _, metadata = load_artifacts()
    thresholds = metadata["segment_thresholds"]
    assert thresholds["medium"] < thresholds["high"]

    assert segment(thresholds["high"]) == "high_propensity"
    assert segment(thresholds["medium"]) == "medium_propensity"
    assert segment(thresholds["medium"] - 1e-6) == "low_propensity"


def test_segment_ordering_is_monotone():
    """A higher score can never land in a lower segment."""
    order = {"low_propensity": 0, "medium_propensity": 1, "high_propensity": 2}
    ranks = [order[segment(s / 100)] for s in range(0, 101)]
    assert ranks == sorted(ranks)


def test_more_support_tickets_never_helps():
    """Sign check against the fitted coefficient, not against an assumption."""
    assert signed_weights()["support_tickets"] < 0

    base = dict(STRONG_ACCOUNT)
    worse = dict(STRONG_ACCOUNT, support_tickets=STRONG_ACCOUNT["support_tickets"] + 5)
    assert predict(build_features(worse)) < predict(build_features(base))


def test_explanation_covers_every_feature_ranked_by_magnitude():
    contributions = explain(build_features(STRONG_ACCOUNT))

    assert [name for name, _ in contributions] != []
    assert set(name for name, _ in contributions) == set(FEATURE_ORDER)

    magnitudes = [abs(value) for _, value in contributions]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_explanation_reconstructs_the_model_logit():
    """The decomposition is exact, not indicative."""
    import math

    features = build_features(STRONG_ACCOUNT)
    detail = explain_detail(features)
    score = predict(features)

    assert detail["logit"] == pytest.approx(math.log(score / (1 - score)), abs=1e-3)


def test_feature_importance_is_a_normalised_distribution():
    importance = feature_importance()
    assert set(importance) == set(FEATURE_ORDER)
    assert all(value >= 0 for value in importance.values())
    assert sum(importance.values()) == pytest.approx(1.0, abs=1e-3)
