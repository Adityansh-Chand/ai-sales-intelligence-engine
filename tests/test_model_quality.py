"""Quality gate: fails if the shipped model regresses.

Metrics are recomputed from the artifact against the held-out split rather than
trusted from metrics.json, so a stale or corrupted metrics file cannot make a
broken model look healthy. Bars sit comfortably below observed values so the
suite does not flake, but high enough that a real regression trips them.

Observed at the time of writing: ROC-AUC 0.861, PR-AUC 0.758, accuracy 0.798.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.scoring import load_artifacts  # noqa: E402
from pipeline.features import FEATURE_ORDER  # noqa: E402
from training.train import RANDOM_STATE, TEST_SIZE  # noqa: E402

MIN_ROC_AUC = 0.80
MIN_PR_AUC = 0.68
MIN_ACCURACY = 0.75
# The generator's Bayes-optimal ROC-AUC on this split is ~0.890 (computed by
# training/train.py). Scoring near or above that means the split leaked or the
# labels stopped being sampled.
MAX_PLAUSIBLE_ROC_AUC = 0.93


@pytest.fixture(scope="module")
def held_out():
    pipeline, metadata = load_artifacts()
    frame = pd.read_csv(ROOT / "datasets" / "accounts.csv")
    x = frame[FEATURE_ORDER].to_numpy(dtype=float)
    y = frame["converted"].to_numpy(dtype=int)
    _, x_test, _, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    scores = pipeline.predict_proba(x_test)[:, 1]
    return metadata, y_test, scores


def test_roc_auc_meets_bar(held_out):
    _, y_test, scores = held_out
    assert roc_auc_score(y_test, scores) >= MIN_ROC_AUC


def test_pr_auc_meets_bar(held_out):
    _, y_test, scores = held_out
    assert average_precision_score(y_test, scores) >= MIN_PR_AUC


def test_accuracy_meets_bar(held_out):
    _, y_test, scores = held_out
    assert accuracy_score(y_test, (scores >= 0.5).astype(int)) >= MIN_ACCURACY


def test_performance_is_not_implausibly_good(held_out):
    """Guards against the failure mode this rebuild existed to remove."""
    _, y_test, scores = held_out
    assert roc_auc_score(y_test, scores) < MAX_PLAUSIBLE_ROC_AUC


def test_committed_metrics_match_the_artifact(held_out):
    """metrics.json must describe the model actually shipped."""
    metadata, y_test, scores = held_out
    recomputed = roc_auc_score(y_test, scores)
    assert recomputed == pytest.approx(metadata["test"]["roc_auc"], abs=0.005)


def test_model_was_trained_on_a_real_split(held_out):
    metadata, _, _ = held_out
    assert metadata["n_train"] > metadata["n_test"] > 0
    assert metadata["n_train"] + metadata["n_test"] == metadata["n_total"]


def test_data_provenance_is_declared_as_synthetic(held_out):
    """The honesty claim is load-bearing, so it is tested."""
    metadata, _, _ = held_out
    assert "synthetic" in metadata["data_source"].lower()
