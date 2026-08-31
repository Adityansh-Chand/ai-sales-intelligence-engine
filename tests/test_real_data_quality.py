"""Quality gate for the real-data track.

Skips cleanly when the dataset is not cached, so a fresh clone still has a green
suite without a network call. CI fetches it, so CI does run these.

The gates that matter here are the *upper* bounds. A ROC-AUC that climbs toward
0.93 on this data does not mean the model improved -- it means `duration` crept
back in or the split stopped being chronological, which is exactly the failure
this track exists to demonstrate.

Observed at the time of writing:
    deployable, chronological : 0.7090   <- the headline
    with duration             : 0.7931
    random split              : 0.7987
    both mistakes together    : 0.9364
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.fetch_real_data import CSV_PATH  # noqa: E402

METRICS_PATH = ROOT / "models" / "artifacts" / "real" / "metrics.json"

MIN_ROC_AUC = 0.66
# Above this, something is leaking. The deployable model on a chronological split
# cannot legitimately reach the leaky variant's score.
MAX_PLAUSIBLE_ROC_AUC = 0.76
MIN_PR_AUC = 0.40


pytestmark = pytest.mark.skipif(
    not CSV_PATH.exists() or not METRICS_PATH.exists(),
    reason="real dataset not cached; run training/fetch_real_data.py then train_real.py",
)


@pytest.fixture(scope="module")
def metrics():
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def test_headline_is_the_deployable_variant(metrics):
    """The published number must be the one without the leaky feature."""
    assert metrics["headline"]["variant"] == "no_duration"
    assert metrics["excluded_feature"]["name"] == "duration"


def test_split_is_chronological(metrics):
    assert "chronological" in metrics["split"]


def test_roc_auc_within_plausible_band(metrics):
    roc_auc = metrics["headline"]["test"]["roc_auc"]
    assert roc_auc >= MIN_ROC_AUC, f"real-data ROC-AUC regressed to {roc_auc}"
    assert roc_auc <= MAX_PLAUSIBLE_ROC_AUC, (
        f"real-data ROC-AUC {roc_auc} is implausibly high for the deployable "
        "model on a chronological split -- check that `duration` is still "
        "excluded and the split is still ordered"
    )


def test_pr_auc_above_base_rate(metrics):
    pr_auc = metrics["headline"]["test"]["pr_auc"]
    base_rate = metrics["test_positive_rate"]
    assert pr_auc >= MIN_PR_AUC
    assert pr_auc > base_rate, (
        f"PR-AUC {pr_auc} is at or below the {base_rate} base rate, so the model "
        "ranks no better than chance on the positive class"
    )


def test_leakage_comparison_actually_measures_something(metrics):
    """The leaky variant must beat the honest one.

    If it does not, `duration` never reached the model and the comparison is
    decorative -- which is how it was first written, and the bug that produced
    identical scores for both variants.
    """
    honest = metrics["headline"]["test"]["roc_auc"]
    leaky = metrics["leakage_reference"]["test"]["roc_auc"]
    assert leaky > honest, "leaky variant did not beat the deployable one"
    assert metrics["leakage_cost_roc_auc"] > 0.02


def test_random_split_inflates_the_score(metrics):
    """The whole argument: choosing the split wrongly changes the number."""
    assert metrics["random_split_inflation_roc_auc"] > 0.02
    assert metrics["both_mistakes_inflation_roc_auc"] > (
        metrics["leakage_cost_roc_auc"]
    ), "combining both mistakes should inflate more than leakage alone"


def test_real_data_is_attributed(metrics):
    """CC BY 4.0 requires attribution, so the artifact carries it."""
    assert "REAL" in metrics["data_source"]
    assert metrics["citation"]
    assert metrics["url"].startswith("https://archive.ics.uci.edu/")
