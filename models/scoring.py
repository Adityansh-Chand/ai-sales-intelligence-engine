"""Serving-side scoring, backed by the fitted model artifact.

There is no hand-written formula here. `predict` runs the logistic regression
pipeline fitted by `training/train.py`, and `segment` uses cut points derived
from the training-set score distribution rather than round numbers someone
picked. If the artifact is missing, this raises rather than silently falling
back to an invented heuristic.
"""
import json
import warnings
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "models" / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "propensity_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"

_MISSING = (
    f"Model artifact not found at {MODEL_PATH}.\n"
    "Train it first:\n"
    "    python training/generate_dataset.py\n"
    "    python training/train.py"
)


@lru_cache(maxsize=1)
def load_artifacts():
    """Load the fitted pipeline and its metadata once per process."""
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        raise FileNotFoundError(_MISSING)

    import joblib
    import sklearn

    metadata = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    trained_with = metadata.get("sklearn_version")
    if trained_with and trained_with != sklearn.__version__:
        # A warning, not an error: joblib artifacts usually load across nearby
        # versions, but a mismatch is worth surfacing because it can change
        # numerical output subtly.
        warnings.warn(
            f"Model was trained with scikit-learn {trained_with} but "
            f"{sklearn.__version__} is installed. Re-run training/train.py if "
            "scores look wrong.",
            RuntimeWarning,
            stacklevel=2,
        )

    return joblib.load(MODEL_PATH), metadata


def predict(features):
    """Return P(conversion) for one feature vector in FEATURE_ORDER order."""
    pipeline, _ = load_artifacts()
    return float(pipeline.predict_proba([list(features)])[0][1])


def segment(score):
    """Bucket a score using the fitted, quantile-derived thresholds."""
    _, metadata = load_artifacts()
    thresholds = metadata["segment_thresholds"]
    if score >= thresholds["high"]:
        return "high_propensity"
    if score >= thresholds["medium"]:
        return "medium_propensity"
    return "low_propensity"


def model_metadata():
    """Metadata for the /health and /metrics surfaces."""
    _, metadata = load_artifacts()
    return {
        "model_type": metadata["model_type"],
        "data_source": metadata["data_source"],
        "trained_on": metadata["n_train"],
        "test_roc_auc": metadata["test"]["roc_auc"],
        "sklearn_version": metadata["sklearn_version"],
    }
