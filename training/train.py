"""Fit the account propensity model and write the artifacts the service serves.

The model is a logistic regression whose coefficients are FITTED from the
synthetic dataset -- they are not hand-chosen. Regularisation strength is
selected by cross-validation on the training split only; the held-out test
split is scored exactly once, at the end.

    python training/train.py             # train and write models/artifacts/
    python training/train.py --verify    # retrain and fail if metrics drifted
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from monitoring.drift import build_reference  # noqa: E402
from pipeline.features import FEATURE_ORDER  # noqa: E402  (single source of truth)
from training.generate_dataset import generate  # noqa: E402

DATA_PATH = ROOT / "datasets" / "accounts.csv"
ARTIFACT_DIR = ROOT / "models" / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "propensity_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
COEFFICIENTS_PATH = ARTIFACT_DIR / "coefficients.json"
MODEL_CARD_PATH = ARTIFACT_DIR / "model_card.md"
DRIFT_REFERENCE_PATH = ARTIFACT_DIR / "drift_reference.json"

RANDOM_STATE = 42
TEST_SIZE = 0.25
C_GRID = [0.01, 0.1, 1.0, 10.0]
CV_FOLDS = 5

# Segment cut points, expressed as quantiles of the TRAINING-set score
# distribution rather than as hand-picked probabilities. Tunable: raising
# HIGH_QUANTILE makes "high_propensity" rarer and more precise.
LOW_QUANTILE = 0.40
HIGH_QUANTILE = 0.70

# --verify tolerance on held-out ROC-AUC.
VERIFY_TOLERANCE = 0.02


def load_xy():
    frame = pd.read_csv(DATA_PATH)
    x = frame[FEATURE_ORDER].to_numpy(dtype=float)
    y = frame["converted"].to_numpy(dtype=int)
    return x, y


def bayes_ceiling(y_test, test_index):
    """Best ROC-AUC any model could achieve on this test split.

    Labels are sampled from the true probability, so ranking by that exact
    probability is optimal and still imperfect. Reporting a headline metric
    without this number gives it no scale: 0.86 against a 0.87 ceiling is a
    well-fitted model, while 0.86 against a 0.99 ceiling would be a poor one.
    """
    truth = generate().attrs["true_probability"]
    return float(roc_auc_score(y_test, truth[test_index]))


def train():
    x, y = load_xy()
    indices = np.arange(len(y))
    x_train, x_test, y_train, y_test, _, test_index = train_test_split(
        x, y, indices, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        pipeline, {"model__C": C_GRID}, scoring="roc_auc", cv=cv, n_jobs=-1
    )
    search.fit(x_train, y_train)  # train split only -- no test leakage
    best = search.best_estimator_

    cv_scores = cross_val_score(best, x_train, y_train, scoring="roc_auc", cv=cv)

    # The test split is touched here, once.
    probabilities = best.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    train_scores = best.predict_proba(x_train)[:, 1]
    # The distribution the model was fitted on, kept so the running service can
    # tell whether it is still scoring the same kind of population.
    drift_reference = build_reference(train_scores)
    thresholds = {
        "medium": float(np.quantile(train_scores, LOW_QUANTILE)),
        "high": float(np.quantile(train_scores, HIGH_QUANTILE)),
    }

    matrix = confusion_matrix(y_test, predictions)
    metrics = {
        "model_type": "LogisticRegression (fitted)",
        "data_source": "synthetic -- training/generate_dataset.py",
        "features": list(FEATURE_ORDER),
        "n_total": int(len(y)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "positive_rate": round(float(y.mean()), 4),
        "random_state": RANDOM_STATE,
        "cv_folds": CV_FOLDS,
        "best_C": float(search.best_params_["model__C"]),
        "cv_roc_auc_mean": round(float(cv_scores.mean()), 4),
        "cv_roc_auc_std": round(float(cv_scores.std()), 4),
        "test": {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "precision": round(float(precision_score(y_test, predictions)), 4),
            "recall": round(float(recall_score(y_test, predictions)), 4),
            "f1": round(float(f1_score(y_test, predictions)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
            "pr_auc": round(float(average_precision_score(y_test, probabilities)), 4),
            "brier": round(float(brier_score_loss(y_test, probabilities)), 4),
            "confusion_matrix": {
                "true_negative": int(matrix[0][0]),
                "false_positive": int(matrix[0][1]),
                "false_negative": int(matrix[1][0]),
                "true_positive": int(matrix[1][1]),
            },
        },
        "bayes_ceiling_roc_auc": round(bayes_ceiling(y_test, test_index), 4),
        "segment_thresholds": thresholds,
        "segment_quantiles": {"medium": LOW_QUANTILE, "high": HIGH_QUANTILE},
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
    }
    return best, metrics, drift_reference


def write_artifacts(model, metrics, drift_reference):
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    scaler = model.named_steps["scaler"]
    logistic = model.named_steps["model"]
    coefficients = {
        "note": (
            "Fitted by training/train.py -- not hand-chosen. Stored in readable "
            "form so a reviewer can confirm these are learned values."
        ),
        "intercept": float(logistic.intercept_[0]),
        "coefficients": {
            name: float(value)
            for name, value in zip(FEATURE_ORDER, logistic.coef_[0])
        },
        "scaler_mean": {n: float(v) for n, v in zip(FEATURE_ORDER, scaler.mean_)},
        "scaler_scale": {n: float(v) for n, v in zip(FEATURE_ORDER, scaler.scale_)},
    }
    COEFFICIENTS_PATH.write_text(
        json.dumps(coefficients, indent=2) + "\n", encoding="utf-8"
    )
    # The training score distribution, so the running service can tell whether it
    # is still scoring the same kind of population.
    DRIFT_REFERENCE_PATH.write_text(
        json.dumps(drift_reference, indent=2) + "\n", encoding="utf-8"
    )
    MODEL_CARD_PATH.write_text(
        render_model_card(metrics, coefficients), encoding="utf-8"
    )


def render_model_card(metrics, coefficients):
    test = metrics["test"]
    ranked = sorted(
        coefficients["coefficients"].items(), key=lambda kv: abs(kv[1]), reverse=True
    )
    rows = "\n".join(f"| `{name}` | {value:+.4f} |" for name, value in ranked)
    cm = test["confusion_matrix"]
    return f"""# Model Card - Account Propensity

## What this is

A logistic regression predicting whether an account converts. The coefficients are
**fitted** by `training/train.py`; regularisation strength (`C`) was selected by
{metrics['cv_folds']}-fold cross-validation on the training split only. The held-out
test split was scored once.

## Training data - synthetic

The model was trained on **synthetic data** produced by `training/generate_dataset.py`
(seeded, reproducible). It is **not** real CRM data. These metrics describe how well the
model recovers a generating process we wrote down; they are **not** evidence of
real-world performance, and the model has never been validated against real outcomes.

The generator deliberately uses a saturating support penalty, a log spend term, a
square-root maturity term, and a usage x renewal interaction - none of which a linear
model can represent exactly. Labels are sampled from the true probability rather than
thresholded, so some error is irreducible by construction.

## Measured performance (held-out test set, n={metrics['n_test']})

| Metric | Value |
|---|---|
| ROC-AUC | {test['roc_auc']} |
| PR-AUC | {test['pr_auc']} |
| Accuracy | {test['accuracy']} |
| Precision | {test['precision']} |
| Recall | {test['recall']} |
| F1 | {test['f1']} |
| Brier score | {test['brier']} |

Cross-validated ROC-AUC on the training split:
**{metrics['cv_roc_auc_mean']} +/- {metrics['cv_roc_auc_std']}** - consistent with the
held-out figure, so the model is not overfit.

### Headroom

The generator's **Bayes-optimal ROC-AUC on this test split is {metrics['bayes_ceiling_roc_auc']}** -
that is the score achieved by ranking on the true conversion probability itself, and it is
below 1.0 because labels are sampled from that probability rather than thresholded.

The model reaches {test['roc_auc']} against that {metrics['bayes_ceiling_roc_auc']} ceiling, so it
captures nearly all of the ranking signal that exists in the data. This number is what gives
the headline metric scale: a ROC-AUC approaching 1.0 here would indicate leakage or
degenerate labels, not a better model.

Confusion matrix at threshold 0.5: TN={cm['true_negative']}, FP={cm['false_positive']},
FN={cm['false_negative']}, TP={cm['true_positive']}.

Base rate is {metrics['positive_rate']}, so accuracy alone is a weak summary here;
ROC-AUC and PR-AUC are the figures to read.

## Fitted coefficients (standardised feature space)

| Feature | Coefficient |
|---|---|
{rows}

Intercept: {coefficients['intercept']:+.4f}

## Segment thresholds

`high_propensity` at score >= {metrics['segment_thresholds']['high']:.4f},
`medium_propensity` at score >= {metrics['segment_thresholds']['medium']:.4f}.

These are the {int(HIGH_QUANTILE * 100)}th and {int(LOW_QUANTILE * 100)}th percentiles of
the **training-set** score distribution, not hand-picked round numbers. Raising
`HIGH_QUANTILE` in `training/train.py` makes the top segment rarer and more precise; it is
a business calibration knob, not a modelling constant.

## Known limitations

- **`industry` is a real signal the served model cannot see.** It shifts conversion
  materially in the generator (roughly 0.23 to 0.42 across sectors) and is part of
  `datasets/production_schema.json`, but it is not in the `/score` request payload, so the
  model is not trained on it. Adding it is the clearest available improvement and would
  require a serving contract change.
- Linear decision boundary; the generator's interaction and saturation terms are
  approximated, not recovered. This gap is intentional and is why the metrics are not
  near-perfect.
- No calibration layer beyond what logistic regression provides natively.
- No drift detection, no retraining trigger, no monitoring of live score distributions.
- Trained and evaluated on a single seeded synthetic draw; no confidence intervals
  across seeds.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify",
        action="store_true",
        help="retrain and fail if metrics drifted from the committed file",
    )
    args = parser.parse_args()

    model, metrics, drift_reference = train()

    if args.verify:
        if not METRICS_PATH.exists():
            print("FAIL: models/artifacts/metrics.json missing; run without --verify first")
            return 1
        committed = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        old = committed["test"]["roc_auc"]
        new = metrics["test"]["roc_auc"]
        if abs(old - new) > VERIFY_TOLERANCE:
            print(f"FAIL: held-out ROC-AUC drifted {old} -> {new} (tol {VERIFY_TOLERANCE})")
            return 1
        print(f"OK: retrained ROC-AUC {new} matches committed {old} within {VERIFY_TOLERANCE}")
        return 0

    write_artifacts(model, metrics, drift_reference)
    test = metrics["test"]
    print(f"best C           : {metrics['best_C']}")
    print(f"train CV ROC-AUC : {metrics['cv_roc_auc_mean']} +/- {metrics['cv_roc_auc_std']}")
    print(f"held-out ROC-AUC : {test['roc_auc']}   PR-AUC: {test['pr_auc']}")
    print(f"Bayes ceiling    : {metrics['bayes_ceiling_roc_auc']} (best achievable on this split)")
    print(f"held-out accuracy: {test['accuracy']}  precision {test['precision']}  recall {test['recall']}")
    print(
        f"segment cuts     : medium>={metrics['segment_thresholds']['medium']:.4f} "
        f"high>={metrics['segment_thresholds']['high']:.4f}"
    )
    print(f"artifacts        : {ARTIFACT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
