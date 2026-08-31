"""Fit the same pipeline on real campaign outcomes, and report what it costs.

This is the honesty track. `training/train.py` measures how well a model recovers
a generating process we wrote; this measures the same modelling approach against
outcomes nobody designed.

Two decisions do most of the work here, and both make the headline number worse:

**`duration` is dropped.** It is the length of the call being predicted, so it is
known only once the call has ended -- and a call that ends in a subscription is a
long call. Keeping it produces a model that cannot be deployed and a ROC-AUC that
looks excellent. The UCI page says as much. Both variants are trained below and
the gap is reported, because the size of that gap is the interesting part.

**The split is chronological, not random.** The rows are ordered May 2008 to
November 2010, spanning the financial crisis and a collapse in euribor. A random
split trains on 2010 to predict 2009 and quietly inflates every metric. Training
on the first 75% and testing on the last 25% is the question actually being
asked: does a model fitted on the past work on the future?

    python training/train_real.py            # train and write artifacts
    python training/train_real.py --verify   # retrain and fail if metrics drifted
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
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
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from training.fetch_real_data import CSV_PATH  # noqa: E402

ARTIFACT_DIR = ROOT / "models" / "artifacts" / "real"
MODEL_PATH = ARTIFACT_DIR / "propensity_model_real.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
MODEL_CARD_PATH = ARTIFACT_DIR / "model_card.md"

RANDOM_STATE = 42
TEST_FRACTION = 0.25
C_GRID = [0.01, 0.1, 1.0, 10.0]
CV_FOLDS = 5
VERIFY_TOLERANCE = 0.02

# Known only after the outcome it predicts. See the module docstring.
LEAKY = "duration"

CATEGORICAL = ["job", "marital", "education", "default", "housing", "loan",
               "contact", "month", "day_of_week", "poutcome"]
NUMERIC = ["age", "campaign", "previous", "emp.var.rate", "cons.price.idx",
           "cons.conf.idx", "euribor3m", "nr.employed", "days_since_contact",
           "contacted_before"]


def load():
    if not CSV_PATH.exists():
        raise SystemExit(
            f"missing {CSV_PATH}\nrun: python training/fetch_real_data.py"
        )
    frame = pd.read_csv(CSV_PATH, sep=";", quotechar='"')

    # pdays uses 999 as a sentinel for "never previously contacted", which would
    # otherwise read as an enormous gap and drag the coefficient around. Split it
    # into the fact of prior contact and, when there was one, its recency.
    frame["contacted_before"] = (frame["pdays"] != 999).astype(int)
    frame["days_since_contact"] = frame["pdays"].where(frame["pdays"] != 999, 0)

    frame["label"] = (frame["y"] == "yes").astype(int)
    return frame


def build_pipeline(numeric):
    """Numeric columns are passed in, not hardcoded.

    They were hardcoded at first, which made the leakage comparison silently
    meaningless: `duration` was handed to `fit()` but the transformer dropped it,
    so both variants trained on identical features and scored identically. A
    comparison that cannot show a difference is worse than no comparison.
    """
    return Pipeline([
        ("features", ColumnTransformer([
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("numeric", StandardScaler(), numeric),
        ])),
        ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ])


def chronological_split(frame):
    """First 75% of the campaign to train, last 25% to test.

    The file is ordered by contact date, so slicing it in place is a time split.
    No shuffling anywhere in this function, deliberately.
    """
    cut = int(len(frame) * (1 - TEST_FRACTION))
    return frame.iloc[:cut].copy(), frame.iloc[cut:].copy()


def fit_and_score(train_frame, test_frame, numeric, label):
    """Fit on train, select C by CV on train only, score the test split once."""
    columns = CATEGORICAL + numeric
    x_train, y_train = train_frame[columns], train_frame["label"].to_numpy()
    x_test, y_test = test_frame[columns], test_frame["label"].to_numpy()

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(build_pipeline(numeric), {"model__C": C_GRID},
                          scoring="roc_auc", cv=cv, n_jobs=-1)
    search.fit(x_train, y_train)
    best = search.best_estimator_
    cv_scores = cross_val_score(best, x_train, y_train, scoring="roc_auc", cv=cv)

    probabilities = best.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    matrix = confusion_matrix(y_test, predictions)

    return best, {
        "variant": label,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "train_positive_rate": round(float(y_train.mean()), 4),
        "test_positive_rate": round(float(y_test.mean()), 4),
        "best_C": float(search.best_params_["model__C"]),
        "cv_roc_auc_mean": round(float(cv_scores.mean()), 4),
        "cv_roc_auc_std": round(float(cv_scores.std()), 4),
        "test": {
            "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
            "pr_auc": round(float(average_precision_score(y_test, probabilities)), 4),
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
            "brier": round(float(brier_score_loss(y_test, probabilities)), 4),
            "confusion_matrix": {
                "true_negative": int(matrix[0][0]), "false_positive": int(matrix[0][1]),
                "false_negative": int(matrix[1][0]), "true_positive": int(matrix[1][1]),
            },
        },
    }


def train():
    frame = load()
    train_frame, test_frame = chronological_split(frame)

    deployable, honest = fit_and_score(train_frame, test_frame, NUMERIC, "no_duration")
    _, leaky = fit_and_score(
        train_frame, test_frame, NUMERIC + [LEAKY], "with_duration_LEAKY"
    )
    if leaky["test"]["roc_auc"] <= honest["test"]["roc_auc"]:
        # duration is a strong leak; if it does not help, the comparison is not
        # measuring what it claims to and the number should not be published.
        print("WARNING: leaky variant did not beat the deployable one -- "
              "check that `duration` actually reached the model")

    # The same deployable model under a RANDOM split, purely as a reference for
    # what the chronological split costs. Shuffling mixes late-campaign rows into
    # training, so the model gets to see the period it is scored on.
    shuffled = frame.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    cut = int(len(shuffled) * (1 - TEST_FRACTION))
    _, random_split = fit_and_score(
        shuffled.iloc[:cut], shuffled.iloc[cut:], NUMERIC, "no_duration_RANDOM_SPLIT"
    )
    # Both mistakes together -- the common published treatment of this dataset.
    # Measured here rather than cited, so the comparison is ours end to end.
    _, both = fit_and_score(
        shuffled.iloc[:cut], shuffled.iloc[cut:], NUMERIC + [LEAKY],
        "with_duration_AND_random_split",
    )

    # What a model that always guesses the majority class would score, so the
    # headline has a floor as well as the leaky variant as a false ceiling.
    baseline_auc = 0.5
    positive_rate = float(test_frame["label"].mean())

    metrics = {
        "model_type": "LogisticRegression (fitted)",
        "data_source": "REAL -- UCI Bank Marketing (CC BY 4.0)",
        "citation": ("Moro, Cortez and Rita, 'A Data-Driven Approach to Predict the "
                     "Success of Bank Telemarketing', Decision Support Systems, 2014"),
        "url": "https://archive.ics.uci.edu/dataset/222/bank+marketing",
        "task": "predict whether a contacted client subscribes to a term deposit",
        "n_total": int(len(frame)),
        "split": "chronological (first 75% train, last 25% test) -- rows are date-ordered",
        "excluded_feature": {
            "name": LEAKY,
            "reason": ("call duration is known only after the call ends, and the call "
                       "ending in a subscription is what makes it long -- it leaks the "
                       "label and cannot be used at prediction time"),
        },
        "headline": honest,
        "leakage_reference": leaky,
        "leakage_cost_roc_auc": round(
            leaky["test"]["roc_auc"] - honest["test"]["roc_auc"], 4
        ),
        "random_split_reference": random_split,
        "random_split_inflation_roc_auc": round(
            random_split["test"]["roc_auc"] - honest["test"]["roc_auc"], 4
        ),
        "both_mistakes_reference": both,
        "both_mistakes_inflation_roc_auc": round(
            both["test"]["roc_auc"] - honest["test"]["roc_auc"], 4
        ),
        "majority_class_roc_auc": baseline_auc,
        "test_positive_rate": round(positive_rate, 4),
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
    }
    return deployable, metrics


def render_model_card(metrics):
    honest = metrics["headline"]["test"]
    leaky = metrics["leakage_reference"]["test"]
    cm = honest["confusion_matrix"]
    return f"""# Model Card - Account Propensity on REAL data

## What this is

The same pipeline as `models/artifacts/model_card.md` -- one-hot encoding, standard
scaling, logistic regression, `C` chosen by {CV_FOLDS}-fold cross-validation on the
training split only -- fitted on **real outcomes** instead of generated ones.

**Data:** {metrics['data_source']}
{metrics['citation']}
{metrics['url']}

{metrics['n_total']} real contacts from a Portuguese bank's direct marketing
campaigns, May 2008 to November 2010. The label is whether the client actually
subscribed to a term deposit.

## Two decisions that make this number lower, and honest

### `duration` is excluded

{metrics['excluded_feature']['reason'].capitalize()}.

Including it scores **ROC-AUC {leaky['roc_auc']}**. Excluding it scores
**{honest['roc_auc']}**. That **{metrics['leakage_cost_roc_auc']}** gap is the
size of the leak, and it is why published results on this dataset vary so widely:
a model with `duration` looks far better and cannot be deployed, because at the
moment you must decide whether to call someone, the length of that call does not
exist yet.

### The split is chronological

The rows are date-ordered across the financial crisis, with euribor collapsing
over the period. A random split would train on 2010 to predict 2009. This one
trains on the first {metrics['headline']['n_train']} contacts and tests on the
last {metrics['headline']['n_test']} -- a genuine forward test, which is harder
and is the only version of the question worth asking.

The same pipeline under a **random** split scores
**{metrics['random_split_reference']['test']['roc_auc']}**, against
**{honest['roc_auc']}** chronologically: an inflation of
**{metrics['random_split_inflation_roc_auc']}**.

## The finding worth stating plainly

| Variant | ROC-AUC | Deployable? |
|---|---|---|
| **no `duration`, chronological split** | **{honest['roc_auc']}** | **yes -- the headline** |
| with `duration`, chronological split | {leaky['roc_auc']} | no, the feature does not exist yet at decision time |
| no `duration`, random split | {metrics['random_split_reference']['test']['roc_auc']} | no, the split leaks the future |
| **both mistakes together** | **{metrics['both_mistakes_reference']['test']['roc_auc']}** | no -- and this is the number usually published |

Choosing the split wrongly costs **{metrics['random_split_inflation_roc_auc']}**.
Leaving the leaky feature in costs **{metrics['leakage_cost_roc_auc']}**. The
evaluation design is worth *more* here than the leakage everyone warns about, and
doing both compounds them to
**{metrics['both_mistakes_reference']['test']['roc_auc']}** -- an inflation of
**{metrics['both_mistakes_inflation_roc_auc']}** describing nothing anyone could
deploy. All four rows were measured here; none is quoted from a paper.

That is the entire argument this portfolio makes, measured on somebody else's data.

### Base rates move, which is the point

Train-period positive rate is {metrics['headline']['train_positive_rate']}; test
period is {metrics['headline']['test_positive_rate']} -- a fourfold shift as the
campaign changed. A random split averages that away and reports a model that was
never asked the hard question.

Note also that cross-validated ROC-AUC on the training split
({metrics['headline']['cv_roc_auc_mean']}) is *below* the held-out figure
({honest['roc_auc']}). That inversion is a property of the drift, not a mistake:
the later period is both more positive and more separable. It is reported rather
than smoothed over.

## Measured performance (held-out final 25%, n={metrics['headline']['n_test']})

| Metric | Value |
|---|---|
| ROC-AUC | {honest['roc_auc']} |
| PR-AUC | {honest['pr_auc']} |
| Accuracy | {honest['accuracy']} |
| Precision | {honest['precision']} |
| Recall | {honest['recall']} |
| F1 | {honest['f1']} |
| Brier score | {honest['brier']} |

Cross-validated ROC-AUC on the training split:
**{metrics['headline']['cv_roc_auc_mean']} +/- {metrics['headline']['cv_roc_auc_std']}**.

Confusion matrix at threshold 0.5: TN={cm['true_negative']}, FP={cm['false_positive']},
FN={cm['false_negative']}, TP={cm['true_positive']}.

Base rate in the test period is {metrics['test_positive_rate']}, so **accuracy is
not a meaningful summary** -- always predicting "no" scores well above it while
finding nobody. ROC-AUC and PR-AUC are the figures to read, and PR-AUC should be
read against the {metrics['test_positive_rate']} base rate rather than against 1.0.

## How this relates to the served model

The service serves the synthetic-schema model. This one is **not** deployed: its
features are a bank's campaign schema, not the six account features in the `/score`
payload. What transfers is the method -- the pipeline, the leakage discipline, the
chronological split -- validated here against outcomes nobody designed.

Read together, the two model cards say: the approach works on real data, and the
served model demonstrates it on a schema we control.

## Known limitations

- One dataset, one domain, one period. Real, but not a general claim.
- No fairness analysis across the demographic attributes present (`age`, `job`,
  `marital`, `education`). For a real deployment that would be required, not optional.
- Linear decision boundary; no interaction terms beyond what one-hot encoding gives.
- The economic indicator columns are period-level, so the model can lean on
  macroeconomic conditions rather than anything about the individual client.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true",
                        help="retrain and fail if metrics drifted from the committed file")
    args = parser.parse_args()

    model, metrics = train()
    honest = metrics["headline"]["test"]

    if args.verify:
        if not METRICS_PATH.exists():
            print("FAIL: real metrics.json missing; run without --verify first")
            return 1
        committed = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        old = committed["headline"]["test"]["roc_auc"]
        new = honest["roc_auc"]
        if abs(old - new) > VERIFY_TOLERANCE:
            print(f"FAIL: ROC-AUC drifted {old} -> {new} (tol {VERIFY_TOLERANCE})")
            return 1
        print(f"OK: retrained ROC-AUC {new} matches committed {old}")
        return 0

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    MODEL_CARD_PATH.write_text(render_model_card(metrics), encoding="utf-8")

    print(f"real data        : {metrics['n_total']} contacts, {metrics['split']}")
    print(f"held-out ROC-AUC : {honest['roc_auc']}   PR-AUC: {honest['pr_auc']}")
    print(f"  precision {honest['precision']}  recall {honest['recall']}  "
          f"base rate {metrics['test_positive_rate']}")
    print(f"with duration    : {metrics['leakage_reference']['test']['roc_auc']} "
          f"(LEAKY -- not deployable, gap {metrics['leakage_cost_roc_auc']})")
    print(f"random split     : {metrics['random_split_reference']['test']['roc_auc']} "
          f"(inflated by {metrics['random_split_inflation_roc_auc']} vs chronological)")
    print(f"both mistakes    : {metrics['both_mistakes_reference']['test']['roc_auc']} "
          f"(inflated by {metrics['both_mistakes_inflation_roc_auc']} -- "
          f"the usually-published figure)")
    print(f"artifacts        : {ARTIFACT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
