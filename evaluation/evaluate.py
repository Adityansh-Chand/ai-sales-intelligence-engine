"""Score the held-out test split with the fitted model and print the metrics.

This rebuilds the exact split used at training time (same seed, same stratify,
same test size), so what is reported here is genuinely held-out data the model
never saw during fitting or hyperparameter selection.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.explainability import feature_importance  # noqa: E402
from models.scoring import load_artifacts, segment  # noqa: E402
from pipeline.features import FEATURE_ORDER  # noqa: E402
from training.train import RANDOM_STATE, TEST_SIZE  # noqa: E402


def main():
    pipeline, metadata = load_artifacts()

    frame = pd.read_csv(ROOT / "datasets" / "accounts.csv")
    x = frame[FEATURE_ORDER].to_numpy(dtype=float)
    y = frame["converted"].to_numpy(dtype=int)

    _, x_test, _, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    probabilities = pipeline.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    print("Model     :", metadata["model_type"])
    print("Data      :", metadata["data_source"], "(SYNTHETIC - not real CRM data)")
    print(f"Test rows : {len(y_test)}  positive rate {y_test.mean():.3f}")
    print()
    print("-- held-out metrics --")
    print(f"ROC-AUC   : {roc_auc_score(y_test, probabilities):.4f}")
    print(f"PR-AUC    : {average_precision_score(y_test, probabilities):.4f}")
    print(f"Accuracy  : {accuracy_score(y_test, predictions):.4f}")
    print(f"Brier     : {brier_score_loss(y_test, probabilities):.4f}")
    print()
    print(classification_report(y_test, predictions, target_names=["no_convert", "convert"]))

    matrix = confusion_matrix(y_test, predictions)
    print("Confusion matrix (rows=actual, cols=predicted)")
    print(f"              no_convert  convert")
    print(f"  no_convert  {matrix[0][0]:10d} {matrix[0][1]:8d}")
    print(f"  convert     {matrix[1][0]:10d} {matrix[1][1]:8d}")
    print()

    print("-- segment distribution on held-out set --")
    segments = [segment(float(p)) for p in probabilities]
    for name in ("high_propensity", "medium_propensity", "low_propensity"):
        mask = np.array([s == name for s in segments])
        if mask.sum():
            print(
                f"  {name:18s} n={mask.sum():4d}  actual conversion "
                f"{y_test[mask].mean():.3f}"
            )
    print()

    print("-- fitted global importance (|coef|, normalised) --")
    for name, value in sorted(
        feature_importance().items(), key=lambda kv: kv[1], reverse=True
    ):
        print(f"  {name:20s} {value:.4f}")


if __name__ == "__main__":
    main()
