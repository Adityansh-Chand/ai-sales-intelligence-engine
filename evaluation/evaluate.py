
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.scoring import predict
from pipeline.features import build_features


df = pd.read_csv(ROOT / "datasets" / "sample_data.csv")
correct = 0
for row in df.to_dict("records"):
    prediction = int(predict(build_features(row)) >= 0.5)
    correct += prediction == int(row["converted"])

print("records:", len(df))
print("accuracy:", correct / len(df))
