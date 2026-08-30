"""Generate the synthetic account dataset used to train the propensity model.

This data is SYNTHETIC. It is not real CRM data and no claim is made that the
model's measured performance transfers to real accounts. What the dataset does
provide is an honest supervised learning problem: the label-generating process
below is deliberately NOT the model that gets fitted, so the trained model has
something real to learn and cannot score perfectly.

Three properties make the resulting metrics meaningful rather than circular:

1. The true log-odds uses a saturating support penalty, a log spend term, a
   square-root maturity term, and a usage x renewal interaction. A linear model
   on the raw features cannot recover any of those exactly.
2. Labels are SAMPLED from the true probability, not thresholded, so there is
   irreducible Bayes error and 100% accuracy is impossible by construction.
3. `industry` genuinely shifts conversion but is not part of the serving
   payload, so the model is missing a real signal -- as deployed models
   usually are. See the model card.

Deterministic: fixed seed, fixed base date, no wall-clock reads.

    python training/generate_dataset.py            # write datasets/accounts.csv
    python training/generate_dataset.py --check    # fail if output would differ
"""
import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "datasets" / "accounts.csv"

SEED = 20260830
N_ACCOUNTS = 5000
BASE_DATE = date(2026, 1, 5)

# Per-industry conversion offset (log-odds) and spend scale. The offsets are the
# signal the served model cannot see, because `industry` is not in the API payload.
INDUSTRIES = {
    "saas":          {"logit": 0.72, "spend": 0.30, "weight": 0.28},
    "finance":       {"logit": 0.32, "spend": 0.55, "weight": 0.17},
    "healthcare":    {"logit": -0.16, "spend": 0.20, "weight": 0.18},
    "manufacturing": {"logit": -0.48, "spend": 0.05, "weight": 0.22},
    "retail":        {"logit": -0.72, "spend": -0.25, "weight": 0.15},
}

# Solved numerically so the positive rate lands at ~0.32 -- mild, realistic imbalance.
INTERCEPT = -6.08


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate(seed: int = SEED, n: int = N_ACCOUNTS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    names = list(INDUSTRIES)
    weights = np.array([INDUSTRIES[k]["weight"] for k in names])
    industry = rng.choice(names, size=n, p=weights / weights.sum())
    ind_logit = np.array([INDUSTRIES[k]["logit"] for k in industry])
    ind_spend = np.array([INDUSTRIES[k]["spend"] for k in industry])

    # One latent "engagement" factor drives both visits and usage, so the
    # features are correlated the way real account telemetry is.
    engagement = rng.normal(0.0, 1.0, n)

    account_age_days = np.clip(
        rng.lognormal(mean=5.7, sigma=0.75, size=n), 10, 2000
    ).astype(int)

    usage_frequency = np.clip(
        50.0 + 22.0 * engagement + rng.normal(0, 9.0, n), 0, 100
    ).round(1)

    visits = np.clip(
        rng.poisson(np.exp(2.05 + 0.50 * engagement)), 0, 60
    ).astype(int)

    spend = np.exp(
        7.0
        + 0.40 * engagement
        + 0.50 * np.log1p(account_age_days / 30.0)
        + 0.010 * usage_frequency
        + ind_spend
        + rng.normal(0, 0.55, n)
    ).round(2)

    # Unhappy, low-usage accounts file more tickets.
    support_tickets = rng.poisson(
        np.exp(1.10 - 0.020 * usage_frequency + rng.normal(0, 0.30, n))
    ).astype(int)

    renewal_days = rng.integers(1, 366, size=n)

    usage_n = usage_frequency / 100.0
    visits_n = np.clip(visits / 30.0, 0, 1)
    renewal_urgency = 1.0 - renewal_days / 365.0
    maturity = np.sqrt(np.clip(account_age_days / 730.0, 0, 1))

    # Coefficient magnitudes matter as much as their shape. An earlier draft used
    # values roughly 2.4x smaller; the true log-odds then had a standard deviation
    # of only ~0.9, which meant Bernoulli sampling washed out almost all of the
    # ranking signal and capped the Bayes-optimal AUC at 0.70. A model scoring 0.70
    # there is not a weak model -- it is a saturated one, and the metric says
    # nothing about the modelling. These magnitudes put the achievable ceiling in a
    # range where the model's own quality is what the metric actually measures.
    true_logit = (
        INTERCEPT
        + 3.60 * usage_n
        + 2.20 * np.log1p(spend) / np.log1p(60000.0)      # diminishing returns on spend
        + 1.70 * maturity                                  # sqrt, not linear
        - 3.20 * (1.0 - np.exp(-support_tickets / 3.0))    # saturating penalty
        + 2.60 * usage_n * renewal_urgency                 # interaction
        + 1.20 * visits_n
        + ind_logit                                        # invisible to the served model
        + rng.normal(0, 0.50, n)                           # unexplained variation
    )

    probability = _sigmoid(true_logit)
    converted = rng.binomial(1, probability)

    opportunity_value = np.where(
        converted == 1,
        (spend * rng.lognormal(0.15, 0.45, n)).round(2),
        0.0,
    )

    last_activity = [
        (BASE_DATE - timedelta(days=int(d))).isoformat()
        for d in rng.integers(0, 30, size=n)
    ]
    closed_at = [
        (BASE_DATE + timedelta(days=int(d))).isoformat() if c == 1 else ""
        for d, c in zip(rng.integers(1, 90, size=n), converted)
    ]

    frame = pd.DataFrame(
        {
            "account_id": [f"acct_{i:05d}" for i in range(1, n + 1)],
            "tenant_id": [f"tenant_{i % 37:03d}" for i in range(1, n + 1)],
            "industry": industry,
            "visits": visits,
            "spend": spend,
            "account_age_days": account_age_days,
            "usage_frequency": usage_frequency,
            "support_tickets": support_tickets,
            "renewal_days": renewal_days,
            "last_activity_at": last_activity,
            "converted": converted,
            "opportunity_value": opportunity_value,
            "closed_at": closed_at,
        }
    )
    # Carried separately from the emitted CSV. Training uses it to measure the
    # Bayes-optimal ceiling -- the best ROC-AUC any model could reach given that
    # labels are sampled rather than thresholded. Without that number, a headline
    # metric has no scale to be judged against.
    frame.attrs["true_probability"] = probability
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate and fail if the committed CSV would change",
    )
    args = parser.parse_args()

    frame = generate()
    text = frame.to_csv(index=False, lineterminator="\n")

    if args.check:
        if not OUT_PATH.exists():
            print(f"FAIL: {OUT_PATH} is missing; run without --check first")
            return 1
        current = OUT_PATH.read_text(encoding="utf-8")
        if current != text:
            print("FAIL: regenerated dataset differs from the committed file")
            return 1
        print(f"OK: {OUT_PATH.name} is reproducible ({len(frame)} rows)")
        return 0

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(text, encoding="utf-8")
    rate = frame["converted"].mean()
    print(f"wrote {OUT_PATH} rows={len(frame)} positive_rate={rate:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
