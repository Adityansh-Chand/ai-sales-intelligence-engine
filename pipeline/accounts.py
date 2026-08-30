"""Account lookup by id, so other services can ask about an account they only
know by identifier rather than having to carry its features around.

This is the feature-store role in miniature: the scoring service owns the
features, and callers pass an id. Backed by the generated dataset here; in a real
deployment this is where a feature store or CRM read would sit.
"""
import csv
from functools import lru_cache
from pathlib import Path

from pipeline.features import FEATURE_ORDER

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "datasets" / "accounts.csv"


@lru_cache(maxsize=1)
def _index():
    """account_id -> row. Loaded once; the dataset is small and static."""
    if not DATA_PATH.exists():
        return {}
    with DATA_PATH.open(encoding="utf-8", newline="") as handle:
        return {row["account_id"]: row for row in csv.DictReader(handle)}


def known_account_ids(limit=5):
    return sorted(_index())[:limit]


def account_count():
    return len(_index())


def lookup(account_id):
    """Return (features, metadata) for an account id, or (None, None)."""
    row = _index().get(account_id)
    if row is None:
        return None, None

    features = [float(row[name]) for name in FEATURE_ORDER]
    metadata = {
        "account_id": row["account_id"],
        "tenant_id": row.get("tenant_id"),
        # Present in the dataset and predictive, but deliberately not a model
        # input -- see the model card. Returned so callers can see it exists.
        "industry": row.get("industry"),
    }
    return features, metadata
