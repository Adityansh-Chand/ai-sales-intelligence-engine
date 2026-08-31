"""Fetch the UCI Bank Marketing dataset -- real outcomes, not generated ones.

Every other number in this repository describes how well a model recovers a
generating process we wrote ourselves. That is a real measurement of a fake
problem. This dataset is the other half: 41,188 real contacts from a Portuguese
bank's direct marketing campaigns (May 2008 - November 2010), each labelled with
whether the client actually subscribed to a term deposit.

    S. Moro, P. Cortez and P. Rita, "A Data-Driven Approach to Predict the
    Success of Bank Telemarketing", Decision Support Systems, 2014.
    https://archive.ics.uci.edu/dataset/222/bank+marketing
    Licensed CC BY 4.0.

Cached under datasets/real/ and NOT committed -- the repository stays small and
the licence terms stay simple. CI caches the same path.

    python training/fetch_real_data.py           # download and cache
    python training/fetch_real_data.py --check   # verify the cache, no network
"""
import argparse
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "datasets" / "real"
CSV_PATH = CACHE_DIR / "bank-additional-full.csv"

URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"

# The archive nests a second zip inside the first.
OUTER_MEMBER = "bank-additional.zip"
INNER_MEMBER = "bank-additional/bank-additional-full.csv"

# Verified on first download and asserted on every later one, so a silently
# changed upstream file surfaces as a checksum failure rather than as quietly
# different metrics.
EXPECTED_SHA256 = "74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8"
EXPECTED_ROWS = 41188


def download():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"downloading {URL}")
    request = urllib.request.Request(URL, headers={"User-Agent": "portfolio-fetch"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()

    with zipfile.ZipFile(io.BytesIO(payload)) as outer:
        names = outer.namelist()
        if OUTER_MEMBER in names:
            with zipfile.ZipFile(io.BytesIO(outer.read(OUTER_MEMBER))) as inner:
                data = inner.read(INNER_MEMBER)
        elif INNER_MEMBER in names:
            data = outer.read(INNER_MEMBER)
        else:
            print(f"unexpected archive layout: {names[:10]}")
            return 1

    CSV_PATH.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    rows = data.count(b"\n") - 1
    print(f"cached  {CSV_PATH}")
    print(f"sha256  {digest}")
    print(f"rows    {rows}")
    if digest != EXPECTED_SHA256:
        print(f"FAIL: checksum mismatch, expected {EXPECTED_SHA256}")
        return 1
    if rows != EXPECTED_ROWS:
        print(f"FAIL: expected {EXPECTED_ROWS} rows, got {rows}")
        return 1
    return 0


def check():
    if not CSV_PATH.exists():
        print(f"MISSING: {CSV_PATH}\nrun: python training/fetch_real_data.py")
        return 1
    data = CSV_PATH.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    rows = data.count(b"\n") - 1
    print(f"cached  {CSV_PATH}")
    print(f"sha256  {digest}")
    print(f"rows    {rows}")
    if digest != EXPECTED_SHA256:
        print(f"FAIL: checksum mismatch, expected {EXPECTED_SHA256}")
        return 1
    if rows != EXPECTED_ROWS:
        print(f"FAIL: expected {EXPECTED_ROWS} rows, got {rows}")
        return 1
    print("OK: real dataset present, checksum and row count match")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="verify the cached copy without downloading")
    args = parser.parse_args()
    return check() if args.check else download()


if __name__ == "__main__":
    sys.exit(main())
