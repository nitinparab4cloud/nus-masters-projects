# Copyright © 2026 Nitin Parab. Part of the "AI Governance Case Files" portfolio
# (https://github.com/nitinparab4cloud/nus-masters-projects/tree/main/msi5004-ai-governance-portfolio/02-fairness-auditor).
# See this project's LICENSE file for reuse terms.

"""
data_loader.py
---------------
Loads the ProPublica COMPAS recidivism dataset and reproduces their core
filtering logic, so the fairness audit starts from the same analytic sample
ProPublica used -- not the raw, noisier table.

ProPublica's own filters (from their methodology writeup) drop rows where:
  - the charge date is more than 30 days from arrest (data quality issue), or
  - recidivism flag is unclear (is_recid == -1), or
  - the case was an "ordinary traffic" charge, or
  - the COMPAS screening was not "Risk of Recidivism" (c_charge_degree == 'O')

Source data: https://github.com/propublica/compas-analysis
(also mirrored on Hugging Face -- see load_from_huggingface() below)
"""

import io
import urllib.request

import pandas as pd

RAW_CSV_URL = (
    "https://raw.githubusercontent.com/propublica/compas-analysis/"
    "master/compas-scores-two-years.csv"
)

# Columns actually needed for the audit -- the raw file has 50+ columns of
# which most are COMPAS's own intermediate fields we don't need.
KEEP_COLUMNS = [
    "id", "sex", "age", "age_cat", "race", "juv_fel_count", "juv_misd_count",
    "juv_other_count", "priors_count", "c_charge_degree", "days_b_screening_arrest",
    "decile_score", "score_text", "is_recid", "two_year_recid",
]


def load_raw(csv_path: str | None = None) -> pd.DataFrame:
    """Load the raw CSV either from a local path or by fetching it."""
    if csv_path:
        return pd.read_csv(csv_path)
    with urllib.request.urlopen(RAW_CSV_URL) as resp:
        raw_bytes = resp.read()
    return pd.read_csv(io.BytesIO(raw_bytes))


def load_from_huggingface():
    """
    Alternative loader using the Hugging Face `datasets` library, useful if
    you want everything -- data, audit code, and the demo -- to live in the
    Hugging Face ecosystem end to end.

        pip install datasets
        from data_loader import load_from_huggingface
        df = load_from_huggingface()
    """
    from datasets import load_dataset

    ds = load_dataset("mstz/compas", split="train")
    return ds.to_pandas()


def apply_propublica_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce ProPublica's analytic-sample filters."""
    filtered = df[
        (df["days_b_screening_arrest"] <= 30)
        & (df["days_b_screening_arrest"] >= -30)
        & (df["is_recid"] != -1)
        & (df["c_charge_degree"] != "O")
        & (df["score_text"] != "N/A")
    ].copy()
    return filtered


def load_compas(csv_path: str | None = None) -> pd.DataFrame:
    """One-call entry point: load, filter, and trim to the columns the audit needs."""
    raw = load_raw(csv_path)
    filtered = apply_propublica_filters(raw)
    cols = [c for c in KEEP_COLUMNS if c in filtered.columns]
    return filtered[cols].reset_index(drop=True)


if __name__ == "__main__":
    df = load_compas()
    print(f"Loaded {len(df)} rows after ProPublica's analytic-sample filters.")
    print(df["race"].value_counts())
