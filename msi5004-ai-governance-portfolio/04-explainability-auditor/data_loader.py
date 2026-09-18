"""
data_loader.py
---------------
Loads the ProPublica COMPAS recidivism dataset and reproduces their core
filtering logic, so this audit starts from the same analytic sample as
Case 02 (the Fairness Auditor) -- not a different, unstated sample.

This is a deliberate near-duplicate of Case 02's data_loader.py rather than
an import from that folder: each project in this portfolio is self-contained
and runnable on its own (see the top-level README's "Layout" section).

ProPublica's own filters (from their methodology writeup) drop rows where:
  - the charge date is more than 30 days from arrest (data quality issue), or
  - recidivism flag is unclear (is_recid == -1), or
  - the case was an "ordinary traffic" charge, or
  - the COMPAS screening was not "Risk of Recidivism" (c_charge_degree == 'O')

Source data: https://github.com/propublica/compas-analysis
"""

import io
import urllib.request

import pandas as pd

RAW_CSV_URL = (
    "https://raw.githubusercontent.com/propublica/compas-analysis/"
    "master/compas-scores-two-years.csv"
)

# Columns needed for this audit. Note what's deliberately NOT here: `race`
# is kept only for post-hoc analysis of the explanations (see app.py), never
# as a model input -- see surrogate_model.py's docstring for why.
KEEP_COLUMNS = [
    "id", "sex", "age", "race", "juv_fel_count", "juv_misd_count",
    "juv_other_count", "priors_count", "c_charge_degree",
    "days_b_screening_arrest", "decile_score", "score_text",
    "is_recid", "two_year_recid",
]


def load_raw(csv_path: str | None = None) -> pd.DataFrame:
    """Load the raw CSV either from a local path or by fetching it."""
    if csv_path:
        return pd.read_csv(csv_path)
    with urllib.request.urlopen(RAW_CSV_URL) as resp:
        raw_bytes = resp.read()
    return pd.read_csv(io.BytesIO(raw_bytes))


def apply_propublica_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce ProPublica's analytic-sample filters (same as Case 02)."""
    filtered = df[
        (df["days_b_screening_arrest"] <= 30)
        & (df["days_b_screening_arrest"] >= -30)
        & (df["is_recid"] != -1)
        & (df["c_charge_degree"] != "O")
        & (df["score_text"] != "N/A")
    ].copy()
    return filtered


def load_compas(csv_path: str | None = None) -> pd.DataFrame:
    """One-call entry point: load, filter, and trim to the columns this audit needs."""
    raw = load_raw(csv_path)
    filtered = apply_propublica_filters(raw)
    cols = [c for c in KEEP_COLUMNS if c in filtered.columns]
    return filtered[cols].reset_index(drop=True)


if __name__ == "__main__":
    df = load_compas()
    print(f"Loaded {len(df)} rows after ProPublica's analytic-sample filters.")
    print(df[["sex", "age", "priors_count", "c_charge_degree"]].describe(include="all"))
