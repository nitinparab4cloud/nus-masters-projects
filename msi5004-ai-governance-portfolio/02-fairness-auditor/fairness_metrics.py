"""
fairness_metrics.py
--------------------
Core fairness metrics for the COMPAS audit, built on `fairlearn`. This module
answers the "aggregate statistics" half of the audit -- fairness_metrics.py
covers group-level metrics; counterfactual_test.py (Week 3, still to build)
covers the individual-level test from the MSI5004 coursework.

The three metrics below map directly onto the AI Verify Testing Framework's
fairness principle and onto Domain III/IV of the AIGP body of knowledge
(bias testing as an operational governance control, not just a stated value).

Definitions used here, in plain terms:

- Demographic parity difference: does the model flag "high risk" at the same
  RATE across groups, regardless of whether the flag is accurate?
  0 = identical rates across groups; larger = bigger gap.

- Equalized odds difference: among people who did NOT reoffend, is the false
  positive rate the same across groups? (And symmetrically for true
  positives.) This is the metric at the heart of ProPublica's original
  finding -- COMPAS's aggregate accuracy was similar across race, but its
  FALSE POSITIVE rate for Black defendants was roughly double that for white
  defendants.

- Disparate impact ratio: the "80% rule" borrowed from US employment
  discrimination law -- the selection rate for the disadvantaged group
  divided by the selection rate for the advantaged group. Below 0.8 is the
  conventional (not legally binding in every context) threshold for concern.
"""

import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)


def prepare_labels(df: pd.DataFrame, score_threshold: int = 5):
    """
    Turn COMPAS's 1-10 decile_score into a binary "flagged high-risk"
    prediction, and two_year_recid into the binary ground-truth outcome.

    score_threshold=5 matches COMPAS's own Low/Medium-High cut used in the
    ProPublica analysis (decile_score >= 5 => "Medium" or "High").
    """
    y_true = df["two_year_recid"].astype(int)
    y_pred = (df["decile_score"] >= score_threshold).astype(int)
    sensitive = df["race"]
    return y_true, y_pred, sensitive


def compute_group_metrics(df: pd.DataFrame, score_threshold: int = 5) -> pd.DataFrame:
    """Per-race-group selection rate, TPR, and FPR -- the building blocks for the report."""
    y_true, y_pred, sensitive = prepare_labels(df, score_threshold)

    frame = MetricFrame(
        metrics={
            "selection_rate": selection_rate,
            "true_positive_rate": true_positive_rate,
            "false_positive_rate": false_positive_rate,
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive,
    )
    return frame.by_group


def compute_headline_metrics(df: pd.DataFrame, score_threshold: int = 5) -> dict:
    """The three headline numbers for the audit report's executive summary."""
    y_true, y_pred, sensitive = prepare_labels(df, score_threshold)

    dp_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive)
    eo_diff = equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive)

    by_group = compute_group_metrics(df, score_threshold)
    selection_rates = by_group["selection_rate"]
    disparate_impact_ratio = selection_rates.min() / selection_rates.max()

    return {
        "demographic_parity_difference": round(float(dp_diff), 4),
        "equalized_odds_difference": round(float(eo_diff), 4),
        "disparate_impact_ratio": round(float(disparate_impact_ratio), 4),
        "disparate_impact_flag": disparate_impact_ratio < 0.8,
    }


if __name__ == "__main__":
    from data_loader import load_compas

    df = load_compas()
    # ProPublica's analysis focused on the Black-vs-white comparison; restrict
    # to those two groups for a clean two-group headline read, and use the
    # full set for the by-group table.
    print("=== By-group metrics (all races) ===")
    print(compute_group_metrics(df).round(3))

    two_group = df[df["race"].isin(["African-American", "Caucasian"])]
    print("\n=== Headline metrics (African-American vs. Caucasian) ===")
    for k, v in compute_headline_metrics(two_group).items():
        print(f"{k}: {v}")
