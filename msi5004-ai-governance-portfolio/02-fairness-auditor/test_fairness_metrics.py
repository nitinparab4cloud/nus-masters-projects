"""
test_fairness_metrics.py
-------------------------
Regression tests for fairness_metrics.py, pinned against ProPublica's own
published 2016 finding on the African-American vs. Caucasian comparison at
the default threshold, plus general sanity properties any correct
selection-rate/TPR/FPR implementation must satisfy regardless of dataset.

Requires network access to fetch the COMPAS CSV (data_loader.py), same as
running data_loader.py or fairness_metrics.py directly.
"""

import pytest

from data_loader import load_compas
from fairness_metrics import compute_group_metrics, compute_headline_metrics


@pytest.fixture(scope="module")
def compas_df():
    return load_compas()


def test_analytic_sample_size(compas_df):
    # ProPublica's own published filtered-sample size -- if this drifts, the
    # upstream CSV or the filter logic changed and every downstream number
    # needs re-checking.
    assert len(compas_df) == 6172


def test_reproduces_propublica_headline_finding(compas_df):
    two_group = compas_df[compas_df["race"].isin(["African-American", "Caucasian"])]
    by_group = compute_group_metrics(two_group, score_threshold=5)

    aa = by_group.loc["African-American"]
    cauc = by_group.loc["Caucasian"]

    # Values from this repo's own README, itself reproducing ProPublica's
    # 2016 analysis -- tight tolerance since this is a deterministic
    # calculation over a fixed public dataset, not a trained model.
    assert aa["selection_rate"] == pytest.approx(0.576, abs=0.001)
    assert aa["true_positive_rate"] == pytest.approx(0.715, abs=0.001)
    assert aa["false_positive_rate"] == pytest.approx(0.423, abs=0.001)
    assert cauc["selection_rate"] == pytest.approx(0.331, abs=0.001)
    assert cauc["true_positive_rate"] == pytest.approx(0.504, abs=0.001)
    assert cauc["false_positive_rate"] == pytest.approx(0.220, abs=0.001)


def test_headline_metrics_disparate_impact(compas_df):
    two_group = compas_df[compas_df["race"].isin(["African-American", "Caucasian"])]
    headline = compute_headline_metrics(two_group, score_threshold=5)

    assert headline["demographic_parity_difference"] == pytest.approx(0.245, abs=0.001)
    assert headline["equalized_odds_difference"] == pytest.approx(0.212, abs=0.001)
    assert headline["disparate_impact_ratio"] == pytest.approx(0.575, abs=0.001)
    assert bool(headline["disparate_impact_flag"])  # below the conventional 0.8 threshold


def test_selection_rate_is_monotonic_in_threshold(compas_df):
    # A general property, true for any group regardless of dataset specifics:
    # raising the score-threshold-to-flag can only flag the same cases or
    # fewer, never more.
    subset = compas_df[compas_df["race"] == "Caucasian"]
    low = compute_group_metrics(subset, score_threshold=1).loc["Caucasian", "selection_rate"]
    high = compute_group_metrics(subset, score_threshold=10).loc["Caucasian", "selection_rate"]
    assert low >= high


def test_disparate_impact_ratio_is_bounded(compas_df):
    two_group = compas_df[compas_df["race"].isin(["Hispanic", "Asian"])]
    headline = compute_headline_metrics(two_group, score_threshold=5)
    assert 0.0 <= headline["disparate_impact_ratio"] <= 1.0
