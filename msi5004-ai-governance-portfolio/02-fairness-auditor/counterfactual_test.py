# Copyright © 2026 Nitin Parab. Part of the "AI Governance Case Files" portfolio
# (https://github.com/nitinparab4cloud/nus-masters-projects/tree/main/msi5004-ai-governance-portfolio/02-fairness-auditor).
# See this project's LICENSE file for reuse terms.

"""
counterfactual_test.py
-----------------------
STUB -- this is Case 02's Week 3 milestone (checklist item p2-3 in the
portfolio tracker). The aggregate metrics in fairness_metrics.py answer
"is the ERROR RATE different across groups, on average." This file answers
a different, individually-scoped question straight out of the MSI5004
coursework's fairness section:

    "Would this specific person's outcome change if their race changed,
     and nothing else did?"

That's a stronger, more legible test for a governance audience than another
aggregate statistic -- it produces a concrete "yes, for N out of M cases,
swapping race alone flips the risk score" finding rather than a percentage
that needs a stats background to interpret.

Two honest ways to build this against a real dataset (COMPAS has no
counterfactual-twin ground truth, so pick one and say so in the report):

  Option A -- Matched pairs (no modeling required, fastest to build):
      For each African-American defendant, find the Caucasian defendant(s)
      with the closest priors_count, age, charge degree, and sex. Compare
      decile_score between the matched pair. This is an *approximation* of
      a counterfactual, not a true one -- say so in the report.

  Option B -- Model-based (more rigorous, more work):
      Train a simple surrogate model that predicts decile_score from the
      other features *including* race, then re-run every row with race
      flipped and compare predictions. This directly answers "does race
      carry independent predictive weight after controlling for the other
      features," which is closer to what COMPAS's own vendor would need to
      answer to defend the tool.

Start with Option A -- it's honest about being an approximation, ships in a
day, and is easy to explain in an interview. Option B is a good "if I had
another week" line in the report.
"""

import pandas as pd


def find_matched_pairs(
    df: pd.DataFrame,
    group_a: str = "African-American",
    group_b: str = "Caucasian",
    match_on: tuple = ("priors_count", "age", "sex", "c_charge_degree"),
) -> pd.DataFrame:
    """
    TODO (Week 3): for each row in group_a, find the closest row in group_b
    on the `match_on` features (exact match on categorical fields like sex
    and c_charge_degree; nearest-neighbour on numeric fields like
    priors_count and age -- consider `sklearn.neighbors.NearestNeighbors`).

    Return a DataFrame with one row per matched pair:
        [group_a_id, group_b_id, group_a_decile_score, group_b_decile_score,
         score_gap, priors_count, age, sex, c_charge_degree]

    Then in the report, summarise:
      - What fraction of matched pairs have a score_gap >= 2 (a materially
        different risk band)?
      - In which direction does the gap skew?
    """
    raise NotImplementedError("Build the nearest-neighbour matching here -- see docstring.")


def summarize_counterfactual_gaps(matched_pairs: pd.DataFrame, gap_threshold: int = 2) -> dict:
    """
    TODO (Week 3): once find_matched_pairs() works, this just needs:
        material_gap = matched_pairs[matched_pairs["score_gap"].abs() >= gap_threshold]
        return {
            "n_pairs": len(matched_pairs),
            "n_material_gap": len(material_gap),
            "pct_material_gap": len(material_gap) / len(matched_pairs),
            "mean_gap_direction": matched_pairs["score_gap"].mean(),
        }
    """
    raise NotImplementedError("Implement once find_matched_pairs() is working.")


if __name__ == "__main__":
    print(__doc__)
