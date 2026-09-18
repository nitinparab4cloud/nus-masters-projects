# Copyright © 2026 Nitin Parab. Part of the "AI Governance Case Files" portfolio
# (https://github.com/nitinparab4cloud/nus-masters-projects/tree/main/msi5004-ai-governance-portfolio/02-fairness-auditor).
# See this project's LICENSE file for reuse terms.

"""
export_artifact.py
-------------------
Builds the JSON artifact for the static JS demo: the filtered COMPAS rows
(race, decile_score, two_year_recid) trimmed to just what the client-side
metric computation needs, plus a few precomputed reference checks so the
JS port can be verified against this script's own output.
"""

import json

from data_loader import load_compas
from fairness_metrics import compute_group_metrics, compute_headline_metrics

RACES = ["African-American", "Caucasian", "Hispanic", "Asian", "Native American", "Other"]

df = load_compas()
df = df[df["race"].isin(RACES)].reset_index(drop=True)

race_index = {r: i for i, r in enumerate(RACES)}

rows_race = [race_index[r] for r in df["race"]]
rows_score = [int(s) for s in df["decile_score"]]
rows_recid = [int(r) for r in df["two_year_recid"]]

# Reference checks: reproduce the README's headline numbers, plus one more
# pair/threshold combo, to verify the JS port against later.
checks = []

two_group = df[df["race"].isin(["African-American", "Caucasian"])]
by_group = compute_group_metrics(two_group, score_threshold=5).round(4)
headline = compute_headline_metrics(two_group, score_threshold=5)
checks.append({
    "label": "African-American vs. Caucasian, threshold=5",
    "group_a": "African-American",
    "group_b": "Caucasian",
    "threshold": 5,
    "by_group": {
        idx: {
            "selection_rate": float(row["selection_rate"]),
            "true_positive_rate": float(row["true_positive_rate"]),
            "false_positive_rate": float(row["false_positive_rate"]),
        }
        for idx, row in by_group.iterrows()
    },
    "headline": headline,
})

two_group2 = df[df["race"].isin(["Hispanic", "Asian"])]
by_group2 = compute_group_metrics(two_group2, score_threshold=7).round(4)
headline2 = compute_headline_metrics(two_group2, score_threshold=7)
checks.append({
    "label": "Hispanic vs. Asian, threshold=7",
    "group_a": "Hispanic",
    "group_b": "Asian",
    "threshold": 7,
    "by_group": {
        idx: {
            "selection_rate": float(row["selection_rate"]),
            "true_positive_rate": float(row["true_positive_rate"]),
            "false_positive_rate": float(row["false_positive_rate"]),
        }
        for idx, row in by_group2.iterrows()
    },
    "headline": headline2,
})

three_group = df[df["race"].isin(["Caucasian", "Native American", "Other"])]
by_group3 = compute_group_metrics(three_group, score_threshold=3).round(4)
headline3 = compute_headline_metrics(three_group, score_threshold=3)
checks.append({
    "label": "Native American vs. Other, threshold=3",
    "group_a": "Native American",
    "group_b": "Other",
    "threshold": 3,
    "by_group": {
        idx: {
            "selection_rate": float(row["selection_rate"]),
            "true_positive_rate": float(row["true_positive_rate"]),
            "false_positive_rate": float(row["false_positive_rate"]),
        }
        for idx, row in by_group3.iterrows()
        if idx in ("Native American", "Other")
    },
    "headline": compute_headline_metrics(
        three_group[three_group["race"].isin(["Native American", "Other"])], score_threshold=3
    ),
})

artifact = {
    "races": RACES,
    "n_rows": len(df),
    "row_race": rows_race,
    "row_score": rows_score,
    "row_recid": rows_recid,
    "checks": checks,
}

def _default(o):
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


with open("fairness_artifact.json", "w") as f:
    json.dump(artifact, f, default=_default)

print(f"Exported {len(df)} rows across {len(RACES)} race groups.")
print(json.dumps(checks, indent=2, default=_default))
