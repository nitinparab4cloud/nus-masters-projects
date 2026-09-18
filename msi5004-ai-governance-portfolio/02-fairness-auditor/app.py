# Copyright © 2026 Nitin Parab. Part of the "AI Governance Case Files" portfolio
# (https://github.com/nitinparab4cloud/nus-masters-projects/tree/main/msi5004-ai-governance-portfolio/02-fairness-auditor).
# See this project's LICENSE file for reuse terms.

"""
app.py
------
Gradio front-end for the fairness audit. Load COMPAS, pick a score threshold
and a pair of groups, see the metrics update as a summary, a bar chart, and
a table.
"""

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd

from data_loader import load_compas
from fairness_metrics import compute_group_metrics, compute_headline_metrics

# Same validated categorical pair the static JS demo uses (dataviz skill,
# slots 1/2 -- see 02-fairness-auditor's index.html and its self-test).
SERIES_A_COLOR = "#2a78d6"
SERIES_B_COLOR = "#eb6834"

_df_cache = {"df": None}


def get_df():
    if _df_cache["df"] is None:
        _df_cache["df"] = load_compas()
    return _df_cache["df"]


RACES = ["African-American", "Caucasian", "Hispanic", "Asian", "Native American", "Other"]


def make_chart(by_group: pd.DataFrame, group_a: str, group_b: str):
    metrics = ["selection_rate", "true_positive_rate", "false_positive_rate"]
    labels = ["Selection rate", "True positive rate", "False positive rate"]
    a_vals = [by_group.loc[group_a, m] for m in metrics]
    b_vals = [by_group.loc[group_b, m] for m in metrics]

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    x = range(len(metrics))
    width = 0.32
    bars_a = ax.bar([i - width / 2 for i in x], a_vals, width, color=SERIES_A_COLOR, label=group_a)
    bars_b = ax.bar([i + width / 2 for i in x], b_vals, width, color=SERIES_B_COLOR, label=group_b)
    for bars in (bars_a, bars_b):
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.legend(frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def run_audit(group_a: str, group_b: str, threshold: int):
    df = get_df()
    subset = df[df["race"].isin([group_a, group_b])]
    if subset["race"].nunique() < 2:
        return "Pick two different groups.", pd.DataFrame(), None

    by_group = compute_group_metrics(subset, score_threshold=threshold).round(3)
    headline = compute_headline_metrics(subset, score_threshold=threshold)

    summary = (
        f"### {group_a} vs. {group_b} — score threshold ≥ {threshold}\n\n"
        f"- **Demographic parity difference:** {headline['demographic_parity_difference']} "
        f"(0 = identical flag rates)\n"
        f"- **Equalized odds difference:** {headline['equalized_odds_difference']} "
        f"(0 = identical error-rate profile)\n"
        f"- **Disparate impact ratio:** {headline['disparate_impact_ratio']} "
        f"({'⚠️ below the 0.8 concern threshold' if headline['disparate_impact_flag'] else 'above the 0.8 threshold'})\n"
    )
    chart = make_chart(by_group, group_a, group_b)
    return summary, by_group.reset_index(), chart


with gr.Blocks(title="Algorithmic Fairness Auditor") as demo:
    gr.Markdown(
        "# Algorithmic Fairness Auditor\n"
        "A fairness audit of the ProPublica COMPAS recidivism dataset. Pick two "
        "groups and a risk-score threshold to see selection rate, true positive "
        "rate, and false positive rate side by side."
    )
    gr.Markdown(
        "_Reproduces ProPublica's 2016 finding: at the default threshold, COMPAS's "
        "false positive rate for African-American defendants runs roughly double "
        "that for Caucasian defendants, despite similar overall accuracy._"
    )

    with gr.Row():
        group_a_dd = gr.Dropdown(RACES, value="African-American", label="Group A")
        group_b_dd = gr.Dropdown(RACES, value="Caucasian", label="Group B")
        threshold_slider = gr.Slider(1, 10, value=5, step=1, label="Flag as high-risk if decile score ≥")

    run_btn = gr.Button("Run audit", variant="primary")
    summary_out = gr.Markdown()
    chart_out = gr.Plot()
    table_out = gr.Dataframe()

    run_btn.click(
        run_audit, inputs=[group_a_dd, group_b_dd, threshold_slider], outputs=[summary_out, table_out, chart_out]
    )
    demo.load(
        run_audit, inputs=[group_a_dd, group_b_dd, threshold_slider], outputs=[summary_out, table_out, chart_out]
    )

if __name__ == "__main__":
    demo.launch()
