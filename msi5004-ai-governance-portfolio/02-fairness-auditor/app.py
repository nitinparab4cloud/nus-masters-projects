"""
app.py
------
Gradio front-end for the fairness audit. Ships a working baseline: load
COMPAS, pick a score threshold and a pair of groups, see the metrics update.

TODO (Week 4, checklist item p2-5): this currently shows numbers in a table.
Add a plotly bar chart (selection rate / TPR / FPR by group) so the gap is
visible at a glance, not just readable in a table -- see the `dataviz`
guidance if you're doing this inside a Claude session.
"""

import gradio as gr
import pandas as pd

from data_loader import load_compas
from fairness_metrics import compute_group_metrics, compute_headline_metrics

_df_cache = {"df": None}


def get_df():
    if _df_cache["df"] is None:
        _df_cache["df"] = load_compas()
    return _df_cache["df"]


RACES = ["African-American", "Caucasian", "Hispanic", "Asian", "Native American", "Other"]


def run_audit(group_a: str, group_b: str, threshold: int):
    df = get_df()
    subset = df[df["race"].isin([group_a, group_b])]
    if subset["race"].nunique() < 2:
        return "Pick two different groups.", pd.DataFrame()

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
    return summary, by_group.reset_index()


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
    table_out = gr.Dataframe()

    run_btn.click(run_audit, inputs=[group_a_dd, group_b_dd, threshold_slider], outputs=[summary_out, table_out])
    demo.load(run_audit, inputs=[group_a_dd, group_b_dd, threshold_slider], outputs=[summary_out, table_out])

if __name__ == "__main__":
    demo.launch()
