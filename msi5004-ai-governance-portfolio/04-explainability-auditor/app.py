# Copyright © 2026 Nitin Parab. Part of the "AI Governance Case Files" portfolio
# (https://github.com/nitinparab4cloud/nus-masters-projects/tree/main/msi5004-ai-governance-portfolio/04-explainability-auditor).
# See this project's LICENSE file for reuse terms.

"""
app.py
------
Gradio front-end for the Explainability Auditor. Two ways in: explain a real
case from the COMPAS dataset by row index, or explain a hypothetical profile
you build yourself with the sliders.

Local run:   python app.py
HF Spaces:   push this file + the other .py files + requirements.txt to a
             Space with SDK "gradio" (see README -- same free-tier caveat as
             the rest of this portfolio applies).
"""

import gradio as gr

from data_loader import load_compas
from explain_engine import explain_instance, explain_row, render_plain_language
from surrogate_model import FEATURE_LABELS, FEATURE_NAMES, train_surrogate

_cache = {"df": None, "surrogate": None}


def get_state():
    if _cache["df"] is None:
        df = load_compas()
        _cache["df"] = df
        _cache["surrogate"] = train_surrogate(df)
    return _cache["df"], _cache["surrogate"]


DISCLAIMER = (
    "**This explains a surrogate model's approximation of COMPAS, not COMPAS's actual "
    "proprietary logic** — no outside party has access to that. See the Methodology "
    "section in the README for why that's the honest thing an outside auditor can do, "
    "and the fidelity number below for how well the approximation holds up. Race is "
    "never used as a model input; where shown, it's context only."
)


def explain_real_case(row_index: int):
    df, surrogate = get_state()
    row_index = int(row_index)
    if row_index < 0 or row_index >= len(df):
        return f"Row index must be between 0 and {len(df) - 1}.", ""
    row = df.iloc[row_index]
    exp = explain_row(surrogate, row)

    context = (
        f"### Case #{row_index}\n\n"
        f"- Race (context only, not a model input): **{row['race']}**\n"
        f"- COMPAS's own decile score: **{int(row['decile_score'])}** "
        f"({'flagged high-risk' if row['decile_score'] >= 5 else 'not flagged high-risk'} "
        f"by COMPAS's own Low vs. Medium/High cut)\n"
        f"- Actual two-year recidivism outcome: "
        f"{'reoffended' if row['two_year_recid'] == 1 else 'did not reoffend'}\n"
    )
    return context, render_plain_language(exp)


def explain_hypothetical(age, priors, juv_fel, juv_misd, juv_other, sex, charge_degree):
    _, surrogate = get_state()
    raw = {
        "age": age,
        "priors_count": priors,
        "juv_fel_count": juv_fel,
        "juv_misd_count": juv_misd,
        "juv_other_count": juv_other,
        "is_male": 1.0 if sex == "Male" else 0.0,
        "is_felony_charge": 1.0 if charge_degree == "Felony" else 0.0,
    }
    exp = explain_instance(surrogate, raw)
    return render_plain_language(exp)


with gr.Blocks(title="Explainability Auditor") as demo:
    gr.Markdown(
        "# Explainability Auditor\n"
        "Per-person, plain-language explanations of an AI risk-classification decision — "
        "the individual-level question, alongside Case 02's group-level fairness audit of "
        "the same COMPAS dataset."
    )
    gr.Markdown(DISCLAIMER)

    with gr.Tab("Explain a real case from the dataset"):
        gr.Markdown(
            "Pick a row from the ProPublica COMPAS analytic sample (0 to 6,171) and see "
            "why the surrogate model would classify that person as it does."
        )
        row_input = gr.Number(value=88, label="Row index", precision=0)
        explain_btn = gr.Button("Explain this case", variant="primary")
        context_out = gr.Markdown()
        real_explanation_out = gr.Markdown()
        explain_btn.click(explain_real_case, inputs=row_input, outputs=[context_out, real_explanation_out])
        demo.load(explain_real_case, inputs=row_input, outputs=[context_out, real_explanation_out])

    with gr.Tab("Explain a hypothetical profile"):
        gr.Markdown("Build a profile yourself and see the same explanation logic applied to it.")
        with gr.Row():
            age_in = gr.Slider(18, 80, value=30, step=1, label="Age")
            priors_in = gr.Slider(0, 38, value=2, step=1, label="Number of prior convictions")
        with gr.Row():
            juv_fel_in = gr.Slider(0, 10, value=0, step=1, label="Juvenile felony count")
            juv_misd_in = gr.Slider(0, 10, value=0, step=1, label="Juvenile misdemeanor count")
            juv_other_in = gr.Slider(0, 10, value=0, step=1, label="Other juvenile offense count")
        with gr.Row():
            sex_in = gr.Radio(["Male", "Female"], value="Male", label="Sex")
            charge_in = gr.Radio(["Felony", "Misdemeanor"], value="Felony", label="Current charge")
        hyp_btn = gr.Button("Explain this profile", variant="primary")
        hyp_explanation_out = gr.Markdown()
        hyp_btn.click(
            explain_hypothetical,
            inputs=[age_in, priors_in, juv_fel_in, juv_misd_in, juv_other_in, sex_in, charge_in],
            outputs=hyp_explanation_out,
        )

if __name__ == "__main__":
    demo.launch()
