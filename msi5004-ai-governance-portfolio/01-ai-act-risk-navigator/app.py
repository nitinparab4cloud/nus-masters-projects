"""
app.py
------
Gradio front-end for the AI Act Risk Navigator. Deployable as-is to a free
Hugging Face Space (CPU basic tier is enough -- there's no model to load).

Local run:   python app.py
HF Spaces:   push this file + risk_engine.py + requirements.txt to a Space
             with SDK "gradio"; it auto-detects app.py as the entry point.
"""

import gradio as gr

from risk_engine import classify
from test_scenarios import SCENARIOS

TIER_BADGE = {
    "prohibited": "🔴 Unacceptable Risk — Prohibited",
    "high-risk": "🟠 High-Risk",
    "limited-risk": "🔵 Limited Risk — Transparency",
    "minimal-risk": "🟢 Minimal Risk",
}

DISCLAIMER = (
    "**This is a portfolio decision-support tool, not legal advice.** "
    "It applies a curated, keyword-based reading of Regulation (EU) 2024/1689 "
    "and is not a substitute for a qualified legal or compliance review. "
    "Always verify against the current consolidated text of the Act."
)


def run_classification(description: str):
    if not description or not description.strip():
        return "_Enter a system description above, or pick a sample scenario._", "", "", ""

    result = classify(description)
    badge = TIER_BADGE[result.tier]

    rationale_md = f"### {badge}\n\n**Why:** {result.rationale}\n\n**Timeline:** {result.timeline}"

    checklist_md = "### Documentation checklist\n\n" + "\n".join(
        f"- {item}" for item in result.checklist
    )

    rmf_md = "### NIST AI RMF — suggested next action per function\n\n" + "\n".join(
        f"- **{fn}** — {action}" for fn, action in result.rmf_actions.items()
    )

    singapore_md = f"### Singapore context\n\n{result.singapore_note}"

    return rationale_md, checklist_md, rmf_md, singapore_md


SAMPLE_CHOICES = {s["name"]: s["description"] for s in SCENARIOS}


def load_sample(name: str) -> str:
    return SAMPLE_CHOICES.get(name, "")


with gr.Blocks(title="AI Act Risk Navigator") as demo:
    gr.Markdown(
        "# AI Act Risk Navigator\n"
        "Describe an AI system in plain language and get its EU AI Act risk tier, "
        "the documentation checklist that tier requires, and a cross-reference "
        "against NIST AI RMF and Singapore's governance framework."
    )
    gr.Markdown(DISCLAIMER)

    with gr.Row():
        with gr.Column(scale=2):
            description_box = gr.Textbox(
                label="AI system description",
                placeholder=(
                    "e.g. \"A recruitment algorithm that screens incoming resumes and "
                    "ranks candidates for interview based on predicted job fit.\""
                ),
                lines=5,
            )
            sample_dropdown = gr.Dropdown(
                choices=list(SAMPLE_CHOICES.keys()),
                label="…or load one of the 10 test scenarios",
            )
            classify_btn = gr.Button("Classify", variant="primary")

        with gr.Column(scale=3):
            rationale_out = gr.Markdown()
            checklist_out = gr.Markdown()
            rmf_out = gr.Markdown()
            singapore_out = gr.Markdown()

    sample_dropdown.change(load_sample, inputs=sample_dropdown, outputs=description_box)
    classify_btn.click(
        run_classification,
        inputs=description_box,
        outputs=[rationale_out, checklist_out, rmf_out, singapore_out],
    )
    description_box.submit(
        run_classification,
        inputs=description_box,
        outputs=[rationale_out, checklist_out, rmf_out, singapore_out],
    )

if __name__ == "__main__":
    demo.launch()
