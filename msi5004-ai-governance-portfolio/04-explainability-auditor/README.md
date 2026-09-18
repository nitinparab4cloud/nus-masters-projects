# Explainability Auditor

Per-person, plain-language explanations of an AI risk-classification decision, built on
the same ProPublica COMPAS dataset audited for group-level fairness in **Project 2** of
this series. Where that project asked "is this model fair across groups?", this one asks
the individual question: **why was this specific person flagged?** — the question
Article 86 of the EU AI Act and the "explicability" principle in AI ethics frameworks
are actually about.

**Live demo:** _add your Hugging Face Space link here once deployed_ (a working,
tested, static `index.html` for a free HF Static Space ships in this folder — see
"Deploying to Hugging Face Spaces" below; this one's ready to upload today)
**Part of:** [AI Governance Case Files](../README.md) — a four-project portfolio series
**Status:** built and tested end to end, including a JS port of the explanation logic
for the live browser demo.

---

## Why this exists

Most fairness audits (Case 02 included) stop at the aggregate level: does the model's
error rate differ across groups? That's the right question for a regulator or an
oversight board, but it's the wrong question for the person actually sitting across
from the decision. They don't experience an "equalized odds difference of 0.212" —
they experience one specific outcome, and they're entitled to know why. This project
builds the individual-level half of that pair.

## The honest starting problem

COMPAS's actual scoring algorithm is proprietary. Northpointe/Equivant has never
published it, and no outside auditor — this project included — has ever had access to
it. That's not a gap in this project specifically; it's the situation every real AI
governance auditor faces with a vendor's black-box system. The credible response isn't
to pretend otherwise — it's to do what's actually done in practice: train a
transparent model on the same publicly known inputs to **approximate the system's
outputs**, report exactly how faithful that approximation is, and explain the
approximation rather than silently presenting it as the real thing.

## How it works

1. **Surrogate model** (`surrogate_model.py`): a plain logistic regression trained to
   predict COMPAS's own published high-risk flag (`decile_score >= 5`) from COMPAS's
   own publicly known input features — age, prior convictions, juvenile offense
   counts, sex, and current charge type. On held-out data it agrees with COMPAS's own
   flag on **75.6%** of cases, against a 55.4% majority-class baseline.

2. **Race-blind by design, on purpose**: race is never a model input, matching
   COMPAS's own stated design. This isn't just fidelity to the original instrument —
   it's the point of the demo. Case 02 already showed COMPAS produces racially
   disparate error rates. This project shows *how* that can happen without race ever
   being "looked at" directly: other race-blind features — prior conviction count,
   above all — correlate with race because of well-documented disparities upstream in
   policing and charging. Explaining a decision with race-blind features doesn't
   explain away a racial disparity; it's how proxy discrimination actually works, and
   this tool is built to surface that, not launder it.

3. **Exact SHAP-style attribution, no `shap` dependency** (`explain_engine.py`): for a
   linear (here, logistic) model, the Shapley value of each feature under a
   feature-independence assumption has a closed form —
   `φᵢ = βᵢ × standardized_valueᵢ` — with no sampling or approximation involved. These
   contributions satisfy SHAP's namesake "local accuracy" property exactly:
   `baseline + Σφᵢ = the model's own output`, for every example
   (`test_explainability.py` checks this to floating-point precision). Implementing
   this directly, rather than depending on the `shap` package, keeps the whole
   explanation pipeline in one auditable file — in keeping with this whole portfolio's
   stance that a governance tool should be inspectable end to end, not "explainable"
   via a second black box.

## Sources

- Lundberg, S. & Lee, S. (2017), ["A Unified Approach to Interpreting Model Predictions"](https://arxiv.org/abs/1705.07874) — the SHAP framework this closed-form calculation reproduces for linear models
- Regulation (EU) 2024/1689, **Article 86** — the right to an explanation of individual decision-making for high-risk AI systems
- Floridi, L. et al. (2018), ["AI4People — An Ethical Framework for a Good AI Society"](https://link.springer.com/article/10.1007/s11023-018-9482-5) — the "explicability" principle
- [ProPublica — "Machine Bias" (2016)](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing) and their [compas-analysis GitHub repo](https://github.com/propublica/compas-analysis) — same dataset and analytic-sample filters as Case 02

## Running it

```bash
pip install -r requirements.txt
python surrogate_model.py       # trains the surrogate, prints fidelity + coefficients
python test_explainability.py   # confirms all 5 sanity checks pass
python app.py                    # launches the interactive Gradio demo locally
```

## Deploying to Hugging Face Spaces

Same free-tier situation as Project 1: Hugging Face moved Gradio and Docker Spaces
behind a paid plan, so `app.py` (the full Gradio version) needs a paid Space or local
use. For the free route, `index.html` in this folder is a complete, self-contained,
already-tested static demo: the trained surrogate's coefficients and a set of real
example cases are embedded directly as JSON, and the explanation logic is a plain-JS
port of `explain_engine.py` — verified to match the Python output exactly on multiple
test cases (see the file's own self-test, logged to the browser console).

1. Create a new Space → SDK: **Static** → template: **Blank**.
2. Upload `index.html` from this folder (drag-and-drop through the HF web UI works
   fine — no git needed).
3. Done. No build step, no server, nothing installed at runtime.

If you retrain the surrogate (e.g. with a different feature set) and want to refresh
the live demo, regenerate the embedded JSON block by re-running the export step
documented at the top of `index.html`'s `<script>` tag, or ask for it to be rebuilt.

## Test scenarios

`test_explainability.py` runs 5 sanity checks rather than worked "scenarios" like
Cases 01/02, since this project doesn't classify a discrete label — it produces a
continuous attribution. The checks: the surrogate meaningfully beats a majority-class
baseline; the local-accuracy property holds exactly; more priors never lowers the risk
estimate; younger never produces a lower estimate than older in this data; and the
predicted flag always agrees with the 0.5 probability cut. All 5 pass.

## Limitations — read before relying on this for anything real

- **This explains a surrogate model's approximation of COMPAS, not COMPAS's actual
  logic.** The 75.6% fidelity number is the single most important caveat here — it
  means roughly 1 in 4 cases, the surrogate's explanation would describe reasoning
  COMPAS itself may not have used. It's surfaced in the UI on every result, not just
  in this file.
- The feature-independence assumption behind the closed-form SHAP calculation is a
  simplification — in reality, age and priors_count are correlated in this dataset.
  `shap`'s own more sophisticated interventional/correlation-aware explainers would
  produce a more precise (though not dramatically different, for a model this size)
  attribution. The trade-off made here is exactness and zero dependencies over that
  extra precision.
- This is a portfolio project, not a real explainability audit of a deployed system —
  a genuine Article 86 compliance exercise would need the actual system's real inputs
  and outputs, not a third party's best reconstruction of them.
- Race and ethnicity categories here are COMPAS's own dataset categories, used only
  as audit context (never as a model input), and aren't used or implied to describe
  any real individual beyond this public, already-published dataset.

## Roadmap / what I'd build next

- [ ] Add `shap`'s own `LinearExplainer` as an optional cross-check against the
      hand-implemented closed-form version, to quantify how much the independence
      assumption actually matters on this dataset.
- [ ] Extend the same technique to the `unitary/toxic-bert` model documented in
      Case 03, for full cross-project continuity.
- [ ] A side-by-side view comparing two hypothetical profiles that differ in exactly
      one feature, to make the "what would change the outcome" question concrete.

## License

All rights reserved — see `LICENSE`. Public here for evaluation by prospective employers and collaborators; not licensed for reuse without permission.
