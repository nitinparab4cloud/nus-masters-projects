# Governance Documentation Suite

A generator that produces the full artifact set an AI governance function has to
produce and keep current for a real model — Model Card, Risk Assessment, Data
Governance Statement, Human Oversight Design, Incident Response Runbook, and Vendor
AI Risk Questionnaire — from a single YAML spec file. The deliverable isn't six
hand-written documents; it's a **process** that regenerates them for any model.

**Live demo:** _add your Hugging Face Space link here once deployed_
**Part of:** [AI Governance Case Files](../README.md) — a three-project portfolio series
**Status:** the generator is built and tested (see below); the worked example ships
now, a second worked example and the Space UI are this project's remaining milestones.

---

## It already works

```bash
pip install -r requirements.txt
python generate.py specs/example_toxic_comment_classifier.yaml
```

This writes six Markdown documents to `generated/toxic-comment-classifier-toxic-bert/`,
built around **[unitary/toxic-bert](https://huggingface.co/unitary/toxic-bert)** — a
real, small, public Hugging Face model (part of the open-source Detoxify project,
trained on the Jigsaw Toxic Comment Classification Challenge dataset) chosen because
it's a genuine content-moderation/recommender-adjacent governance case, continuous
with the MSI5004 Week 4 material on recommender systems and harmful-content
amplification.

Read the generated documents to see the shape of the output before writing your own
spec — they're checked into `generated/` in this repo on purpose, as a worked example.

## Spec-driven, not hand-written

Each document is a Jinja2 template (`templates/*.md.j2`) rendered against one YAML
spec file (`specs/*.yaml`). `generate.py` also computes the **overall risk level**
from `risk_probability` × `risk_severity` using a simple, legible matrix
(`generate.py::RISK_MATRIX`) — deliberately a lookup table a reviewer can sanity-check
in ten seconds, not a black-box score.

To document a second model, copy `specs/example_toxic_comment_classifier.yaml`,
fill in your own model's details, and run `generate.py` against it. The most useful
second example for this portfolio series is the model audited in Case 02 (the
fairness auditor) — using the same model across both projects makes the series read
as one continuous governance exercise rather than three disconnected demos.

### Spec schema, by section

| Section | Feeds into | Key fields |
|---|---|---|
| Model basics | Model Card | `system_name`, `description`, `model_type`, `intended_use`, `out_of_scope_uses` |
| Training/eval | Model Card | `training_data_summary`, `evaluation_data_summary`, `performance_metrics` |
| Risk | Risk Assessment | `risk_probability`, `risk_severity` (Low/Medium/High — drives the computed `overall_risk_level`), `affected_parties`, `mitigations` |
| Data | Data Governance Statement | `data_sources`, `pii_present`, `data_governance_measures` |
| Oversight | Human Oversight Design | `oversight_model` (in-the-loop / over-the-loop / out-of-the-loop), `escalation_path`, `contestability_mechanism` |
| Incidents | Incident Response Runbook | `incident_severity_levels`, `incident_contacts`, `containment_action` |
| Sourcing | Vendor AI Risk Questionnaire | `vendor_name` (use `"Internal"` when not vendor-sourced — the questionnaire still ships, ready for the day it isn't) |

See `specs/example_toxic_comment_classifier.yaml` for every field filled in with a
real worked example, including the reasoning behind each choice in inline comments.

## Sources

- Mitchell, M. et al. (2019), ["Model Cards for Model Reporting"](https://arxiv.org/abs/1810.03993) — the Model Card format
- IMDA AI Verify Testing Framework, control 4.3.1 — probability × severity risk scoring
- MSI5004 seminar material — human agency & oversight models (in/over/out-of-the-loop), contestability

## Deploying to Hugging Face Spaces (remaining milestone)

Wrap `generate.py` in a small Gradio form (fields matching the spec schema above) so
a reviewer can fill in a model's details in a browser and download the six documents,
instead of hand-editing YAML. New Space → SDK **Gradio** → CPU basic.

## Remaining milestones

- [ ] Apply the generator to the model audited in Case 02, for cross-project continuity.
- [ ] Build the Gradio form front-end and deploy to HF Spaces.
- [ ] Publish the repo.

## Limitations

- This is a documentation *process*, not an automated compliance guarantee — every
  generated document still needs a human governance reviewer to have actually done
  the underlying risk thinking; the tool formats and cross-links their answers
  consistently, it doesn't produce them.
- The risk matrix (`RISK_MATRIX` in `generate.py`) is intentionally simple. A real
  program might use a finer-grained or domain-specific scoring model — the point
  here is that whatever matrix you use should stay legible enough for a reviewer
  to check by hand, not that this exact 3×3 grid is the right one everywhere.
- `unitary/toxic-bert`'s own published performance numbers are carried through the
  worked example as reported on its Hugging Face model card; they haven't been
  independently re-validated here (the spec's own "Caveats and Recommendations"
  field says so).

## License

MIT — see `LICENSE`.
