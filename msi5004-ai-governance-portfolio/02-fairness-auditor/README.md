# Algorithmic Fairness Auditor

A technical fairness audit of the ProPublica COMPAS recidivism dataset — the same
case study covered in MSI5004's Week 5 seminar — that reproduces ProPublica's 2016
finding with `fairlearn`, then goes past the aggregate numbers with a counterfactual
test on individual cases.

**Live demo:** _add your Hugging Face Space link here once deployed_
**Part of:** [AI Governance Case Files](../README.md) — a three-project portfolio series
**Status:** core data pipeline and group-level metrics are built and tested below;
the counterfactual test and audit report are this project's remaining milestones
(see `counterfactual_test.py` and the checklist in the portfolio tracker).

---

## What's already working

```bash
pip install -r requirements.txt
python data_loader.py       # loads + filters the dataset, prints group counts
python fairness_metrics.py  # prints the by-group table and headline metrics
python app.py                # launches the interactive Gradio demo locally
```

Running `fairness_metrics.py` against the full dataset reproduces ProPublica's
headline finding almost exactly:

| Group | Selection rate | True positive rate | False positive rate |
|---|---|---|---|
| African-American | 0.576 | 0.715 | **0.423** |
| Caucasian | 0.331 | 0.504 | **0.220** |

Comparing just these two groups: **demographic parity difference = 0.245**,
**equalized odds difference = 0.212**, **disparate impact ratio = 0.575** (below
the conventional 0.8 concern threshold). In plain terms: African-American
defendants who did *not* reoffend within two years were flagged "high risk" at
close to double the rate of Caucasian defendants who also did not reoffend — the
same asymmetry ProPublica's original investigation found.

## Methodology

**Data.** ProPublica's public COMPAS release (`compas-scores-two-years.csv`),
filtered to their own analytic sample: charge date within 30 days of arrest,
recidivism flag not missing, and excluding ordinary-traffic charges
(`data_loader.py::apply_propublica_filters`). This matches their published
methodology so the numbers are directly comparable to their original write-up
rather than to a different, unstated sample.

**Binary framing.** COMPAS outputs a 1–10 decile score; a score ≥ 5 is treated as
a "high risk" flag, matching COMPAS's own Low vs. Medium/High cutoff. The
Gradio demo lets you move this threshold and watch the metrics shift — useful
for showing that the disparity isn't an artifact of one specific cutoff choice.

**Metrics** (see `fairness_metrics.py` docstring for the full definitions):

- **Demographic parity difference** — gap in flag rate between groups.
- **Equalized odds difference** — gap in error-rate profile (false positive
  rate and true positive rate together) between groups. This is the metric
  ProPublica's own analysis centers on.
- **Disparate impact ratio** — the "80% rule" borrowed from US employment
  discrimination law: (selection rate, disadvantaged group) / (selection
  rate, advantaged group).

**Counterfactual test (Week 3 — to build).** The metrics above are all
*aggregate*: they describe the dataset, not any one person. The MSI5004
coursework's fairness section asks a sharper, individual-level question —
"would this person's outcome change if their race changed, all else equal?"
`counterfactual_test.py` sketches two ways to approximate this against COMPAS
(nearest-neighbour matched pairs vs. a surrogate model); the matched-pairs
approach is scoped to ship in the Week 3 checklist item.

## Sources

- [ProPublica — "Machine Bias" (2016)](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing) and their [compas-analysis GitHub repo](https://github.com/propublica/compas-analysis)
- [fairlearn documentation](https://fairlearn.org/) — metric definitions and `MetricFrame` API
- IMDA AI Verify Testing Framework — fairness principle, counterfactual fairness test
  (from MSI5004 seminar material)

## Deploying to Hugging Face Spaces

Same pattern as Case 01: new Space → SDK **Gradio** → CPU basic → push `app.py`,
`data_loader.py`, `fairness_metrics.py`, and `requirements.txt`.

## Remaining milestones

- [ ] Implement `counterfactual_test.py::find_matched_pairs` (nearest-neighbour
      matching on priors/age/sex/charge degree) and summarize the gap.
- [ ] Write the audit report: methodology, findings, mitigation recommendations
      (e.g. equalized-odds post-processing, or removing the feature entirely and
      re-auditing what replaces it).
- [ ] Add a chart to the Gradio demo (see `app.py` TODO).
- [ ] Publish the repo and the report.

## Limitations

- COMPAS's proprietary scoring logic isn't public — this audits the *outputs*
  (published scores vs. outcomes), which is what an external governance review
  would actually have access to, not the model internals.
- The matched-pairs counterfactual test is an approximation, not a true
  counterfactual (COMPAS was never run twice on the same person with race
  flipped) — the report should say this plainly rather than overclaim.
- Race and ethnicity categories here are COMPAS's own dataset categories, used
  only for the audit's protected-attribute analysis, and are not used or implied
  to describe any real individual beyond this public, already-published dataset.

## License

MIT — see `LICENSE`.
