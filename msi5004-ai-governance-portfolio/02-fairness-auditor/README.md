# Algorithmic Fairness Auditor

A technical fairness audit of the ProPublica COMPAS recidivism dataset — the same
case study covered in MSI5004's Week 5 seminar — that reproduces ProPublica's 2016
finding with `fairlearn`, then goes past the aggregate numbers with a counterfactual
test on individual cases.

**Live demo:** _add your Hugging Face Space link here once deployed_ (a working,
tested, static `index.html` for a free HF Static Space ships in this folder — see
"Deploying to Hugging Face Spaces" below; this one's ready to upload today)
**Part of:** [AI Governance Case Files](../README.md) — a four-project portfolio series
**Status:** data pipeline and group-level metrics are built and tested, both in the
local Gradio app (now with a bar chart) and in a browser-based static demo. The
counterfactual test and audit report are this project's remaining milestones (see
`counterfactual_test.py` and the checklist in the portfolio tracker).

---

## What's already working

```bash
pip install -r requirements.txt
python data_loader.py             # loads + filters the dataset, prints group counts
python fairness_metrics.py        # prints the by-group table and headline metrics
python app.py                      # launches the interactive Gradio demo locally
pytest test_fairness_metrics.py   # regression tests, pinned to ProPublica's own published numbers
```

The Gradio demo shows a bar chart (selection rate / true positive rate / false
positive rate, grouped by the two selected races) alongside the summary text and
table, so the gap is visible at a glance rather than only readable in a table.

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

Same free-tier situation as Cases 01 and 04: Hugging Face moved Gradio and Docker
Spaces behind a paid plan, so `app.py` (the full Gradio version, now with the bar
chart) needs a paid Space or local use. For the free route, `index.html` in this
folder is a complete, self-contained, already-tested static demo: the filtered
COMPAS rows (race, decile score, two-year recidivism flag — 6,172 cases) are
embedded directly as JSON, and the metric calculations are a plain-JS port of
`fairness_metrics.py`'s `prepare_labels` / `compute_group_metrics` /
`compute_headline_metrics` — verified to match the Python output exactly on
multiple group/threshold combinations (see the file's own self-test, logged to
the browser console). It reproduces the same interaction as the Gradio app —
pick two groups and a threshold, see selection rate / TPR / FPR update, now with
a chart, a table, and a disparate-impact badge — entirely client-side.

1. Create a new Space → SDK: **Static** → template: **Blank**.
2. Upload `index.html` from this folder (drag-and-drop through the HF web UI
   works fine — no git needed).
3. Done. No build step, no server, nothing installed at runtime.

If you retrain against a different dataset or add a group, regenerate the
embedded JSON block by re-running `export_artifact.py` and re-embedding its
output into `index.html`'s `<script>` tag, or ask for it to be rebuilt.

## Remaining milestones

- [ ] Implement `counterfactual_test.py::find_matched_pairs` (nearest-neighbour
      matching on priors/age/sex/charge degree) and summarize the gap.
- [ ] Write the audit report: methodology, findings, mitigation recommendations
      (e.g. equalized-odds post-processing, or removing the feature entirely and
      re-auditing what replaces it).
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
