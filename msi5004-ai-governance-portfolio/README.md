# AI Governance Case Files

A four-project portfolio built to demonstrate AI governance experience for the job
search — classify an AI system's regulatory risk, technically audit a real model for
fairness, explain its individual decisions in plain language, then document the full
governance artifact set. Grew out of NUS's MSI5004 (AI Governance and Ethics) module,
with Case 04 extending into material from a separate AI ethics course. Full plan,
weekly milestones, and progress tracker:
**[AI Governance Case Files (working plan)](https://claude.ai/code/artifact/6c79a3b0-6071-46a6-ba06-e443a4afd022)**.

| # | Project | Track | Status |
|---|---|---|---|
| 01 | [AI Act Risk Navigator](01-ai-act-risk-navigator/) | Regulatory classification | Built, tested, live |
| 02 | [Algorithmic Fairness Auditor](02-fairness-auditor/) | Technical audit | Core pipeline + metrics built; report + counterfactual test remaining |
| 03 | [Governance Documentation Suite](03-governance-documentation-suite/) | Documentation & process | Generator built and tested with a worked example |
| 04 | [Explainability Auditor](04-explainability-auditor/) | Technical audit — individual explanations | Built and tested, including a live browser demo |

## Why four, in this order

The build order (01–04, by folder number) isn't quite the narrative order — Case 04
was added after Case 03. Read as a story, the arc mirrors NIST AI RMF's own function
sequence (Govern → **Map** → **Measure** → **Manage**) and the actual order an AI
governance program works in:

1. **Classify** (Case 01) — determine what regulatory tier a system falls into.
2. **Audit** (Case 02) — test a real model for group-level fairness with the rigor a
   compliance review would demand.
3. **Explain** (Case 04) — answer the individual-level question group metrics can't:
   why was this specific person's outcome what it was.
4. **Document** (Case 03) — produce the paper trail that keeps a system governed,
   drawing on what the previous three established.

## Sources

Every input across all four projects is public: the EU AI Act's text, NIST's
published RMF, the ProPublica COMPAS dataset, and an open-weight Hugging Face model.
Each project's own README lists its specific sources.

## Layout

Each project folder is self-contained (its own `README.md`, `requirements.txt`, and
`LICENSE`) and can be run independently — `cd` into a folder and follow its README.
They're kept together here, under `msi5004-ai-governance-portfolio/`, because they
form one continuous narrative and share this top-level plan and progress tracker.
