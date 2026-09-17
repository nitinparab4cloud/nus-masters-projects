# AI Governance Case Files

A three-project portfolio built to demonstrate AI governance experience for the job
search — classify an AI system's regulatory risk, technically audit a real model for
fairness, then document the full governance artifact set. Full plan, weekly
milestones, and progress tracker: **[AI Governance Case Files (working plan)](https://claude.ai/code/artifact/6c79a3b0-6071-46a6-ba06-e443a4afd022)**.

Each subfolder below is a standalone project meant to become its own GitHub repo —
see "Splitting into separate repos" below.

| # | Project | Track | Status |
|---|---|---|---|
| 01 | [AI Act Risk Navigator](01-ai-act-risk-navigator/) | Regulatory classification | Built, tested, ready to deploy |
| 02 | [Algorithmic Fairness Auditor](02-fairness-auditor/) | Technical audit | Core pipeline + metrics built; report + counterfactual test remaining |
| 03 | [Governance Documentation Suite](03-governance-documentation-suite/) | Documentation & process | Generator built and tested with a worked example |

## Why three, in this order

This mirrors NIST AI RMF's own function sequence (Govern → **Map** → **Measure** →
**Manage**) and the actual order an AI governance program works in:

1. **Classify** — Case 01 determines what regulatory tier a system falls into.
2. **Audit** — Case 02 tests a real model for fairness with the rigor a compliance
   review would demand.
3. **Document** — Case 03 produces the paper trail that keeps a system governed.

## IP note

Every input across all three projects is public: the EU AI Act's text, NIST's
published RMF, the ProPublica COMPAS dataset, and an open-weight Hugging Face model.
Nothing here is Salesforce data, code, or confidential material.

## Splitting into separate repos

Each project folder is self-contained (its own `README.md`, `requirements.txt`, and
`LICENSE`). To publish them as three separate GitHub repos rather than one monorepo:

```bash
# from inside e.g. 01-ai-act-risk-navigator/
git init
git add .
git commit -m "Initial commit: AI Act Risk Navigator"
git branch -M main
git remote add origin https://github.com/<your-username>/ai-act-risk-navigator.git
git push -u origin main
```

Repeat for each of the other two folders with their own repo name
(`algorithmic-fairness-auditor`, `governance-documentation-suite`). Three separate
repos read better on a GitHub profile than one monorepo — a recruiter scanning your
pinned repos sees three distinct, named pieces of work.
