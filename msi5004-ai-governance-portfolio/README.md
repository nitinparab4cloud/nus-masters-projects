# AI Governance Case Files

[![Tests](https://github.com/nitinparab4cloud/nus-masters-projects/actions/workflows/tests.yml/badge.svg)](https://github.com/nitinparab4cloud/nus-masters-projects/actions/workflows/tests.yml)

A six-project portfolio built to demonstrate AI governance experience for the job
search — classify an AI system's regulatory risk, technically audit a real model for
fairness, explain its individual decisions in plain language, document the full
governance artifact set, build and red-team a small governed agent of its own, then
build the incident-response controls for when an agentic AI system misbehaves despite
that governance. Grew out of NUS's MSI5004 (AI Governance and Ethics) module, with
Case 04 extending into material from a separate AI ethics course and Cases 05–06
going beyond it entirely.
Full plan, weekly milestones, and progress tracker:
**[AI Governance Case Files (working plan)](https://claude.ai/code/artifact/6c79a3b0-6071-46a6-ba06-e443a4afd022)**.

| # | Project | Track | Status |
|---|---|---|---|
| 01 | [AI Act Risk Navigator](01-ai-act-risk-navigator/) | Regulatory classification | Built, tested, live |
| 02 | [Algorithmic Fairness Auditor](02-fairness-auditor/) | Technical audit | Core pipeline, metrics, and a live browser demo built; audit report + counterfactual test remaining |
| 03 | [Governance Documentation Suite](03-governance-documentation-suite/) | Documentation & process | Generator built and tested with a worked example |
| 04 | [Explainability Auditor](04-explainability-auditor/) | Technical audit — individual explanations | Built and tested, including a live browser demo |
| 05 | [Governed AI Helpdesk Agent](05-governed-rag-agent/) | GenAI & agentic governance | Built and tested, including a live browser demo and a 13-case red-team suite |
| 06 | [Agentic AI Incident Response](06-agentic-incident-response/) | Agentic AI security & incident response | Built and tested, including a live browser demo and an 11-case red-team suite |

## Why six, in this order

The build order (01–06, by folder number) isn't quite the narrative order — Case 04
was added after Case 03, and Cases 05–06 after all four. Read as a story, the arc
mirrors NIST AI RMF's own function sequence (Govern → **Map** → **Measure** →
**Manage**) and the actual order an AI governance program works in, then closes on
the two capability areas none of the first four touch — governing a system that can
act, and responding when governance around one wasn't enough:

1. **Classify** (Case 01) — determine what regulatory tier a system falls into.
2. **Audit** (Case 02) — test a real model for group-level fairness with the rigor a
   compliance review would demand.
3. **Explain** (Case 04) — answer the individual-level question group metrics can't:
   why was this specific person's outcome what it was.
4. **Document** (Case 03) — produce the paper trail that keeps a system governed,
   drawing on what the previous three established.
5. **Govern an agent** (Case 05) — build a small tool-using AI system with the input
   screening, least-privilege tooling, grounded-or-refuse answering, and audit trail
   a governance review would require of one, then red-team it against its own controls.
6. **Respond** (Case 06) — take a real, publicly documented agentic AI security
   incident and build the specific governance controls (scoped credentials,
   default-deny egress, a forced stop on stuck retries, coordination-channel
   detection, and an enforced escalation path) that would have caught it, proven
   against a red-team suite rather than argued in the abstract.

## Employer skills demonstrated

Cross-referenced against recurring requirements in recent AI Governance / AI Risk /
Responsible AI / Model Risk job postings, not just the MSI5004 syllabus. Built means
there's working, tested code behind the claim; Planned means it's scoped but not
built yet — listed here on purpose rather than left out, so the gap is visible
instead of implied away.

| Skill domain | Where it's demonstrated | Status |
|---|---|---|
| Regulatory mapping | Case 01 maps a system description to EU AI Act articles/Annex III, a NIST RMF action per function, and a Singapore Model AI Governance Framework note; Case 04 ties individual explanations to Article 86 | Built |
| AI risk assessment & control mapping | Case 01's tier classification with cited article/category; Case 03's probability × severity risk matrix (`generate.py::RISK_MATRIX`) | Built |
| Bias & fairness testing | Case 02: demographic parity difference, equalized odds difference, disparate impact ratio, and subgroup selection/TPR/FPR, on the real ProPublica COMPAS dataset | Built |
| Explainability & transparency | Case 04: closed-form SHAP-style per-case attribution with an exact local-accuracy guarantee, plus Case 03's Model Card generation | Built |
| Model/AI evaluation | Case 02's surrogate-fidelity check (agreement vs. COMPAS, vs. a majority-class baseline); Case 04's 5 automated sanity checks (fidelity, local accuracy, monotonicity, flag consistency) | Built |
| Human oversight | Case 03's Human Oversight Design doc (in/over/out-of-the-loop model, escalation path, contestability mechanism) | Built |
| Privacy & data governance | Case 03's Data Governance Statement (PII flag, data governance measures) | Built |
| Audit & evidence readiness | Case 03 generates a six-document evidence set (Model Card, Risk Assessment, Data Governance Statement, Human Oversight Design, Incident Response Runbook, Vendor AI Risk Questionnaire) from one spec file, reproducibly | Built |
| Third-party / vendor AI risk | Case 03's Vendor AI Risk Questionnaire template | Built |
| Technical foundations | Python throughout; every live demo is a dependency-free static JS port verified against its Python source (Cases 01, 02, 04, 05, 06); automated tests in every project; reproducible via `requirements.txt` | Built |
| Stakeholder & policy translation | Each project's README states the plain-language "why this exists" case before the technical detail; the portfolio's own Classify → Audit → Explain → Document → Govern-an-agent → Respond narrative | Built |
| Agentic AI governance (least privilege, policy-as-code, audit log) | Case 05: single-entry tool allowlist with fail-closed refusal on unknown/malformed calls, risk-tier-based human-approval gate, SHA-256 hash-chained audit trail with tamper detection | Built |
| GenAI security (prompt injection, jailbreaks, data leakage) | Case 05: 15-pattern injection screen (OWASP LLM01), structural data-leakage prevention (the answering path has no code path into the intake/audit stores), 13-case red-team suite (`redteam_suite.py`) across 5 risk categories | Built |
| Agentic AI security & incident response | Case 06: a synthetic sandbox modeling a real 2026 agentic security incident's failure modes, a policy engine enforcing scoped credentials, default-deny egress, stuck-loop detection, coordination-channel detection, and an alert-escalation-to-autonomous-shutdown path, plus an 11-case red-team suite (`redteam_suite.py`) | Built |
| AI governance operating model (intake, inventory, approval workflow, RACI) | Case 01 + Case 03 cover classify → document; Case 05 adds a working stateful intake/approval-workflow application (submit → risk-tier route → human approve/reject → audit) | Built |
| Model risk / independent validation | Case 02's fidelity numbers are validation-flavored; a formal builder-vs-validator report split, robustness/stress tests, and drift simulation are scoped, not built | Planned |
| Monitoring & observability (drift, degradation, control effectiveness) | Not built | Planned |
| Governance reporting (executive dashboard, risk heatmap) | Not built | Planned |

## Roadmap

Two workstreams are scoped on top of the six built projects, in this order:

1. **Cases 02 + 04 → Model Risk Validation Lab**. Add a model card, a
   robustness/stress test, a drift simulation, and a builder-vs-validator report
   split on top of the existing fairness and explainability work.
2. **Case 03 → Regulatory Compliance & Evidence Copilot**. Add explicit NIST AI
   RMF / ISO 42001 mapping tables and a citation-grounded LLM layer that answers
   from a supplied regulatory corpus rather than free-generating — reusing Case 05's
   retrieval and grounding pattern.

## Sources

Every input across all six built projects is public: the EU AI Act's text, NIST's
published RMF, the OWASP LLM Top 10, the ProPublica COMPAS dataset, an open-weight
Hugging Face model, and (Case 06) OpenAI's and Hugging Face's own public postmortems
on the incident it models. Each project's own README lists its specific sources.

## Layout

Each project folder is self-contained (its own `README.md`, `requirements.txt`, and
`LICENSE`) and can be run independently — `cd` into a folder and follow its README.
They're kept together here, under `msi5004-ai-governance-portfolio/`, because they
form one continuous narrative and share this top-level plan and progress tracker.
