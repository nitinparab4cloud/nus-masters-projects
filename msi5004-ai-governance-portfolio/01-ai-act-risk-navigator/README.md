# AI Act Risk Navigator

A decision-support tool that classifies an AI system under the **EU AI Act's four risk
tiers** (Regulation (EU) 2024/1689), cites the specific article or Annex III category
driving the call, generates the compliance documentation checklist that tier requires,
and cross-references the result against **NIST AI RMF**'s four functions and
**Singapore's Model AI Governance Framework**.

**Live demo:** https://huggingface.co/spaces/nparab/ai-act-risk-navigator
**Part of:** [AI Governance Case Files](../README.md) — a three-project portfolio series

---

## Why this exists

Most explainers describe the Act's four risk tiers. This applies them: give it a
plain-language description of an AI system and it tells you *which specific rule fires
and why*, the way a compliance reviewer would have to.

## How it works

Classification is deliberately **rule-based, not an LLM call** — every decision is
keyword/phrase matching against a curated taxonomy built from the Act's own text
(`risk_engine.py`). That trade-off is the point: a governance tool that can't explain
*which clause* produced its answer isn't auditable, and auditability is what a real
compliance review demands.

The check order follows the Act's own severity ordering:

1. **Article 5** prohibited practices ("unacceptable risk") — 10 sub-points, including
   the two added in the 2026 amendment (non-consensual intimate imagery, CSAM).
2. **Annex III** high-risk categories (Article 6(2)) — the 8 sectors: biometrics,
   critical infrastructure, education, employment, essential services, law
   enforcement, migration/border control, and administration of justice.
3. **Article 50** transparency triggers ("limited risk") — chatbots, synthetic media,
   emotion recognition/biometric categorisation outside Annex III.
4. Otherwise: **minimal risk**.

Each result also carries:
- The **compliance deadline** that applies to that tier (the Act's obligations phase
  in on different dates through 2028 — see `risk_engine.py::TIMELINE_NOTES`).
- A **NIST AI RMF** action for each of Govern / Map / Measure / Manage, so the result
  reads as "what do I do next," not just "here's a label."
- A **Singapore context** note, since the Model AI Governance Framework is soft law
  with no binding tiers of its own — it's a second lens, not a second engine.

## Sources

- [Regulation (EU) 2024/1689](https://artificialintelligenceact.eu/) (consolidated text and article-by-article commentary via artificialintelligenceact.eu)
- [NIST AI Risk Management Framework 1.0](https://airc.nist.gov/airmf-resources/airmf/5-sec-core/) — Govern/Map/Measure/Manage core functions
- Singapore IMDA — Model AI Governance Framework & AI Verify Testing Framework

## Running it

```bash
pip install -r requirements.txt
python test_scenarios.py   # confirms all 10 sample scenarios classify as expected
python app.py               # launches the Gradio demo locally
```

## Deploying to Hugging Face Spaces

   The live demo above runs as a free **Static** Space rather than a Gradio SDK Space.
   Hugging Face moved Gradio and Docker Spaces behind a paid plan, so the classification
   logic in `risk_engine.py` is ported line-for-line into plain JavaScript in a single
   `index.html`, tested against the same 10 scenarios as `test_scenarios.py` to confirm
   it matches. `risk_engine.py` itself (imported by `app.py`) remains the source of
   truth for the logic — the HTML page is a static, dependency-free copy of it for
   demo purposes, not a separate implementation to maintain by hand.

   If Gradio Spaces become free again, or you have a paid plan, the original path also
   works: new Space → SDK **Gradio** → CPU basic → push `app.py`, `risk_engine.py`,
   `test_scenarios.py`, and `requirements.txt`.

## License

MIT — see `LICENSE`.
