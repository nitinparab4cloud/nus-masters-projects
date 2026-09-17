# AI Act Risk Navigator

A decision-support tool that classifies an AI system under the **EU AI Act's four risk
tiers** (Regulation (EU) 2024/1689), cites the specific article or Annex III category
driving the call, generates the compliance documentation checklist that tier requires,
and cross-references the result against **NIST AI RMF**'s four functions and
**Singapore's Model AI Governance Framework**.

**Live demo:** _add your Hugging Face Space link here once deployed_
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

1. Create a new Space → SDK: **Gradio** → hardware: **CPU basic** (free tier; there's
   no model to load, so this never needs a GPU).
2. Push `app.py`, `risk_engine.py`, `test_scenarios.py`, and `requirements.txt` to the
   Space's repo (or connect it to this GitHub repo directly).
3. The Space builds automatically and serves `app.py`.

## Test scenarios

`test_scenarios.py` carries 10 worked examples spanning all four tiers — three
prohibited, four high-risk, two limited-risk, one minimal-risk — used both as the
regression suite (`python test_scenarios.py`) and as the sample dropdown in the demo.

## Limitations — read before relying on this for anything real

- **This is a portfolio project, not legal advice.** It is a simplified, keyword-based
  reading of a long and actively-amended regulation. Delegated acts and guidance
  documents continue to refine Annex III's exact scope.
- Keyword matching produces false negatives on descriptions that don't use the
  taxonomy's phrasing, and it can't weigh the **Article 6(3) narrow-procedural-task
  exception**, which requires human judgment about materiality of influence on the
  outcome — the tool flags where that exception needs to be actively assessed, but
  doesn't decide it.
- A production version of this would need: a broader taxonomy (synonyms, multilingual
  input), a confidence score instead of a binary match, and a human review step before
  any classification is relied on operationally.

## Roadmap / what I'd build next

- [ ] Optional LLM-assisted mode for free-text reasoning with citation-checking against
      the retrieved article text (RAG over the Act itself), as a complement to the
      rule-based engine rather than a replacement for it.
- [ ] A confidence score and "closest alternative tier" output when multiple rules
      partially match.
- [ ] Batch mode: classify a CSV of AI system descriptions (an actual AI inventory) at
      once.

## License

MIT — see `LICENSE`.
