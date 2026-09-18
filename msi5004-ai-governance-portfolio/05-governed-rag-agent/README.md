# Governed AI Helpdesk Agent

A small "agent" — really a chat surface plus one allowlisted tool — that answers AI
governance policy questions grounded only in a fixed, cited corpus, and routes new
AI-system registrations through the same risk-tiered human-approval gate Cases 01 and
03 use elsewhere in this portfolio. Where Cases 01–04 audit and document AI systems,
this one **is** an AI system with agentic capability, built with the controls a
governance review would actually demand of one: an input screen against
prompt-injection, a tool allowlist against excessive agency, grounded-or-refuse
answering against hallucinated policy advice, and a tamper-evident audit trail over
every action it takes.

**Live demo:** [`index.html`](index.html) in this folder — a complete, self-contained,
already-tested static demo; upload it to a free Hugging Face **Static** Space in two
clicks (see "Deploying to Hugging Face Spaces" below), or just open the file locally.
**Part of:** [AI Governance Case Files](../README.md) — a five-project portfolio series
**Status:** built and tested end to end, including a JS port for the live browser demo
and a 13-case adversarial red-team suite.

---

## Why this exists

The first four projects in this series classify, audit, explain, and document AI
systems from the outside. None of them *is* one with any agentic capability — and
agentic AI is exactly where a growing share of real governance work now sits: a
chatbot that can take actions, not just answer questions, is a materially different
risk surface than a static classifier. This project builds the smallest system that's
honestly representative of that surface, so the controls below are demonstrated
against real, executable code rather than described in the abstract.

## How it works

The system is two independent halves, matching the design of a real governed agent:
an **answering** half that can only ever retrieve and cite, never act, and an
**acting** half that can only ever take one specific, auditable action.

1. **`policy_engine.py` — retrieval and grounding.** A dependency-free TF-IDF index
   (hand-rolled, no `sklearn`/vector database — a dozen short passages don't need
   one, and hand-rolling keeps the path auditable end to end and portable to the JS
   demo) over a 12-passage corpus drawn from this portfolio's own published outputs:
   the EU AI Act tiers and articles (Case 01), NIST AI RMF, and the governance
   document purposes Case 03 generates. A message is answered **only** from the
   best-matching passage, with a citation, when the retrieval score clears a
   threshold (0.20 — see the constant's comment for how that number was tuned); below
   it, the engine refuses rather than guesses. Every message is screened against 15
   prompt-injection pattern families (OWASP LLM01) *before* anything else runs — an
   attack dressed as an ordinary question doesn't get a pass.

2. **`tools.py` — the one action this agent can take.** `ALLOWED_TOOLS` has exactly
   one entry, `submit_ai_use_case_intake`; any other tool name is refused and logged
   before its arguments are even inspected (OWASP LLM08, excessive agency). A
   submitted intake's risk tier is assessed from its description using a trimmed copy
   of Case 01's own Article 5 / Annex III keyword taxonomy — a Prohibited or
   High-Risk description is queued `pending_approval` and can only leave that state
   through an explicit `.approve()`/`.reject()` call from a named human reviewer;
   nothing in the chat surface, and no claim embedded in the submission text itself
   ("pre-approved by legal"), can move it there.

3. **`audit_log.py` — the trail.** Every refusal, submission, approval, and rejection
   is appended to a hash chain: each entry's SHA-256 hash covers its own content plus
   the previous entry's hash, so altering or deleting a past entry breaks every hash
   after it. Documented honestly rather than oversold — this is tamper-evident within
   one process's memory, not a real WORM store or blockchain, and both the code and
   the live demo say so directly.

4. **`app.py` — the interface.** A four-tab Gradio app: chat (the requester persona),
   a registration form (deliberately a form, not free-text slot-filling — the chat
   layer classifies *intent*, not structured fields, and pretending otherwise would
   misrepresent what NLU can reliably do here), a reviewer queue, and the audit log
   with a one-click chain-verification check.

## Risk taxonomy and controls

| Risk (OWASP LLM Top 10 category) | Control | Where |
|---|---|---|
| Prompt injection (LLM01) | Regex screen over 15 instruction-override pattern families, checked before retrieval or intent classification | `policy_engine.py::screen_for_injection` |
| Excessive agency (LLM08) | Single-entry tool allowlist; any other tool name refused and logged; malformed/unexpected arguments fail closed instead of crashing | `tools.py::IntakeStore.call_tool` |
| Misinformation / ungrounded generation (LLM09) | Answers only from the top retrieval hit, only above a tuned grounding threshold; refuses otherwise | `policy_engine.py::answer_policy_question` |
| Sensitive information disclosure (LLM06) | Structural, not just prompted: the answering path has no code reference to `IntakeStore`/`AuditLog` at all, so it cannot leak intake or audit records even if asked to | `policy_engine.py` (architecture) |
| Approval bypass (portfolio-specific) | Risk tier is assessed from description keywords, not from claims in the text; `approve()`/`reject()` require an explicit call with a named reviewer | `tools.py::assess_risk_tier`, `IntakeStore.approve/reject` |

`redteam_suite.py` is the executable version of this table: 13 adversarial prompts
across these five categories, each asserting the specific safe behavior expected of
it (not just "no crash"). Two of the cases found real gaps during development — a
grounding threshold loose enough to let two off-topic probes through as confidently
"answered," and a malformed tool call that crashed instead of failing closed — and
both are fixed in the modules above, with the fix explained inline in the case that
caught it, rather than the case being narrowed away.

## Sources

- Regulation (EU) 2024/1689 (the EU AI Act) — risk tiers, Article 5, Annex III,
  Article 50, reused from Case 01's own `risk_engine.py` taxonomy
- NIST AI Risk Management Framework 1.0 — the four-function structure referenced in
  the corpus
- OWASP Top 10 for Large Language Model Applications — the injection-pattern and
  excessive-agency control categories this project implements against
- This portfolio's own Case 01 and Case 03 outputs, as the corpus's governance content

## Running it

```bash
pip install -r requirements.txt
python -m pytest test_policy_engine.py test_tools_and_audit.py redteam_suite.py -v
python app.py   # launches the interactive Gradio demo locally
```

## Deploying to Hugging Face Spaces

Same free-tier situation as the rest of this portfolio: Hugging Face moved Gradio and
Docker Spaces behind a paid plan, so `app.py` needs a paid Space or local use. For the
free route, `index.html` in this folder is a complete, self-contained static demo: the
corpus is embedded directly, and the retrieval, injection-screening, intent
classification, tool-allowlist, risk-tier, and audit-hash-chain logic are all plain-JS
ports of the Python modules above — verified against the same worked queries, injection
attempts, and tamper-detection scenario as the Python test suites (see the page's own
self-test banner, which runs 35 checks on every load using the browser's built-in
`crypto.subtle` for hashing — no external library, no CDN, no network calls at all).

1. Create a new Space → SDK: **Static** → template: **Blank**.
2. Upload `index.html` from this folder (drag-and-drop through the HF web UI works
   fine — no git needed).
3. Done. No build step, no server, nothing installed at runtime.

## Test scenarios

- `test_policy_engine.py` (20 tests): retrieval accuracy against 6 worked queries,
  off-topic queries scoring zero, grounded-refusal behavior, citation correctness,
  intent classification (including the "how do I register" vs. "we are building and
  want to register" distinction), and 6 injection-attempt / false-positive checks.
- `test_tools_and_audit.py` (16 tests): risk-tier assessment, the tool allowlist gate,
  auto-approval vs. queued-for-review routing, the full approve/reject state machine
  (including double-approval and approving-a-never-queued-intake rejections), and the
  audit chain's linkage and tamper detection.
- `redteam_suite.py` (14 tests, 13 adversarial cases + 1 coverage check): the risk
  taxonomy table above, executed as adversarial prompts with pass/fail scoring per
  case rather than aggregate pass/fail.

All 50 tests pass.

## Limitations — read before relying on this for anything real

- **The corpus is 12 short passages this project wrote itself**, not a real
  regulatory or policy database — the retrieval and grounding mechanics are real and
  portable to a larger corpus, but the specific answers here shouldn't be treated as
  legal advice or an authoritative restatement of the EU AI Act.
- **The injection screen is regex-based**, matching the same pattern families
  `redteam_suite.py` tests against — it is not a learned classifier and will miss
  novel phrasings that don't match any of the 15 patterns. A production system would
  layer this with a model-based classifier, not rely on regex alone.
- **The intent classifier doesn't extract structured fields from free text** — the
  registration form exists because reliably turning "we're building a resume tool for
  our hiring team, mostly for engineering roles" into `{system_name, description,
  requester}` needs real NLU (or a real LLM call), which this dependency-free,
  deterministic design deliberately doesn't attempt.
- **There is no real authentication** — `approve()`/`reject()` take a reviewer name as
  a string, not a verified identity. A real deployment would gate the reviewer UI
  behind SSO/RBAC, not trust a free-text field.
- This is a portfolio project demonstrating governance *patterns* an agentic AI system
  needs, not a production-hardened agent framework.

## Roadmap / what I'd build next

- [ ] A `flag_for_review` escalation tool alongside the intake tool, so the allowlist
      demonstrates least-privilege with more than one entry.
- [ ] Extend the injection screen with a small learned classifier as a second layer,
      to quantify how much it catches that the regex screen misses.
- [ ] Reuse this project's retrieval and grounding pattern for the planned Case 03
      extension (a citation-grounded compliance copilot over a larger regulatory
      corpus) — see the top-level [Roadmap](../README.md#roadmap).

## License

All rights reserved — see `LICENSE`. Public here for evaluation by prospective employers and collaborators; not licensed for reuse without permission.
