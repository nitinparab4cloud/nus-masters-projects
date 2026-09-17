# Incident Response Runbook: Toxic Comment Classifier (toxic-bert)

Version 1.0 · Owner: Nitin Parab (portfolio project — not affiliated with Unitary AI) · Last updated: 2026-09-09

## Severity Levels

- **Sev 1:** Systematic over-flagging or under-flagging affecting a large share of a protected group's comments — target response time: 4 hours

- **Sev 2:** Isolated but harmful misclassification (e.g. a credible threat missed) — target response time: 24 hours

- **Sev 3:** Isolated false positive with no downstream harm (comment queued, later cleared on review) — target response time: 5 business days


## Detection
Moderator-reported misclassifications logged in the moderation tool; periodic sampling audit (see Case 02 methodology) run quarterly to catch systematic drift the moderator-reporting channel would miss.


## Response Steps
1. **Triage** — assign severity level and an incident owner.
2. **Contain** — Raise the flagging threshold or disable auto-surfacing for the affected category pending review.
3. **Notify** — escalate per severity level to: Trust & Safety lead, Model owner
4. **Remediate** — fix the root cause; if the root cause touches the model itself,
   trigger a re-assessment of the [Risk Assessment](risk_assessment.md).
5. **Report** — log the incident and outcome in the AI system inventory; notify
   affected parties where required.

## Post-Incident Review
Every incident above Severity 2 triggers a
post-incident review within 5 business days, covering: what
happened, why the existing controls didn't catch it, and what changes to this
runbook or the [Human Oversight Design](human_oversight.md) follow from it.

---
_See also: [Risk Assessment](risk_assessment.md), [Human Oversight Design](human_oversight.md)._