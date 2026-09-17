# Risk Assessment: Toxic Comment Classifier (toxic-bert)

_Scoring method follows AI Verify Testing Framework 4.3.1: risk = probability × severity,
read alongside the scope of affected parties._
Version 1.0 · Owner: Nitin Parab (portfolio project — not affiliated with Unitary AI) · Last updated: 2026-09-09

## Risk Rating

| Factor | Rating |
|---|---|
| Probability of harm occurring | Medium |
| Severity if it occurs | Medium |
| Affected parties | Platform users whose comments are scored, Users targeted by content the system fails to catch, Human moderators relying on the signal |
| **Overall risk level** | **Medium** |

## Rationale
Probability rated Medium: false positives/negatives are a routine, expected occurrence for any moderation classifier, not a rare edge case. Severity rated Medium rather than High because the current design keeps a human moderator in the loop before any account-level action is taken -- severity would move to High if this were reconfigured for automatic removal or bans without review.


## Regulatory context

- **EU AI Act tier:** Likely outside Annex III (not squarely one of the 8 listed sectors) -- but re-run this through the AI Act Risk Navigator (Case 01) if scope expands, e.g. into automated account suspension, which starts to resemble an access-to-essential-services or employment-adjacent use case. — Content moderation is not itself an Annex III category as of the current text; monitor for sector-specific delegated acts.

- **Review cadence:** re-assess this rating on any material change to intended use,
  training data, deployment context, or after any incident logged in the
  [Incident Response Runbook](incident_response.md).

## Mitigations in place

- Human moderator reviews every flagged comment before removal (see Human Oversight Design)

- Confidence threshold for surfacing to moderators, not for auto-action

- Planned: identity-term false-positive audit before any expansion in scope (mirrors Case 02's methodology)


---
_See also: [Model Card](model_card.md), [Human Oversight Design](human_oversight.md),
[Incident Response Runbook](incident_response.md)._