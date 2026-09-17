# Human Oversight Design: Toxic Comment Classifier (toxic-bert)

_Model chosen from the three standard oversight patterns: human-in-the-loop
(approve every decision), human-over-the-loop (monitor and can override),
human-out-of-the-loop (fully autonomous, reviewed periodically)._
Version 1.0 · Owner: Nitin Parab (portfolio project — not affiliated with Unitary AI) · Last updated: 2026-09-09

## Oversight Model
**Selected pattern:** Human-over-the-loop

**Rationale:** Matched to a Medium overall risk level: moderators are not required to approve every single score, but they review every comment the model surfaces above the flagging threshold, and can override or retrain the threshold. This sits deliberately short of full autonomy (human-out-of- the-loop) given the documented identity-term false-positive risk noted in the Model Card.


This pattern is matched to the system's overall risk level of
**Medium** (see [Risk Assessment](risk_assessment.md)) — the
higher the risk, the closer to human-in-the-loop the design should sit.

## Escalation Path
Flagged comment > moderation queue > human moderator decision. Comments scoring in the top confidence band for "threat" are escalated to a senior moderator within 1 hour regardless of queue volume.


## Contestability
Individuals affected by this system's outputs can challenge a decision via:
A user whose comment was removed can appeal through the platform's standard content-appeal form; appeals are reviewed by a moderator who did not make the original removal decision.


---
_See also: [Risk Assessment](risk_assessment.md), [Incident Response Runbook](incident_response.md)._