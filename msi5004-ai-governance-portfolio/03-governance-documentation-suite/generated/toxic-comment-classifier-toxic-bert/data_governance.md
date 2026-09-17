# Data Governance Statement: Toxic Comment Classifier (toxic-bert)

Version 1.0 · Owner: Nitin Parab (portfolio project — not affiliated with Unitary AI) · Last updated: 2026-09-09

## Data Sources

- Jigsaw Toxic Comment Classification Challenge dataset (public, Kaggle-hosted, Wikipedia talk-page comments)


## Personal Data
- **Contains personal data:** Yes

- **Basis / handling:** Wikipedia talk-page comments may incidentally contain personal information volunteered by the original commenters. No additional PII is collected by this classifier itself; it processes platform comment text at inference time and does not persist it beyond the moderation queue's own retention policy.



## Data Governance Measures

- Inference-time text is retained only as long as the platform's own moderation-queue retention policy requires

- No re-training on live platform data without a separate data governance review

- Model checkpoint version pinned and logged for every deployment


## Known Limitations
Training data is English-only, sourced from a single platform (Wikipedia) and time period (~2017) -- performance on a different platform's comment style, slang, or non-English text is unvalidated.


---
_See also: [Model Card](model_card.md), [Risk Assessment](risk_assessment.md)._