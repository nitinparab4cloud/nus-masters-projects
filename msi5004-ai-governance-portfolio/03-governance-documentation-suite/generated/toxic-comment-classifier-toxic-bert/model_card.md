# Model Card: Toxic Comment Classifier (toxic-bert)

_Format follows Mitchell et al. (2019), "Model Cards for Model Reporting"._
Version 1.0 · Owner: Nitin Parab (portfolio project — not affiliated with Unitary AI) · Last updated: 2026-09-09

## Model Details
A BERT-based multi-label classifier that scores a piece of text for six categories of toxicity (toxic, severe_toxic, obscene, threat, insult, identity_hate), intended as a content-moderation signal for comment sections or chat platforms.


- **Model type:** Fine-tuned BERT-base, multi-label text classification
- **Developed by:** Nitin Parab (portfolio project — not affiliated with Unitary AI)
- **Base model:** bert-base-uncased

## Intended Use
**Primary intended uses:** Flagging candidate-toxic comments for human moderator review on a moderate-traffic community platform. Output is a confidence score per category, not an automatic removal decision.


**Out-of-scope uses:** Fully automated content removal without human review; use as the sole basis for account bans; use on languages other than English (the base model and training data are English-only); use as a general sentiment classifier.


## Training Data
Jigsaw Toxic Comment Classification Challenge dataset (Wikipedia talk-page comments, publicly released by Jigsaw/Conversation AI via Kaggle), human-labelled across six toxicity categories.


## Evaluation Data
Held-out split of the same Jigsaw dataset; the model card on Hugging Face reports per-category ROC-AUC. Re-validate on a sample of your own platform's comments before relying on these numbers -- the training distribution (Wikipedia talk pages, ~2017) will not match every deployment's actual comment style or period.


## Performance

- **ROC-AUC (toxic):** 0.98 (per published model card; re-validate for your data)

- **ROC-AUC (identity_hate):** 0.97 (per published model card; smallest-support category -- treat with the most caution)


## Ethical Considerations
Toxicity classifiers trained on this kind of data are documented in the literature to have disparate false-positive rates on text that mentions identity terms neutrally (e.g. "I am a gay man" scoring higher than a comparable neutral sentence) -- this is a known failure mode of the training data, not specific to this checkpoint, and should be tested for directly (see Risk Assessment) before this model gates any real action.


## Caveats and Recommendations
Use as a human-in-the-loop triage signal only at first deployment (see Human Oversight Design). Do not fully automate removals until a platform-specific fairness audit -- structured the same way as Case 02 in this portfolio -- has been run against real traffic.


---
_See also: [Risk Assessment](risk_assessment.md), [Data Governance Statement](data_governance.md),
[Human Oversight Design](human_oversight.md)._