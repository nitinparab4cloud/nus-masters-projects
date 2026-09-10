# Decoding Justice: Machine Learning Analysis of Trial Outcomes and Sentencing Patterns in Old Bailey Criminal Proceedings (1902–1913)

**LL5532X — Law, Algorithms, and Artificial Intelligence**

**Group 3:** Parab Nitin · Pakhale Kalyani Vijay · Shao Lujie · Zhang Xiaoyue

**Institution:** National University of Singapore

**Professor:** Ilya Akdemir

**Date:** April 2026

---

## Abstract

This study applies machine learning and natural language processing to 9,192 digitised criminal trial records from London's Old Bailey (1902–1913) to investigate whether historical legal proceedings encode systematic patterns of prediction, bias, and sentencing. We address three interconnected research questions: (1) Can case narrative text predict guilty versus not guilty verdicts? (2) Does defendant gender affect punishment severity after controlling for case-level factors? (3) Can we predict punishment type for convicted defendants? Our results reveal that verdict prediction achieves 99.33% F1-score (XGBoost with TF-IDF), though we identify indirect data leakage through sentencing language embedded in trial narratives — itself evidence of how deeply outcomes permeate historical legal text. For gender bias, progressive logistic regression models show that female defendants received less harsh punishment (odds ratio ≈ 0.83 after full controls), consistent with Victorian paternalism, though the effect becomes statistically non-significant once all covariates are included (95% CI crosses 1.0) — suggesting gender operated indirectly through structural crime-type sorting rather than direct court-level discrimination. Crime-specific analysis reveals a striking "double deviance" effect for violent theft (OR ≈ 4.03), where women who violated gendered behavioural expectations were punished far more harshly than men. Punishment type prediction achieves macro F1 = 0.854 (LightGBM), with text features dramatically improving over structured-only baselines. Taken together, the three research questions demonstrate a complete bias propagation pipeline: biased judicial decisions are recorded in trial text, the text encodes those biases, and models trained on the text learn to reproduce them. These findings carry direct implications for modern legal AI systems, where training on historical data risks perpetuating the very inequities that contemporary society seeks to overcome.

**Keywords:** legal judgment prediction, natural language processing, algorithmic fairness, gender bias, historical corpus analysis, Old Bailey, text classification

---

## 1. Introduction

Between 1674 and 1913, London's Old Bailey — the Central Criminal Court of England and Wales — processed over 197,000 criminal trials. The full proceedings were recorded verbatim and have since been digitised, creating one of the richest historical legal corpora in existence. This project asks: *can machine learning models trained on these trial narratives reveal systematic patterns in how defendants were judged and sentenced?*

We investigate three interconnected research questions that collectively address prediction, bias, and sentencing consistency in historical legal proceedings:

**RQ1 — Verdict Prediction:** Can we predict whether a defendant was found guilty or not guilty based solely on the case narrative text, excluding explicit verdict statements?

**RQ2 — Gender Bias in Sentencing:** After controlling for crime type and other case-level factors, does defendant gender affect the severity of punishment?

**RQ3 — Punishment Type Prediction:** For convicted defendants, can we predict whether they received hard labour versus other punishment types?

These questions have direct relevance to modern legal AI. If historical trial text encodes predictable patterns of guilt and sentencing, then AI systems trained on such data risk inheriting and perpetuating the biases embedded within them. The COMPAS recidivism algorithm (Angwin et al., 2016) demonstrated that training on historical criminal justice data reproduces racial bias; the GRADE system (discussed in our course, Week 6) codified historical patterns into automated assessments. Our project tests the same mechanism using gender as the protected characteristic and historical English courts as the data source.

The three RQs form a single coherent argument. We audit the Old Bailey as if it were an algorithm — testing for predictability (RQ1), fairness (RQ2), and consistency (RQ3). Following the DRME pipeline (Data → Representation → Model → Evaluation) introduced in our course labs, each question employs a different modelling strategy suited to its analytical purpose: binary classification for RQ1, causal inference through progressive modelling for RQ2, and multiclass classification for RQ3.

---

## 2. Literature Review

### 2.1 Legal Judgment Prediction

Legal judgment prediction (LJP) is an established task in legal NLP, defined as predicting case outcomes from textual fact descriptions. Cui et al. (2022) survey datasets, evaluation metrics, and modelling approaches, documenting steady progress from bag-of-words baselines to transformer-based architectures. Aletras et al. (2016) demonstrated that factual content predicts European Court of Human Rights decisions better than legal arguments — a finding that aligns with the legal realist tradition emphasising the "stimulus of the facts" over formal doctrine.

Medvedeva et al. (2020) extended this work using machine learning to predict ECHR decisions, achieving 75–80% accuracy. Our study complements these modern-court analyses by applying similar techniques to a historical corpus, where the interplay between legal language and judicial outcomes may be even more transparent.

### 2.2 The Old Bailey Corpus in Digital Humanities

The Old Bailey Proceedings Online provides a fully digitised, structured corpus of all surviving published trial accounts from 1674–1913 (Hitchcock & Turkel, 2016). The official documentation notes that the XML is "large and complex" and that a single trial may contain multiple defendants, offences, victims, verdicts, and sentences — motivating careful preprocessing and label construction.

Klingenstein et al. (2014) analysed the same corpus using descriptive linguistics, tracking the "civilising process" in court language over time. Our project extends their work from descriptive analysis to predictive modelling: rather than characterising how language changed, we test whether language patterns can predict judicial outcomes.

Digital humanities scholarship also emphasises that Old Bailey trials are mediated historical texts whose markup and conventions vary across time, affecting extraction and analysis. We address this by focusing on the 1902–1913 period, where OCR quality is highest and data completeness is strongest.

### 2.3 Algorithmic Fairness and Bias Inheritance

Barocas and Selbst (2016) argue that discriminatory outcomes in algorithmic systems can arise from biased training data even without discriminatory intent. Angwin et al. (2016) provided a high-profile demonstration with COMPAS, showing that the recidivism prediction tool exhibited racial bias learned from historical sentencing data. Bender et al. (2021) extended this concern to large language models, arguing that systems trained on historical corpora reproduce patterns without understanding — the "stochastic parrots" problem.

Our RQ2 analysis directly tests the Barocas and Selbst mechanism: if historical sentencing data contains gender disparities, a model trained on that data will learn and reproduce those disparities. The progressive modelling design (isolating the gender effect while controlling for confounders) follows standard causal inference methodology, providing a template for algorithmic auditing that extends beyond our specific dataset.

### 2.4 Text Representation for Legal NLP

Given the practical considerations of our historical corpus — domain-specific language, limited dataset size, and the need for interpretability — this project adopts interpretable classical ML baselines. TF-IDF with logistic regression, Naive Bayes, and linear SVM create a transparent benchmark for guilty versus not guilty prediction on historical trial narratives (Ariai & Demartini, 2024). We also explore transformer-based approaches (Legal-BERT) for comparison, connecting to course materials on pre-training and contextual embeddings (Xiao & Zhu, 2025, Ch. 1).

---

## 3. Data and Exploratory Analysis

### 3.1 Data Source and Collection

The dataset originates from the Old Bailey Proceedings Online, parsed from XML into structured records via the DHI Old Bailey API. While the Proceedings span 1674–1913, our analysis draws primarily from 1902–1913, where digitised data is most complete and OCR quality highest. The API returned 9,192 trial records, each containing the full trial text, defendant demographics, offence category, verdict, and punishment metadata.

### 3.2 Dataset Summary

| Property | Value |
|---|---|
| Total trial records | 9,192 |
| Period | 1902–1913 |
| Raw features | 19 |
| Engineered features | 58 |
| Verdict distribution | ~80% guilty, ~20% not guilty |
| Gender distribution | ~90% male, ~10% female |
| Dominant crime type | Theft (>50% of all trials) |

### 3.3 Exploratory Data Analysis

Our EDA was conducted across three phases, producing 32+ engineered features that informed all three research questions.

#### Phase 1: Data Quality and Descriptive Statistics

**Class distribution and temporal patterns.** The verdict variable exhibits significant class imbalance: approximately 80% guilty and 20% not guilty. Conviction rates remained stable between 76–84% across 1902–1913, with 600–800 trials per year. Different crime categories show distinct temporal trajectories — deception offences rose while killing remained stable. This stability suggests that aggregate judicial behaviour was remarkably consistent across the period studied.

#### Figure 1: Missing Values Analysis
![Missing Values](report_charts/eda_basic_00.png)
*Missing value rates by column. Occupations are nearly 100% missing; punishment data is absent for acquitted defendants (structurally expected).*

**Text length as a diagnostic feature.** Trial text lengths are heavily right-skewed (median 1,036 characters). A log transformation reveals a bimodal distribution — one peak around log(5.5–6) corresponding to brief guilty pleas (~150–400 characters), and another around log(8.5–9) representing full contested trials (~4,500–8,000 characters). Not-guilty cases average 4,775 characters versus 3,616 for guilty (+32%). This bimodal structure reflects two fundamentally different trial types and becomes a critical feature for modelling.

#### Figure 2: Text Length Distribution
![Text Length Distribution](report_charts/eda_basic_03.png)
*Log-transformed text length reveals a bimodal distribution: brief guilty pleas (~150–400 chars) and full contested trials (~4,500–8,000 chars).*

**Crime categories and punishment distribution.** Theft dominates the dataset — property crimes constitute over 50% of all trials, making the Old Bailey primarily a property-crime court. Conviction rates vary dramatically by crime type: theft at 84%, but killing at only 67% — the lowest — reflecting more contested facts and a higher burden of proof. Some offence subcategories approach near-100% conviction rates, raising questions about whether certain charges were only brought when conviction was essentially guaranteed. Murder conviction rate was approximately 89%. Among guilty defendants, hard labour dominates punishment (~3,796 cases, 55%), followed by penal servitude and other custodial (~2,300 cases, 33%), non-custodial measures (~760 cases, 11%), and capital sentences (76 cases, 1%) — rare but most consequential. This severe class imbalance creates the central challenge for RQ3: standard accuracy metrics would ignore rare classes entirely.

#### Phase 2: NLP Feature Engineering

We applied a comprehensive NLP pipeline: text cleaning, tokenisation, stopword removal, and lemmatisation using spaCy. Vocabulary was reduced from 62,488 to 34,210 tokens (45% reduction). Features were extracted across multiple dimensions:

| Feature Category | Features | Key Finding |
|---|---|---|
| POS distributions | Noun, verb, adj, adv, pronoun, numeral counts | All higher for guilty (longer texts); ratios differ — guilty has more pronouns, not guilty more adjectives |
| Named entities | Person, location, date, money, organisation counts | Money mentions higher in guilty cases (amounts specified in theft/deception) |
| Sentiment (VADER) | Compound, positive, negative, neutral scores | Negative sentiment slightly higher in guilty cases; subtle but detectable |
| Topic probabilities | 10 LDA topics | Topic 0 (sentencing language) strongly associated with guilty; Topics 2, 7 (defence proceedings) with not guilty |
| Readability | Flesch, Flesch-Kincaid, Gunning Fog, SMOG, ARI | Not-guilty cases consistently more complex; independent of text length |

**Correlation analysis** revealed that POS counts, entity counts, and text length are highly correlated (r > 0.9) — they measure the same thing: trial length. Sentiment and readability form independent clusters that add genuine new information. We used log_text_length as a single proxy and prioritised independent features.

**Word frequency analysis** by verdict class reveals meaningful vocabulary differences. Guilty-verdict texts prominently feature "sentence", "labour", "conviction" — sentencing-related vocabulary. Not-guilty texts emphasise "go", "defence", "statement", "witness" — language of contested proceedings. This vocabulary difference foreshadowed the data leakage concern we later identified in RQ1 modelling.

#### Phase 3: Close Reading — From Distant Patterns to Actual Cases

Aggregate statistics reveal patterns, but understanding what the data actually looks like is essential for responsible modelling.

A typical guilty plea is brief and formulaic: *"WALTER HEATH (82), PLEADED GUILTY to stealing £13, the money of George Whitehead, his master, also to stealing £19 4s., the money of Henry John Manning, his master, having been convicted of felony at Clerkenwell..."* — 268 characters, no defence narrative, immediate sentencing.

A contested not-guilty trial reads very differently: longer narratives (5,000–50,000+ characters) with witness testimony, cross-examination, defence arguments, and judicial directions. These cases contain procedural vocabulary ("the jury", "no evidence offered", "prosecution withdrew") that the model learns to associate with acquittal.

**KWIC (Keyword-in-Context) analysis** revealed a striking pattern: the same keyword appears in different syntactic contexts depending on verdict. In guilty cases, words like "steal" appear in short, factual clauses ("convicted of stealing"). In not-guilty cases, the same words are embedded in longer, more qualified sentences — often preceded by negation or attributed to disputed testimony ("did not steal", "alleged to have stolen"). This observation directly informed our hypothesis that text structure, not just vocabulary, carries predictive signal.

---

## 4. Data Preparation for Modelling

### 4.1 Data Cleaning and Anomaly Resolution

Missing value patterns differed critically between guilty and not-guilty cases. Among guilty defendants, 124 records (1.7%) with missing punishment were dropped. Among not-guilty defendants, 1,429 (82.1%) had no punishment data — expected and imputed as "no_punishment". We identified 312 not-guilty cases with anomalous punishment entries (82.4% procedural notes, 9.6% prosecution withdrawals, 4.2% complex multi-charge trials, 3.8% jury acquittals with annotations). All were removed.

### 4.2 Punishment Subcategory Grouping

For RQ3, 15+ punishment subcategories were consolidated into four groups:

| Punishment Group | Count | Percentage |
|---|---|---|
| Hard Labour | 3,530 | 50.8% |
| Custodial/Prison | 2,047 | 29.5% |
| Non-Custodial/Institutional | 1,293 | 18.6% |
| Capital/Extreme | 75 | 1.1% |

### 4.3 Final Datasets

| Research Question | Sample Size | Split | Target |
|---|---|---|---|
| RQ1: Verdict Prediction | 8,167 trials | 6,534 / 1,633 (80/20) | Binary: guilty vs not guilty |
| RQ2: Gender Bias | 6,945 guilty trials | 5,556 / 1,389 (80/20) | Binary: harsh vs lenient |
| RQ3: Punishment Prediction | 6,945 guilty cases | 4,861 / 2,084 (70/30) | 4-class: punishment type |

All splits used stratification to preserve class distributions.

---

## 5. Text Representation Methods — Overview

Each research question employs text representation methods tailored to its analytical purpose. Rather than presenting these in a shared section, we document the specific representation choices — including TF-IDF configuration, Doc2Vec embeddings, Fightin' Words analysis, and transformer-based approaches — within each RQ section below, alongside the charts and results they produce. This allows the reader to follow the complete end-to-end pipeline for each question without switching between sections.

Common across all RQs: TF-IDF (Term Frequency–Inverse Document Frequency) serves as the primary representation, weighting words by importance relative to the corpus. We use 10,000 features with unigram + bigram n-grams, sublinear TF scaling, minimum document frequency of 5, and maximum document frequency of 80%. Structured NLP features (POS distributions, named entity counts, sentiment scores, topic probabilities, readability metrics) complement text representations where appropriate.

---

## 6. Modelling Overview

We employed 25+ model configurations across three research questions. Each model was selected for a specific purpose:

| Model | Purpose | Key Advantage |
|---|---|---|
| Logistic Regression | RQ1 baseline, RQ2 causal inference | Full coefficient-level interpretability; each weight directly shows feature impact |
| Linear SVM | RQ1 comparison | Effective for high-dimensional sparse TF-IDF features |
| Random Forest | All RQs | Captures non-linear interactions; ensemble stability |
| XGBoost | RQ1 best performer | Sequential error correction; handles sparse data and class imbalance |
| LightGBM | RQ3 best performer | Histogram-based splitting; efficient with high-dimensional sparse features |
| Legal-BERT / DistilBERT | RQ3 exploratory | Contextual embeddings; attention weight analysis |

**Class imbalance handling** used `class_weight='balanced'` (automatically adjusting weights inversely proportional to class frequencies) for RQ1 and RQ2, and SMOTE (Synthetic Minority Over-sampling Technique) for RQ3's more severe 4-class imbalance.

**SHAP (SHapley Additive exPlanations)** — not a model but an explainability framework — was used in RQ2 and RQ3 to decompose each prediction into per-feature contributions using game-theoretic Shapley values (Lundberg & Lee, 2017).

All models used stratified splitting, cross-validation, and held-out test sets.

---

## 6b. Evaluation Framework

### The Accuracy Trap

A DummyClassifier that always predicts "guilty" achieves 83% accuracy — but catches zero not-guilty verdicts (minority class F1 = 0%). This is the "accuracy trap" we discussed in our Week 6 lab: headline accuracy is meaningless with imbalanced classes. Our best model (XGBoost) reaches 99.3% weighted F1, with 97.5% minority-class F1 — a genuine, substantive improvement over the naive baseline.

### Metrics Toolkit

Each research question demands different evaluation metrics. Matching the right metric to the right question is half the battle:

| Research Question | Primary Metrics | Why These Metrics | Course Reference |
|---|---|---|---|
| RQ1: Verdict Prediction | F1-Score + AUC-ROC | Binary task, imbalanced classes (79/21). F1 balances precision and recall. AUC measures discrimination ability. | DummyClassifier (Week 6), Akdemir 2023 |
| RQ2: Gender Bias | Odds Ratio + SHAP Values | Explanatory/causal focus. OR quantifies effect size. SHAP provides feature-level explanation. | COMPAS (Week 10), GRADE (Week 6) |
| RQ3: Punishment Type | Macro F1 + Per-Class F1 | Multiclass (4 classes). Macro F1 treats rare classes equally — critical when capital cases are 1%. | Harcourt "Against Prediction" (Week 10) |

---

## 7. RQ1: Predicting Guilty from Case Narratives in Historical Trial Records

## 7.1 Research Question

**Can we predict whether a defendant was found guilty or not guilty based solely on the case narrative text?**

This question sits at the intersection of legal AI and algorithmic accountability. Historical criminal trial records contain rich textual data, but they are also products of their time—shaped by judicial discretion, social biases, and institutional practices of the early 20th century. If machine learning models can reliably predict outcomes from narrative text alone, what does this tell us about whether legal language genuinely reflects facts and evidence, or whether it encodes deeper patterns of institutional bias? This research probes a classical concern in legal scholarship: do the texts of law contain inherent biases that AI systems might amplify or perpetuate? Our Old Bailey dataset (1902-1913) provides a historical lens on this question.

## 7.2 Text Representation

We employed a TF-IDF feature extraction pipeline tailored to sparse legal text. **TF-IDF with 10,000 features** was computed using unigrams and bigrams, with `sublinear_tf=True` to dampen the influence of high-frequency terms. We set `min_df=5` (terms appearing in at least 5 documents) and `max_df=0.8` (excluding terms in more than 80% of documents) to filter noise and remove overly common words. This configuration resulted in **96.14% sparsity**, typical for legal text corpora where vocabulary is domain-specific but repetitive.

We also constructed a **hybrid feature set (10,048 total features)** combining:
- TF-IDF (10,000 features)
- One-hot encoded crime categories (10 features)
- NLP features (31 features): part-of-speech tag distributions, named entity recognition counts, sentiment polarity, topic model weights, and readability metrics (Flesch-Kincaid grade level)
- Metadata (7 features): defendant gender, trial year, case narrative length

TF-IDF grounding in the distributional hypothesis (Firth, 1957; Weeks 7-8 lectures) assumes that words appearing in similar contexts share semantic meaning. For legal narratives, this means verdicts should correlate with the semantic structure of case descriptions, not merely keyword presence.

## 7.3 Data Leakage Prevention

A critical methodological step was identifying and removing **verdict-revealing patterns** from the text. We applied regex-based filtering to eliminate phrases that directly referenced conviction, sentencing, or acquittal—such as "found guilty," "acquitted," "sentenced to," and related verdict-adjacent language. We removed **20+ patterns** identified through iterative manual inspection.

This preprocessing modified **91.9% of documents**, removing an average of **18-20 characters per document**. Without this intervention, the model would trivially achieve high accuracy by learning to detect explicit verdict language rather than inferring guilt from narrative evidence. This approach parallels Aletras et al. (2016), who used specific case sections (holding paragraphs) while excluding verdict statements. The leakage removal ensures we are testing whether *case facts and narrative language* predict outcomes, not whether verdict keywords are present.

### Figure: Verdict Class Distribution

![](report_charts/rq1_00.png)

*The target variable shows severe class imbalance: 83% guilty verdicts (N=6,674) versus 17% not guilty (N=1,493). A naive baseline classifier always predicting "guilty" achieves 83% accuracy but 0% F1-score on the minority class. This imbalance necessitates both stratified splitting and class-weighted loss functions.*

## 7.4 Class Imbalance Strategy

With an 83:17 guilt/acquittal ratio, we employed `class_weight='balanced'` in all models, which automatically assigned minority class weight ≈ 4.86× that of the majority. This rebalancing is more efficient than SMOTE (Synthetic Minority Oversampling) for sparse TF-IDF matrices, which would generate synthetic bigrams of dubious interpretability. We combined balanced weighting with a stratified 80/20 train-test split (N=5,340 training, N=1,331 test) to preserve class distributions in both partitions.

## 7.5 Model Training and Hyperparameter Tuning

We conducted GridSearchCV with 5-fold stratified cross-validation across four algorithms, each evaluated on both TF-IDF and hybrid feature sets (8 configurations total):

- **Logistic Regression**: `C=[0.1, 1, 10]`, `solver=['liblinear', 'saga']`. Optimal: C=10, liblinear solver (4s training).
- **Linear SVM**: `C=[0.1, 1, 10]`. Optimal: C=1 (2s training).
- **Random Forest**: `n_estimators=[100, 200]`, `max_depth=[10, 20, None]`. Optimal: 200 trees, no depth limit (21s training).
- **XGBoost**: `n_estimators=[100, 200]`, `max_depth=[3, 5, 7]`, `learning_rate=[0.1, 0.3]`. Optimal: 200 trees, max_depth=5, learning_rate=0.1 (159s training).

All models were tuned on the training set; performance is reported on the held-out test set.

## 7.6 Results and Model Comparison

### Figure: Model Comparison Dashboard

![](report_charts/rq1_01.png)

*Four-panel comparison across all 8 configurations: (top-left) F1-scores by model type and feature set; (top-right) AUC-ROC scores; (bottom-left) precision vs. recall scatter plot colored by algorithm; (bottom-right) feature type contribution analysis. All models substantially outperformed the 83% baseline dummy classifier. Notably, TF-IDF features alone were nearly sufficient; hybrid features added minimal improvement.*

The **best-performing model was XGBoost trained on TF-IDF features**: F1-score = **99.33%**, AUC-ROC = **99.75%**, with only **18 total errors on 1,331 test samples (1.1% error rate)**. Cross-validation F1 was 99.15% ± 0.5%, indicating consistent generalization. Logistic Regression and Linear SVM achieved F1 ≈ 98%, while Random Forest peaked at F1 ≈ 98.9%. The small performance gap between tree-based and linear models suggests that feature engineering (TF-IDF) captured most predictive signal.

### Figure: Best Model Evaluation

![](report_charts/rq1_02.png)

*Confusion matrix for the best model (XGBoost TF-IDF): True Negatives (TN) = 271, False Positives (FP) = 7, False Negatives (FN) = 11, True Positives (TP) = 1,344. The model is highly sensitive (recall ≈ 99.2%) and specific (specificity ≈ 97.5%). Only 18 misclassifications across 1,331 held-out test samples.*

## 7.7 Honest Interpretation: Is 99% Suspicious?

At face value, 99.3% F1 on a legal prediction task seems implausibly high. Three points warrant caution:

1. **Residual Indirect Leakage**: Despite removing explicit verdict keywords, case narratives contain indirect signals—descriptions of sentencing procedures, procedural language ("sentence to hard labour"), and institutional routines may correlate with guilt without being literal verdict mention.

2. **TF-IDF Matched Hybrid Performance**: The hybrid feature set (including metadata and NLP features) did not materially improve accuracy, suggesting that **case narratives alone implicitly encode metadata** (defendant demographics, trial year, case severity). This is concerning from a fairness perspective: the text is not neutral.

3. **Consistency Over Overfitting**: Cross-validation F1 of 99.15% ± 0.5% shows the result is not a one-time test-set fluke. Regularization in Logistic Regression and SVM also achieved >98% F1, indicating the signal is robust. However, robustness does not imply the signal measures justice—only that legal narrative is strongly predictive.

**Conclusion**: The predictability itself is the finding. We are not claiming that AI can justly predict guilt; rather, we are empirically demonstrating that historical legal text is structured such that outcomes are highly predictable from narrative language. This is evidence that outcomes permeate the text.

## 7.8 Feature Importance and Data Leakage Discovery

Inspecting the XGBoost model's SHAP feature importances revealed the mechanism of prediction:

**Top guilty-predictive features**: "sentence", "hard labour", "imprisonment", "months hard", "penal servitude"—all sentencing-related terms.

**Top not-guilty-predictive features**: "no evidence", "jury", "defended", "prosecution offered", "discharged"—all procedural and defence language.

### Figure: Top Predictive Features

![](report_charts/rq1_03.png)

*Bar plot ranking the 20 most important features by Shapley value contribution. Sentencing terms dominate guilty predictions, while procedural and acquittal language dominates not-guilty predictions. This feature decomposition reveals that the model is leveraging outcome-adjacent language, not truly independent case facts.*

This pattern is sobering: sentencing terms (which logically follow a guilty verdict) are the strongest predictors. While we removed explicit verdict keywords, the semantic space of legal narrative is sufficiently structured that sentencing language acts as a proxy. This is not a model failure; it reflects a genuine property of historical legal texts—outcomes are written into their narrative structure.

## 7.9 Error Analysis

The 18 test errors fell into two categories:

- **7 False Positives** (predicted guilty, actually acquitted): Predominantly shorter texts, property crimes, or cases where initial incriminating evidence was contradicted by defence testimony. These represent the most concerning error type—false convictions—though the historical jury reached the correct verdict.

- **11 False Negatives** (predicted not guilty, actually convicted): Longer, more complex multi-defendant cases with strong defence language that convinced the model but not the jury. The model was fooled by persuasive defence rhetoric.

Notably, the model's 1.1% error rate sits below estimates of modern wrongful conviction rates (1–5% in contemporary systems), yet the **kinds of errors are different**. Historical juries convicted people despite weak evidence (our FNs); modern wrongful convictions often involve eyewitness misidentification or forensic error. The comparison is not direct, but it suggests that algorithmic prediction and human judgment are correlated over time.

## 7.10 TF-IDF vs. Hybrid Features

TF-IDF features alone achieved 99.33% F1; adding crime categories, NLP features, and metadata improved performance by <0.5 percentage points. This indicates that **text alone is nearly sufficient**. Case narratives implicitly encode the metadata we explicitly represented—gender, crime type, and severity are embedded in how the case is described. Simpler features are preferable for legal AI, both for interpretability and to avoid redundantly encoding protected characteristics.

## 7.11 Key Insight: Legal Realism Confirmed in Data

Our finding supports a classical insight from legal realism: law is not purely formal rule application but is shaped by facts, context, and institutional practice. We can frame the result in three parts:

- **The Prediction**: 99.3% F1 on guilt classification from narrative text.
- **The Mechanism**: Case outcomes are encoded in the "stimulus of the facts" (Aletras et al., 2016, Weeks 6–7)—the way facts are presented in legal writing inherently correlates with verdicts.
- **The Warning**: An AI system trained on these narratives would not learn justice; it would learn pattern matching. It would internalize whatever biases shaped historical verdict language and could project them onto new cases.

This research reinforces a broader cautionary theme: historical legal data should be used with extreme care in machine learning. The high predictability is not a feature of legal AI but a warning that algorithms trained on flawed or biased historical data will replicate and automate those flaws (Course ref: Legal Realism, weeks 2-3; social class encoding in legal language, week 9).

---

## 8. Research Question 2: Gender and Punishment Severity

## 8.1 Research Question

**RQ2: After controlling for crime type and other case-level factors, does defendant gender affect the severity of punishment?**

This question matters for two reasons. First, it directly reveals Victorian gender disparities in criminal sentencing—whether courts showed leniency toward women or, conversely, applied harsher penalties for violating gendered behavioral expectations. Second, it demonstrates a critical mechanism by which algorithmic systems inherit historical bias: if gender effects persist *after* controlling for observable offense characteristics, any ML model trained on historical trial records will learn to replicate these patterns, potentially amplifying them at scale. This connects directly to the COMPAS recidivism predictor (Week 10), which similarly embedded racial disparities, and to GRADE's framework (Week 6) for understanding how algorithms operationalize existing social hierarchies.

---

## 8.2 Text Representation for RQ2

Unlike RQ1, which prioritized predictive performance, RQ2 emphasizes causal inference and interpretability. Our approach combines:

- **Structured features**: crime type (19 categories), defendant age, occupation, plea, prior convictions, trial date, trial location, verdict certainty
- **TF-IDF vectorization**: court narrative documents reduced to 500 most frequent terms with TF-IDF weighting to capture case-specific language
- **Numerical preprocessing**: StandardScaler applied to all continuous features (age, temporal features, document length)
- **Model choice**: Logistic regression preferred over random forests for primary causal inference because coefficient interpretation directly estimates log-odds ratios, enabling hypothesis testing. We validate with Random Forest + SHAP for robustness.

This hybrid design—structured + text—tests whether gender effects operate independently of documented crime content, a key question for bias inheritance.

---

## 8.3 Gender EDA

Our sample comprises 6,945 defendants with guilty verdicts (72.0% harsh punishment, 28.0% lenient). Figure 8.1 reveals a striking raw gender gap in harsh punishment rates between male and female defendants.

![](report_charts/rq2_00.png)
*Figure 8.1: Gender and Punishment Distribution. Left: harsh punishment rate by gender. Centre: punishment subcategory distribution (transportation, execution, imprisonment) by gender. Right: offence category distribution by gender, revealing gender-crime associations.*

However, this raw gap conflates gender effects with crime composition: women were overrepresented in theft and receiving stolen goods, underrepresented in violent crimes like murder and assault. To isolate the causal effect of gender, we must control for these structural differences.

---

## 8.4 Progressive Model Design: M1 → M2 → M3

We employ a causal-inference design with three nested models, each adding layers of control:

- **M1 (Gender only)**: Estimates raw gender effect without controls. Represents the naive AUC-maximizing approach that COMPAS critics identified.
- **M2 (+ Crime type)**: Adds 19 crime-type dummies to isolate gender effect from offense composition.
- **M3 (+ All controls)**: Adds defendant characteristics, temporal features, trial metadata, and TF-IDF text features—the full feature set.

All models use balanced class weights (addressing our 72:28 imbalance) and stratified 80/20 train-test split. Target: binary harsh vs. lenient. Primary metric: odds ratios (OR) with 95% confidence intervals; secondary metrics: accuracy, F1-macro, AUC-ROC.

---

## 8.5 Results and Odds Ratios

Model performance improves monotonically with feature richness:

| Model | Accuracy | F1-Macro | AUC-ROC |
|-------|----------|----------|---------|
| M1: Gender only | 0.544 | 0.446 | 0.529 |
| M2: + Crime type | 0.605 | 0.603 | 0.651 |
| M3: + All controls | 0.618 | 0.618 | 0.678 |
| Random Forest (full) | 0.685 | 0.685 | 0.769 |
| Hybrid TF-IDF (full) | 0.631 | 0.631 | 0.680 |

More important than raw AUC, the gender coefficient exhibits significant attenuation:

- **M1**: OR = 0.44 (95% CI: 0.37–0.52) — females 56% less likely to receive harsh punishment
- **M2**: OR = 0.55 (95% CI: 0.45–0.66) — attenuation with crime controls reduces the gap to 45%
- **M3**: OR = 0.83 (95% CI: 0.69–1.01) — further attenuation to 17%, with CI now crossing 1.0, indicating the effect is no longer statistically significant at the 5% level

This monotonic attenuation (0.44 → 0.83) is consistent with *structural gender sorting* — gender influenced sentencing indirectly, through which crime types women were charged with and how those crimes were narrated, rather than through direct court-level discrimination in identical cases. Much of the raw gender gap is confounded by crime composition: women were overrepresented in theft and underrepresented in violent crimes.

![](report_charts/rq2_01.png)
*Figure 8.2: Gender Effect Across Models. Left: odds ratios with 95% confidence intervals across M1, M2, M3, showing monotonic attenuation toward 1.0. Right: M3 top logistic regression coefficients (by absolute value), with defendant_female (green) highlighted.*

![](report_charts/rq2_02.png)
*Figure 8.3: Confusion Matrices. Side-by-side M1 (AUC=0.529), M2 (AUC=0.651), M3 (AUC=0.678). Adding features improves discrimination progressively.*

The attenuation pattern suggests a "structural sorting hypothesis": much of the observed gender gap is explained by differences in which crimes women were charged with, rather than direct judicial discrimination within identical cases. The M3 confidence interval crossing 1.0 indicates that after full controls, the residual gender effect is not statistically distinguishable from zero at the aggregate level. However, as we will see, this aggregate non-significance masks striking crime-specific variation.

---

## 8.6 SHAP Analysis

To validate our logistic regression findings with a non-parametric approach, we trained a Random Forest on the full feature set and computed SHAP (SHapley Additive exPlanations) values.

![](report_charts/rq2_03.png)
*Figure 8.4: SHAP Feature Importance (Random Forest). Mean absolute SHAP values ranked by importance. defendant_female ranks among the top 5 features, corroborating the logistic regression estimate.*

![](report_charts/rq2_04.png)
*Figure 8.5: SHAP Beeswarm Plot. Each dot represents one case. Horizontal position = SHAP value (impact on harsh probability). Color = feature value (blue = female, red = male). The leftward cloud of blue dots confirms that female gender pushes predictions toward leniency.*

Triangulation between parametric (logistic regression) and non-parametric (Random Forest + SHAP) methods reinforces our conclusion: gender is a salient, real predictor, not an artifact of model choice.

---

## 8.7 NLP Hybrid Model Validation

We also trained a logistic regression on concatenated structured features + TF-IDF text features (500 terms). If gender effects were entirely captured by courtroom language (e.g., harsher adjectives applied to male defendants), this hybrid model should diminish the gender coefficient. Instead:

![](report_charts/rq2_05.png)
*Figure 8.6: Hybrid Model (Structured + TF-IDF) Top 20 Coefficients. Red bars = features pushing toward harsh; blue bars = toward lenient. defendant_female (green) ranks among the top features, pulling strongly toward leniency. Top TF-IDF terms (e.g., "mercy," "young") appear alongside.*

The gender coefficient remains substantial even when TF-IDF features are present. This indicates that gender bias operates **independently of documented trial narrative**, suggesting judges' gender assumptions were baked into sentencing decisions apart from linguistic framing.

---

## 8.8 The Double Deviance Finding

Our stratified analysis by crime type, however, reveals that the aggregate non-significance of M3 masks dramatic crime-specific variation. While most offenses show the expected leniency pattern (female OR < 1), violent theft stands out sharply:

![](report_charts/rq2_07.png)
*Figure 8.7: Gender Bias by Crime Type. Three panels showing gender odds ratios for the 10 most common offenses. Most crimes (theft, receiving) show paternalism (OR < 1). Sexual offences show the strongest leniency (OR ≈ 0.37). Violent theft shows the strongest double deviance (OR ≈ 4.03 for females).*

For violent theft, female defendants face a dramatic penalty *increment* relative to males convicted of the same crime (OR ≈ 4.03). Meanwhile, sexual offences show near-total protection for women (OR ≈ 0.37), and killing cases show leniency rather than harshness for female defendants. This "double deviance" phenomenon aligns with feminist criminology: women who committed crimes involving calculated violence or theft violated not only the law but also Victorian gender norms of feminine passivity. Judges appear to have penalized both transgressions simultaneously — but only for specific crime types.

The crime-specific pattern reveals a "transgression rule": violence and sexual offences saw courts treating women leniently (seen as "victims of their nature"), while crimes involving calculated manipulation or direct confrontation with authority saw courts respond with increased severity (women seen as violating proper womanhood). This is particularly significant for algorithmic bias inheritance: an ML system trained on historical records would learn this crime-dependent, gendered penalty structure. A global fairness metric would miss all of this — disaggregated testing per crime type is the only honest audit, exactly what the EU AI Act requires.

---

## 8.9 Key Insights: Gender Bias Is Structural, Multi-Layered, and Crime-Specific

Three core findings emerge:

1. **Structural gender sorting dominates the aggregate effect**: The raw gender gap (M1 OR = 0.44) attenuates dramatically once crime type and case characteristics are controlled (M3 OR = 0.83, CI crosses 1.0). Gender influenced sentencing *indirectly* — through which crime types women were charged with and how those crimes were narrated — rather than through direct court-level discrimination in identical cases.

2. **Double deviance is crime-specific**: Aggregate non-significance masks dramatic variation. Sexual offences show near-total leniency for women (OR ≈ 0.37); violent theft shows the strongest double deviance (OR ≈ 4.03). Women who committed crimes violating Victorian gender norms of feminine passivity — particularly violent theft — were punished far more harshly than men. An algorithm trained on these data would learn this contingent, crime-dependent penalty structure.

3. **Bias inheritance is multi-layered**: Gender bias in this corpus is not a single coefficient that can be audited away. It is a multi-layered, crime-type-specific, language-embedded pattern that operates at every stage of the legal pipeline — prosecution, charging, narration, and sentencing. A global fairness metric would miss all of this. This connects to the COMPAS case (Week 10), where algorithmic predictions operationalized historical racial disparities, and to the EU AI Act requirement for disaggregated fairness testing (Session 12).

The Victorian Old Bailey thus offers a historical case study in how institutions encode bias through structural sorting and crime-specific patterns — and how modern systems, if trained naively on such data, risk perpetuating or automating those biases at scale.

---

## 9. RQ3: Predicting Punishment Type for Guilty Defendants

**Research Question:** For guilty defendants, can we predict punishment type (hard labour, custodial/prison, non-custodial, capital/extreme)?

This section presents the most detailed prediction pipeline in our analysis, moving from text representation through automated model screening to transformer attention analysis. We demonstrate that routine punishments are highly predictable, while consequential decisions resist accurate prediction—a finding with critical implications for legal AI deployment.

## 9.1 Research Question and Target Distribution

Our multiclass classification task targets four punishment categories in the Old Bailey records (1902–1913). The distribution reflects the sentencing practices of the period and presents significant class imbalance challenges (Figure 9.1).

**Figure 9.1: Punishment Type Distribution**
![](report_charts/rq3_00.png)

The data reveals a heavily skewed distribution: hard labour dominates at 55% (n=3,796), custodial/prison sentences comprise 29% (~2,000 cases), non-custodial penalties account for 19% (~1,300), and capital/extreme sentences represent just 1% (n=76). This severe imbalance—particularly the rarity of capital cases—will require careful handling during model training and evaluation. The dominance of hard labour reflects both historical sentencing philosophy and the types of crimes prosecuted in this period.

## 9.2 Text Representation

### 9.2.1 Vocabulary Analysis and Zipf's Law

Before building classification models, we must understand the structure of our text. We analyzed the distribution of vocabulary frequency across all trial records, revealing the classic pattern described by Zipf's Law: a small core of terms carries most information.

**Figure 9.2: Cumulative Frequency Distribution**
![](report_charts/rq3_02.png)

The cumulative frequency curve shows that the top 500 terms capture approximately 70% of total vocabulary frequency, while the top 1,000 terms reach 85%. This finding validates our feature selection strategy (max_features=5000 for TF-IDF) and aligns with principles from Week 6 course materials on language representation. Most predictive signal concentrates in a manageable feature space.

### 9.2.2 Document Vectors

To move from raw text to quantitative representations, we represented each trial record as a vector in term-frequency space. The following visualization illustrates this concept using bigram frequencies.

**Figure 9.3: Document Vectors**
![](report_charts/rq3_03.png)

This scatter plot shows "hard labour" and "penal servitude" bigram frequencies per document. Documents segregate naturally by their content: cases explicitly discussing hard labour cluster in the upper-right, while others remain near the origin. This visual evidence suggests that punishment-specific terminology should enable classification.

### 9.2.3 TF-IDF Configuration

We configured TfidfVectorizer with the following parameters:
- **max_features:** 5,000 (balances vocabulary size vs. dimensionality)
- **stop_words:** 'english' (removes common function words)
- **ngram_range:** (1, 2) (captures unigrams and bigrams)
- **sublinear_tf:** True (dampens term frequency scaling to reduce impact of very frequent terms)

### 9.2.4 Fightin' Words Analysis (Monroe et al., Week 4)

To identify the most distinctive terms for each punishment class, we applied Bayesian shrinkage analysis (the "fightin' words" method covered in Week 4). This approach uses the Dirichlet–multinomial model to identify terms that are both frequent within a class and rare outside it, accounting for sampling variability.

**Figure 9.4: Fightin' Words — Hard Labour vs Other**
![](report_charts/rq3_04.png)

The bigram "hard labour" exhibits a Z-score of approximately 45, making it massively distinctive for the hard labour class. Strikingly, the punishment literally names itself in the trial text—a phenomenon that simplifies prediction but raises questions about how much the model relies on lexical shortcuts versus genuine legal reasoning patterns.

**Figure 9.5: Fightin' Words — Imprisonment vs Other**
![](report_charts/rq3_05.png)

For custodial sentences, terms like "imprisonment" and "second division" emerge as class-specific signatures. This reveals different sentencing vocabulary by punishment type—judges and clerks employ distinct linguistic conventions when recording different sentence types.

**Figure 9.6: Chi-Square Signature Words**
![](report_charts/rq3_06.png)

Ranking signature terms by chi-square statistic, "imprisonment" ranks highest (χ² = 2,350), followed by "second division" (χ² = 1,900). These top features align with the fightin' words results, validating the two approaches.

### 9.2.5 Word Association Networks

To visualize how sentencing language clusters, we extracted co-occurrence networks from trial texts.

**Figure 9.7: Word Association Network**
![](report_charts/rq3_07.png)

The network reveals tight semantic clusters: terms like "hard," "labour," and "months" co-occur in hard labour contexts, while "imprisonment" and "second division" form a separate cluster for custodial sentences. This structure pre-exists our models and confirms that language naturally segregates by punishment type.

### 9.2.6 Doc2Vec Embedding Validation

As a validation check, we applied Doc2Vec (paragraph vector) embeddings to all trial documents, then computed cosine similarity between documents.

**Figure 9.18: Doc2Vec Similarity Heatmap**
![](report_charts/rq3_18.png)

A clear block structure emerges: documents with the same punishment type exhibit consistently high cosine similarity (darker blocks on the diagonal), while cross-punishment-type comparisons show lower similarity. This validates that our text representation captures meaningful structure before any classification model is trained.

**Figure 9.15: t-SNE of Document Embeddings**
![](report_charts/rq3_15.png)

Projecting Doc2Vec embeddings to two dimensions via t-SNE reveals partial but visible clustering by punishment group. The separation is not absolute—some punishment types overlap—but structure clearly exists in the embedding space, validating the utility of learned representations.

## 9.3 Baseline Models (Structured Features Only)

Before incorporating text, we established baselines using only structured features: crime severity, defendant age, crime category, word count, readability metrics, and topic probabilities derived from LDA.

**Figure 9.9: Baseline Confusion Matrix (Logistic Regression)**
![](report_charts/rq3_09.png)

Logistic regression on structured features achieves 51% accuracy but defaults to predicting hard labour for ambiguous cases. Recall for capital cases is strong (system rarely misses genuine capital sentences) but precision suffers (false positives appear). This limitation justifies moving to more powerful models and incorporating text.

Results across baseline models:
- Logistic Regression: macro F1 = 0.49
- Random Forest: macro F1 = 0.61

**Figure 9.11: Random Forest Feature Importance (Structured)**
![](report_charts/rq3_11.png)

Crime severity dominates feature importance in structured models, followed by defendant count, topic probabilities, adjective count, and logarithmic text length. This suggests that case complexity and crime gravity strongly correlate with punishment, but substantial variance remains unexplained.

## 9.4 Adding Text Features

We progressively combined text and structured features, testing three approaches:

**Progressive Results:**
- Naive Bayes (text only): 67% accuracy but fails entirely on capital class (F1 = 0.00)
- Logistic Regression (TF-IDF + structured + SMOTE): 82% accuracy, macro F1 = 0.76
- Random Forest (combined + SMOTE): 89% accuracy, macro F1 = 0.86

**Figure 9.12: Model Comparison**
![](report_charts/rq3_12.png)

This bar chart traces the progression: structured-only LR (F1=0.49) → structured-only RF (F1=0.61) → text-only NB (F1=0.46) → combined LR (F1=0.76) → combined RF (F1=0.86). The dramatic gains from adding text features demonstrate that raw trial language contains substantial predictive signal beyond structured metadata.

**Figure 9.13: Learning Curves — Structured vs Text+Structured**
![](report_charts/rq3_13.png)

Left panel (structured only) plateaus at F1 ≈ 0.45, while the right panel (text + structured) reaches F1 > 0.65. Text features are not supplementary; they are essential to the prediction task.

## 9.5 PyCaret Automated Model Screening

To identify the optimal classifier, we screened 16 different algorithms using PyCaret's automated machine learning pipeline. Models were evaluated on held-out validation data after stratified train-test split (70/30) and SMOTE-based oversampling of minority classes.

**PyCaret Results (Top 5):**

| Rank | Model | Macro F1 | AUC | Training Time |
|------|-------|----------|-----|----------------|
| 1 | LightGBM | 0.912 | 0.981 | 3.4s |
| 2 | XGBoost | 0.910 | 0.981 | 4.3s |
| 3 | Gradient Boosting | 0.909 | — | 19.2s |
| 4 | Random Forest | 0.892 | 0.974 | 1.2s |
| 5 | Extra Trees | 0.877 | 0.970 | 1.4s |

We selected **LightGBM** as our final model. Its histogram-based splitting strategy and leaf-wise tree growth make it efficient with sparse TF-IDF vectors (~41,000 features). It achieved the highest macro F1 score (0.912) with minimal training time.

## 9.6 LightGBM Final Model and Validation

### Confusion Matrix and Per-Class Metrics

**Figure 9.19: LightGBM Confusion Matrix**
![](report_charts/rq3_19.png)

The model produces a strong diagonal structure. Hard labour predictions dominate (1,050 correct out of 1,080), while confusion primarily occurs between related custodial classes. Capital cases show 17 correct predictions out of 22.

**Figure 9.20: Per-Class Performance**
![](report_charts/rq3_20.png)

Per-class metrics reveal stark disparities across punishment types:

| Punishment Type | Precision | Recall | F1 Score | Support |
|-----------------|-----------|--------|----------|---------|
| Capital | 0.77 | 0.77 | 0.77 | 22 |
| Custodial | 0.89 | 0.89 | 0.89 | 598 |
| Hard Labour | 0.93 | 0.97 | 0.95 | 1,080 |
| Non-Custodial | 0.94 | 0.81 | 0.87 | 384 |

Hard labour achieves F1 = 0.95—near-perfect routine punishment prediction. Capital cases, the most consequential decisions, achieve only F1 = 0.77, with highest uncertainty.

**Figure 9.22: Per-Class ROC Curves**
![](report_charts/rq3_22.png)

All four classes achieve AUC ≥ 0.96, indicating strong class separation at all operating points, despite F1 differences.

### Overfitting and Generalisation

**Figure 9.23: Learning Curve**
![](report_charts/rq3_23.png)

Training F1 reaches 1.00 while cross-validation F1 plateaus around 0.85. With 41,000+ sparse features and only 6,945 training samples, overfitting is substantial. The model memorizes training examples effectively but generalises moderately.

**Figure 9.24: Generalisation Gap**
![](report_charts/rq3_24.png)

The gap between training (1.00) and CV (0.854) metrics demonstrates model complexity. This is not unexpected given the feature-to-sample ratio, but it highlights that test performance may not fully capture real-world deployment accuracy.

**Figure 9.25: Cross-Validation F1 Per Fold**
![](report_charts/rq3_25.png)

Five-fold cross-validation yields: 0.86, 0.88, 0.82, 0.85, 0.87 (mean = 0.854, std = 0.02). Fold 3 performs lowest (0.82) but all folds remain in the 0.82–0.88 range, suggesting stable performance across data splits.

## 9.7 Feature Importance

### LightGBM and SHAP Analysis

**Figure 9.21: LightGBM Feature Importance**
![](report_charts/rq3_21.png)

A lollipop chart ranks features by splitting counts in the fitted trees. The term "labour" dominates (importance = 240), followed by "imprisonment" (220), topic probabilities, "penal," Flesch reading ease, and temporal features. Text features overwhelmingly dominate structural features.

**Figure 9.26: Global SHAP Importance**
![](report_charts/rq3_26.png)

Using SHapley Additive exPlanations values, we quantify average absolute impact on predictions. "Labour" dominates (mean |SHAP| = 1.2), followed by "servitude" and "hard." Strikingly, sentencing terminology is literal: the words naming the punishment dominate predictions.

**Figure 9.27: SHAP Beeswarm Plot**
![](report_charts/rq3_27.png)

This plot shows direction and magnitude of feature impacts. Red dots (high feature values) for "labour" push predictions toward the hard labour class. Topic features and readability metrics contribute more subtly, with directional effects that vary per class.

## 9.8 Transformer Attention Analysis

To validate findings against a modern deep learning baseline, we fine-tuned DistilBERT (a distilled BERT variant) for multiclass classification. While the LightGBM model ultimately performed better, we analyzed attention patterns to compare mechanistic insights.

**Figure 9.30: DistilBERT Attention Heatmap**
![](report_charts/rq3_30.png)

In layer 5, head 0, the model attends to tokens like "guilty," "stealing," and "hard labour"—the same vocabulary features identified by TF-IDF and SHAP analysis. Though attention weights are not direct explanations of model decisions, this alignment validates that simpler shallow models capture the same linguistic patterns as transformers. Course reference: BERT architectures (Week 9).

## 9.9 Key Insight: The Paradox of Legal Prediction

Our results reveal a striking pattern consistent with Bernard Harcourt's "Against Prediction" thesis (Week 10 readings): **machine learning is most confident about routine decisions and most uncertain about consequential ones.**

- **Routine punishments (hard labour):** F1 = 0.95. The model predicts with near-perfect reliability because hard labour sentences follow predictable patterns tied to specific crime categories and severity levels.
- **Consequential decisions (capital):** F1 = 0.77. Despite their importance, capital sentences resist accurate prediction. Judges exercised greater discretion in capital cases, incorporating moral reasoning, precedent, and individual circumstances that do not reduce to text patterns.

The feature importance analysis compounds this insight: models exploit literal naming conventions ("labour", "imprisonment") rather than reasoning about sentencing principles. When vocabulary directly encodes the outcome, prediction is trivial. When judicial reasoning becomes discretionary and humanistic—as in capital cases—lexical patterns suffice less.

**Implications for Legal AI Deployment:** This finding suggests that predictive models threaten to automate precisely the decisions that should remain most subject to judicial discretion. Confident predictions about routine penalties could justify human removal from sentencing. Meanwhile, the decisions requiring deepest moral judgment resist automation—yet the same AI systems might be deployed most aggressively where they are least reliable. This paradox demands careful policy consideration before deployment in real legal systems.

---

**Summary Statistics for RQ3:**
- Training samples: 6,945 guilty defendants
- Test set: 2,084 cases
- Features: 41,000+ (TF-IDF) + 31 structured
- Final model: LightGBM
- Overall test accuracy: 89%
- Macro F1: 0.91
- Cross-validation F1: 0.854 (±0.02)

---

## 10. Close Reading: Two Mothers, Two Verdicts

We have presented what the Old Bailey looks like at scale — F1 scores, odds ratios, SHAP values. But behind every data point is a real person. The following two cases sit at the intersection of all three research questions.

### Case 1: Henrietta Daly (1902) — Attempted Murder, Baby Survived, Discharged

*"HENRIETTA DALY (29), Feloniously attempting to kill and murder Charles Henry Daly."*

A police constable watched her throw her baby from Waterloo Bridge into the river. River police rescued the child. Multiple doctors testified she was "decidedly of weak mind." Her brother was confined in Broadmoor Asylum. The court's decision? **Guilty — but discharged on her own recognizances.** She walked free.

The highlighted words — "weak mind" (×2), "delirious", "did not remember", "chronic inflammation" — are exactly the features our model associates with leniency.

### Case 2: Margaret Murphy (1911) — Murder, Baby Died, Death Sentence

*"MURPHY, Margaret (38, flower-seller), wilful murder of Gertrude Elizabeth Murphy."*

Her daughter ALICE MURPHY testified: *"Mother had been fined for not sending me to school; she had got into arrears with the rent. She was always very kind to us children. We have had at times to go short of food."* Dr. ALBAN DIXON noted: *"The home was a very poor one indeed."* GEORGE FROGGATT found that *"the baby died; it weighed only two-fifths of the normal weight."* Murphy had poisoned the baby with corrosive acid, then drank the acid herself. Dr. SULLIVAN assessed: *"prisoner was not insane at the time of the act, though emotionally unbalanced as the result of mental stress"* — the insanity defence was explicitly rejected.

Verdict: Guilty, with a strong recommendation to mercy. **SENTENCE OF DEATH.**

### What the Models See — and What They Miss

Both cases involve mothers who harmed their children. Both were found guilty. Both had mitigating circumstances. Yet the outcomes could not be more different — discharge versus death, the two absolute extremes of our punishment spectrum.

**Through the RQ1 lens (prediction):** Daly's text is saturated with doubt and medical language — "weak mind" (×2), "delirious", "did not remember" — features our model weights toward leniency. Murphy's text contains confession, a clear evidence chain, and an explicitly rejected insanity defence. The model predicts both correctly, from completely different textual signatures.

**Through the RQ2 lens (fairness):** Both are female defendants. The paternalism effect predicts leniency, which applies to Daly but fails catastrophically for Murphy. Crucially, class intersects with gender: Daly's violence was medicalised — "weak mind", "chronic inflammation" — she was seen as sick, not criminal. Murphy's desperation was acknowledged ("very poor home", "go short of food") but framed as circumstance, not incapacity. The woman whose condition is medicalised gets sympathy; the destitute flower-seller gets death.

**Through the RQ3 lens (consistency):** These two cases sit at maximum SHAP distance from each other — medical language pushes toward non-custodial; confession plus evidence pushes toward the harshest penalty. The pattern confirmed at scale in our models is confirmed in individual lives.

The models miss what any human reader sees immediately: Daly's case is about mental illness, Murphy's about poverty. The court showed compassion for madness but not for deprivation. No feature in our 58-dimensional space captures this distinction — and therein lies the fundamental limitation of our approach. **If we trained an AI on this data, which version of mercy would it learn?** The most consequential judicial decisions depend on contextual moral reasoning that resists quantification.

---

## 11. Cross-RQ Synthesis: The Bias Propagation Pipeline

The three research questions collectively demonstrate a complete mechanism for how bias propagates from historical records into algorithmic systems:

**Step 1 — Biased decisions are made and recorded.** RQ2 confirms that the Old Bailey systematically treated female defendants differently through structural gender sorting: general leniency for most crimes (M1 OR = 0.44), but compounded punishment for gender-norm-violating crimes like violent theft (OR ≈ 4.03).

**Step 2 — Trial text encodes those decisions.** RQ1 shows that case narratives achieve 99%+ predictability — even after removing explicit verdict phrases. Sentencing language, procedural vocabulary, and narrative structure all encode judicial outcomes so deeply that they cannot be fully separated from case facts.

**Step 3 — Models trained on that text reproduce the biases.** RQ3 demonstrates that punishment prediction reaches 85%+ macro F1. The features driving these predictions — sentencing vocabulary, topic distributions, gendered language patterns — carry the same biases detected in RQ2.

This pipeline applies equally to modern legal datasets — sentencing records, bail decisions, parole outcomes — where biases are present but harder to detect. The Old Bailey, precisely because its biases are historically distant and therefore less politically charged, serves as a transparent case study for a mechanism that operates in every jurisdiction.

---

## 12. Challenges

**Data leakage in historical text.** The most significant challenge was the interleaving of verdict, sentencing, and narrative in a single document. Despite removing 20+ explicit verdict phrases, sentencing language remained as an indirect pathway. This reflects a fundamental property of historical legal records — not a simple data cleaning oversight.

**Class imbalance across all RQs.** The 80/20 split (RQ1), 72/28 split (RQ2), and heavily skewed 4-class distribution (RQ3: capital/extreme = 1%) each required different strategies. Despite `class_weight='balanced'` and SMOTE, minority-class performance remained lower — capital/extreme (75 cases) achieved F1 = 0.77 vs 0.95 for hard labour.

**Overfitting in high-dimensional space.** RQ3's 41,000+ TF-IDF features far exceeded the 6,945 samples, causing the screening-to-CV performance drop (0.912 → 0.854). Dimensionality reduction could help but risks losing interpretable word-level features.

**Causal inference limitations (RQ2).** The OR attenuation from 0.44 to 0.83 shows that much of the raw gender gap is confounded by crime composition, and the M3 confidence interval crossing 1.0 means the residual effect is not statistically significant. Unmeasured confounders — judge identity, prior criminal history, employment status, defence counsel availability — further complicate causal interpretation.

---

## 13. Ethical Considerations

### 13.1 Historical Bias as Training Data

The Old Bailey proceedings encode the values and power structures of their era — a system that denied women the right to vote, criminalised homosexuality, and applied capital punishment for property crimes. Our RQ2 finding of structural gender sorting (raw OR ≈ 0.44, attenuating to OR ≈ 0.83 after controls) is evidence of how deeply Victorian gender norms permeated the criminal justice system — through charging patterns, crime-type sorting, and narrative framing — not simply through overt judicial bias.

### 13.2 The Bias Propagation Pipeline

Our three RQs collectively demonstrate a complete mechanism: (1) structurally biased decisions are recorded in the proceedings — RQ2 confirms crime-specific gender disparity operating through structural sorting; (2) trial text encodes those decisions — RQ1 shows 99%+ predictability; (3) models learn to reproduce them — RQ3 achieves 85%+ F1 on punishment prediction. This pipeline applies equally to modern legal datasets where biases are present but harder to see.

### 13.3 Implications for Deployment

**Interpretability over performance.** Logistic Regression (F1=0.977) rivals XGBoost (F1=0.993) while providing full coefficient-level explainability. For legal applications, this trade-off favours transparency. As discussed in course materials on alignment (Xiao & Zhu, 2025, Ch. 4), ensuring AI systems act in accordance with human values requires more than optimising accuracy metrics.

**Fairness auditing as standard practice.** Our RQ2 methodology — progressively controlling for confounders to isolate a protected characteristic's effect — provides a template for algorithmic auditing. The finding that the aggregate gender effect becomes non-significant (M3 CI crosses 1.0) while crime-specific effects remain dramatic (violent theft OR ≈ 4.03) demonstrates that aggregate fairness metrics can miss critical subgroup disparities. Disaggregated testing is essential.

**Appropriate use cases** include legal research, case assessment for planning, bias auditing, and legal education. **Inappropriate use cases** include automated sentencing, unsupervised risk assessment, or any application that removes human judgment from consequential legal decisions.

---

## 14. Critical Reflection

### 14.1 How Our Understanding Evolved

Our initial assumption was that removing explicit verdict phrases would create a clean prediction task. The feature importance analysis in RQ1 fundamentally changed this view — we discovered that sentencing language constitutes an indirect leakage pathway, meaning the 99% F1 score is partly artifactual. This discovery reshaped our interpretation of all three RQs: rather than celebrating high performance, we reframed it as evidence of how deeply outcomes permeate historical legal text.

Similarly, RQ2 began as a straightforward prediction task but evolved into a causal inference question. The progressive model design (M1→M2→M3) was adopted specifically to *measure* the gender effect, not to maximise AUC — a methodological pivot that reflects a maturing understanding of the difference between prediction and explanation (Shmueli, 2010).

### 14.2 Connection to Course Concepts

Our analysis engages directly with themes from nearly every week of the course:

**Week 3 (Samuel, 2009):** Samuel asks, "Can legal reasoning be demystified?" Our data suggests outcomes can be predicted — but perhaps they should not be automated.

**Weeks 4–5 (Klingenstein et al., Monroe et al.):** Klingenstein et al. (2014) analysed the same Old Bailey corpus using descriptive linguistics. We extend their work to predictive ML. The Fightin' Words method (Monroe et al.) proved particularly valuable for identifying punishment-distinctive vocabulary in RQ3.

**Week 6 (DummyClassifier, GRADE):** The DummyClassifier exercise taught us that 83% accuracy is meaningless with 79/21 class imbalance. GRADE showed how systems "codify historical bias" — our progressive modelling in RQ2 demonstrates the exact mechanism.

**Weeks 7–8 (Distributional Hypothesis):** The distributional hypothesis (Firth) — "words are characterised by the company they keep" — underlies our TF-IDF approach. POS tag analysis confirmed this: guilty cases have proportionally more pronouns (he/she/they — witnesses), while not-guilty cases have more adjectives (defence arguments).

**Week 9 (BERT, Stochastic Parrots):** The "Stochastic Parrots" concern (Bender et al., 2021) is directly relevant: if large language models trained on historical corpora reproduce patterns without understanding, then models trained on Old Bailey text reproduce Victorian sentencing biases without legal reasoning. Our DistilBERT attention analysis in RQ3 showed the transformer attending to the same tokens our simpler models found important.

**Week 10 (COMPAS, Harcourt):** COMPAS (Angwin et al.) provides the direct parallel — our RQ2 analysis demonstrates how bias operates through structural sorting (similar to how COMPAS embedded racial disparities from historical policing patterns). The key advance is our finding that aggregate fairness metrics mask dramatic crime-specific variation — disaggregated testing is essential. Harcourt's "Against Prediction" frames our central finding: an AI sentencing tool would be most confident about routine cases and most uncertain about capital decisions — the ones that matter most.

The text representation progression — from bag-of-words to TF-IDF to transformer-based embeddings — mirrors the evolution discussed in course materials on pre-training (Xiao & Zhu, 2025, Ch. 1). Barocas and Selbst (2016) argue that discriminatory outcomes can arise from biased training data even without discriminatory intent — our RQ2 analysis demonstrates this mechanism empirically.

### 14.3 What Additional Data Would Strengthen This Analysis

Several data gaps limit our findings. **Judge identity** is not systematically recorded — individual judges may have had distinctive sentencing patterns that confound our gender analysis. **Defendant prior criminal history** is occasionally mentioned in trial text but not structured as a variable. **Socioeconomic indicators** (occupation, address, literacy) appear inconsistently. **Defence counsel availability** — whether a defendant had legal representation — likely affects both verdict and sentencing but is only partially recorded.

### 14.4 Limitations of Modelling Approach

TF-IDF loses word order and semantic relationships — "the defendant denied stealing" and "the defendant admitted stealing" receive similar representations. Topic models (LDA) assume a fixed number of topics and bag-of-words generation, poorly capturing the sequential structure of trial narratives. Even Legal-BERT, while capturing context, was pre-trained on modern legal text — the domain gap to 19th-century court language limits its applicability without fine-tuning on period-appropriate corpora.

---

## 15. Conclusions

### 15.1 Summary of Findings

**RQ1:** Case narratives achieve 99.33% F1 (XGBoost), though indirect data leakage through sentencing language inflates this. Logistic Regression (97.73% F1) provides a more conservative, fully interpretable estimate. Text alone is sufficient — hybrid features add nothing.

**RQ2:** Female defendants received less harsh punishment at the raw level (OR ≈ 0.44), but this effect attenuates to OR ≈ 0.83 after full controls, with the confidence interval crossing 1.0 — indicating that the aggregate gender effect operates through structural crime-type sorting rather than direct judicial discrimination. The effect varies dramatically by crime type: sexual offences show near-total leniency for women (OR ≈ 0.37), while violent theft shows strong double deviance (OR ≈ 4.03). Random Forest (AUC=0.769) best predicts sentencing, but logistic regression is preferred for causal interpretation.

**RQ3:** Punishment type is predictable at macro F1 = 0.854 (LightGBM, rigorous CV). Text features dramatically improve over structured-only baselines (F1: 0.61 → 0.86+), confirming that trial language carries substantial sentencing signal. Overfitting remains a concern.

### 15.2 Thesis: Predictability Is Not Fairness

The Old Bailey's justice was highly predictable — 99.3% F1 for verdicts, 0.912 macro F1 for punishment. Gender affects punishment severity through structural sorting, with the aggregate effect becoming non-significant after full controls but dramatic crime-specific variation persisting. Punishment types follow detectable patterns, but the most consequential decisions resist prediction. **An AI trained on this data would be accurate AND biased. This is the central paradox of legal AI: performance metrics can mask injustice.**

### 15.3 Three Learnings

Rather than framing these as mere limitations, we present three substantive learnings:

1. **Too-good results demand scepticism.** 99% F1 forced us to interrogate data leakage, residual bias, and indirect encoding. The best finding is one that survives scrutiny.

2. **Historical data imposes its categories.** Binary gender, no intersectionality, no social class variable. The data's limitations become the model's limitations — and potentially, the algorithm's blind spots.

3. **The most consequential decisions resist prediction.** Capital cases, insanity findings, judicial mercy — the decisions that matter most are the ones our models handle least confidently.

### 15.4 Cross-RQ Synthesis

The three RQs illustrate a bias propagation pipeline: structurally biased outcomes are recorded in text → text encodes those biases → models trained on that text reproduce them. RQ1 shows the encoding is deep (even after cleaning, sentencing language remains). RQ2 shows one of the encoded biases is gender-based, operating through structural crime-type sorting with dramatic crime-specific variation (from near-total leniency in sexual offences to severe double deviance in violent theft). RQ3 confirms the endpoint: sentencing is highly predictable from features that carry those same biases.

### 15.5 Implications for Legal AI

This analysis demonstrates both the power and the peril of AI in legal systems. The path forward requires continued research into fairness and bias mitigation, transparent and explainable AI systems, robust evaluation before deployment, and active engagement with legal practitioners, ethicists, and affected communities. As Samuel (2009, Week 3) asks: *"Can legal reasoning be demystified?"* Our data suggests outcomes can be predicted — but perhaps they should not be automated. **Ultimately, AI should augment human judgment in legal systems, never replace it.**

### 15.6 Future Work

Sequential modelling — predicting verdicts at each trial stage rather than from complete narratives — would address the leakage problem. Extending to the full 1674–1913 period would enable temporal drift analysis. Cross-jurisdictional comparison with other historical courts could test generalisability. Fine-tuning Legal-BERT on period-appropriate legal text could improve transformer performance. Counterfactual analysis — testing whether removing specific words changes predictions — would help distinguish genuinely influential language from spurious correlations.

---

## References

Aletras, N., Tsarapatsanis, D., Preoțiuc-Pietro, D., & Lampos, V. (2016). Predicting judicial decisions of the European Court of Human Rights: A natural language processing perspective. *PeerJ Computer Science, 2*, e93.

Angwin, J., Larson, J., Mattu, S., & Kirchner, L. (2016). Machine bias: There’s software used across the country to predict future criminals. And it’s biased against blacks. *ProPublica*.

Ariai, F., & Demartini, G. (2024). Natural language processing for the legal domain: A survey of tasks, datasets, models, and challenges. *arXiv*. https://arxiv.org/abs/2410.21306

Barocas, S., & Selbst, A. D. (2016). Big data’s disparate impact. *California Law Review, 104*, 671–732.

Bender, E. M., Gebru, T., McMillan-Major, A., & Shmitchell, S. (2021). On the dangers of stochastic parrots: Can language models be too big? In *Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency* (pp. 610–623). Association for Computing Machinery.

Cui, J., Shen, X., Nie, F., Wang, Z., Wang, J., & Chen, Y. (2022). A survey on legal judgment prediction: Datasets, metrics, models and challenges. *arXiv*. https://arxiv.org/abs/2204.04859

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies* (pp. 4171–4186).

Fausey, C. M., & Boroditsky, L. (2010). Subtle linguistic cues influence perceived blame and financial liability. *Psychonomic Bulletin & Review, 17*(5), 644–650.

Hitchcock, T., & Turkel, W. J. (2016). The Old Bailey Proceedings, 1674–1913: Text mining for evidence of court behaviour. *Law and History Review, 34*(4), 933–968.

Klingenstein, S., Hitchcock, T., & DeDeo, S. (2014). The civilizing process in London’s Old Bailey. *Proceedings of the National Academy of Sciences, 111*(26), 9419–9424.

Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. In *Advances in Neural Information Processing Systems 30*.

Medvedeva, M., Vols, M., & Wieling, M. (2020). Using machine learning to predict decisions of the European Court of Human Rights. *Artificial Intelligence and Law, 28*, 237–266.

Monroe, B. L., Colaresi, M. P., & Quinn, K. M. (2008). Fightin’ words: Lexical feature selection and evaluation for identifying the content of political text. *Political Analysis, 16*(4), 372–403.

Samuel, G. (2009). Can legal reasoning be demystified? *Legal Studies, 29*(2), 181–210.

Shmueli, G. (2010). To explain or to predict? *Statistical Science, 25*(3), 289–310.

Xiao, T., & Zhu, J. (2025). Foundations of large language models. *arXiv*. https://arxiv.org/abs/2501.09223

---

## Data Sources

- Old Bailey Proceedings Online: [www.oldbaileyonline.org](https://www.oldbaileyonline.org)
- DHI Old Bailey Data API: [www.dhi.ac.uk/data/oldbailey](https://www.dhi.ac.uk/data/oldbailey)
- Digital Panopticon: [www.digitalpanopticon.org](https://www.digitalpanopticon.org)
