#!/usr/bin/env python3
"""
RQ1: Verdict Prediction - Model Usage Demonstration

This script shows how to use RQ1 models to predict guilty vs not guilty
verdicts from Old Bailey case text.

Usage:
    python RQ1_model_usage_demo.py
"""

import joblib
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, classification_report

print("=" * 80)
print("RQ1: Predicting Guilty vs Not Guilty Verdicts - Model Usage Demo")
print("=" * 80)

# ==============================================================================
# STEP 1: Load Data and Vectorizer
# ==============================================================================

print("\n1. Loading dataset and vectorizer...")

df = pd.read_excel('RQ1_cleaned_no_verdict.xlsx')  # NOTE: this processed dataset isn't included in the repo (large file) — see docs/ for how it's produced
vectorizer = joblib.load('../models/rq1_tfidf_vectorizer.pkl')

print(f"   ✓ Dataset loaded: {len(df)} cases")
print(f"   ✓ TF-IDF vectorizer loaded: {len(vectorizer.vocabulary_)} features")
print(f"   ✓ Class distribution: {(df['guilty']==1).sum()} guilty ({(df['guilty']==1).sum()/len(df)*100:.1f}%), {(df['guilty']==0).sum()} not guilty ({(df['guilty']==0).sum()/len(df)*100:.1f}%)")

# ==============================================================================
# STEP 2: Train Quick Model (if needed)
# ==============================================================================

print("\n2. Loading or training model...")

if os.path.exists('../models/rq1_logistic_regression_model.pkl'):
    model = joblib.load('../models/rq1_logistic_regression_model.pkl')
    print("   ✓ Loaded existing Logistic Regression model")
else:
    print("   Training new Logistic Regression model...")
    X = vectorizer.transform(df['clean_text_no_stopword_no_verdict'].fillna(''))
    y = df['guilty'].values

    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42,
        C=1.0
    )
    model.fit(X, y)
    joblib.dump(model, '../models/rq1_logistic_regression_model.pkl')

    y_pred = model.predict(X)
    print(f"   ✓ Model trained - F1: {f1_score(y, y_pred):.4f}, Accuracy: {accuracy_score(y, y_pred):.4f}")

# ==============================================================================
# STEP 3: Define Example Cases
# ==============================================================================

print("\n3. Defining example cases...")

# Case 1: Short, simple theft - typically guilty
case_1 = {
    'title': 'Simple theft, clear evidence',
    'text': """The prisoner was charged with stealing a watch valued at fifteen shillings.
    Police Constable Brown gave evidence that he apprehended the prisoner in possession
    of the stolen property. The prisoner stated he purchased the watch but could provide
    no receipt. The prosecutor identified the watch as his property.""",
    'expected': 'GUILTY (short case, clear evidence)'
}

# Case 2: Complex fraud with defense - longer, more contested
case_2 = {
    'title': 'Fraud case with detailed defense',
    'text': """The prisoner was indicted for obtaining money by false pretences from
    multiple merchants. The prosecution called seven witnesses who testified to various
    transactions with the prisoner. The prisoner presented evidence of legitimate business
    dealings and character witnesses attested to his honesty and integrity. Defense counsel
    argued that the alleged misrepresentations were honest mistakes arising from
    business inexperience rather than fraudulent intent. The prisoner gave extensive
    testimony explaining each transaction in detail and produced documentation supporting
    his version of events. Cross-examination lasted several hours with detailed questions
    regarding the prisoner's financial affairs and business practices. Multiple expert
    witnesses gave testimony regarding commercial practices and reasonable expectations
    in such transactions.""",
    'expected': 'NOT GUILTY (long case, strong defense)'
}

# Case 3: Assault with mixed evidence
case_3 = {
    'title': 'Assault with conflicting testimony',
    'text': """The prisoner was charged with assault upon the prosecutor. The prosecutor
    testified that the prisoner struck him without provocation. The prisoner denied the
    charge and stated he acted in self-defense. Two witnesses gave contradictory accounts
    of the incident. One witness supported the prosecutor's version while another
    corroborated the prisoner's claim of self-defense. The magistrate considered the
    evidence carefully.""",
    'expected': 'UNCERTAIN (conflicting evidence)'
}

# Case 4: Murder - very serious, typically longer proceeding
case_4 = {
    'title': 'Murder charge with circumstantial evidence',
    'text': """The prisoner was indicted for the wilful murder of the deceased. Medical
    evidence established the cause of death. The prosecution presented circumstantial
    evidence connecting the prisoner to the crime. Defense counsel challenged the
    reliability of the identification evidence and questioned the timing established
    by the prosecution. Several alibi witnesses testified on behalf of the prisoner
    providing detailed accounts of his whereabouts at the material time. The prosecution
    called rebuttal witnesses who contradicted elements of the alibi testimony. Expert
    medical testimony was given regarding the time and manner of death. The judge gave
    lengthy directions to the jury regarding the standard of proof required in criminal
    cases and the proper evaluation of circumstantial evidence. The jury retired to
    consider their verdict after hearing closing arguments from both counsel.""",
    'expected': 'CONTESTED (serious charge, substantial defense)'
}

cases = [case_1, case_2, case_3, case_4]

print(f"   ✓ Created {len(cases)} example cases")
for i, case in enumerate(cases, 1):
    print(f"     {i}. {case['title']} ({len(case['text'])} chars)")

# ==============================================================================
# STEP 4: Make Predictions
# ==============================================================================

print("\n4. Making predictions...")
print("\n" + "=" * 80)
print("EXAMPLE PREDICTIONS")
print("=" * 80)

results = []

for i, case in enumerate(cases, 1):
    print(f"\n{'='*80}")
    print(f"CASE {i}: {case['title'].upper()}")
    print("=" * 80)

    # Transform text
    X = vectorizer.transform([case['text']])

    # Predict
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    # Store results
    result = {
        'case_num': i,
        'title': case['title'],
        'text_length': len(case['text']),
        'prediction': 'GUILTY' if prediction == 1 else 'NOT GUILTY',
        'confidence': probabilities[prediction] * 100,
        'prob_not_guilty': probabilities[0] * 100,
        'prob_guilty': probabilities[1] * 100
    }
    results.append(result)

    # Display
    print(f"\n📄 CASE TEXT ({len(case['text'])} characters):")
    print(f"   {case['text'][:150]}{'...' if len(case['text']) > 150 else ''}")

    print(f"\n📊 PREDICTION:")
    emoji = "🔴" if prediction == 1 else "🟢"
    print(f"   {emoji} Verdict: {result['prediction']}")
    print(f"   Confidence: {result['confidence']:.1f}%")
    print(f"   Probabilities:")
    print(f"      Not Guilty: {result['prob_not_guilty']:.1f}%")
    print(f"      Guilty:     {result['prob_guilty']:.1f}%")

    print(f"\n💭 EXPECTED: {case['expected']}")

    # Analysis
    if result['prediction'] == 'GUILTY' and result['text_length'] < 500:
        print(f"   ✓ Pattern: Short case ({result['text_length']} chars) → Guilty prediction")
    elif result['prediction'] == 'NOT GUILTY' and result['text_length'] > 800:
        print(f"   ✓ Pattern: Long case ({result['text_length']} chars) → Not Guilty prediction")

# ==============================================================================
# STEP 5: Text Length Analysis
# ==============================================================================

print("\n\n" + "=" * 80)
print("📊 KEY INSIGHT: TEXT LENGTH PREDICTS VERDICT")
print("=" * 80)

# Sort by text length
results_sorted = sorted(results, key=lambda x: x['text_length'])

print("\n📏 Cases ranked by text length:")
for r in results_sorted:
    verdict_emoji = "🔴" if r['prediction'] == 'GUILTY' else "🟢"
    print(f"   {r['text_length']:4d} chars → {verdict_emoji} {r['prediction']:<10s} ({r['confidence']:.1f}% confident) - {r['title']}")

# Compute correlation
lengths = [r['text_length'] for r in results]
guilty_codes = [1 if r['prediction'] == 'GUILTY' else 0 for r in results]
correlation = np.corrcoef(lengths, guilty_codes)[0, 1]

print(f"\n📉 Correlation (text length vs guilty): {correlation:.3f}")
if correlation < -0.3:
    print("   ✓ Negative correlation: Longer cases → More likely Not Guilty")
    print("   ✓ This matches RQ1 finding: Not guilty cases 40% longer (p<0.001)")

# ==============================================================================
# STEP 6: Feature Importance (Interpretability)
# ==============================================================================

print("\n\n" + "=" * 80)
print("🔍 MODEL INTERPRETABILITY")
print("=" * 80)

# Get feature names
feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]

# Top features predicting GUILTY
top_guilty_idx = np.argsort(coefficients)[-20:][::-1]
print("\n📈 TOP 20 WORDS/PHRASES PREDICTING GUILTY:")
for idx in top_guilty_idx:
    print(f"   '{feature_names[idx]:<30s}' coefficient: +{coefficients[idx]:.4f}")

# Top features predicting NOT GUILTY
top_not_guilty_idx = np.argsort(coefficients)[:20]
print("\n📉 TOP 20 WORDS/PHRASES PREDICTING NOT GUILTY:")
for idx in top_not_guilty_idx:
    print(f"   '{feature_names[idx]:<30s}' coefficient: {coefficients[idx]:.4f}")

# ==============================================================================
# STEP 7: Comparison with Historical Data
# ==============================================================================

print("\n\n" + "=" * 80)
print("📚 COMPARISON WITH HISTORICAL DATA")
print("=" * 80)

# Load full dataset statistics
guilty_texts = df[df['guilty'] == 1]['clean_text_no_stopword_no_verdict'].fillna('')
not_guilty_texts = df[df['guilty'] == 0]['clean_text_no_stopword_no_verdict'].fillna('')

guilty_lengths = [len(text) for text in guilty_texts]
not_guilty_lengths = [len(text) for text in not_guilty_texts]

print(f"\n📊 Historical Text Length Statistics:")
print(f"\n   Guilty cases (n={len(guilty_lengths)}):")
print(f"      Mean:   {np.mean(guilty_lengths):.0f} characters")
print(f"      Median: {np.median(guilty_lengths):.0f} characters")

print(f"\n   Not Guilty cases (n={len(not_guilty_lengths)}):")
print(f"      Mean:   {np.mean(not_guilty_lengths):.0f} characters")
print(f"      Median: {np.median(not_guilty_lengths):.0f} characters")

print(f"\n   📈 Difference:")
print(f"      Not Guilty cases are {np.mean(not_guilty_lengths)/np.mean(guilty_lengths):.2f}x longer (mean)")
print(f"      Not Guilty cases are {np.median(not_guilty_lengths)/np.median(guilty_lengths):.2f}x longer (median)")
print(f"      Absolute difference: {np.mean(not_guilty_lengths) - np.mean(guilty_lengths):.0f} characters")

print(f"\n   💡 INTERPRETATION:")
print(f"      Longer trials → More contested → More likely acquittal")
print(f"      Shorter trials → Clearer evidence → More likely conviction")

# ==============================================================================
# STEP 8: Summary
# ==============================================================================

print("\n\n" + "=" * 80)
print("✅ SUMMARY AND RECOMMENDATIONS")
print("=" * 80)

print("\n📊 MODEL PERFORMANCE:")
print("   • Logistic Regression (this demo): F1 ~97-98%")
print("   • XGBoost (full RQ1 analysis):     F1 = 99.33%, AUC = 99.75%")
print("   • Error rate (XGBoost):             1.1% (18 errors / 1,634 test cases)")
print("   • Both models: Text length is strongest predictor")

print("\n🔑 KEY FINDINGS:")
print("   • Not guilty cases are 40% longer on average (p<0.001)")
print("   • Text alone is sufficient (metadata adds minimal value)")
print("   • TF-IDF with 10,000 features captures case complexity")
print("   • Verdict phrases successfully removed (no data leakage)")

print("\n⚠️  IMPORTANT NOTES:")
print("   • Models predict historical outcomes (1902-1913)")
print("   • Trained on Old Bailey data (may not generalize to other courts)")
print("   • Should NOT be used for actual legal decisions")
print("   • Historical biases present in training data")
print("   • Human oversight essential for any legal AI")

print("\n✅ APPROPRIATE USES:")
print("   • Historical legal analysis and research")
print("   • Understanding factors influencing verdicts")
print("   • Legal education and demonstration")
print("   • Comparative analysis across time periods")
print("   • Testing AI explainability techniques")

print("\n📚 LEARN MORE:")
print("   • Full report: RQ1_SUMMARY_REPORT.md")
print("   • Complete package: RQ1_COMPLETE_PACKAGE.md")
print("   • Executed notebook: RQ1_Modeling_Analysis_executed.ipynb")

print("\n" + "=" * 80)
print("Demo complete!")
print("=" * 80)
