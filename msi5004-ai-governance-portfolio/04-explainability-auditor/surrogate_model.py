# Copyright © 2026 Nitin Parab. Part of the "AI Governance Case Files" portfolio
# (https://github.com/nitinparab4cloud/nus-masters-projects/tree/main/msi5004-ai-governance-portfolio/04-explainability-auditor).
# See this project's LICENSE file for reuse terms.

"""
surrogate_model.py
-------------------
Trains a transparent surrogate model that approximates COMPAS's own
high-risk flag from its publicly known input features.

Why a surrogate at all? COMPAS's actual scoring algorithm is proprietary --
Northpointe/Equivant has never published it, and no outside auditor (this
one included) has ever had access to it. That's not a limitation specific
to this project; it's the standard situation any real AI governance auditor
faces with a vendor's black-box system. The honest move in that situation is
the one used here and documented in real algorithmic-audit literature: train
an interpretable model on the same inputs to approximate the system's
*outputs*, then explain the approximation -- clearly labelled as an
approximation, with its fidelity to the real system reported up front, never
silently treated as if it were the real thing.

Why race is deliberately NOT a model input: COMPAS's own designers have
stated race is not a direct input to their instrument. Training this
surrogate the same way -- race-blind -- lets the demo make a genuinely
important point from the fairness literature: a model can produce racially
disparate outcomes (Case 02 measured exactly that) without ever "looking at"
race directly, because other race-blind features (most notably prior
convictions) correlate with race due to well-documented disparities upstream
in policing and charging. Explaining individual predictions with race-blind
features doesn't explain away that disparity -- it's how proxy
discrimination actually works, and this tool is built to surface that, not
hide it.

Model choice: plain logistic regression, not gradient boosting. This keeps
the SHAP-style attribution in explain_engine.py exact and closed-form rather
than an approximation, and keeps the surrogate itself inspectable (the
coefficients ARE the global explanation) -- consistent with this whole
portfolio's stance that a governance tool should be auditable, not just
locally explainable after the fact.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# The race-blind feature set, in a fixed order used everywhere downstream.
FEATURE_NAMES = [
    "age",
    "priors_count",
    "juv_fel_count",
    "juv_misd_count",
    "juv_other_count",
    "is_male",
    "is_felony_charge",
]

FEATURE_LABELS = {
    "age": "Age",
    "priors_count": "Number of prior convictions",
    "juv_fel_count": "Juvenile felony count",
    "juv_misd_count": "Juvenile misdemeanor count",
    "juv_other_count": "Other juvenile offense count",
    "is_male": "Sex recorded as male",
    "is_felony_charge": "Current charge is a felony (vs. misdemeanor)",
}


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw COMPAS columns into the race-blind numeric feature matrix."""
    out = pd.DataFrame(index=df.index)
    out["age"] = df["age"].astype(float)
    out["priors_count"] = df["priors_count"].astype(float)
    out["juv_fel_count"] = df["juv_fel_count"].astype(float)
    out["juv_misd_count"] = df["juv_misd_count"].astype(float)
    out["juv_other_count"] = df["juv_other_count"].astype(float)
    out["is_male"] = (df["sex"] == "Male").astype(float)
    out["is_felony_charge"] = (df["c_charge_degree"] == "F").astype(float)
    return out[FEATURE_NAMES]


@dataclass
class SurrogateModel:
    """
    A fitted logistic-regression surrogate plus the standardization stats
    needed to compute exact linear SHAP values against it later.
    """
    model: LogisticRegression
    feature_means: np.ndarray   # mean of each raw feature over the training set
    feature_stds: np.ndarray    # std of each raw feature over the training set
    fidelity: dict = field(default_factory=dict)  # how well it matches real COMPAS


def train_surrogate(df: pd.DataFrame, score_threshold: int = 5, random_state: int = 0) -> SurrogateModel:
    """
    Train the surrogate to predict COMPAS's own high-risk flag
    (decile_score >= score_threshold), and report how faithfully it
    reproduces that flag on held-out data -- this fidelity number is the
    single most important caveat in the whole project, and it's surfaced
    in the UI, not buried in code.
    """
    X_raw = build_feature_matrix(df)
    y = (df["decile_score"] >= score_threshold).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.25, random_state=random_state, stratify=y
    )

    means = X_train.mean(axis=0).to_numpy()
    stds = X_train.std(axis=0).replace(0, 1.0).to_numpy()

    X_train_std = (X_train.to_numpy() - means) / stds
    X_test_std = (X_test.to_numpy() - means) / stds

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_std, y_train)

    train_acc = model.score(X_train_std, y_train)
    test_acc = model.score(X_test_std, y_test)
    # Baseline: always predicting the majority class in the test set.
    baseline_acc = max(y_test.mean(), 1 - y_test.mean())

    fidelity = {
        "train_agreement_with_compas": round(float(train_acc), 4),
        "test_agreement_with_compas": round(float(test_acc), 4),
        "majority_class_baseline": round(float(baseline_acc), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    return SurrogateModel(model=model, feature_means=means, feature_stds=stds, fidelity=fidelity)


if __name__ == "__main__":
    from data_loader import load_compas

    df = load_compas()
    surrogate = train_surrogate(df)
    print("Surrogate fidelity (agreement with COMPAS's own high-risk flag):")
    for k, v in surrogate.fidelity.items():
        print(f"  {k}: {v}")
    print("\nCoefficients (standardized scale -- sign and relative size are what matter):")
    for name, coef in zip(FEATURE_NAMES, surrogate.model.coef_[0]):
        print(f"  {FEATURE_LABELS[name]:45s} {coef:+.3f}")
