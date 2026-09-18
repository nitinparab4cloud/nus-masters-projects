"""
explain_engine.py
------------------
Per-person, plain-language explanations of the surrogate model's prediction
-- the technical core of this project, and the part that maps directly onto
Article 86 of the EU AI Act (the right to an explanation of individual
decision-making for high-risk AI systems) and onto the "explicability"
principle from AI ethics frameworks such as Floridi et al.'s AI4People.

No `shap` package dependency, and this is a deliberate choice, not a
workaround: for a linear (here, logistic) model, the Shapley value of each
feature has an exact closed form once you assume feature independence --
there's no sampling, no approximation, nothing to validate against a
reference implementation. It's the same maths `shap.LinearExplainer` uses
under its default independence assumption; it's just implemented here
directly so the whole explanation pipeline stays inspectable in one file,
in keeping with this portfolio's stance that a governance tool should be
auditable end to end, not "explainable" via a second black box.

The maths, briefly:
  For a logistic regression with standardized inputs z_i = (x_i - mean_i) / std_i,
  the model's log-odds output is:  logit(x) = intercept + sum_i(beta_i * z_i)

  The Shapley value (contribution) of feature i for a specific person is:
      phi_i = beta_i * (z_i_person - mean(z_i_background))
  Since z is already mean-centered over the training set, mean(z_i_background) = 0,
  so phi_i = beta_i * z_i_person exactly.

  These contributions satisfy the local accuracy property SHAP is named for:
      logit(x) = baseline + sum_i(phi_i)
  where baseline = intercept (the model's average prediction over the
  background/training population, in log-odds). test_explainability.py
  checks this equality holds to floating-point precision for every example.
"""

from dataclasses import dataclass

import numpy as np

from surrogate_model import FEATURE_LABELS, FEATURE_NAMES, SurrogateModel, build_feature_matrix


@dataclass
class FeatureContribution:
    feature: str
    label: str
    raw_value: float
    contribution: float  # in log-odds


@dataclass
class Explanation:
    predicted_flag: str            # "High-Risk" or "Not High-Risk" (per the surrogate)
    predicted_probability: float   # surrogate's estimated probability of the high-risk flag
    baseline_logit: float
    total_logit: float
    contributions: list[FeatureContribution]
    fidelity: dict


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


def explain_instance(surrogate: SurrogateModel, raw_features: dict) -> Explanation:
    """
    raw_features: a dict with the same keys as surrogate_model.FEATURE_NAMES,
    in original (non-standardized) units, e.g.
        {"age": 24, "priors_count": 5, "juv_fel_count": 0, "juv_misd_count": 0,
         "juv_other_count": 0, "is_male": 1, "is_felony_charge": 1}
    """
    x = np.array([raw_features[name] for name in FEATURE_NAMES], dtype=float)
    z = (x - surrogate.feature_means) / surrogate.feature_stds

    coefs = surrogate.model.coef_[0]
    intercept = float(surrogate.model.intercept_[0])

    phi = coefs * z  # exact per-feature Shapley contributions, see module docstring
    total_logit = intercept + float(phi.sum())
    prob = _sigmoid(total_logit)

    contributions = [
        FeatureContribution(
            feature=name,
            label=FEATURE_LABELS[name],
            raw_value=raw_features[name],
            contribution=float(c),
        )
        for name, c in zip(FEATURE_NAMES, phi)
    ]
    contributions.sort(key=lambda c: abs(c.contribution), reverse=True)

    return Explanation(
        predicted_flag="High-Risk" if prob >= 0.5 else "Not High-Risk",
        predicted_probability=round(float(prob), 4),
        baseline_logit=round(intercept, 4),
        total_logit=round(total_logit, 4),
        contributions=contributions,
        fidelity=surrogate.fidelity,
    )


def explain_row(surrogate: SurrogateModel, df_row) -> Explanation:
    """Convenience wrapper: explain a single row from the loaded COMPAS dataframe."""
    features_df = build_feature_matrix(df_row.to_frame().T)
    raw = {name: float(features_df.iloc[0][name]) for name in FEATURE_NAMES}
    return explain_instance(surrogate, raw)


def render_plain_language(explanation: Explanation, top_n: int = 4) -> str:
    """
    Turns an Explanation into the kind of plain-language paragraph a
    non-technical, affected person could actually read -- the point of
    Article 86, not just a table of numbers.
    """
    lines = []
    pct = round(explanation.predicted_probability * 100)
    lines.append(
        f"**Surrogate model prediction: {explanation.predicted_flag}** "
        f"(estimated {pct}% probability of being flagged high-risk by a model "
        f"trained to approximate COMPAS's own classification)."
    )
    lines.append("")
    lines.append(f"**The {top_n} factors that most influenced this specific prediction:**")
    for c in explanation.contributions[:top_n]:
        direction = "increased" if c.contribution > 0 else "decreased"
        lines.append(
            f"- {c.label} (value: {c.raw_value:g}) **{direction}** the risk estimate "
            f"(contribution: {c.contribution:+.3f} log-odds)."
        )
    lines.append("")
    lines.append(
        f"_Model fidelity: this surrogate agrees with COMPAS's own high-risk flag on "
        f"{round(explanation.fidelity['test_agreement_with_compas'] * 100)}% of held-out cases "
        f"(vs. a {round(explanation.fidelity['majority_class_baseline'] * 100)}% baseline from always "
        f"guessing the majority class). This explains the surrogate's own reasoning, not COMPAS's "
        f"actual proprietary logic, which no outside party has access to._"
    )
    return "\n".join(lines)
