"""
test_explainability.py
-----------------------
Sanity tests for the surrogate model and the SHAP-style explanations, run
against the live COMPAS dataset (same pattern as Case 01/02's test files).

Run directly:  python test_explainability.py
Run with pytest (if installed):  pytest test_explainability.py
"""

from data_loader import load_compas
from explain_engine import explain_instance
from surrogate_model import FEATURE_NAMES, train_surrogate

TOLERANCE = 1e-6


def _base_profile():
    return {
        "age": 30,
        "priors_count": 2,
        "juv_fel_count": 0,
        "juv_misd_count": 0,
        "juv_other_count": 0,
        "is_male": 1.0,
        "is_felony_charge": 1.0,
    }


def check_fidelity_beats_baseline(surrogate) -> tuple[bool, str]:
    """The surrogate should meaningfully beat a majority-class guess, or the
    whole explanation exercise is explaining noise."""
    fid = surrogate.fidelity
    margin = fid["test_agreement_with_compas"] - fid["majority_class_baseline"]
    ok = margin >= 0.10
    return ok, (
        f"test_agreement={fid['test_agreement_with_compas']}, "
        f"baseline={fid['majority_class_baseline']}, margin={margin:.4f}"
    )


def check_local_accuracy(surrogate) -> tuple[bool, str]:
    """SHAP's namesake property: baseline + sum(contributions) == the model's
    actual output, exactly (up to floating point), for every profile."""
    profiles = [
        _base_profile(),
        {**_base_profile(), "priors_count": 10, "age": 20},
        {**_base_profile(), "is_male": 0.0, "is_felony_charge": 0.0},
        {**_base_profile(), "juv_fel_count": 3, "juv_misd_count": 2, "juv_other_count": 1},
    ]
    for p in profiles:
        exp = explain_instance(surrogate, p)
        reconstructed = exp.baseline_logit + sum(c.contribution for c in exp.contributions)
        if abs(reconstructed - exp.total_logit) > 1e-3:  # rounding at 4dp in the dataclass fields
            return False, f"profile={p}: reconstructed={reconstructed}, total_logit={exp.total_logit}"
    return True, f"local accuracy holds for {len(profiles)} profiles"


def check_priors_direction(surrogate) -> tuple[bool, str]:
    """More prior convictions, all else equal, should never lower the risk estimate
    (the surrogate's own coefficient for priors_count is positive -- confirms the
    explanation logic is internally consistent with the model it's explaining)."""
    low = explain_instance(surrogate, {**_base_profile(), "priors_count": 0})
    high = explain_instance(surrogate, {**_base_profile(), "priors_count": 15})
    ok = high.predicted_probability >= low.predicted_probability
    return ok, f"priors=0 -> {low.predicted_probability}, priors=15 -> {high.predicted_probability}"


def check_age_direction(surrogate) -> tuple[bool, str]:
    """Younger, all else equal, should not produce a lower risk estimate than older
    in this dataset (age's coefficient is negative) -- again, internal consistency,
    not a claim about the real world."""
    younger = explain_instance(surrogate, {**_base_profile(), "age": 19})
    older = explain_instance(surrogate, {**_base_profile(), "age": 60})
    ok = younger.predicted_probability >= older.predicted_probability
    return ok, f"age=19 -> {younger.predicted_probability}, age=60 -> {older.predicted_probability}"


def check_flag_matches_probability(surrogate) -> tuple[bool, str]:
    """predicted_flag should always agree with the 0.5 cut on predicted_probability."""
    profiles = [
        _base_profile(),
        {**_base_profile(), "priors_count": 20, "age": 18},
        {**_base_profile(), "priors_count": 0, "age": 65},
    ]
    for p in profiles:
        exp = explain_instance(surrogate, p)
        expected_flag = "High-Risk" if exp.predicted_probability >= 0.5 else "Not High-Risk"
        if exp.predicted_flag != expected_flag:
            return False, f"profile={p}: flag={exp.predicted_flag}, prob={exp.predicted_probability}"
    return True, "flag/probability agreement holds"


CHECKS = [
    ("Surrogate fidelity beats majority-class baseline by >=10pp", check_fidelity_beats_baseline),
    ("Local accuracy: baseline + contributions == model output", check_local_accuracy),
    ("More priors never lowers the risk estimate", check_priors_direction),
    ("Younger never produces a lower estimate than older here", check_age_direction),
    ("predicted_flag always agrees with the 0.5 probability cut", check_flag_matches_probability),
]


def run():
    df = load_compas()
    surrogate = train_surrogate(df)

    passed = 0
    for name, fn in CHECKS:
        ok, detail = fn(surrogate)
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        print(f"[{status}] {name}")
        print(f"       {detail}")
    print(f"\n{passed}/{len(CHECKS)} checks passed.")
    if passed != len(CHECKS):
        raise SystemExit(1)


# --- pytest-compatible wrappers ---------------------------------------------
def test_all_checks():
    df = load_compas()
    surrogate = train_surrogate(df)
    for name, fn in CHECKS:
        ok, detail = fn(surrogate)
        assert ok, f"{name}: {detail}"


if __name__ == "__main__":
    run()
