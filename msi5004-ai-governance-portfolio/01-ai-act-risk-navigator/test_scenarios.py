# Copyright © 2026 Nitin Parab. Part of the "AI Governance Case Files" portfolio
# (https://github.com/nitinparab4cloud/nus-masters-projects/tree/main/msi5004-ai-governance-portfolio/01-ai-act-risk-navigator).
# See this project's LICENSE file for reuse terms.

"""
test_scenarios.py
------------------
Ten worked scenarios spanning all four EU AI Act risk tiers, used both as a
regression test suite and as the "sample rows" shown in the Gradio demo.

Run directly:  python test_scenarios.py
Run with pytest (if installed):  pytest test_scenarios.py
"""

from risk_engine import classify

SCENARIOS = [
    {
        "name": "Real-time public facial recognition for policing",
        "description": (
            "A live facial recognition system deployed on city CCTV that continuously "
            "scans crowds in public squares to identify persons of interest for police, "
            "outside of any specific search for a victim or suspect."
        ),
        "expected_tier": "prohibited",
    },
    {
        "name": "Emotion-scoring interview screener",
        "description": (
            "Software used by an employer that infers candidates' emotional state during "
            "video interviews to score their suitability for a role."
        ),
        "expected_tier": "prohibited",
    },
    {
        "name": "National social trust score",
        "description": (
            "A government platform that assigns citizens a social scoring value based on "
            "their online behaviour, used to determine access to unrelated public services."
        ),
        "expected_tier": "prohibited",
    },
    {
        "name": "CV screening and candidate ranking tool",
        "description": (
            "A recruitment algorithm that screens incoming resumes and ranks candidates for "
            "interview based on predicted job fit."
        ),
        "expected_tier": "high-risk",
    },
    {
        "name": "Consumer credit scoring model",
        "description": (
            "A creditworthiness model used by a bank to approve or reject personal loan "
            "applications."
        ),
        "expected_tier": "high-risk",
    },
    {
        "name": "Exam proctoring and cheating detection",
        "description": (
            "A university system that monitors students during online exams to detect "
            "cheating and flags irregular behaviour for review."
        ),
        "expected_tier": "high-risk",
    },
    {
        "name": "Recidivism risk assessment for parole boards",
        "description": (
            "A recidivism risk assessment tool that scores defendants for use by parole "
            "boards when deciding early release."
        ),
        "expected_tier": "high-risk",
    },
    {
        "name": "Customer-support chatbot",
        "description": (
            "A conversational agent on a retail website that answers customer questions "
            "about orders and returns."
        ),
        "expected_tier": "limited-risk",
    },
    {
        "name": "AI-generated marketing video",
        "description": (
            "A tool that produces an AI-generated video of a spokesperson for a marketing "
            "campaign, without any real person's likeness being altered."
        ),
        "expected_tier": "limited-risk",
    },
    {
        "name": "Internal meeting-notes summarizer",
        "description": (
            "An internal tool that summarizes uploaded meeting transcripts into a short "
            "bullet-point recap for the team that uploaded them."
        ),
        "expected_tier": "minimal-risk",
    },
]


def run():
    passed = 0
    for case in SCENARIOS:
        result = classify(case["description"])
        ok = result.tier == case["expected_tier"]
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        print(f"[{status}] {case['name']}")
        print(f"       expected={case['expected_tier']!r} got={result.tier!r} ({result.tier_label})")
        if not ok:
            print(f"       rationale: {result.rationale}")
    print(f"\n{passed}/{len(SCENARIOS)} scenarios classified as expected.")
    if passed != len(SCENARIOS):
        raise SystemExit(1)


# --- pytest-compatible wrapper ---------------------------------------------
def test_all_scenarios():
    for case in SCENARIOS:
        result = classify(case["description"])
        assert result.tier == case["expected_tier"], (
            f"{case['name']}: expected {case['expected_tier']}, got {result.tier} "
            f"({result.rationale})"
        )


if __name__ == "__main__":
    run()
