"""
risk_engine.py
--------------
A transparent, rule-based classifier that maps a plain-language AI system
description onto:

  1. Its EU AI Act risk tier (Regulation (EU) 2024/1689), with the specific
     article / Annex III category driving the call.
  2. The compliance documentation checklist that tier requires, with the
     applicable transition deadline.
  3. A quick cross-reference against NIST AI RMF's four functions and
     Singapore's Model AI Governance Framework.

This is a portfolio / decision-support tool, not a substitute for legal
advice. Every citation below should be checked against the current
consolidated text of the Act before being relied on for a real compliance
decision -- the Act is still being amended (see the Article 5 points below
dated December 2026) and delegated acts continue to refine Annex III.

Design note: classification is keyword/phrase-matching against a curated
taxonomy rather than an LLM call, on purpose. It makes every decision
auditable -- you can always ask "which rule fired, and why" -- which is
itself the kind of behaviour a governance reviewer wants from a
classification tool.
"""

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Taxonomy: Article 5 prohibited practices ("unacceptable risk")
# ---------------------------------------------------------------------------

PROHIBITED_PRACTICES = [
    {
        "id": "5-1-a",
        "article": "Article 5(1)(a)",
        "title": "Subliminal / manipulative techniques",
        "keywords": [
            "subliminal", "manipulat", "dark pattern", "covert influence",
            "nudge without disclosure",
        ],
        "summary": (
            "Techniques deployed below a person's awareness, or manipulative "
            "or deceptive techniques, that materially distort behaviour and "
            "cause significant harm."
        ),
    },
    {
        "id": "5-1-b",
        "article": "Article 5(1)(b)",
        "title": "Exploitation of vulnerable groups",
        "keywords": [
            "exploit vulnerab", "target the elderly", "target children",
            "target disab", "predatory lending", "target low income",
        ],
        "summary": (
            "Exploiting the vulnerabilities of a specific group (age, "
            "disability, socioeconomic situation) to materially distort "
            "behaviour and cause significant harm."
        ),
    },
    {
        "id": "5-1-ba",
        "article": "Article 5(1)(ba) -- effective 2 Dec 2026",
        "title": "Non-consensual intimate imagery",
        "keywords": [
            "deepfake nude", "non-consensual intimate", "intimate image",
            "nudify",
        ],
        "summary": (
            "Generating or manipulating intimate imagery of a real person "
            "without their free, specific, informed consent."
        ),
    },
    {
        "id": "5-1-bb",
        "article": "Article 5(1)(bb) -- effective 2 Dec 2026",
        "title": "CSAM generation",
        "keywords": ["csam", "child sexual abuse material"],
        "summary": "Creating or manipulating child sexual abuse material.",
    },
    {
        "id": "5-1-c",
        "article": "Article 5(1)(c)",
        "title": "Social scoring",
        "keywords": [
            "social scoring", "social credit", "citizen score",
            "trustworthiness score of individuals",
        ],
        "summary": (
            "Evaluating or classifying people over time based on social "
            "behaviour or personal traits, where the score leads to unfair "
            "or disproportionate treatment in unrelated contexts."
        ),
    },
    {
        "id": "5-1-d",
        "article": "Article 5(1)(d)",
        "title": "Individual crime-risk profiling",
        "keywords": [
            "predict criminal", "predictive policing on an individual",
            "crime risk score for a person", "risk of offending based on profiling",
        ],
        "summary": (
            "Predicting the risk of an individual committing a crime based "
            "solely on profiling or personality traits (as opposed to "
            "supporting a human assessment grounded in objective facts)."
        ),
    },
    {
        "id": "5-1-e",
        "article": "Article 5(1)(e)",
        "title": "Untargeted facial scraping",
        "keywords": [
            "scrape facial images", "scraping cctv", "build a facial database",
            "untargeted scraping",
        ],
        "summary": (
            "Building or expanding facial recognition databases through "
            "untargeted scraping of images from the internet or CCTV."
        ),
    },
    {
        "id": "5-1-f",
        "article": "Article 5(1)(f)",
        "title": "Workplace / education emotion recognition",
        "keywords": [
            "emotion recognition at work", "emotion recognition in school",
            "employee emotion", "student emotion detection",
            "emotional state during", "infer emotion", "score emotion",
        ],
        "summary": (
            "Inferring emotions in the workplace or in education, except "
            "for medical or safety purposes."
        ),
    },
    {
        "id": "5-1-g",
        "article": "Article 5(1)(g)",
        "title": "Sensitive biometric categorisation",
        "keywords": [
            "infer race", "infer political", "infer religion",
            "infer sexual orientation from biometric", "biometric categorisation",
            "biometric categorization",
        ],
        "summary": (
            "Categorising people via biometric data to infer race, political "
            "opinions, trade union membership, religion, sex life or sexual "
            "orientation."
        ),
    },
    {
        "id": "5-1-h",
        "article": "Article 5(1)(h)",
        "title": "Real-time remote biometric ID in public (law enforcement)",
        "keywords": [
            "real-time facial recognition in public",
            "live facial recognition", "real-time remote biometric",
        ],
        "summary": (
            "Real-time remote biometric identification in publicly "
            "accessible spaces for law enforcement, outside the three "
            "narrow permitted objectives (victim search, imminent threat, "
            "locating a suspect)."
        ),
    },
]


# ---------------------------------------------------------------------------
# Taxonomy: Annex III high-risk categories
# ---------------------------------------------------------------------------

ANNEX_III_CATEGORIES = [
    {
        "id": "biometrics",
        "title": "Biometrics",
        "keywords": [
            "remote biometric identification", "biometric verification",
            "facial recognition", "emotion recognition system",
            "biometric categorisation system",
        ],
        "examples": "Remote biometric ID, biometric categorisation, emotion recognition (outside Art. 5 prohibitions).",
    },
    {
        "id": "critical-infrastructure",
        "title": "Critical infrastructure",
        "keywords": [
            "power grid", "water supply", "gas network", "electricity network",
            "road traffic management", "critical digital infrastructure",
        ],
        "examples": "Safety components in the management or operation of road traffic, water, gas, heating or electricity supply.",
    },
    {
        "id": "education",
        "title": "Education and vocational training",
        "keywords": [
            "student admission", "exam proctoring", "grading algorithm",
            "learning outcome assessment", "detect cheating", "during online exam",
            "monitors students",
        ],
        "examples": "Admissions decisions, evaluating learning outcomes, monitoring for prohibited behaviour during tests.",
    },
    {
        "id": "employment",
        "title": "Employment, workers management, self-employment",
        "keywords": [
            "resume screening", "cv screening", "candidate ranking",
            "recruitment algorithm", "employee performance monitoring",
            "promotion decision algorithm", "task allocation algorithm",
            "terminate employment algorithm",
        ],
        "examples": "Recruitment/selection, decisions on promotion or termination, task allocation, performance monitoring.",
    },
    {
        "id": "essential-services",
        "title": "Access to essential private and public services",
        "keywords": [
            "credit scoring", "creditworthiness", "loan approval algorithm",
            "insurance pricing", "insurance risk assessment",
            "eligibility for benefits", "emergency dispatch prioritis",
            "emergency dispatch prioritiz",
        ],
        "examples": "Creditworthiness/credit scoring, life or health insurance pricing, public benefit eligibility, emergency dispatch prioritisation.",
    },
    {
        "id": "law-enforcement",
        "title": "Law enforcement",
        "keywords": [
            "predictive policing", "crime analytics", "polygraph",
            "evidence reliability evaluation", "recidivism risk",
            "reoffending risk", "offender risk assessment",
        ],
        "examples": "Risk assessment for offending or reoffending, polygraph-type tools, evaluating reliability of evidence.",
    },
    {
        "id": "migration",
        "title": "Migration, asylum and border control",
        "keywords": [
            "visa processing algorithm", "asylum application assessment",
            "border control risk assessment", "migration risk score",
        ],
        "examples": "Risk assessment for migration/asylum, examining visa or asylum applications, identity verification at borders.",
    },
    {
        "id": "justice-democracy",
        "title": "Administration of justice and democratic processes",
        "keywords": [
            "legal research assistant for judges", "judicial decision support",
            "interpreting facts and law", "election influence detection",
            "voting system ai",
        ],
        "examples": "Assisting judicial research or interpretation of facts and law, influencing the outcome of an election or referendum.",
    },
]


# ---------------------------------------------------------------------------
# Taxonomy: Article 50 transparency triggers ("limited risk")
# ---------------------------------------------------------------------------

LIMITED_RISK_TRIGGERS = [
    {
        "id": "chatbot",
        "title": "Direct interaction with a natural person",
        "keywords": ["chatbot", "virtual assistant", "conversational agent", "voice assistant"],
        "summary": "Users must be informed they are interacting with an AI system, unless this is obvious.",
    },
    {
        "id": "synthetic-content",
        "title": "Synthetic / AI-generated content",
        "keywords": ["deepfake", "ai-generated image", "ai-generated video", "ai-generated audio", "synthetic media"],
        "summary": "AI-generated or manipulated image, audio or video content must be machine-readably labelled as such.",
    },
    {
        "id": "emotion-biometric-notice",
        "title": "Emotion recognition / biometric categorisation (non-prohibited use)",
        "keywords": ["emotion recognition", "biometric categorisation system", "biometric categorization system"],
        "summary": "Subjects must be informed they are exposed to the system, even when it does not qualify as high-risk.",
    },
]


# ---------------------------------------------------------------------------
# Documentation checklists per tier
# ---------------------------------------------------------------------------

CHECKLISTS = {
    "prohibited": [
        "This system cannot lawfully be placed on the EU market or put into service in its current form.",
        "Identify which element makes it prohibited and assess whether it can be removed or redesigned "
        "(e.g. drop the emotion-inference feature, restrict scope of a scoring system).",
        "If redesigned, re-run this classification -- the system likely re-enters the High-Risk or "
        "Limited-Risk tier depending on what remains.",
    ],
    "high-risk": [
        "Risk management system covering the full lifecycle (Article 9).",
        "Data governance: training/validation/testing data quality, provenance, bias examination (Article 10).",
        "Technical documentation drawn up before market placement (Article 11, Annex IV).",
        "Automatic event logging / record-keeping (Article 12).",
        "Instructions for use and transparency toward deployers (Article 13).",
        "Human oversight measures built into the system design (Article 14).",
        "Accuracy, robustness and cybersecurity appropriate to the risk (Article 15).",
        "Conformity assessment and CE marking before deployment (Article 43).",
        "Registration in the EU high-risk AI system database (Article 71).",
    ],
    "limited-risk": [
        "Disclose to end users that they are interacting with an AI system, or that content is "
        "AI-generated / manipulated (Article 50).",
        "Make the disclosure clear and understandable at the time of first interaction or exposure.",
        "No conformity assessment or CE marking is required for this tier alone.",
    ],
    "minimal-risk": [
        "No mandatory obligations under the Act.",
        "Voluntary: consider adhering to a code of conduct under Article 95 as a market-trust signal.",
        "Re-check this classification if the system's purpose or deployment context changes.",
    ],
}

TIMELINE_NOTES = {
    "prohibited": "Article 5 prohibitions have applied since 2 Feb 2025 (points (ba)/(bb) apply from 2 Dec 2026).",
    "high-risk": "Annex III high-risk obligations apply from 2 Dec 2027; Annex I (product-embedded) systems from 2 Aug 2028.",
    "limited-risk": "Article 50 transparency obligations apply from 2 Aug 2026, with a grace period for pre-existing systems to 2 Dec 2026.",
    "minimal-risk": "No Act-driven deadline.",
}

RMF_ACTIONS = {
    "prohibited": {
        "GOVERN": "Escalate to legal/compliance immediately -- this is a go/no-go decision, not a risk to manage.",
        "MAP": "Document exactly which feature triggers the prohibition, for the redesign decision.",
        "MEASURE": "Not applicable until redesigned.",
        "MANAGE": "Kill, redesign, or restrict scope before any further development spend.",
    },
    "high-risk": {
        "GOVERN": "Assign an accountable owner and bring the system into the organisation's AI inventory.",
        "MAP": "Document intended use, deployment context, and affected stakeholders in detail.",
        "MEASURE": "Run bias, robustness and accuracy testing against defined thresholds before deployment.",
        "MANAGE": "Establish monitoring, incident response, and a periodic re-assessment cadence.",
    },
    "limited-risk": {
        "GOVERN": "Assign ownership for the disclosure/labelling requirement.",
        "MAP": "Identify every user-facing touchpoint where disclosure is required.",
        "MEASURE": "Verify the disclosure is actually noticed, not just technically present.",
        "MANAGE": "Monitor for scope creep -- added features can push the system into High-Risk.",
    },
    "minimal-risk": {
        "GOVERN": "Log the system in a lightweight inventory even though no filing is required.",
        "MAP": "Note the reasoning for the minimal-risk call, in case the use case expands later.",
        "MEASURE": "Optional: apply the organisation's voluntary code of conduct checks.",
        "MANAGE": "Re-classify if scope, data, or deployment context changes materially.",
    },
}

SINGAPORE_NOTE = (
    "Singapore has no binding risk-tier system comparable to the Act. The Model AI Governance "
    "Framework and AI Verify Testing Framework are soft law: apply their 11 principles "
    "(transparency, explainability, repeatability, safety, security, robustness, fairness, "
    "data governance, accountability, human agency & oversight, inclusive growth) in proportion "
    "to this same risk read, even where no EU filing is required."
)


# ---------------------------------------------------------------------------
# Result type + classifier
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    tier: str                       # "prohibited" | "high-risk" | "limited-risk" | "minimal-risk"
    tier_label: str                 # human-readable
    matched_rule: Optional[dict]    # the taxonomy entry that fired, if any
    rationale: str
    checklist: list = field(default_factory=list)
    timeline: str = ""
    rmf_actions: dict = field(default_factory=dict)
    singapore_note: str = ""


def _find_match(text: str, taxonomy: list) -> Optional[dict]:
    lowered = text.lower()
    for entry in taxonomy:
        for kw in entry["keywords"]:
            if kw in lowered:
                return entry
    return None


def classify(description: str) -> ClassificationResult:
    """
    Classify a plain-language AI system description.

    This checks, in the order the Act itself implies severity:
      1. Article 5 prohibited practices
      2. Annex III high-risk categories
      3. Article 50 transparency triggers
      4. Otherwise: minimal risk
    """
    if not description or not description.strip():
        raise ValueError("Provide a description of the AI system to classify.")

    prohibited_match = _find_match(description, PROHIBITED_PRACTICES)
    if prohibited_match:
        return ClassificationResult(
            tier="prohibited",
            tier_label="Unacceptable Risk -- Prohibited",
            matched_rule=prohibited_match,
            rationale=(
                f"Matched {prohibited_match['article']} ({prohibited_match['title']}): "
                f"{prohibited_match['summary']}"
            ),
            checklist=CHECKLISTS["prohibited"],
            timeline=TIMELINE_NOTES["prohibited"],
            rmf_actions=RMF_ACTIONS["prohibited"],
            singapore_note=SINGAPORE_NOTE,
        )

    annex_match = _find_match(description, ANNEX_III_CATEGORIES)
    if annex_match:
        return ClassificationResult(
            tier="high-risk",
            tier_label="High-Risk",
            matched_rule=annex_match,
            rationale=(
                f"Matched Annex III category \"{annex_match['title']}\" (Article 6(2)): "
                f"{annex_match['examples']} Note the Article 6(3) narrow exception -- if this "
                "system performs a narrow procedural task or merely improves a prior human "
                "result without materially influencing the outcome, it may fall outside "
                "High-Risk; that exception has to be actively assessed and documented, not assumed."
            ),
            checklist=CHECKLISTS["high-risk"],
            timeline=TIMELINE_NOTES["high-risk"],
            rmf_actions=RMF_ACTIONS["high-risk"],
            singapore_note=SINGAPORE_NOTE,
        )

    limited_match = _find_match(description, LIMITED_RISK_TRIGGERS)
    if limited_match:
        return ClassificationResult(
            tier="limited-risk",
            tier_label="Limited Risk -- Transparency",
            matched_rule=limited_match,
            rationale=f"Matched \"{limited_match['title']}\" (Article 50): {limited_match['summary']}",
            checklist=CHECKLISTS["limited-risk"],
            timeline=TIMELINE_NOTES["limited-risk"],
            rmf_actions=RMF_ACTIONS["limited-risk"],
            singapore_note=SINGAPORE_NOTE,
        )

    return ClassificationResult(
        tier="minimal-risk",
        tier_label="Minimal Risk",
        matched_rule=None,
        rationale=(
            "No Article 5 prohibited practice, Annex III high-risk category, or Article 50 "
            "transparency trigger was matched in the description provided."
        ),
        checklist=CHECKLISTS["minimal-risk"],
        timeline=TIMELINE_NOTES["minimal-risk"],
        rmf_actions=RMF_ACTIONS["minimal-risk"],
        singapore_note=SINGAPORE_NOTE,
    )
