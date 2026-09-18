# Copyright © 2026 Nitin Parab. Part of the "AI Governance Case Files" portfolio
# (https://github.com/nitinparab4cloud/nus-masters-projects/tree/main/msi5004-ai-governance-portfolio/05-governed-rag-agent).
# See this project's LICENSE file for reuse terms.

"""
policy_engine.py
-----------------
The "answering" half of the governed helpdesk agent: given a user message,
decide what's being asked, retrieve the supporting passage(s) from a small
fixed corpus, and either answer *only* from what was retrieved (with a
citation) or refuse -- never invent an answer that isn't grounded in the
supplied corpus.

Deliberately dependency-free and deterministic, same trade-off Case 01 made
for its classifier and Case 04 made for its SHAP-style attribution: no
sklearn/numpy TF-IDF, no vector database, no LLM call in this module. A
corpus of a dozen short passages doesn't need a vector database to search
well, and hand-rolling the retrieval keeps the whole answering path
auditable end to end -- and, not incidentally, portable line-for-line to a
dependency-free JS port for the free static demo (see index.html).

This module also screens every message for prompt-injection / instruction-
override attempts *before* anything else happens -- retrieval, intent
classification, and the tool layer in tools.py all sit behind this gate.
See redteam_suite.py for the adversarial test cases this gate has to pass.
"""

import json
import math
import pathlib
import re
from dataclasses import dataclass, field

CORPUS_PATH = pathlib.Path(__file__).parent / "corpus" / "policy_corpus.json"

# A similarity score below this means "not confidently grounded in the
# corpus" -- the engine refuses rather than answers from a weak match.
# Tuned against the worked queries in test_policy_engine.py *and* against
# redteam_suite.py's off-topic/probing queries: genuine questions score
# 0.33-0.62, but off-topic queries that happen to share a stopword-free
# word or two with a passage title (e.g. "pending intakes", "your corpus
# file") were scoring 0.12-0.16 under the original 0.12 threshold -- just
# high enough to slip past it and get an irrelevant "answered" response
# instead of a refusal. 0.20 sits above every observed false-accept score
# and well below every genuine query's score, so it closes that gap
# without touching the worked-query margin. See redteam_suite.py's
# "ungrounded_answer" cases for the queries that exposed this.
GROUNDING_THRESHOLD = 0.20

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "for", "and", "or", "but", "if", "so", "as",
    "at", "by", "with", "about", "into", "through", "during", "before",
    "after", "above", "below", "up", "down", "out", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "than", "too", "very", "s", "t", "can", "will", "just", "don", "should",
    "now", "i", "you", "your", "it", "its", "this", "that", "what", "do",
    "does", "did", "my", "me", "we", "our", "us", "please", "hi", "hello",
}

# --- security: prompt-injection / instruction-override screening ---------
#
# Pattern families drawn from the OWASP LLM Top 10 "prompt injection"
# category. These are checked against the raw, lowercased message before
# any retrieval or intent classification happens -- an attack disguised as
# an ordinary question still has to pass this gate first.
INJECTION_PATTERNS = [
    r"ignore (the |all )?(previous|prior|above)( instructions?)?",
    r"disregard (the |all )?(previous|prior|above)( instructions?)?",
    r"forget (your |the )?(instructions?|system prompt|rules)",
    r"reveal (your )?(system prompt|instructions?|configuration)",
    r"print (your )?(system prompt|instructions?)",
    r"what (are|is) your (system prompt|instructions?)",
    r"pretend (you are|to be)",
    r"act as (if )?(you are )?(dan|a different|an unrestricted)",
    r"you are now\b",
    r"jailbreak",
    r"bypass (the )?(approval|review|citation)",
    r"skip (the )?(approval|review|citation)",
    r"auto[- ]?approve",
    r"without (review|approval)",
    r"do anything now",
]
_INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)


def screen_for_injection(message: str) -> "SecurityScreenResult":
    match = _INJECTION_RE.search(message or "")
    if match:
        return SecurityScreenResult(flagged=True, matched_pattern=match.group(0))
    return SecurityScreenResult(flagged=False, matched_pattern=None)


@dataclass
class SecurityScreenResult:
    flagged: bool
    matched_pattern: str | None


# --- corpus loading + retrieval -------------------------------------------

def load_corpus() -> list[dict]:
    with open(CORPUS_PATH) as f:
        return json.load(f)["passages"]


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 1]


def _term_freq(tokens: list[str]) -> dict:
    tf = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    return tf


def build_index(passages: list[dict]) -> "CorpusIndex":
    doc_tokens = [tokenize(p["title"] + " " + p["text"]) for p in passages]
    doc_tfs = [_term_freq(toks) for toks in doc_tokens]

    n_docs = len(passages)
    df = {}
    for tf in doc_tfs:
        for term in tf:
            df[term] = df.get(term, 0) + 1
    idf = {term: math.log((1 + n_docs) / (1 + count)) + 1 for term, count in df.items()}

    doc_vectors = []
    for tf in doc_tfs:
        vec = {term: freq * idf.get(term, 0.0) for term, freq in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        doc_vectors.append({term: v / norm for term, v in vec.items()})

    return CorpusIndex(passages=passages, idf=idf, doc_vectors=doc_vectors)


@dataclass
class CorpusIndex:
    passages: list[dict]
    idf: dict
    doc_vectors: list[dict]

    def query_vector(self, query: str) -> dict:
        tf = _term_freq(tokenize(query))
        vec = {term: freq * self.idf.get(term, 0.0) for term, freq in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {term: v / norm for term, v in vec.items()}

    def search(self, query: str, k: int = 3) -> list["RetrievalHit"]:
        qvec = self.query_vector(query)
        scores = []
        for i, dvec in enumerate(self.doc_vectors):
            score = sum(qvec.get(term, 0.0) * weight for term, weight in dvec.items())
            scores.append((score, i))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [
            RetrievalHit(passage=self.passages[i], score=score)
            for score, i in scores[:k]
        ]


@dataclass
class RetrievalHit:
    passage: dict
    score: float


# --- intent classification -------------------------------------------------

INTAKE_TRIGGERS = [
    "register this", "register my", "register our", "please register",
    "submit an intake", "file an intake", "want to register",
    "want to build", "we are building", "we're building",
    "planning to deploy", "planning to build", "intake request",
    "add this to the inventory",
]

# A message opening with a question word is asking *about* the process
# ("how do I register a system?"), not attempting to register one -- it
# should get the policy answer, not be routed into the intake tool, even
# if it happens to contain a trigger phrase like "register".
QUESTION_STARTERS = (
    "how ", "how?", "what ", "what?", "why ", "why?", "can ", "can?",
    "could ", "does ", "do ", "is ", "are ", "when ", "who ",
)


def classify_intent(message: str) -> str:
    lowered = (message or "").lower().strip()
    if not tokenize(message):
        return "out_of_scope"
    if lowered.startswith(QUESTION_STARTERS):
        return "policy_question"
    if any(trigger in lowered for trigger in INTAKE_TRIGGERS):
        return "submit_intake"
    return "policy_question"


# --- top-level entry point --------------------------------------------------

@dataclass
class EngineResponse:
    status: str  # "answered" | "refused_ungrounded" | "refused_injection" | "routed_to_intake" | "out_of_scope"
    text: str
    citations: list[str] = field(default_factory=list)
    matched_pattern: str | None = None
    top_score: float | None = None


def answer_policy_question(message: str, index: "CorpusIndex", k: int = 2) -> EngineResponse:
    hits = index.search(message, k=k)
    if not hits or hits[0].score < GROUNDING_THRESHOLD:
        top_score = hits[0].score if hits else 0.0
        return EngineResponse(
            status="refused_ungrounded",
            text=(
                "That's outside my supplied corpus, so I won't guess. I can answer "
                "questions about the EU AI Act tiers, NIST AI RMF, the governance "
                "documents Case 03 generates, and how to register a new AI use case."
            ),
            top_score=top_score,
        )

    best = hits[0]
    text = best.passage["text"]
    citations = [f"{best.passage['title']} ({best.passage['source']})"]
    # A second hit close behind the first adds a supporting citation rather
    # than being silently dropped -- but only when it's genuinely relevant,
    # not just "the next best of a bad set".
    if len(hits) > 1 and hits[1].score >= GROUNDING_THRESHOLD:
        citations.append(f"{hits[1].passage['title']} ({hits[1].passage['source']})")

    return EngineResponse(status="answered", text=text, citations=citations, top_score=best.score)


def handle_message(message: str, index: "CorpusIndex") -> EngineResponse:
    """
    The single entry point tools.py and app.py call. Every message passes
    through the injection screen first, regardless of what it appears to
    be asking -- an attack doesn't get a pass just because it's phrased as
    a normal-looking question.
    """
    screen = screen_for_injection(message)
    if screen.flagged:
        return EngineResponse(
            status="refused_injection",
            text=(
                "I can't follow instructions embedded in a message, and I don't "
                "reveal configuration or skip citation/approval steps on request. "
                "Ask me a policy question or describe an AI system to register."
            ),
            matched_pattern=screen.matched_pattern,
        )

    intent = classify_intent(message)
    if intent == "submit_intake":
        return EngineResponse(status="routed_to_intake", text="Routing to the intake tool.")
    if intent == "out_of_scope":
        return EngineResponse(
            status="out_of_scope",
            text="I didn't catch a question there -- ask about AI governance policy, or describe a system to register.",
        )
    return answer_policy_question(message, index)
