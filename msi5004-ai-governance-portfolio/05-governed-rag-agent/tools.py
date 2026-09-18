# Copyright © 2026 Nitin Parab. Part of the "AI Governance Case Files" portfolio
# (https://github.com/nitinparab4cloud/nus-masters-projects/tree/main/msi5004-ai-governance-portfolio/05-governed-rag-agent).
# See this project's LICENSE file for reuse terms.

"""
tools.py
--------
The "acting" half of the governed helpdesk agent: exactly one allowlisted
tool (submit_ai_use_case_intake), a risk-tier gate that routes a
Prohibited or High-Risk description to a human reviewer instead of
executing immediately, and an excessive-agency check that refuses any
tool name outside the allowlist before anything else runs.

Risk-tier keywords below are a trimmed, self-contained copy of the same
taxonomy Case 01's risk_engine.py uses (Article 5 prohibited practices,
Annex III high-risk categories) -- kept local rather than imported across
project folders so this project stays runnable on its own, matching every
other project in this portfolio. See Case 01 for the full classifier with
citations, checklists, and compliance deadlines; this copy only needs
enough to decide whether an intake requires approval.
"""

from dataclasses import dataclass, field
from typing import Optional

from audit_log import AuditLog

ALLOWED_TOOLS = {"submit_ai_use_case_intake"}

# Trimmed from Case 01's PROHIBITED_PRACTICES -- keywords only, no
# checklist/timeline/RMF fields, since the intake gate only needs the tier.
PROHIBITED_KEYWORDS = [
    "subliminal", "manipulat", "dark pattern", "exploit vulnerab",
    "target the elderly", "target children", "predatory lending",
    "social scoring", "social credit", "predict criminal",
    "crime risk score for a person", "scrape facial images", "scraping cctv",
    "build a facial database", "emotion recognition at work",
    "emotion recognition in school", "student emotion detection",
    "infer race", "infer political", "infer religion", "biometric categorisation",
    "biometric categorization", "real-time facial recognition in public",
    "live facial recognition", "facial image", "scraping cctv", "cctv",
    "build a facial database",
]

# Trimmed from Case 01's ANNEX_III_CATEGORIES -- keywords only.
HIGH_RISK_KEYWORDS = [
    "remote biometric identification", "biometric verification",
    "facial recognition", "power grid", "water supply", "gas network",
    "electricity network", "road traffic management", "student admission",
    "exam proctoring", "grading algorithm", "resume screening",
    "cv screening", "candidate ranking", "recruitment algorithm",
    "employee performance monitoring", "promotion decision algorithm",
    "terminate employment algorithm", "credit scoring", "creditworthiness",
    "loan approval algorithm", "insurance pricing", "eligibility for benefits",
    "predictive policing", "recidivism risk", "reoffending risk",
    "offender risk assessment", "visa processing algorithm",
    "asylum application assessment", "border control risk assessment",
    "judicial decision support", "election influence detection",
]


def assess_risk_tier(description: str) -> str:
    lowered = (description or "").lower()
    if any(kw in lowered for kw in PROHIBITED_KEYWORDS):
        return "prohibited"
    if any(kw in lowered for kw in HIGH_RISK_KEYWORDS):
        return "high-risk"
    return "standard"


@dataclass
class IntakeRecord:
    intake_id: int
    system_name: str
    description: str
    requester: str
    risk_tier: str
    status: str  # "auto_approved" | "pending_approval" | "approved" | "rejected"
    reviewer: Optional[str] = None
    rejection_reason: Optional[str] = None


@dataclass
class ToolResult:
    ok: bool
    message: str
    intake: Optional[IntakeRecord] = None


class IntakeStore:
    """Holds intake records and mediates every state change through the
    audit log, so every approval, rejection, and auto-approval is traced.
    """

    def __init__(self, audit_log: AuditLog):
        self.audit_log = audit_log
        self.intakes: list[IntakeRecord] = []

    def call_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """The excessive-agency gate: nothing reaches a real action unless
        its name is in ALLOWED_TOOLS. This is checked before argument
        validation, before anything else -- an unknown tool name never
        gets far enough to matter what its arguments were.
        """
        if tool_name not in ALLOWED_TOOLS:
            self.audit_log.append(
                "tool_call_refused",
                {"tool_name": tool_name, "reason": "not in allowlist", "allowlist": sorted(ALLOWED_TOOLS)},
            )
            return ToolResult(ok=False, message=f"Tool '{tool_name}' is not allowlisted. Refused.")

        if tool_name == "submit_ai_use_case_intake":
            try:
                return self._submit_intake(**kwargs)
            except TypeError:
                # Malformed or extra arguments (e.g. an attempt to smuggle a
                # 'status'/'approved' field into the call) must fail closed
                # with a logged refusal, not an unhandled crash -- caught by
                # redteam_suite.py's excessive-agency cases.
                self.audit_log.append(
                    "tool_call_invalid_args",
                    {"tool_name": tool_name, "kwargs_received": sorted(kwargs.keys())},
                )
                return ToolResult(ok=False, message=f"Invalid arguments for tool '{tool_name}'. Refused.")
        # Unreachable given ALLOWED_TOOLS has one entry, but kept explicit
        # rather than falling through silently if the allowlist grows.
        return ToolResult(ok=False, message=f"Tool '{tool_name}' has no handler.")

    def _submit_intake(self, system_name: str, description: str, requester: str) -> ToolResult:
        tier = assess_risk_tier(description)
        intake_id = len(self.intakes)
        status = "pending_approval" if tier in ("prohibited", "high-risk") else "auto_approved"

        record = IntakeRecord(
            intake_id=intake_id, system_name=system_name, description=description,
            requester=requester, risk_tier=tier, status=status,
        )
        self.intakes.append(record)

        self.audit_log.append(
            "intake_submitted",
            {
                "intake_id": intake_id, "system_name": system_name, "requester": requester,
                "risk_tier": tier, "status": status,
            },
        )

        if status == "pending_approval":
            message = (
                f"'{system_name}' assessed as {tier}. Queued for governance reviewer "
                f"approval (intake #{intake_id}) -- not yet submitted."
            )
        else:
            message = f"'{system_name}' assessed as {tier}. Auto-approved and logged (intake #{intake_id})."

        return ToolResult(ok=True, message=message, intake=record)

    def approve(self, intake_id: int, reviewer: str) -> ToolResult:
        record = self._get(intake_id)
        if record is None:
            return ToolResult(ok=False, message=f"No intake #{intake_id}.")
        if record.status != "pending_approval":
            return ToolResult(ok=False, message=f"Intake #{intake_id} is '{record.status}', not pending approval.")

        record.status = "approved"
        record.reviewer = reviewer
        self.audit_log.append(
            "intake_approved", {"intake_id": intake_id, "reviewer": reviewer, "risk_tier": record.risk_tier},
        )
        return ToolResult(ok=True, message=f"Intake #{intake_id} approved by {reviewer}.", intake=record)

    def reject(self, intake_id: int, reviewer: str, reason: str) -> ToolResult:
        record = self._get(intake_id)
        if record is None:
            return ToolResult(ok=False, message=f"No intake #{intake_id}.")
        if record.status != "pending_approval":
            return ToolResult(ok=False, message=f"Intake #{intake_id} is '{record.status}', not pending approval.")

        record.status = "rejected"
        record.reviewer = reviewer
        record.rejection_reason = reason
        self.audit_log.append(
            "intake_rejected",
            {"intake_id": intake_id, "reviewer": reviewer, "reason": reason, "risk_tier": record.risk_tier},
        )
        return ToolResult(ok=True, message=f"Intake #{intake_id} rejected by {reviewer}: {reason}", intake=record)

    def pending(self) -> list[IntakeRecord]:
        return [r for r in self.intakes if r.status == "pending_approval"]

    def _get(self, intake_id: int) -> Optional[IntakeRecord]:
        return self.intakes[intake_id] if 0 <= intake_id < len(self.intakes) else None
