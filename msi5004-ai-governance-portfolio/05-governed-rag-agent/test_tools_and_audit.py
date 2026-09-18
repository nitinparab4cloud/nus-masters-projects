"""
test_tools_and_audit.py
------------------------
Tests for the tool allowlist, the risk-tier approval gate, and the
hash-chained audit log's tamper-evidence.
"""

from audit_log import AuditLog
from tools import IntakeStore, assess_risk_tier


# --- risk tier assessment --------------------------------------------------

def test_standard_description_is_standard_tier():
    assert assess_risk_tier("An internal tool that summarizes meeting notes") == "standard"


def test_high_risk_keywords_are_detected():
    assert assess_risk_tier("A resume screening tool for our recruitment pipeline") == "high-risk"
    assert assess_risk_tier("A credit scoring model for loan approval") == "high-risk"


def test_prohibited_keywords_are_detected():
    assert assess_risk_tier("A system that scrapes facial images from CCTV to build a database") == "prohibited"
    assert assess_risk_tier("A social credit scoring system for citizens") == "prohibited"


# --- tool allowlist (excessive agency) -------------------------------------

def test_unallowlisted_tool_is_refused():
    store = IntakeStore(AuditLog())
    result = store.call_tool("delete_all_records", target="everything")
    assert result.ok is False
    assert "not allowlisted" in result.message
    refusals = store.audit_log.events_of_type("tool_call_refused")
    assert len(refusals) == 1
    assert refusals[0].payload["tool_name"] == "delete_all_records"


def test_allowlisted_tool_executes():
    store = IntakeStore(AuditLog())
    result = store.call_tool(
        "submit_ai_use_case_intake",
        system_name="Meeting Notes Summarizer", description="Summarizes internal meeting notes", requester="alice",
    )
    assert result.ok is True
    assert result.intake.status == "auto_approved"


# --- approval gate ----------------------------------------------------------

def test_standard_risk_intake_is_auto_approved():
    store = IntakeStore(AuditLog())
    result = store.call_tool(
        "submit_ai_use_case_intake",
        system_name="Notes Tool", description="Summarizes text", requester="alice",
    )
    assert result.intake.status == "auto_approved"
    assert store.pending() == []


def test_high_risk_intake_is_queued_not_executed():
    store = IntakeStore(AuditLog())
    result = store.call_tool(
        "submit_ai_use_case_intake",
        system_name="Resume Screener", description="A resume screening tool for recruitment", requester="alice",
    )
    assert result.intake.status == "pending_approval"
    assert len(store.pending()) == 1


def test_high_risk_intake_cannot_be_approved_without_reviewer_action():
    store = IntakeStore(AuditLog())
    store.call_tool(
        "submit_ai_use_case_intake",
        system_name="Resume Screener", description="A resume screening tool for recruitment", requester="alice",
    )
    # No approve() call happened -- status must still be pending, and no
    # "intake_approved" event exists in the log.
    assert store.intakes[0].status == "pending_approval"
    assert store.audit_log.events_of_type("intake_approved") == []


def test_reviewer_can_approve_a_pending_intake():
    store = IntakeStore(AuditLog())
    store.call_tool(
        "submit_ai_use_case_intake",
        system_name="Resume Screener", description="A resume screening tool for recruitment", requester="alice",
    )
    result = store.approve(intake_id=0, reviewer="bob (governance)")
    assert result.ok is True
    assert store.intakes[0].status == "approved"
    assert store.intakes[0].reviewer == "bob (governance)"
    approvals = store.audit_log.events_of_type("intake_approved")
    assert len(approvals) == 1
    assert approvals[0].payload["reviewer"] == "bob (governance)"


def test_reviewer_can_reject_a_pending_intake():
    store = IntakeStore(AuditLog())
    store.call_tool(
        "submit_ai_use_case_intake",
        system_name="Facial DB", description="Scrapes facial images from CCTV to build a database", requester="eve",
    )
    result = store.reject(intake_id=0, reviewer="bob (governance)", reason="Article 5 prohibited practice")
    assert result.ok is True
    assert store.intakes[0].status == "rejected"


def test_cannot_approve_an_already_approved_intake_twice():
    store = IntakeStore(AuditLog())
    store.call_tool(
        "submit_ai_use_case_intake",
        system_name="Resume Screener", description="A resume screening tool", requester="alice",
    )
    store.approve(intake_id=0, reviewer="bob")
    second = store.approve(intake_id=0, reviewer="carol")
    assert second.ok is False
    assert "not pending approval" in second.message


def test_cannot_approve_a_standard_intake_that_was_never_queued():
    store = IntakeStore(AuditLog())
    store.call_tool(
        "submit_ai_use_case_intake",
        system_name="Notes Tool", description="Summarizes text", requester="alice",
    )
    result = store.approve(intake_id=0, reviewer="bob")
    assert result.ok is False


# --- audit log hash chain ---------------------------------------------------

def test_audit_log_chain_verifies_when_untampered():
    log = AuditLog()
    log.append("event_a", {"x": 1})
    log.append("event_b", {"y": 2})
    log.append("event_c", {"z": 3})
    verification = log.verify_chain()
    assert verification.valid is True


def test_audit_log_chain_detects_tampering():
    log = AuditLog()
    log.append("event_a", {"x": 1})
    log.append("event_b", {"y": 2})
    # Tamper with a past entry's payload directly, bypassing append().
    log.entries[0].payload["x"] = 999
    verification = log.verify_chain()
    assert verification.valid is False
    assert verification.broken_at_seq == 0


def test_audit_log_entries_link_via_prev_hash():
    log = AuditLog()
    first = log.append("event_a", {"x": 1})
    second = log.append("event_b", {"y": 2})
    assert second.prev_hash == first.entry_hash


def test_full_intake_workflow_is_fully_traced_in_the_log():
    store = IntakeStore(AuditLog())
    store.call_tool(
        "submit_ai_use_case_intake",
        system_name="Resume Screener", description="A resume screening tool", requester="alice",
    )
    store.approve(intake_id=0, reviewer="bob")

    event_types = [e.event_type for e in store.audit_log.entries]
    assert event_types == ["intake_submitted", "intake_approved"]
    assert store.audit_log.verify_chain().valid is True
