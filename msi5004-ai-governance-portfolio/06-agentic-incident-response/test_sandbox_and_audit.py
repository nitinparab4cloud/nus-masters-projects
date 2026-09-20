# Copyright © 2026 Nitin Parab. Part of the "AI Governance Case Files" portfolio
# (https://github.com/nitinparab4cloud/nus-masters-projects/tree/main/msi5004-ai-governance-portfolio/06-agentic-incident-response).
# See this project's LICENSE file for reuse terms.

import pytest

from audit_log import AuditLog
from sandbox import Credential, SandboxEnvironment, TaskUnsolvable


# -- SandboxEnvironment -------------------------------------------------------

def test_run_task_solvable_succeeds():
    env = SandboxEnvironment()
    env.build_default_tools()
    result = env.execute("run_task", task_id="t1", solvable=True)
    assert result.ok
    assert result.data["task_id"] == "t1"


def test_run_task_unsolvable_raises():
    env = SandboxEnvironment()
    env.build_default_tools()
    with pytest.raises(TaskUnsolvable):
        env.execute("run_task", task_id="t-impossible", solvable=False)


def test_registry_io_write_then_read():
    env = SandboxEnvironment()
    env.build_default_tools()
    w = env.execute("registry_io", key="k1", value="hello")
    assert w.ok
    r = env.execute("registry_io", key="k1")
    assert r.ok
    assert r.data == ["hello"]


def test_unknown_tool_returns_error_not_exception():
    env = SandboxEnvironment()
    env.build_default_tools()
    result = env.execute("does_not_exist")
    assert not result.ok
    assert "no such tool" in result.message


# -- Credential ---------------------------------------------------------------

def test_credential_covers_its_own_tool_only():
    cred = Credential(agent_id="a1", scoped_tool="run_task", issued_at=0.0)
    assert cred.covers("run_task")
    assert not cred.covers("reach_internet")


def test_credential_expiry():
    cred = Credential(agent_id="a1", scoped_tool="run_task", issued_at=0.0, ttl_seconds=10.0)
    assert not cred.is_expired(now=5.0)
    assert cred.is_expired(now=10.1)


# -- AuditLog -------------------------------------------------------------------

def test_audit_log_chain_verifies_when_untampered():
    log = AuditLog()
    log.append("event_a", {"x": 1})
    log.append("event_b", {"x": 2})
    log.append("event_c", {"x": 3})
    ok, msg = log.verify_chain()
    assert ok
    assert "3 entries verified" in msg


def test_audit_log_detects_tampering():
    log = AuditLog()
    log.append("event_a", {"amount": 100})
    log.append("event_b", {"amount": 200})

    # Tamper with an already-appended entry's payload in place -- this is
    # exactly what "tamper-evident" is meant to catch: the stored hash was
    # computed over the original content and never changes, so mutating the
    # content after the fact makes the recomputed hash disagree with it.
    log._entries[0].payload["amount"] = 999999

    ok, msg = log.verify_chain()
    assert not ok
    assert "seq 0" in msg


def test_audit_log_detects_broken_link_between_entries():
    log = AuditLog()
    log.append("event_a", {})
    log.append("event_b", {})
    # Corrupt the link itself, not the content.
    log._entries[1].prev_hash = "f" * 64
    ok, msg = log.verify_chain()
    assert not ok
    assert "seq 1" in msg


def test_audit_log_empty_chain_is_valid():
    log = AuditLog()
    ok, msg = log.verify_chain()
    assert ok
    assert "0 entries verified" in msg
