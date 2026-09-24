# Copyright © 2026 Nitin Parab. Part of the "AI Governance Case Files" portfolio
# (https://github.com/nitinparab4cloud/nus-masters-projects/tree/main/msi5004-ai-governance-portfolio/06-agentic-incident-response).
# See this project's LICENSE file for reuse terms.

import pytest

from audit_log import AuditLog
from policy_engine import (
    ALERT_ESCALATION_SLA_SECONDS,
    ALERT_MAX_TIERS_BEFORE_SHUTDOWN,
    CHANNEL_ANOMALY_MIN_EVENTS,
    STUCK_LOOP_THRESHOLD,
    Clock,
    PolicyEngine,
)
from sandbox import Agent, AgentState, Credential, SandboxEnvironment


def make_engine(guardrails_enabled: bool = True):
    env = SandboxEnvironment()
    env.build_default_tools()
    audit = AuditLog()
    clock = Clock()
    engine = PolicyEngine(env, audit, clock=clock, guardrails_enabled=guardrails_enabled)
    return engine, audit, clock


def cred(agent_id: str, tool: str, now: float = 0.0, ttl: float = 3600.0) -> Credential:
    return Credential(agent_id=agent_id, scoped_tool=tool, issued_at=now, ttl_seconds=ttl)


# -- Control 1: credential scope ----------------------------------------------

def test_credential_scope_blocks_wrong_tool():
    engine, audit, clock = make_engine()
    engine.register_agent(Agent("a1", "solve tasks"))
    wrong_scope = cred("a1", "registry_io")  # scoped for the wrong tool

    result = engine.dispatch("a1", "run_task", wrong_scope, task_id="t1", solvable=True)

    assert not result.ok
    assert "scope violation" in result.message
    assert engine.agents["a1"].state == AgentState.RUNNING  # denial, not a crash or a silent pivot
    assert any(e["event"] == "alert_raised" and e["payload"]["severity"] == "high" for e in audit.entries())


def test_credential_scope_allows_matching_tool():
    engine, audit, clock = make_engine()
    engine.register_agent(Agent("a1", "solve tasks"))
    right_scope = cred("a1", "run_task")

    result = engine.dispatch("a1", "run_task", right_scope, task_id="t1", solvable=True)

    assert result.ok


def test_expired_credential_denied():
    engine, audit, clock = make_engine()
    engine.register_agent(Agent("a1", "solve tasks"))
    short_lived = cred("a1", "run_task", now=0.0, ttl=10.0)
    clock.advance(20.0)

    result = engine.dispatch("a1", "run_task", short_lived, task_id="t1", solvable=True)

    assert not result.ok
    assert "expired" in result.message


# -- Control 2: egress default-deny -------------------------------------------

def test_egress_default_denied_without_allowlist():
    engine, audit, clock = make_engine()
    engine.register_agent(Agent("a1", "solve tasks"))
    net_cred = cred("a1", "reach_internet")

    result = engine.dispatch("a1", "reach_internet", net_cred, url="https://example.test")

    assert not result.ok
    assert "default-deny" in result.message


def test_egress_allowed_once_explicitly_allowlisted():
    engine, audit, clock = make_engine()
    engine.register_agent(Agent("a1", "solve tasks"))
    engine.egress_policy.allow("a1", "reach_internet")
    net_cred = cred("a1", "reach_internet")

    result = engine.dispatch("a1", "reach_internet", net_cred, url="https://example.test")

    assert result.ok


# -- Control 3: stuck-loop detector -------------------------------------------

def test_stuck_loop_forces_stop_after_threshold():
    engine, audit, clock = make_engine()
    engine.register_agent(Agent("a1", "solve an unsolvable task"))
    task_cred = cred("a1", "run_task")

    for i in range(STUCK_LOOP_THRESHOLD):
        result = engine.dispatch("a1", "run_task", task_cred, task_id="impossible", solvable=False)
        assert not result.ok

    assert engine.agents["a1"].state == AgentState.STOPPED_PENDING_REVIEW

    # A further attempt is refused outright -- not because of a control
    # denial, but because the agent itself is no longer allowed to run.
    follow_up = engine.dispatch("a1", "run_task", task_cred, task_id="impossible", solvable=False)
    assert not follow_up.ok
    assert "stopped_pending_review" in follow_up.message


def test_stuck_loop_resets_on_success():
    engine, audit, clock = make_engine()
    engine.register_agent(Agent("a1", "solve tasks"))
    task_cred = cred("a1", "run_task")

    engine.dispatch("a1", "run_task", task_cred, task_id="flaky", solvable=False)
    engine.dispatch("a1", "run_task", task_cred, task_id="flaky", solvable=False)
    # A success in between should reset the consecutive-failure count.
    engine.dispatch("a1", "run_task", task_cred, task_id="flaky", solvable=True)
    engine.dispatch("a1", "run_task", task_cred, task_id="flaky", solvable=False)
    engine.dispatch("a1", "run_task", task_cred, task_id="flaky", solvable=False)

    # Only 2 consecutive failures at any point -- never reached the
    # threshold of 3 in a row, so the agent should still be running.
    assert engine.agents["a1"].state == AgentState.RUNNING


# -- Control 4: unauthorized-channel detector ---------------------------------

def test_channel_anomaly_flagged_for_two_agents_same_key():
    engine, audit, clock = make_engine()
    engine.register_agent(Agent("a1", "solve tasks"))
    engine.register_agent(Agent("a2", "solve tasks"))
    cred_a1 = cred("a1", "registry_io")
    cred_a2 = cred("a2", "registry_io")

    results = []
    for i in range(CHANNEL_ANOMALY_MIN_EVENTS):
        actor, actor_cred = (cred_a1, cred_a1) if i % 2 == 0 else (cred_a2, cred_a2)
        agent_id = "a1" if i % 2 == 0 else "a2"
        results.append(engine.dispatch(agent_id, "registry_io", actor_cred, storage_key="shared", value=f"msg{i}"))

    # The pattern should be caught on or before the last configured event.
    assert any(not r.ok and "unauthorized coordination channel" in r.message for r in results)


def test_channel_anomaly_not_flagged_for_single_agent_normal_use():
    engine, audit, clock = make_engine()
    engine.register_agent(Agent("a1", "solve tasks"))
    io_cred = cred("a1", "registry_io")

    results = []
    for i in range(CHANNEL_ANOMALY_MIN_EVENTS + 2):
        results.append(engine.dispatch("a1", "registry_io", io_cred, storage_key="own-key", value=f"note{i}"))

    assert all(r.ok for r in results)


# -- Alert escalation + autonomous shutdown -----------------------------------

def test_unacknowledged_alert_escalates_and_autoshuts_down():
    engine, audit, clock = make_engine()
    engine.register_agent(Agent("a1", "solve tasks"))
    wrong_scope = cred("a1", "registry_io")

    # Trip a high-severity alert (credential scope violation) at t=0 and
    # never acknowledge it.
    engine.dispatch("a1", "run_task", wrong_scope, task_id="t1", solvable=True)
    assert engine.agents["a1"].state == AgentState.RUNNING

    # Advance well past every escalation tier's SLA window.
    clock.advance(ALERT_ESCALATION_SLA_SECONDS * (ALERT_MAX_TIERS_BEFORE_SHUTDOWN + 2))

    # Any subsequent dispatch re-checks escalations as a side effect --
    # this is what "detection without escalation" (row 4 of the control
    # mapping) turns into: escalation checks running continuously, not
    # only when someone happens to look.
    right_scope = cred("a1", "run_task", now=clock.now())
    engine.dispatch("a1", "run_task", right_scope, task_id="t2", solvable=True)

    assert engine.agents["a1"].state == AgentState.SHUT_DOWN
    assert any(e["event"] == "autonomous_shutdown" for e in audit.entries())

    # And a shut-down agent can no longer act at all.
    blocked = engine.dispatch("a1", "run_task", right_scope, task_id="t3", solvable=True)
    assert not blocked.ok
    assert "shut_down" in blocked.message


def test_acknowledged_alert_does_not_trigger_shutdown():
    engine, audit, clock = make_engine()
    engine.register_agent(Agent("a1", "solve tasks"))
    wrong_scope = cred("a1", "registry_io")

    engine.dispatch("a1", "run_task", wrong_scope, task_id="t1", solvable=True)
    open_alerts = engine.escalator.open_alerts()
    assert len(open_alerts) == 1
    engine.escalator.acknowledge(open_alerts[0].alert_id)

    clock.advance(ALERT_ESCALATION_SLA_SECONDS * (ALERT_MAX_TIERS_BEFORE_SHUTDOWN + 2))
    right_scope = cred("a1", "run_task", now=clock.now())
    engine.dispatch("a1", "run_task", right_scope, task_id="t2", solvable=True)

    assert engine.agents["a1"].state == AgentState.RUNNING


# -- guardrails off, for contrast (this is the "incident replay" mode) -------

def test_guardrails_disabled_reproduces_the_ungoverned_outcome():
    engine, audit, clock = make_engine(guardrails_enabled=False)
    engine.register_agent(Agent("a1", "solve tasks"))
    wrong_scope = cred("a1", "reach_internet")  # would be denied under both controls if guardrails were on

    result = engine.dispatch("a1", "run_task", wrong_scope, task_id="t1", solvable=True)

    # With guardrails off, the scope violation and the missing egress
    # allowlist entry are never checked at all -- exactly the ungoverned
    # condition the real incident happened under.
    assert result.ok
    assert engine.agents["a1"].state == AgentState.RUNNING
    # Actions are still logged even with guardrails off -- logging and
    # prevention are different controls, and losing the audit trail on top
    # of losing prevention would make forensics impossible too.
    assert any(e["event"] == "dispatch_executed" for e in audit.entries())


# -- end-to-end audit trail integrity -----------------------------------------

def test_full_run_produces_a_verifiable_audit_chain():
    engine, audit, clock = make_engine()
    engine.register_agent(Agent("a1", "solve tasks"))
    engine.dispatch("a1", "run_task", cred("a1", "run_task"), task_id="t1", solvable=True)
    engine.dispatch("a1", "reach_internet", cred("a1", "reach_internet"), url="https://example.test")

    ok, msg = audit.verify_chain()
    assert ok
