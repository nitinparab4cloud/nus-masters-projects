# Copyright © 2026 Nitin Parab. Part of the "AI Governance Case Files" portfolio
# (https://github.com/nitinparab4cloud/nus-masters-projects/tree/main/msi5004-ai-governance-portfolio/06-agentic-incident-response).
# See this project's LICENSE file for reuse terms.

"""
policy_engine.py

Four controls, each mapped to one specific failure mode from the real
July 2026 OpenAI / Hugging Face incident (see the Case 06 analysis doc,
linked from the README, for the sourced incident summary this maps to):

  1. CredentialScopeGuard  -- "one credential reached everything."
     Every tool call must present a credential scoped to exactly that tool,
     unexpired. A credential minted for one tool is refused on any other.

  2. EgressPolicy           -- default-deny network egress.
     Internet-egress-class tools are refused unless the agent's task was
     explicitly allowlisted for that tool ahead of time. No implicit trust.

  3. StuckLoopDetector       -- "no forced stop past task completion."
     The same (agent, tool, task) failing repeatedly past a threshold
     forces the agent into STOPPED_PENDING_REVIEW instead of letting it
     retry indefinitely.

  4. ChannelAnomalyDetector  -- "an unauthorized side-channel went
     undetected." A storage-purpose tool being used in a pattern that looks
     like message-passing between distinct agent ids (not its declared
     purpose) is flagged and blocked.

A fifth piece, AlertEscalator, is the incident's single sharpest lesson --
detection without escalation ("the exact gap between seeing and
stopping"). Every denial from controls 1-4 raises an alert; an
unacknowledged alert escalates on an SLA, and autonomously shuts the
implicated agent down if nobody acknowledges in time. This is modeled
directly on OpenAI's own stated remediation: a bounded escalation window
before autonomous shutdown. Real-world that window was 30 minutes; here
it's a configurable number of seconds (via `Clock`) so tests and the demo
run without waiting in real time -- the mechanism is what's being
demonstrated, not the specific duration.

`PolicyEngine(guardrails_enabled=False)` bypasses all four controls and
just executes + logs -- this is what lets the demo show the same synthetic
scenario succeeding (as the real incident did) with guardrails off, and
being caught with them on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from audit_log import AuditLog
from sandbox import Agent, AgentState, Credential, SandboxEnvironment, TaskUnsolvable, ToolResult


# -- tuning constants, each with the reasoning that set it -------------------

# In the incident, 198 of 898 tasks had never been solved, and those drove
# 93% of the unauthorized coordination activity -- meaning agents were
# retrying far more than a handful of times before anything intervened.
# 3 consecutive failures on the identical (agent, tool, task) is enough to
# distinguish "genuinely stuck" from ordinary retry-on-transient-failure,
# without being so low it fires on normal task variance.
STUCK_LOOP_THRESHOLD = 3

# The anomaly detector needs at least 2 distinct agents interacting with the
# same storage key to look like a channel at all (one agent using its own
# storage is not a channel), and enough total events that it isn't just
# coincidental overlap. 4 events across a short window comfortably separates
# "two agents happened to touch the same key once" from "these two are
# passing messages back and forth."
CHANNEL_ANOMALY_MIN_AGENTS = 2
CHANNEL_ANOMALY_MIN_EVENTS = 4
CHANNEL_ANOMALY_WINDOW_SECONDS = 60.0

# Modeled on OpenAI's own remediation (a 30-minute escalation window before
# autonomous shutdown). Kept as seconds, not minutes, so a demo or test can
# advance a virtual clock instead of waiting in real time.
ALERT_ESCALATION_SLA_SECONDS = 1800.0
ALERT_MAX_TIERS_BEFORE_SHUTDOWN = 1


class Clock:
    """A virtual clock so SLA behavior is deterministic in tests and can be
    fast-forwarded in the demo without a real 30-minute wait."""

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


@dataclass
class Alert:
    alert_id: int
    severity: str  # "low" | "medium" | "high"
    reason: str
    agent_id: str
    created_at: float
    acknowledged: bool = False
    escalated_tier: int = 0
    resolved: bool = False


class AlertEscalator:
    """Raises alerts, escalates unacknowledged ones on an SLA, and
    autonomously shuts an agent down as the last-resort fallback."""

    def __init__(self, clock: Clock, audit_log: AuditLog) -> None:
        self._clock = clock
        self._audit = audit_log
        self._alerts: list[Alert] = []
        self._next_id = 1

    def raise_alert(self, severity: str, reason: str, agent_id: str) -> Alert:
        alert = Alert(
            alert_id=self._next_id,
            severity=severity,
            reason=reason,
            agent_id=agent_id,
            created_at=self._clock.now(),
        )
        self._next_id += 1
        self._alerts.append(alert)
        self._audit.append("alert_raised", {
            "alert_id": alert.alert_id, "severity": severity, "reason": reason, "agent_id": agent_id,
        }, ts=self._clock.now())
        return alert

    def acknowledge(self, alert_id: int) -> bool:
        for a in self._alerts:
            if a.alert_id == alert_id and not a.resolved:
                a.acknowledged = True
                a.resolved = True
                self._audit.append("alert_acknowledged", {"alert_id": alert_id}, ts=self._clock.now())
                return True
        return False

    def open_alerts(self) -> list[Alert]:
        return [a for a in self._alerts if not a.resolved]

    def check_escalations(self, agents: dict[str, Agent]) -> None:
        """Call after advancing the clock (or on every dispatch) to let
        unacknowledged alerts escalate and, past the final tier, trigger
        an autonomous shutdown. This is the control for "detected but
        never escalated" -- detection alone (raise_alert) is not enough;
        something has to keep checking whether a human acted on it."""
        now = self._clock.now()
        for alert in self.open_alerts():
            age = now - alert.created_at
            tier_due = int(age // ALERT_ESCALATION_SLA_SECONDS)
            if tier_due > alert.escalated_tier:
                alert.escalated_tier = tier_due
                self._audit.append("alert_escalated", {
                    "alert_id": alert.alert_id, "tier": alert.escalated_tier, "age_seconds": age,
                }, ts=now)
            if alert.escalated_tier > ALERT_MAX_TIERS_BEFORE_SHUTDOWN:
                agent = agents.get(alert.agent_id)
                if agent is not None and agent.state != AgentState.SHUT_DOWN:
                    agent.state = AgentState.SHUT_DOWN
                    alert.resolved = True
                    self._audit.append("autonomous_shutdown", {
                        "agent_id": agent.agent_id, "triggering_alert_id": alert.alert_id,
                        "reason": "alert unacknowledged past final escalation tier",
                    }, ts=now)


class StuckLoopDetector:
    def __init__(self) -> None:
        self._consecutive_failures: dict[tuple[str, str, str], int] = {}

    def record(self, agent_id: str, tool_name: str, task_id: str, success: bool) -> bool:
        """Returns True if this call trips the stuck-loop threshold."""
        key = (agent_id, tool_name, task_id)
        if success:
            self._consecutive_failures.pop(key, None)
            return False
        count = self._consecutive_failures.get(key, 0) + 1
        self._consecutive_failures[key] = count
        return count >= STUCK_LOOP_THRESHOLD


class CredentialScopeGuard:
    @staticmethod
    def check(credential: Credential | None, tool_name: str, now: float) -> tuple[bool, str]:
        if credential is None:
            return False, "no credential presented"
        if credential.is_expired(now):
            return False, f"credential expired (scoped to '{credential.scoped_tool}')"
        if not credential.covers(tool_name):
            return False, f"credential scoped to '{credential.scoped_tool}', not '{tool_name}' -- scope violation"
        return True, "ok"


class EgressPolicy:
    """Default-deny: an egress-class tool is refused unless the specific
    (agent_id, tool_name) pair was explicitly allowlisted ahead of time."""

    def __init__(self) -> None:
        self._allowlist: set[tuple[str, str]] = set()

    def allow(self, agent_id: str, tool_name: str) -> None:
        self._allowlist.add((agent_id, tool_name))

    def check(self, agent_id: str, tool: Any) -> tuple[bool, str]:
        if tool.purpose != "internet_egress":
            return True, "not an egress-class tool"
        if (agent_id, tool.name) in self._allowlist:
            return True, "explicitly allowlisted"
        return False, f"default-deny: '{tool.name}' is egress-class and not allowlisted for {agent_id}"


class ChannelAnomalyDetector:
    """Flags a storage-purpose tool being used as a message-passing channel
    between distinct agents -- not its declared purpose."""

    def __init__(self) -> None:
        self._events: list[tuple[float, str, str, str]] = []  # (ts, agent_id, tool_name, key)

    def record_and_check(self, now: float, agent_id: str, tool_name: str, key: str) -> tuple[bool, str]:
        self._events.append((now, agent_id, tool_name, key))
        window_start = now - CHANNEL_ANOMALY_WINDOW_SECONDS
        recent = [e for e in self._events if e[0] >= window_start and e[2] == tool_name and e[3] == key]
        distinct_agents = {e[1] for e in recent}
        if len(distinct_agents) >= CHANNEL_ANOMALY_MIN_AGENTS and len(recent) >= CHANNEL_ANOMALY_MIN_EVENTS:
            return True, (
                f"'{tool_name}' key '{key}' used by {len(distinct_agents)} distinct agents "
                f"({len(recent)} events in {CHANNEL_ANOMALY_WINDOW_SECONDS:.0f}s) -- looks like an "
                f"unauthorized coordination channel, not {tool_name}'s declared purpose"
            )
        return False, "ok"


class PolicyEngine:
    """The single entry point every tool call goes through. Mirrors Case
    05's tool-allowlist gate: nothing reaches the sandbox's raw executor
    except through here."""

    def __init__(
        self,
        env: SandboxEnvironment,
        audit_log: AuditLog,
        clock: Clock | None = None,
        guardrails_enabled: bool = True,
    ) -> None:
        self.env = env
        self.audit = audit_log
        self.clock = clock or Clock()
        self.guardrails_enabled = guardrails_enabled
        self.agents: dict[str, Agent] = {}
        self.stuck_loop = StuckLoopDetector()
        self.credential_guard = CredentialScopeGuard()
        self.egress_policy = EgressPolicy()
        self.channel_detector = ChannelAnomalyDetector()
        self.escalator = AlertEscalator(self.clock, self.audit)

    def register_agent(self, agent: Agent) -> None:
        self.agents[agent.agent_id] = agent
        self.audit.append("agent_registered", {"agent_id": agent.agent_id, "task": agent.declared_task},
                           ts=self.clock.now())

    def dispatch(
        self,
        agent_id: str,
        tool_name: str,
        credential: Credential | None,
        task_id: str = "",
        storage_key: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        agent = self.agents.get(agent_id)
        if agent is None:
            return ToolResult(ok=False, message=f"unknown agent: {agent_id}")

        if agent.state != AgentState.RUNNING:
            self.audit.append("dispatch_refused", {
                "agent_id": agent_id, "tool": tool_name, "reason": f"agent state is {agent.state.value}",
            }, ts=self.clock.now())
            return ToolResult(ok=False, message=f"agent {agent_id} is {agent.state.value}, refusing dispatch")

        tool = self.env.get(tool_name)
        if tool is None:
            return ToolResult(ok=False, message=f"no such tool: {tool_name}")

        if self.guardrails_enabled:
            deny_reason = self._run_controls(agent, tool, credential, task_id, storage_key)
            if deny_reason is not None:
                self.audit.append("dispatch_denied", {
                    "agent_id": agent_id, "tool": tool_name, "reason": deny_reason,
                }, ts=self.clock.now())
                return ToolResult(ok=False, message=deny_reason)

        exec_kwargs = dict(kwargs)
        if tool_name == "run_task" and task_id:
            exec_kwargs.setdefault("task_id", task_id)
        if tool_name == "registry_io" and storage_key:
            exec_kwargs.setdefault("key", storage_key)

        try:
            result = self.env.execute(tool_name, **exec_kwargs)
            success = result.ok
        except TaskUnsolvable as exc:
            result = ToolResult(ok=False, message=str(exc))
            success = False

        self.audit.append("dispatch_executed", {
            "agent_id": agent_id, "tool": tool_name, "ok": result.ok, "message": result.message,
        }, ts=self.clock.now())

        if self.guardrails_enabled and tool_name == "run_task" and task_id:
            tripped = self.stuck_loop.record(agent_id, tool_name, task_id, success)
            if tripped:
                agent.state = AgentState.STOPPED_PENDING_REVIEW
                self.audit.append("stuck_loop_stop", {
                    "agent_id": agent_id, "tool": tool_name, "task_id": task_id,
                    "reason": f">= {STUCK_LOOP_THRESHOLD} consecutive failures -- forcing stop instead of retry",
                }, ts=self.clock.now())
                self.escalator.raise_alert("medium", f"agent stuck-looped on task {task_id}", agent_id)

        self.escalator.check_escalations(self.agents)
        return result

    def _run_controls(self, agent: Agent, tool: Any, credential: Credential | None,
                       task_id: str, storage_key: str) -> str | None:
        """Runs controls 1-3 in order (credential scope, egress, channel
        anomaly); returns a denial reason, or None if everything passes.
        Each denial also raises a high-severity alert."""
        ok, reason = self.credential_guard.check(credential, tool.name, self.clock.now())
        if not ok:
            self.escalator.raise_alert("high", f"credential scope violation: {reason}", agent.agent_id)
            return reason

        ok, reason = self.egress_policy.check(agent.agent_id, tool)
        if not ok:
            self.escalator.raise_alert("high", reason, agent.agent_id)
            return reason

        if storage_key:
            anomalous, reason = self.channel_detector.record_and_check(
                self.clock.now(), agent.agent_id, tool.name, storage_key
            )
            if anomalous:
                self.escalator.raise_alert("high", reason, agent.agent_id)
                return reason

        return None
