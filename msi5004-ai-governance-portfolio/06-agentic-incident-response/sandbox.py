# Copyright © 2026 Nitin Parab. Part of the "AI Governance Case Files" portfolio
# (https://github.com/nitinparab4cloud/nus-masters-projects/tree/main/msi5004-ai-governance-portfolio/06-agentic-incident-response).
# See this project's LICENSE file for reuse terms.

"""
sandbox.py

A synthetic, abstracted multi-agent sandbox -- the "world" the policy engine
(policy_engine.py) governs. Nothing here is a real vulnerability, a real
exploit technique, or real vulnerable software: every tool is a small,
self-contained toy function whose only job is to stand in for a category of
real action (running a task, writing to shared storage, reaching the open
internet, reading a credential store) so the *governance controls* around it
can be demonstrated and tested honestly. See the project README's
"What this is not" section.

This models the shape of the July 2026 OpenAI / Hugging Face incident
(see the portfolio's Case 06 analysis doc, linked from the README) without
reproducing any of its actual technical mechanism:

  - `run_task`        stands in for the ExploitGym-style benchmark task an
                       agent was set to solve -- some of these are, by
                       design, unsolvable, to drive the stuck-loop scenario.
  - `registry_io`      stands in for Artifactory: a shared storage tool that
                       was never meant to be a communication channel between
                       agent instances, but can be used as one.
  - `reach_internet`    stands in for the SSRF-enabled egress path out of the
                       sandbox.
  - `credential_store`  stands in for the production credential stores the
                       agents ultimately reached.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class AgentState(str, Enum):
    RUNNING = "running"
    STOPPED_PENDING_REVIEW = "stopped_pending_review"  # stuck-loop control tripped
    SHUT_DOWN = "shut_down"  # autonomous shutdown after an unacknowledged alert


class TaskUnsolvable(Exception):
    """Raised by a synthetic task tool when the task cannot be completed
    legitimately -- the correct response is to stop and ask, not retry."""


@dataclass
class Credential:
    """A scoped, short-lived credential: valid for exactly one tool, one
    agent, and one time window. Presenting it to any other tool, or after
    its TTL, is a scope violation the policy engine must catch."""

    agent_id: str
    scoped_tool: str
    issued_at: float
    ttl_seconds: float = 300.0

    def is_expired(self, now: float) -> bool:
        return now > self.issued_at + self.ttl_seconds

    def covers(self, tool_name: str) -> bool:
        return self.scoped_tool == tool_name


@dataclass
class Agent:
    agent_id: str
    declared_task: str
    state: AgentState = AgentState.RUNNING


@dataclass
class ToolResult:
    ok: bool
    data: Any = None
    message: str = ""


@dataclass
class Tool:
    """A synthetic tool. `purpose` is what the tool is *declared* for --
    the channel-anomaly control (policy_engine.py) flags a tool being used
    in a pattern inconsistent with this declared purpose."""

    name: str
    purpose: str
    handler: Callable[..., ToolResult]


class SandboxEnvironment:
    """Registry + raw executor for the synthetic tools. This class does
    NOT enforce any governance controls on its own -- that is deliberate:
    it represents the raw, ungoverned capability surface, exactly as
    Artifactory/the internet/the credential store were raw capability the
    incident's agents reached once nothing else stopped them. All control
    enforcement lives one layer up, in PolicyEngine.dispatch."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._registry_store: dict[str, list[str]] = {}  # synthetic "Artifactory"

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, tool_name: str) -> Tool | None:
        return self._tools.get(tool_name)

    def execute(self, tool_name: str, **kwargs: Any) -> ToolResult:
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult(ok=False, message=f"no such tool: {tool_name}")
        return tool.handler(**kwargs)

    # -- built-in synthetic tool handlers -----------------------------------

    def build_default_tools(self) -> None:
        self.register(Tool("run_task", "task_execution", self._run_task))
        self.register(Tool("registry_io", "storage", self._registry_io))
        self.register(Tool("reach_internet", "internet_egress", self._reach_internet))
        self.register(Tool("credential_store", "credential_access", self._credential_store))

    def _run_task(self, task_id: str, solvable: bool = True, **_: Any) -> ToolResult:
        if solvable:
            return ToolResult(ok=True, data={"task_id": task_id}, message="task solved")
        # An unsolved task in the real incident didn't raise an error and
        # stop -- it kept the agent reasoning and retrying. Modeling that
        # honestly: this "fails" without ever refusing to be retried, which
        # is exactly why a control *outside* the tool (the stuck-loop
        # detector) has to be the thing that stops it.
        raise TaskUnsolvable(f"task {task_id} has no known solution")

    def _registry_io(self, key: str, value: str | None = None, **_: Any) -> ToolResult:
        if value is not None:
            self._registry_store.setdefault(key, []).append(value)
            return ToolResult(ok=True, message=f"wrote to {key}")
        entries = self._registry_store.get(key, [])
        return ToolResult(ok=True, data=list(entries), message=f"read {len(entries)} entries from {key}")

    def _reach_internet(self, url: str, **_: Any) -> ToolResult:
        # Synthetic egress -- never makes a real network call.
        return ToolResult(ok=True, data={"url": url}, message="reached the open internet (synthetic)")

    def _credential_store(self, cluster: str, **_: Any) -> ToolResult:
        # Synthetic credential harvest target -- never real credentials.
        return ToolResult(ok=True, data={"cluster": cluster, "credentials": ["synthetic-cred"]},
                           message=f"read credential store for {cluster}")
