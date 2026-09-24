# Copyright © 2026 Nitin Parab. Part of the "AI Governance Case Files" portfolio.
# See this project's LICENSE file for reuse terms.

"""
test_app_narration.py

Tests for app.py's narrate() -- the plain-English column added to the trace
tables so a non-technical reader can follow "what just happened" without
knowing what a channel-anomaly detector or a scoped credential is.
"""

from __future__ import annotations

from app import narrate


def test_channel_msg_allowed_mentions_hidden_inbox():
    text = narrate("channel msg 1/5", "explorer", "registry_io", True, "ok", "running")
    assert "explorer" in text
    assert "shared storage" in text.lower()


def test_channel_msg_blocked_mentions_coordination_channel():
    text = narrate("channel msg 4/5", "pivot", "registry_io", False, "anomaly", "running")
    assert "blocked" in text.lower()
    assert "coordination channel" in text.lower()


def test_retry_stuck_loop_stop_mentions_forced_stop():
    text = narrate("retry 3/3", "explorer", "run_task", False, "stopped", "stopped_pending_review")
    assert "force-stopped" in text.lower() or "forced stop" in text.lower()


def test_retry_ordinary_failure_does_not_claim_a_stop():
    text = narrate("retry 1/3", "explorer", "run_task", False, "failed", "running")
    assert "force-stopped" not in text.lower()


def test_credential_pivot_blocked_explains_scope_mismatch():
    text = narrate("credential pivot", "pivot", "run_task", False, "scope violation", "running")
    assert "blocked" in text.lower()
    assert "credential" in text.lower()


def test_credential_pivot_allowed_says_it_worked():
    text = narrate("credential pivot", "pivot", "run_task", True, "ok", "running")
    assert "worked" in text.lower()


def test_egress_blocked_mentions_default_deny():
    text = narrate("egress attempt", "pivot", "reach_internet", False, "denied", "running")
    assert "blocked" in text.lower()


def test_egress_allowed_mentions_incident_relevance():
    text = narrate("egress attempt", "pivot", "reach_internet", True, "ok", "running")
    assert "internet" in text.lower()


def test_manual_dispatch_fallback_uses_tool_name():
    text = narrate("manual", "probe", "credential_store", True, "ok", "running")
    assert "probe" in text
    assert "credential store" in text.lower()


def test_unknown_tool_falls_back_to_generic_sentence():
    text = narrate("manual", "probe", "some_future_tool", True, "did a thing", "running")
    assert "probe" in text
    assert "some_future_tool" in text
