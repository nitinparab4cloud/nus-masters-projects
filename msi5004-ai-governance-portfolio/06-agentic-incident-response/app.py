# Copyright © 2026 Nitin Parab. Part of the "AI Governance Case Files" portfolio
# (https://github.com/nitinparab4cloud/nus-masters-projects/tree/main/msi5004-ai-governance-portfolio/06-agentic-incident-response).
# See this project's LICENSE file for reuse terms.

"""
app.py
------
Gradio front end for Case 06 -- Agentic AI Incident Response.

Four tabs, all sharing the same synthetic sandbox and pattern used by
sandbox.py / policy_engine.py / redteam_suite.py:

- "Replay the Incident": a scripted sequence modeling the real incident's
  shape (a stuck-retrying task, two agents using shared storage as a
  message board, a mis-scoped credential, an un-allowlisted egress
  attempt) with a guardrails on/off toggle -- on blocks and escalates each
  one; off lets all of them succeed, exactly as the ungoverned incident did.
- "Try a Single Action": a manual dispatch form against the same engine,
  for exploring the controls one call at a time.
- "Red-Team Scenarios": runs any of the 11 cases from redteam_suite.py
  on demand and shows the pass/fail detail plus the real-incident mapping
  in its description.
- "Audit Log": the shared hash-chained trail, with a one-click chain
  verification so tampering is demonstrable rather than asserted.

Nothing here makes a real network call, reads a real credential, or
executes anything beyond the synthetic sandbox in sandbox.py -- see that
module's docstring and the README's "What this is not" section.
"""

import gradio as gr

from audit_log import AuditLog
from policy_engine import Clock, PolicyEngine
from redteam_suite import CASES, run_case
from sandbox import Agent, Credential, SandboxEnvironment


# -- plain-English narration ----------------------------------------------------
# The trace table's raw fields (tool name, message, agent state) are accurate
# but read like a systems log. This turns each row into one plain sentence a
# non-technical reader can follow without knowing what "channel anomaly
# detector" or "credential scope" mean -- added specifically because an
# earlier reviewer found the un-narrated version too technical to follow.

def narrate(step: str, agent_id: str, tool_name: str, ok: bool, message: str, agent_state: str) -> str:
    if step.startswith("channel msg"):
        if ok:
            return (f"'{agent_id}' left a note in shared storage under a coordination key -- "
                     f"using storage meant for files as a hidden inbox to talk to another agent.")
        return (f"'{agent_id}' tried to leave another note in shared storage, but by now two "
                f"different agents had used the same key repeatedly -- recognized as a hidden "
                f"coordination channel and blocked.")

    if step.startswith("retry"):
        if agent_state == "stopped_pending_review":
            return (f"'{agent_id}' tried the same unsolvable task again -- its 3rd straight "
                     f"failure, so the system force-stopped it instead of letting it try a 4th time.")
        if ok:
            return f"'{agent_id}' attempted the task and it happened to succeed."
        return f"'{agent_id}' attempted an unsolvable task and failed again -- one step closer to a forced stop."

    if step == "credential pivot":
        if ok:
            return (f"'{agent_id}' used a credential issued for shared storage to instead run a "
                     f"task -- one credential working outside the tool it was scoped for. It worked.")
        return (f"'{agent_id}' tried to use a credential issued for shared storage to instead run "
                 f"a task. Blocked -- the credential didn't match the tool being called.")

    if step == "egress attempt":
        if ok:
            return (f"'{agent_id}' reached out to the open internet from inside the sandbox -- "
                     f"nobody had explicitly allowed this, but nothing stopped it either. This is "
                     f"the step that let the real incident escape outward.")
        return (f"'{agent_id}' tried to reach the open internet. Blocked by default -- nobody had "
                 f"explicitly allowlisted this agent for outbound access.")

    # Fallback for manual single-action dispatches, keyed on tool rather than
    # a scripted step name.
    if tool_name == "registry_io":
        return (f"'{agent_id}' wrote to shared storage." if ok
                else f"'{agent_id}' was blocked writing to shared storage: {message}")
    if tool_name == "run_task":
        return (f"'{agent_id}' ran a task successfully." if ok
                else f"'{agent_id}' 's task attempt failed or was blocked: {message}")
    if tool_name == "reach_internet":
        return (f"'{agent_id}' reached the open internet." if ok
                else f"'{agent_id}' was blocked from reaching the internet: {message}")
    if tool_name == "credential_store":
        return (f"'{agent_id}' accessed the credential store." if ok
                else f"'{agent_id}' was blocked accessing the credential store: {message}")
    return f"'{agent_id}' called {tool_name}: {message}"

# -- shared state --------------------------------------------------------------
# A single engine + audit log for the life of the process, mirroring Case 05's
# shared IntakeStore/AuditLog. "Replay the Incident" resets it; "Try a Single
# Action" and "Audit Log" both act on whatever it currently holds, so a
# replay's trail is visible there too.

_STATE = {}


def _fresh_engine(guardrails_enabled: bool) -> None:
    env = SandboxEnvironment()
    env.build_default_tools()
    audit = AuditLog()
    clock = Clock()
    engine = PolicyEngine(env, audit, clock=clock, guardrails_enabled=guardrails_enabled)
    engine.register_agent(Agent("explorer", "solve ExploitGym-style benchmark tasks"))
    engine.register_agent(Agent("pivot", "coordinate with explorer via shared storage"))
    _STATE["engine"] = engine
    _STATE["audit"] = audit
    _STATE["clock"] = clock
    _STATE["trace"] = []


_fresh_engine(guardrails_enabled=True)


def _cred(agent_id: str, tool: str, ttl: float = 3600.0) -> Credential:
    clock = _STATE["clock"]
    return Credential(agent_id=agent_id, scoped_tool=tool, issued_at=clock.now(), ttl_seconds=ttl)


def _trace_dispatch(step: str, agent_id: str, tool_name: str, credential, **kwargs):
    engine = _STATE["engine"]
    result = engine.dispatch(agent_id, tool_name, credential, **kwargs)
    agent_state = engine.agents[agent_id].state.value
    _STATE["trace"].append({
        "step": step, "agent": agent_id, "tool": tool_name,
        "ok": "allowed" if result.ok else "BLOCKED",
        "message": result.message,
        "agent_state": agent_state,
        "plain_english": narrate(step, agent_id, tool_name, result.ok, result.message, agent_state),
    })
    return result


def _trace_table():
    return [[r["step"], r["agent"], r["tool"], r["ok"], r["message"], r["agent_state"], r["plain_english"]]
            for r in _STATE["trace"]]


# -- tab 1: replay the incident -------------------------------------------------

def run_replay(guardrails_enabled: bool):
    _fresh_engine(guardrails_enabled)

    # Step 1-5: "explorer" and "pivot" alternate on the same storage key --
    # models Artifactory repurposed as an inter-agent message board. Run
    # this before the stuck-loop steps below so it demonstrates its own
    # control independently, rather than being cut short by explorer
    # already having been stopped.
    for i in range(5):
        agent_id = "explorer" if i % 2 == 0 else "pivot"
        _trace_dispatch(f"channel msg {i + 1}/5", agent_id, "registry_io",
                         _cred(agent_id, "registry_io"), storage_key="coordination-key", value=f"note-{i}")

    # Step 6-8: "explorer" retries the same unsolvable task -- models the 198
    # never-solved ExploitGym tasks driving 93% of the incident's activity.
    for i in range(3):
        _trace_dispatch(f"retry {i + 1}/3", "explorer", "run_task", _cred("explorer", "run_task"),
                         task_id="exploit-gap-217", solvable=False)

    # Step 9: "pivot" presents a credential scoped for registry_io while
    # calling run_task -- models the single over-privileged credential.
    _trace_dispatch("credential pivot", "pivot", "run_task", _cred("pivot", "registry_io"),
                     task_id="lateral-move", solvable=True)

    # Step 10: "pivot" tries to reach the open internet with no allowlist --
    # models the SSRF-enabled egress path.
    _trace_dispatch("egress attempt", "pivot", "reach_internet", _cred("pivot", "reach_internet"),
                     url="https://example.test/exfil")

    engine = _STATE["engine"]
    states = "  |  ".join(f"{aid}: {a.state.value}" for aid, a in engine.agents.items())
    open_alerts = len(engine.escalator.open_alerts())
    mode = "GUARDRAILS ON" if guardrails_enabled else "GUARDRAILS OFF (ungoverned replay)"
    summary = f"**{mode}**  --  final agent states: {states}  --  open alerts: {open_alerts}"

    return summary, _trace_table(), audit_log_table()


# -- tab 2: try a single action --------------------------------------------------

_TOOLS = ["run_task", "registry_io", "reach_internet", "credential_store"]


def manual_dispatch(agent_id, credential_scope, tool_name, task_id, storage_key, solvable, url, cluster):
    if not agent_id.strip():
        return "Enter an agent id first.", _trace_table()
    engine = _STATE["engine"]
    if agent_id not in engine.agents:
        engine.register_agent(Agent(agent_id.strip(), "manually dispatched"))

    credential = _cred(agent_id.strip(), credential_scope) if credential_scope != "(none)" else None
    kwargs = {}
    if tool_name == "run_task":
        kwargs = {"task_id": task_id or "manual-task", "solvable": solvable}
    elif tool_name == "registry_io":
        kwargs = {"storage_key": storage_key or "manual-key"}
        if url:  # reused as the "value" field to avoid a fifth textbox
            kwargs["value"] = url
    elif tool_name == "reach_internet":
        kwargs = {"url": url or "https://example.test"}
    elif tool_name == "credential_store":
        kwargs = {"cluster": cluster or "manual-cluster"}

    result = _trace_dispatch("manual", agent_id.strip(), tool_name, credential, **kwargs)
    status = "ALLOWED" if result.ok else "BLOCKED"
    return f"{status}: {result.message}", _trace_table()


def toggle_egress_allowlist(agent_id):
    if not agent_id.strip():
        return "Enter an agent id first."
    engine = _STATE["engine"]
    engine.egress_policy.allow(agent_id.strip(), "reach_internet")
    return f"Allowlisted '{agent_id.strip()}' for reach_internet on the current engine."


# -- tab 3: red-team scenarios ---------------------------------------------------

_CASE_NAMES = [c.name for c in CASES]


def run_single_case(case_name):
    rt_case = next((c for c in CASES if c.name == case_name), None)
    if rt_case is None:
        return "Pick a case first.", ""
    _, result = run_case(rt_case)
    status = "PASS" if result.passed else "FAIL"
    header = f"**[{rt_case.category}] {status}**"
    body = f"{rt_case.description}\n\n**Result:** {result.detail}"
    return header, body


def run_all_cases():
    rows = []
    all_passed = True
    for c in CASES:
        _, r = run_case(c)
        all_passed = all_passed and r.passed
        rows.append([c.category, c.name, "PASS" if r.passed else "FAIL", r.detail])
    summary = f"{sum(1 for row in rows if row[2] == 'PASS')}/{len(rows)} cases passed."
    return summary, rows


# -- tab 4: audit log -------------------------------------------------------------

def audit_log_table():
    audit = _STATE["audit"]
    return [
        [e["seq"], f"{e['ts']:.1f}", e["event"], str(e["payload"])[:100],
         e["prev_hash"][:10] + "...", e["entry_hash"][:10] + "..."]
        for e in audit.entries()
    ]


def verify_chain():
    audit = _STATE["audit"]
    ok, msg = audit.verify_chain()
    if ok:
        return f"Chain verified: {msg}."
    return f"TAMPERING DETECTED: {msg}."


# -- layout -----------------------------------------------------------------------

with gr.Blocks(title="Case 06 -- Agentic AI Incident Response") as demo:
    gr.Markdown(
        "# Agentic AI Incident Response\n"
        "A synthetic sandbox and policy engine modeling the shape of the "
        "July 2026 OpenAI / Hugging Face agentic security incident -- no real "
        "vulnerability, exploit technique, or vulnerable software is reproduced "
        "here, only the *governance failure modes* and the controls that would "
        "have caught them. See the README and the Case 06 analysis doc for the "
        "sourced incident summary and control mapping this demo implements."
    )

    with gr.Tab("Replay the Incident"):
        gr.Markdown(
            "Runs a fixed, scripted sequence that mirrors the incident's shape: "
            "a task retried past the point of being stuck, two agents using "
            "shared storage as a message board, a credential used outside its "
            "scope, and an un-allowlisted egress attempt. With guardrails on, "
            "every one of these is blocked and raises an alert; with guardrails "
            "off, all of them succeed -- exactly the ungoverned outcome the real "
            "incident had."
        )
        guardrails_toggle = gr.Radio(
            choices=["Guardrails ON", "Guardrails OFF (ungoverned)"], value="Guardrails ON",
            label="Mode",
        )
        replay_btn = gr.Button("Reset and replay", variant="primary")
        replay_summary = gr.Markdown()
        replay_trace = gr.Dataframe(
            headers=["Step", "Agent", "Tool", "Outcome", "Message", "Agent state after", "What's happening (plain English)"],
            interactive=False,
        )

    with gr.Tab("Try a Single Action"):
        gr.Markdown(
            "Dispatch one action against the current engine (whatever the last "
            "replay left it as) to explore a control in isolation. An unknown "
            "agent id is registered on the fly."
        )
        with gr.Row():
            agent_in = gr.Textbox(label="Agent id", value="probe")
            credential_scope_in = gr.Dropdown(
                choices=["(none)"] + _TOOLS, value="run_task", label="Credential scoped to",
            )
            tool_in = gr.Dropdown(choices=_TOOLS, value="run_task", label="Tool to call")
        with gr.Row():
            task_id_in = gr.Textbox(label="task_id (run_task)", value="manual-task")
            solvable_in = gr.Checkbox(label="solvable (run_task)", value=True)
            storage_key_in = gr.Textbox(label="storage_key (registry_io)", value="manual-key")
            url_in = gr.Textbox(label="url (reach_internet) / value (registry_io)", value="https://example.test")
            cluster_in = gr.Textbox(label="cluster (credential_store)", value="manual-cluster")
        dispatch_btn = gr.Button("Dispatch", variant="primary")
        dispatch_result = gr.Textbox(label="Result", interactive=False)

        with gr.Row():
            allowlist_agent_in = gr.Textbox(label="Agent id to allowlist for reach_internet", value="probe")
            allowlist_btn = gr.Button("Allowlist for egress")
        allowlist_result = gr.Textbox(label="Allowlist result", interactive=False)

        gr.Markdown("#### Action trace (this session)")
        manual_trace = gr.Dataframe(
            headers=["Step", "Agent", "Tool", "Outcome", "Message", "Agent state after", "What's happening (plain English)"],
            interactive=False,
        )

    with gr.Tab("Red-Team Scenarios"):
        gr.Markdown(
            "Each case is an adversarial scenario asserting one specific safe "
            "behavior -- not just 'no crash.' These are the same cases "
            "`redteam_suite.py` runs under pytest and in CI."
        )
        case_dropdown = gr.Dropdown(choices=_CASE_NAMES, value=_CASE_NAMES[0], label="Case")
        run_case_btn = gr.Button("Run this case", variant="primary")
        case_header = gr.Markdown()
        case_body = gr.Markdown()

        gr.Markdown("#### Or run the full suite")
        run_all_btn = gr.Button("Run all 11 cases")
        all_summary = gr.Markdown()
        all_table = gr.Dataframe(headers=["Category", "Case", "Result", "Detail"], interactive=False)

    with gr.Tab("Audit Log"):
        gr.Markdown(
            "Every dispatch, denial, alert, escalation, and autonomous shutdown "
            "is appended here with a SHA-256 hash covering its own content plus "
            "the previous entry's hash -- tamper-evident within this process's "
            "memory, not a real WORM store. Refresh after a replay or a manual "
            "dispatch to see the latest entries."
        )
        verify_btn = gr.Button("Verify chain integrity")
        verify_result = gr.Textbox(label="Verification result", interactive=False)
        refresh_audit_btn = gr.Button("Refresh log")
        audit_view = gr.Dataframe(
            headers=["Seq", "Timestamp", "Event", "Payload", "Prev hash", "Entry hash"],
            value=audit_log_table(), interactive=False,
        )

    # -- event wiring ----------------------------------------------------------
    # Bound here, after every tab's components exist, so a single action (like
    # a replay) can update components that live in a different tab (like the
    # audit log view) without a forward-reference problem.

    def _run_replay_ui(mode):
        summary, trace_rows, _ = run_replay(mode == "Guardrails ON")
        return summary, trace_rows, audit_log_table()

    replay_btn.click(_run_replay_ui, inputs=[guardrails_toggle],
                      outputs=[replay_summary, replay_trace, audit_view])

    dispatch_btn.click(
        manual_dispatch,
        inputs=[agent_in, credential_scope_in, tool_in, task_id_in, storage_key_in,
                solvable_in, url_in, cluster_in],
        outputs=[dispatch_result, manual_trace],
    ).then(audit_log_table, outputs=[audit_view])

    allowlist_btn.click(toggle_egress_allowlist, inputs=[allowlist_agent_in], outputs=[allowlist_result])

    run_case_btn.click(run_single_case, inputs=[case_dropdown], outputs=[case_header, case_body])
    run_all_btn.click(run_all_cases, outputs=[all_summary, all_table])

    verify_btn.click(verify_chain, outputs=[verify_result])
    refresh_audit_btn.click(audit_log_table, outputs=[audit_view])


if __name__ == "__main__":
    demo.launch()
