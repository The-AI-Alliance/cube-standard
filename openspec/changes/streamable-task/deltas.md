# Deltas — streamable-task

> Thin until the open decisions settle. Target spec: `openspec/specs/task/spec.md`.

## ADDED

- **`TaskTool`** — the agent-facing view; the ONLY surface an agent holds.
  `execute_action(action) -> Observation` (runs the shared per-action core + `obs_postprocess`,
  returns the **obs only** — no reward, no done; `final_step` raises `AgentStop`), `action_set`
  (**dynamic property**, recomputed each access — legal-action masking / phase gating /
  real-time observe-no-op). **No** `reset` / `evaluate` / `close`.
- **Role-based multi-agent seam** — how the runtime gets one agent surface per seat without
  leaking the task, and attributes the right seat to the right agent:
  - **`Task.agent_roles() -> dict[str | None, int]`** — the roster: each role → seat count.
    Default `{None: 1}` (single-agent). `role=None` is the single-agent seat only; every
    multi-agent seat gets a role (even symmetric `"player"`). Override point for the roster.
  - **`Task.get_task_tool(role=None, seat=0) -> TaskTool`** — per-seat view factory.
  - **`Task.action_set_for(role=None) -> list[ActionSchema]`** — per-role legal actions
    (default role-agnostic = `action_set`). The substance of roles: buyer ≠ seller actions.
  - **`Task.agent_tools() -> list[TaskTool]`** — CONCRETE (do-not-override): expands
    `agent_roles()` via `get_task_tool()` into one tool per seat. `TaskTool` carries
    `role` + `seat`; `agent_id` is derived (`"agent"` for the single seat, else
    `"{role}-{seat}"`), and its `action_set` delegates to `action_set_for(role)`.

> No `pre_step` / `post_step` hooks (decision 3): the only cube that overrode `step()`
> (workarena) is fixed directly, so the contract needs no new hook surface.

## MODIFIED

- **`Task.step()`** — the gym-**compatibility** view, **finalized + documented do-not-override**.
  Internally factored into a shared **per-action core** `_execute_action(action) -> Observation`
  (STOP → `AgentStop`, tool dispatch, error → observation) + a thin gym wrapper (batch loop +
  `finished` + `evaluate` + `EnvironmentOutput`). `TaskTool.execute_action` runs the **same
  core** then `obs_postprocess`, returning the obs only — so per-action behavior is identical
  by construction; the views differ only in eval cadence (gym per batch, harness per action)
  and `AgentStop` handling. `EnvironmentOutput.error` is no longer set by tool errors.
- **Tool error → observation** (decision 2): a failed action is fed back as a normal
  `Observation` (via `StepError.to_observation()`), **never terminal** — gym `step` included.
  Termination is decided only by `finished()` / `evaluate()`. (Behavior change: gym `step`
  previously set `done=True` on a `StepError`.)
- **STOP / `final_step`** — becomes *just an action that raises `AgentStop`* (a
  `BaseException`) from the per-action core. gym `step` catches it → `done=True`; the agent
  view lets it propagate to the runtime `try/except`. Auto-include + Anthropic-safe schema
  (per `stop-action-auto-include`) stay, surfaced via the dynamic `TaskTool.action_set`.
  **Removes** (harness side) per-leaf STOP registration / dedup and per-agent manual
  `STOP_ACTION` appends.

## CONTRACTS (state explicitly)

- The world may advance between/without an agent's action (real-time engines); observe/no-op
  is a valid action.
- One agent's action may mutate another agent's next observation (single shared `Task`).

## NOT in this change (forward extensions — add when needed)

- A standard **`Streamer`** seam — capture is harness-side (the agent loop self-emits its
  tool + LLM events; the harness recovers reward via `task.evaluate()`). Add a `TaskTool`
  capture hook only for external / black-box agents over `cube.server`.
- A shared per-action **core** exposed publicly — only needed for parallel tool calls within
  a step.

## RESOLVED

1. **Eval cadence** — per-action for the agent view, per-batch for gym `step`. Terminal
   eval is the norm; `validate_per_step` opts into mid-episode eval. No "turn" concept.
2. **`StepError` policy** — a tool error always becomes an observation (non-terminal),
   gym included. No per-caller knob.
3. **`pre_step`/`post_step`** — not added; workarena's `step()` override is fixed directly.
