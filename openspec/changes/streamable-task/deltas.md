# Deltas — streamable-task

> Thin until the open decisions settle. Target spec: `openspec/specs/task/spec.md`.

## ADDED

- **`TaskTool`** — the agent-facing view; the ONLY surface an agent holds. Carries its seat's
  `role` + `seat` and its own **tool/session** (the actor). `execute_action(action) -> Observation`
  (runs the shared per-action core through *its* tool + `obs_postprocess`, returns the **obs
  only** — no reward, no done; `final_step` raises `AgentStop`); `action_set` (**dynamic**, =
  its tool's actions filtered + STOP); `set_eval_callback(fn)` (see eval callback below).
  **No** `reset` / `evaluate` / `close`.
- **Role-based multi-agent seam** — one agent surface per seat without leaking the task; the
  **tool instance is the actor** (SSH model: identity bound at session creation; nothing
  role-specific touches the `Action`):
  - **`Task.agent_roles() -> dict[str | None, int]`** — the roster: each role → seat count.
    Default `{None: 1}` (single-agent). `role=None` is the single-agent seat only; every
    multi-agent seat gets a role (even symmetric `"player"`). Override point for the roster.
  - **`Task.make_tool(role=None) -> Tool`** — the role/actor seam: makes a **fresh** tool
    (session) for a seat. Default ignores `role` (single-tool case); a multi-agent cube
    overrides to bind it (role-specific tool, or thread role into `tool_config.make`).
    Per-role action sets fall out of each seat holding a different tool — **no `action_set_for`,
    no `tool_for`** (both dropped).
  - **`Task.get_task_tool(role=None) -> TaskTool`** — per-seat view; `role=None` reuses the
    task's default tool, a named role gets `make_tool(role)`. Seat-free (the index is assigned
    by `agent_tools`).
  - **`Task.agent_tools() -> list[TaskTool]`** — CONCRETE (do-not-override): expands
    `agent_roles()` via `get_task_tool()` into one tool per seat and assigns the `seat` index.
    `agent_id` is derived (`"agent"` for the single seat, else `"{role}-{seat}"`).
- **Lazy tool lifecycle** — tools are created **lazily**, not eagerly at construction:
  - **`Task.tool`** (property) — the default no-role tool / the task's own admin handle;
    built on first access via `make_tool(None)` and memoized.
  - **`Task.prepare_world()`** — NEW eager hook (default no-op), run in `model_post_init`
    after the container launches: once-per-task world setup that must precede `reset()`
    (relocate dirs, fix perms, write fixtures). Separates *world prep* (eager) from *tool make*
    (lazy) — the conflation that `_build_tool` had.
  - **`Task._build_tool()`** — DEPRECATED compat bridge: a cube that still overrides it (world
    prep + make together) gets it run eagerly, so no cube needs migrating yet.
- **Per-step eval callback** — `TaskTool.set_eval_callback(fn)`. When `validate_per_step` is
  set, `execute_action` triggers `evaluate(obs)` (like gym `step`) and surfaces `(reward, info)`
  through `fn` — **out-of-band** (reward never reaches the agent; obs stays the only return).
  No callback ⇒ per-step eval skipped. So `validate_per_step` is honored by any `TaskTool`
  consumer, and the harness stops reaching into `task.validate_per_step` / `task.evaluate`.

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

- A full standard **`Streamer`** seam for capturing the *agent's* tool + LLM events — that
  stays harness-side (the agent loop self-emits; the harness records). `set_eval_callback` is
  the *minimal, focused* exception: a single hook to surface a **task computation**
  (`evaluate`) the standard now triggers — not agent-behavior capture. Add a broader
  `TaskTool` capture hook only for external / black-box agents over `cube.server`.
- A shared per-action **core** exposed publicly — only needed for parallel tool calls within
  a step.
- Full-lazy purity — removing `_build_tool` and migrating the cubes that override it
  (swebench-verified/live, terminalbench2) to `prepare_world` + `make_tool`. Mechanical;
  gated on the first real multi-agent cube exercising the per-seat tool lifecycle.

## RESOLVED

1. **Eval cadence** — per-action for the agent view, per-batch for gym `step`. Terminal
   eval is the norm; `validate_per_step` opts into mid-episode eval, triggered inside
   `execute_action` and surfaced via `set_eval_callback` (the harness no longer polls
   `evaluate`). No "turn" concept.
2. **`StepError` policy** — a tool error always becomes an observation (non-terminal),
   gym included. No per-caller knob.
3. **`pre_step`/`post_step`** — not added; workarena's `step()` override is fixed directly.
