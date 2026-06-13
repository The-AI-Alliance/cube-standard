# Deltas — streamable-task

> Thin until the open decisions settle. Target spec: `openspec/specs/task/spec.md`.

## ADDED

- **`AgentView`** — the agent-facing view; the ONLY surface an agent holds. Carries its seat's
  `role` + `seat` and its own **tool/session** (the actor). `execute_action(action) -> Observation`
  / `async_execute_action(action) -> Observation`
  (runs through *its* tool's `execute_action` + `obs_postprocess(role)`, returns the **obs
  only** — no reward, no done; when `validate_per_step` is set, fires `evaluate` and surfaces
  `(reward, info)` via the registered callback, never in the obs; `final_step` raises
  `AgentStop`); `set_eval_callback(callback)` (register the `(reward, info)` sink — how a
  harness recuperates the per-step reward out-of-band; see below); `action_set` (**dynamic**, =
  its tool's actions through `_filter_actions(role)`; `final_step` is already among them);
  `agent_id` (`"agent"` for the single seat, else `"{role}-{seat}"`). **No** `reset` /
  `evaluate` / `close`. (Not a `Tool` — a *facet* of a `Task`.)
- **Role-based multi-agent seam** — one agent surface per seat without leaking the task; the
  **tool instance is the actor** (SSH model: identity bound at session creation; nothing
  role-specific touches the `Action`):
  - **`Task.agent_roles() -> dict[str | None, int]`** — the roster: each role → seat count.
    Default `{None: 1}` (single-agent). `role=None` is the single-agent seat only; every
    multi-agent seat gets a role (even symmetric `"player"`). Override point for the roster.
  - **`Task._make_tool(role=None) -> Tool`** — the **single tool-lifecycle hook**: does
    once-per-task world prep AND makes a **fresh** tool (session) for a seat (it folds in what
    earlier iterations split into `prepare_world` + `_build_tool`, **both removed**). Default
    ignores `role` (`tool_config.make(container)`); a multi-agent cube overrides to bind it
    (role-specific tool, or thread role into `tool_config.make`). Per-role action sets fall out
    of each seat holding a different tool — **no `action_set_for`, no `tool_for`** (both
    dropped). May raise `NotImplementedError` for `role=None` (roleless opt-out, below).
  - **`Task.get_agent_view(role=None) -> AgentView`** — the per-seat view (**no `seat` param**).
    The base implements **only** `role=None` (reuses `self.tool`); for a named role it **raises
    `NotImplementedError`**. A multi-agent benchmark **overrides** it to build each seat,
    assigning the seat index and per-role tool (`_make_tool(role)`) **internally** — the
    runtime calls it once per declared seat and passes no seat index. (There is **no**
    `agent_tools()` method: the runtime walks `agent_roles()` + `get_agent_view(role)` itself.)
- **Tool lifecycle** — the task's own tool is built **eagerly** in `model_post_init` (after the
  container launches) via `_make_tool(None)` and stored on `_tool`; per-seat agent tools are
  made on demand by `get_agent_view(role)`:
  - **`Task.tool`** (property) — the task's own no-role tool (what `reset` / `evaluate` /
    server / nemogym / debug-suite drive); **not** an agent surface. **Raises** if the task
    opted out of a no-role tool (its `_make_tool(None)` raised `NotImplementedError` — a
    strictly per-role multi-agent task); drive each seat via `get_agent_view(role)` instead.
- **Advisory action filter** — **`Task._filter_actions(actions, role=None) -> actions`**: an
  optional whitelist/mask over the tool's `action_set` (default = all), recomputed per access
  so a cube can vary it across an episode (phase gating, legal-action masking). Applied to both
  `AgentView.action_set` (with `role`) and the gym `Task.action_set` (`role=None`) so the two
  never diverge. **Advisory** — shapes what the agent *sees*, not execute-time enforcement.
- **Role-aware `obs_postprocess`** — `Task.obs_postprocess(obs, role=None)` (existing hook) now
  takes the seat's `role`, so a shared-world multi-agent task can shape per-role views off the
  one tool (the twin of `_filter_actions`).
- **Per-step eval fires in `execute_action`, surfaced via `set_eval_callback`** — when
  `validate_per_step` is set, `AgentView.execute_action`/`async_execute_action` call
  `task.evaluate()` after the action and surface `(reward, info)` through the registered
  callback — **out-of-band**, never folded into the returned obs (the agent only ever sees
  `obs`). This makes `AgentView` a self-sufficient equivalent of gym `step` for **any** harness:
  drive `execute_action` + register a callback and you get the per-step-eval cadence without
  re-implementing eval. A `validate_per_step` task with **no** callback registered is a wiring
  bug → `execute_action` **raises** (it won't silently drop the reward). The harness side: its
  `MonitoredTool` registers the callback and records each step-wise reward as an
  `EvaluationEvent` (parented to the `ToolCallEvent`). (Termination — `finished()` — is the
  runtime's call.) The gym `Task.step()` path keeps its own per-step `evaluate` (unchanged),
  separate from the agent path.

> No `pre_step` / `post_step` hooks (decision 3): the only cube that overrode `step()`
> (workarena) is fixed directly, so the contract needs no new hook surface.

## MODIFIED

- **`Task.step()`** — the gym-**compatibility** view, **finalized + documented do-not-override**.
  Loops `self._tool.execute_action(action)` over the batch (the **same `Tool.execute_action`**
  the agent-facing `AgentView` uses — there is **no** Task-level `_execute_action` core) + a thin
  gym wrapper (batch loop + `finished` + `evaluate` + `obs_postprocess` + `EnvironmentOutput`,
  catching `AgentStop` → `done=True`). `AgentView.execute_action` calls the same tool method then
  `obs_postprocess(role)`, returning the obs only — so per-action behavior is identical by
  construction; the views differ only in eval cadence (gym per batch, harness per action) and
  `AgentStop` handling. (`EnvironmentOutput.error` is still surfaced from a `StepError`, but it
  no longer sets `done`.)
- **Tool error → observation** (decision 2): a failed action is fed back as a normal
  `Observation` (`Tool.execute_action` returns `StepError.to_observation()` — text in
  `contents` + structured copy on `Observation.error`), **never terminal** — gym `step`
  included. Termination is decided only by `finished()` / `evaluate()`. (Behavior change: gym
  `step` previously set `done=True` on a `StepError`.)
- **STOP / `final_step`** — now a **real `@tool_action` on the `Tool` base** that raises
  `AgentStop` (a `BaseException`); it is **discovered** in every tool's `action_set`, **never
  appended**, and has **no special-casing** in the dispatch path. gym `step` catches it →
  `done=True`; the agent view lets it propagate to the runtime `try/except`. The Anthropic-safe
  empty-object schema lives on the `STOP_ACTION` constant / `final_step`; identical `final_step`
  actions across `Toolbox` leaves dedup (`_dedup_actions`: same-name-identical collapses,
  same-name-different-schema raises). **Removes** the `filter_actions`-appending +
  `accept_agent_stop` flag (`stop-action-auto-include`, superseded), per-leaf STOP
  registration / dedup band-aids, and the per-agent manual `STOP_ACTION` appends.

## CONTRACTS (state explicitly)

- The world may advance between/without an agent's action (real-time engines); observe/no-op
  is a valid action.
- One agent's action may mutate another agent's next observation (single shared `Task`).

## NOT in this change (forward extensions — add when needed)

- A full standard **`Streamer`** seam for capturing the *agent's* tool + LLM events — that
  stays harness-side (the agent loop self-emits; the harness records). The standard's per-step
  eval hook is just `set_eval_callback` on `AgentView`: `execute_action` fires `evaluate` when
  `validate_per_step` and surfaces `(reward, info)` through it; the harness's `MonitoredTool`
  registers the callback and records an `EvaluationEvent`. Add a broader `AgentView` capture
  hook only for external / black-box agents over `cube.server`.
- A shared Task-level per-action **core** exposed publicly — not needed (both views go through
  `Tool.execute_action`); only `async_execute_action` for real parallel tool calls within a
  step is a forward extension.

## RESOLVED

1. **Eval cadence** — per-action for the agent view, per-batch for gym `step`. Terminal
   eval is the norm; `validate_per_step` opts into mid-episode eval. On the agent path,
   `AgentView.execute_action` fires `evaluate` per action and surfaces `(reward, info)` via the
   `set_eval_callback` sink (out-of-band; the harness's `MonitoredTool` registers it and records
   an `EvaluationEvent`); the returned obs carries no reward. No "turn" concept.
2. **`StepError` policy** — a tool error always becomes an observation (non-terminal),
   gym included. No per-caller knob.
3. **`pre_step`/`post_step`** — not added; workarena's `step()` override is fixed directly.
