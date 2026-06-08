# Deltas — streamable-task

> Thin until the open decisions settle. Target spec: `openspec/specs/task/spec.md`.

## ADDED

- **`TaskTool`** — the agent-facing tool view; the ONLY surface an agent holds.
  `execute_action(action) -> Observation | StepError` (relays to step's per-action
  sub-function, returns the **obs only** — no reward; `final_step` raises `AgentStop`),
  `action_set` (**dynamic property**, recomputed each step — legal-action masking / phase
  gating / real-time observe-no-op). **No** `reset` / `evaluate` / `close`.
- **`Task.agent_tools() -> list[TaskTool]`** — one view per agent (single-agent = N=1); how
  the runtime obtains the agent surface without leaking the task.
- **`Task.pre_step()` / `Task.post_step()`** — **optional** hooks (default no-op), run
  before / after a step's actions. Per-step cube setup/teardown (cache invalidation, phase /
  clock tick). The supported replacement for overriding `step()`. (No "turn" in the contract.)

## MODIFIED

- **`Task.step()`** — the complete gym view, **finalized + documented do-not-override**.
  Internally factored into a **per-action sub-function** (tool dispatch + `obs_postprocess` +
  STOP) + a gym wrapper (`evaluate` + `EnvironmentOutput`). `TaskTool.execute_action` relays
  to the **sub-function only** (returns obs, no evaluate) — gym and agent paths share one
  implementation; the harness owns eval cadence (no double eval). Per-caller knob: the
  `StepError` policy (gym: error ⇒ done; agent loop: error ⇒ returned to the agent).
- **STOP / `final_step`** — becomes *just an action that raises `AgentStop`* (renamed
  `TaskDone`, a `BaseException`) from the per-action sub-function. gym `step` catches it →
  `done=True`; the agent loop lets it propagate to the harness `try/except`. Auto-include +
  Anthropic-safe schema (per `stop-action-auto-include`) stay, surfaced via the dynamic
  `TaskTool.action_set`. **Removes** per-leaf STOP registration / dedup and the per-agent
  manual `STOP_ACTION` appends.

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

## OPEN

1. `finished` / `evaluate` cadence — per-step (proposed) vs per-action.
2. `StepError` policy default.
