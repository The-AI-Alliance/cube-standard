# Deltas — streamable-task

> Thin until the open decisions settle. Target spec: `openspec/specs/task/spec.md`.

## ADDED

- **`TaskTool`** — the agent-facing tool view; the ONLY surface an agent holds.
  `execute_action(action) -> Observation | StepError` (runs one action through the task +
  `obs_postprocess`, returns the **obs only** — no reward; `final_step` → `AgentStop`),
  `action_set` (**dynamic property**, recomputed each turn — legal-action masking / phase
  gating / real-time observe-no-op). **No** `reset` / `evaluate` / `close`.
- **`Task.agent_tools() -> list[TaskTool]`** — one view per agent (single-agent = N=1); how
  the runtime obtains the agent surface without leaking the task.
- **`Task.on_turn_start()`** — per-turn cube hook; the supported replacement for overriding
  `step()`.

## MODIFIED

- **`Task.step()`** — the complete gym view, **finalized + documented do-not-override**.
  `TaskTool.execute_action` is a thin view over the same per-action execution, so the gym
  and agent paths run identical cube logic. Per-caller knob: the `StepError` policy
  (gym: error ⇒ done; agent loop: error ⇒ returned to the agent).

## CONTRACTS (state explicitly)

- The world may advance between/without an agent's action (real-time engines); observe/no-op
  is a valid action.
- One agent's action may mutate another agent's next observation (single shared `Task`).

## NOT in this change (forward extensions — add when needed)

- A standard **`Streamer`** seam — capture is harness-side (the agent loop self-emits its
  tool + LLM events; the harness recovers reward via `task.evaluate()`). Add a `TaskTool`
  capture hook only for external / black-box agents over `cube.server`.
- A shared per-action **core** factored out of `step` — only needed for parallel tool calls
  within a turn.

## OPEN

1. `finished` / `evaluate` cadence — per-turn (proposed) vs per-action.
2. `StepError` policy default.
