# Deltas — streamable-task

> Intentionally thin at this stage. The proposal is high-level; concrete spec deltas
> firm up once open decisions (1)–(4) are settled. Target spec:
> `openspec/specs/task/spec.md` (+ possibly a new `streamer` note).

## ADDED (provisional)

- **`Streamer` (abstract)** — `on_action(action, result)` + `on_eval(reward, info)`.
  Seam only: no event types, no storage, no budget. Concrete impls live downstream.
- **`TaskTool`** — the agent-facing tool facet over a task; the ONLY surface the agent
  holds. `execute_action(action) -> Observation | StepError` (delegates to the task's
  per-action execution + `obs_postprocess`, emits `on_action`, raises `AgentStop` on
  `final_step`), `action_set`, `attach_streamer`. **No** `reset`/`evaluate`/`close` —
  lifecycle stays on `Task`, runtime-driven.
- **`Task.agent_tools() -> list[TaskTool]`** — one facet per agent (single-agent = N=1);
  how the runtime obtains the agent surface without leaking the task.
- **`Task.on_turn_start()`** — per-turn cube hook; the supported replacement for
  overriding `step()`.

## MODIFIED (provisional)

- **`Task.step()`** (already "concrete; do not override") — re-expressed over the same
  execution path as `execute_action`, so the gym view and the tool view share one
  implementation. Rule becomes enforced once `on_turn_start` ships + cubes migrate.

## OPEN (block firming up the deltas)

1. Trajectory capture is split: `TaskTool` stream (env) vs agent (LLM) — who merges?
2. ~~Restricted facet vs whole task~~ — RESOLVED: agent holds a `TaskTool`, never the task.
3. Accept `Streamer` + `TaskTool` in cube-standard (reverses `agent-owns-loop`
   "monitoring is not a cube-standard concern").
4. `finished`/`evaluate` cadence — per-turn vs per-action.
5. Multi-agent: task = set of agent-tools (single-agent = N=1). Per-agent obs/action/
   reward + agent-id on streamer events. Scheduler/turn-policy (async/turn/batch),
   shared-state serialization, joint reward, inter-agent comms — cube-standard vs
   harness ownership TBD.
