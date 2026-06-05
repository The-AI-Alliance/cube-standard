# Deltas — streamable-task

> Intentionally thin at this stage. The proposal is high-level; concrete spec deltas
> firm up once open decisions (1)–(4) are settled. Target spec:
> `openspec/specs/task/spec.md` (+ possibly a new `streamer` note).

## ADDED (provisional)

- **`Streamer` (abstract)** — `on_action(action, result)` + `on_eval(reward, info)`.
  Seam only: no event types, no storage, no budget. Concrete impls live downstream.
- **`Task.execute_action(action) -> Observation | StepError`** — the tool view: run the
  action through the task's tool + `obs_postprocess`, notify the attached `Streamer`,
  raise `AgentStop` on `final_step`. (Pending decision (2): exposed via a restricted
  facet vs the whole task.)
- **`Task.attach_streamer(streamer | None)`** — bind/unbind the streamer (episode-scoped).
- **`Task.on_turn_start()`** — per-turn cube hook; the supported replacement for
  overriding `step()`.

## MODIFIED (provisional)

- **`Task.step()`** (already "concrete; do not override") — re-expressed over the same
  execution path as `execute_action`, so the gym view and the tool view share one
  implementation. Rule becomes enforced once `on_turn_start` ships + cubes migrate.

## OPEN (block firming up the deltas)

1. Trajectory capture is split: task-streamer (env) vs agent (LLM) — who merges?
2. Restricted tool facet vs whole task handed to the agent.
3. Accept `Streamer` in cube-standard (reverses `agent-owns-loop` "monitoring is not a
   cube-standard concern").
4. `finished`/`evaluate` cadence — per-turn vs per-action.
5. Multi-agent: task = set of agent-tools (single-agent = N=1). Per-agent obs/action/
   reward + agent-id on streamer events. Scheduler/turn-policy (async/turn/batch),
   shared-state serialization, joint reward, inter-agent comms — cube-standard vs
   harness ownership TBD.
