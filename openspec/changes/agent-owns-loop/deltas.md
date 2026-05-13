# Deltas: Agent Owns the Loop (cube-standard companion)

Applies to:

- `openspec/specs/tool/spec.md`
- `openspec/specs/server/spec.md`

See primary RFC: `cube-harness/openspec/changes/agent-owns-loop/`.

---

## MODIFIED — `openspec/specs/tool/spec.md`

### New invariant: monitoring is not a cube-standard concern

Add to the Invariants section:

> Trajectory capture, persistence, and per-call instrumentation are NOT
> responsibilities of `Tool` / `AsyncTool` / `Toolbox`. Runtimes that drive
> tools (cube-harness, future remote runners) attach monitoring by composing
> wrappers around tools — they do not subclass `Tool` to add side-effects.
> Adding storage, summary, or trajectory hooks inside a `Tool` subclass is a
> review-blocking design error.

Rationale: cube-harness's new `MonitoredTool` composes around any
`cube.tool.Tool` or `AsyncTool`. Keeping `Tool` free of monitoring concerns
means the same task can be driven by the in-process harness, by a future
remote runner over `cube.server`, or by any third-party runtime, without
duplicating capture logic inside the tool implementation.

No code change.

---

## MODIFIED — `openspec/specs/server/spec.md`

### New note in Public API

Add at the end of the Public API section:

> The JSON-RPC endpoints `tools/call` and `cube/step` are the canonical surface
> for external agents that do not run in the same process as the task. A
> harness driving an agent in-process is free to compose monitoring wrappers
> around the task's `Toolbox` (see cube-harness `MonitoredToolbox`); those
> wrappers are not part of this contract. Remote-agent monitoring, when added,
> will attach on the harness side of the connection and is out of scope for
> the server protocol itself.

No code change.

---

## ADDED — none

## REMOVED — none
