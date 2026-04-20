# Deltas — Core Extensions

**Targets:** `openspec/specs/core/spec.md`, `openspec/specs/task/spec.md`, `openspec/specs/tool/spec.md`, `openspec/specs/server/spec.md`

Applied when each phase lands. Four independent extensions; each can merge separately.

## ADDED — Observation streaming (phase 1)
**Spec:** task, tool, server

Optional streaming observation protocol: `execute_action()` may return an
`AsyncGenerator[Content]` in addition to `Observation | StepError`. RPC layer surfaces
via WebSocket / SSE.

- `AbstractTool.execute_action` return type widens to `Observation | StepError | AsyncGenerator[Content, None]`
- `Task.step` concatenates streamed content into the final observation
- Server adds `tools/call_stream` WebSocket method

Backwards-compatible: existing sync tools unaffected.

## ADDED — Async core (phase 2)
**Spec:** task, tool

`AsyncTask` parallel to `Task` with `async def reset/step/evaluate/close`.
`AsyncToolConfig.make` already exists (see tool spec).

- New `AsyncTask(TypedBaseModel, ABC)` with coroutine abstract methods
- `AsyncBenchmark` optional
- Server adds async dispatch path

Backwards-compatible: sync path retained.

## ADDED — Multi-agent schema (phase 3)
**Spec:** core, task

- `Action.agent_id: str | None` — identifies which agent produced the action
- `Observation.for_agent: str | None` — filter by recipient
- `Task.step(actions: dict[str, Action | list[Action]])` multi-agent overload

Backwards-compatible: `agent_id=None` and `for_agent=None` preserve current single-agent semantics.

## ADDED — Multi-dimensional reward (phase 4)
**Spec:** core, task

- `EnvironmentOutput.reward` widens from `float` to `float | dict[str, float]`
- `Task.evaluate` return type widens accordingly
- Scalar reward still accepted everywhere; dict reward opts into multi-dim

Backwards-compatible.

---

See `proposal.md` for rationale and detailed design.
