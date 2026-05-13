# Auto-include `STOP_ACTION` in `Task.action_set`

**Status:** Proposed
**Date:** 2026-05-13
**Scope:** `cube.task`
**Targets:** cube-standard PR #139

## Problem

Two coupled gaps in the current contract force every cube author to write
the same workaround:

1. `STOP_ACTION` is defined as
   `ActionSchema(name="final_step", description="Stop the task execution.")`
   with no `parameters`. LiteLLM passes the empty `parameters` dict through
   verbatim. Anthropic's API rejects an `input_schema` that is not
   `{"type": "object", ...}`, so any cube that surfaces `STOP_ACTION` to a
   Claude-backed agent fails at the LLM call.

2. The spec says (`openspec/specs/task/spec.md` Gotchas):
   > STOP_ACTION is not automatically in the tool's action set — the harness
   > / agent framework is responsible for including it in the action list
   > shown to the LLM.

   But `Task.step()` already treats `STOP_ACTION` specially when
   `accept_agent_stop=True` (it terminates the task without dispatching to
   the tool). So the action exists in the protocol — the agent just has to
   guess that it is callable, since `action_set` doesn't declare it.

Cube authors work around (1) and (2) together by overriding
`filter_actions()` to append a hand-rolled `ActionSchema` with the
Anthropic-compatible empty-object schema. `swebench-verified-cube` and
`swebench-live-cube` carry identical overrides; every new cube that wants
agent stop hits the same wall.

## Solution

Move both fixes into the base class:

1. **`STOP_ACTION` carries a valid empty-object schema** —
   `parameters={"type": "object", "properties": {}}`. Same payload every
   cube was hand-rolling, now centralised.

2. **`Task.action_set` auto-appends `STOP_ACTION`** when
   `self.accept_agent_stop` is `True` and the action is not already
   present. This closes the loop: the field that controls *whether* the
   agent can stop also controls *whether* the agent sees the stop action.
   The dedup branch keeps existing overrides working through the
   transition.

3. **`filter_actions` docstring** explicitly tells future cube authors not
   to re-add `STOP_ACTION`.

## Backwards compatibility

- Cubes whose `filter_actions` still appends `STOP_ACTION` (the existing
  workaround) keep working — the new code dedupes by name. The override
  can be deleted in a follow-up cleanup PR but is not load-bearing.
- Cubes that set `accept_agent_stop=False` see no change: `action_set`
  does not include `STOP_ACTION` and `step()` does not handle it.
- Default `Task` instances (`accept_agent_stop=True`, no `filter_actions`
  override) now expose `STOP_ACTION` where they previously did not. This
  is the intended behavior change: the agent can in fact stop these
  tasks; `action_set` now reflects that.

**Downstream coordination (cube-harness):** several agents in cube-harness
(`legacy_generic_agent`, `react`, `genny2`) unconditionally append
`STOP_ACTION` to whatever `action_set` they receive from the task. With
this change, tasks that previously did not expose `STOP_ACTION` now do,
so those agents will produce a duplicate `final_step` tool definition in
the LLM tool list. A follow-up cube-harness PR should drop the manual
appends (or dedupe on receipt). Today this duplicate already exists for
the two SWE-bench cubes; this change widens it to every cube with
`accept_agent_stop=True` until the cube-harness agents are updated.

## Migration

**This PR (cube-standard):**

- `cube.task`: extend `STOP_ACTION` with the empty-object schema; auto-
  append in `Task.action_set` when `accept_agent_stop=True` with dedup;
  update the `filter_actions` and `action_set` docstrings.
- Realign `openspec/specs/task/spec.md`: invert the "not automatically in
  the action set" gotcha; document the new `Task.action_set` contract;
  note the schema on the `STOP_ACTION` constant.
- `tests/test_server.py::test_tools_list` updated — the task server now
  surfaces `final_step` alongside the tool's own actions.
- 5 new tests in `tests/test_task.py` (default includes, disabled
  excludes, dedup, schema, regression on `test_task_action_set_comes_from_tool`).

**Follow-up PRs (out of scope here):**

- cube-harness: remove the manual `STOP_ACTION` append in
  `legacy_generic_agent.py`, `react.py`, `genny2.py` so the LLM tool list
  doesn't carry a duplicate `final_step` entry. Alternatively dedup on
  receipt.
- swebench-verified-cube, swebench-live-cube: delete the
  `filter_actions` override entirely. The base class now does the right
  thing.

See [deltas.md](deltas.md) for the spec changes.
