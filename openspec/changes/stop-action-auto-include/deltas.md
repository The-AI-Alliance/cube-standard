# Deltas — auto-include `STOP_ACTION` in `Task.action_set`

**Targets:** `openspec/specs/task/spec.md`

## MODIFIED — `STOP_ACTION` carries a valid empty-object schema

**Spec:** task

The `STOP_ACTION` module constant gains a non-empty `parameters` schema:

```python
STOP_ACTION = ActionSchema(
    name="final_step",
    description="Stop the task execution.",
    parameters={"type": "object", "properties": {}},
)
```

The schema is the minimal payload that satisfies Anthropic's
`input_schema` requirement (`{"type": "object", ...}`). LiteLLM
passes `parameters` through verbatim — an empty dict (the previous value)
makes Anthropic-backed agents fail at the LLM call. Every cube currently
hand-rolls this same schema in a `filter_actions` override; the constant
now ships with it.

## ADDED — `Task.action_set` auto-appends `STOP_ACTION`

**Spec:** task

```python
@property
def action_set(self) -> list[ActionSchema]:
    actions = self.filter_actions(self.tool.action_set)
    if self.accept_agent_stop and not any(a.name == STOP_ACTION.name for a in actions):
        actions = [*actions, STOP_ACTION]
    return actions
```

When `self.accept_agent_stop` is `True` (the default), the property
appends `STOP_ACTION` to whatever `filter_actions()` returns, unless an
action with the same name is already present. The dedup branch keeps
existing `filter_actions` overrides that append `STOP_ACTION` working
during the transition.

Couples the `accept_agent_stop` flag — which already gates the
termination branch in `Task.step()` — to action-set visibility:
*if the agent is allowed to stop, the agent sees the stop action.*

## MODIFIED — `filter_actions` no longer responsible for `STOP_ACTION`

**Spec:** task

```python
def filter_actions(self, actions: list[ActionSchema]) -> list[ActionSchema]:
    """(Optional) Whitelist subset of tool actions.
    By default keeps all tool actions. STOP_ACTION is appended automatically
    by action_set when accept_agent_stop=True — do not add it here.
    """
    return actions
```

## REMOVED — "STOP_ACTION is not automatically in the tool's action set" gotcha

**Spec:** task

The Gotchas bullet:

> STOP_ACTION is not automatically in the tool's action set — the harness
> / agent framework is responsible for including it in the action list
> shown to the LLM.

is removed. Replaced by the contract described above and a positive
statement in the `STOP_ACTION` section that the base class auto-includes
it when `accept_agent_stop=True`.
