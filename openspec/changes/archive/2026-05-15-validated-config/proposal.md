# Add `ValidatedConfig` — a `TypedBaseModel` that validates attribute assignment

**Status:** Accepted
**Date:** 2026-05-15
**Scope:** `cube.core`, `cube.tool`, `cube.benchmark`, `cube.resource`
**Targets:** `refactor/validated-config` branch in cube-standard; unblocks the declarative-recipe refactor in cube-harness (`refactor/declarative-recipes`).

---

## Problem

Config objects are routinely tweaked by attribute assignment in recipes:

```python
agent = GENNY_CONFIGS["swe"]
agent.budget.cost_limit = 2.0
```

Plain Pydantic models only validate at construction. `validate_assignment` is
off by default, so `agent.max_actions = "oops"` silently succeeds and the bad
value only surfaces much later — typically deep inside a Ray worker, far from
the line that caused it. As recipes become thin declarative config files that
mutate canonical configs by assignment, this failure mode moves onto the hot
path for everyday use.

## Decision

Add `ValidatedConfig(TypedBaseModel)` to `cube.core` with
`model_config = ConfigDict(validate_assignment=True)`. The user-facing,
mutable config ABCs subclass it instead of `TypedBaseModel`:

- `ToolConfig`, `AsyncToolConfig` (`cube.tool`)
- `InfraConfig` (`cube.resource`)
- `BenchmarkConfig` (`cube.benchmark`)

`TypedBaseModel` is unchanged — non-config serializable types (`Action`,
`ResourceConfig`, `BenchmarkMetadata`, …) keep construction-only validation
and pay no per-assignment cost.

## Consequences

- Bad attribute assignment on a config raises `ValidationError` at the
  assignment site instead of failing later in a worker.
- For nested writes (`cfg.sub.field = ...`) to be validated, every model in
  the tree must subclass `ValidatedConfig`. cube-harness adopts it for
  `AgentConfig` / `LLMConfig` / budget in the companion PR.
- `model_copy(update=...)` does **not** trigger assignment validation (Pydantic
  behaviour). `BenchmarkConfig.subset_from_list` / `subset_from_glob` use
  `model_copy`, so the subsetting path is unaffected.
- `TypedBaseModel`'s polymorphic `_type` round-trip is preserved: on
  assignment the wrap validator receives a model instance (not a `_type`
  dict) and falls through to the normal handler. Covered by a regression test.

## Alternatives considered

- **Flip `validate_assignment` on `TypedBaseModel` globally.** Rejected:
  imposes per-assignment validation on every serializable type, including
  hot-path data objects that are never mutated by users.
- **Keep it cube-harness-only.** Rejected: `BenchmarkConfig` / `InfraConfig` /
  `ToolConfig` are cube-standard ABCs; nested-tree validation requires the
  base to live here.
