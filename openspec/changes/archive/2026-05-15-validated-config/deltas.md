# Deltas — `ValidatedConfig`

**Spec:** core
**Target:** `openspec/specs/core/spec.md`

Applied when the change lands.

## ADDED — `ValidatedConfig`

`cube.core` gains a new public type:

```python
class ValidatedConfig(TypedBaseModel):
    model_config = ConfigDict(validate_assignment=True)
```

A `TypedBaseModel` that validates attribute assignment in addition to
construction. Subclass this (instead of `TypedBaseModel`) for any config a
user mutates after construction. For validation to reach nested writes
(`cfg.sub.field = ...`), every model in the tree must subclass it.

Invariants:

- Preserves `TypedBaseModel`'s polymorphic `_type` serialization round-trip.
- `model_copy(update=...)` bypasses assignment validation (standard Pydantic
  behaviour); subsetting helpers that use it are unaffected.

## MODIFIED — config ABCs subclass `ValidatedConfig`

The user-facing mutable config base classes now extend `ValidatedConfig`
rather than `TypedBaseModel`:

- `ToolConfig`, `AsyncToolConfig` — spec: tool
- `InfraConfig` — spec: resource
- `BenchmarkConfig` — spec: benchmark

No field or method signatures change; only the validation behaviour of
post-construction attribute assignment. Non-config `TypedBaseModel` types are
unchanged.
