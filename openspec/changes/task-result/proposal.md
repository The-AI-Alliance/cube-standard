# Structured evaluation results — `TaskResult` replaces `tuple[float, dict]`

**Status:** Proposed
**Date:** 2026-05-27
**Scope:** `cube.core`, `cube.task`
**Targets:** `main`

---

## Problem

`Task.evaluate()` returns `tuple[float, dict]`. The scalar reward is
typed, but the evaluation details — what was checked, what passed,
what failed, what was expected vs actual — are buried in an untyped
dict that varies per cube. The harness, telemetry, and analysis tools
can't inspect or display evaluation details without per-cube knowledge.

This matters for three reasons:

1. **Debugging failures.** When an agent scores 0.0, the researcher
   needs to know *which* checks failed and *why* — not just "reward=0".
   Today this requires reading cube-specific code to understand what
   the dict keys mean.

2. **Cross-benchmark analysis.** Aggregating evaluation patterns
   (e.g. "submission format errors account for 30% of failures") is
   impossible when each cube reports results in its own schema.

3. **Telemetry and UX.** The harness attaches `env_output.info` to
   OTel spans as an opaque JSON blob. A structured result lets Bench
   render per-check pass/fail tables without cube-specific parsing.

## Solution

Two new types in `cube.core` and a signature change on `Task.evaluate()`:

```python
class EvaluationCheck(TypedBaseModel):
    name: str
    passed: bool
    expected: str | None
    actual: str | None
    comment: str | None

class TaskResult(TypedBaseModel):
    reward: float
    checks: list[EvaluationCheck]
    info: dict[str, Any]
```

`evaluate()` changes from `-> tuple[float, dict]` to `-> TaskResult`.

`Task.step()` adapts: `reward, info = self.evaluate(obs)` becomes
`result = self.evaluate(obs)` reading `result.reward` / `result.info`.

No defaults on any `TaskResult` field — every cube must explicitly
pass all three.

## Backwards compatibility

**Breaking for all Task subclasses.** Every `evaluate()` override must
change its return from `return reward, info` to
`return TaskResult(reward=reward, checks=[], info=info)`.

This is mechanical: ~12 cubes + ~10 test fixtures, one-line change at
each return site.

**Breaking for callers that unpack the tuple.** `Task.step()` and
`server.py` unpack `reward, info = self.evaluate(obs)`. These change
to `result = self.evaluate(obs)`.

## Non-goals

- Defining how checks are displayed in the harness UI — that's the
  harness's concern.
- Making `EvaluationCheck` fields typed beyond `str` (e.g. numeric
  expected/actual) — `str` is universal for display; cubes that need
  typed comparisons do them internally and report the result as
  pass/fail.

## Migration

**This PR (cube-standard):**

- `cube.core`: add `EvaluationCheck`, `TaskResult`.
- `cube.task`: change `evaluate()` return type, adapt `step()`.
- `cube.server`: adapt `evaluate()` call site.
- Examples and template: migrate return values.
- Tests: migrate return values.

**Follow-up PR (cube-harness):**

- All cubes under `cubes/`: migrate `evaluate()` return values.
- Tests: migrate fixtures.
- Episode: read `result.checks` for telemetry/storage.
