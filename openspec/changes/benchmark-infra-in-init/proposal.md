# Thread `infra` through `Benchmark.__init__`

**Status:** Proposed
**Date:** 2026-04-30
**Scope:** `cube.benchmark`
**Targets:** `main`
**Related:** PR #118 (`feat/benchmark-config` — defined the split). Unblocks deletion of
duplicated boilerplate in cube-harness PR #322 / #323 / #324 and existing on-main cubes
(`osworld-cube`, `webarena-verified`).

---

## Problem

Every cube that needs `infra` in `runtime_context` for per-task container or VM
launches today carries the same triple-override of cube-standard's `Benchmark`
contract:

1. `__init__(self, config, infra: InfraConfig | None = None)` — stash `self._infra`.
2. `_setup()` — publish `self._runtime_context["infra"] = self._infra`.
3. `BenchmarkConfig.make(infra)` — re-implement the base's resource-provisioning
   loop verbatim, just to be able to construct
   `MyBenchmark(config=self, infra=resolved_infra)` instead of the base's
   `benchmark_class(config=self)`.

(1) and (3) are pure plumbing — no cube-specific logic. They exist solely
because the base `Benchmark.__init__` doesn't accept `infra`, so the base
`make()` has no way to thread it through to the runtime.

Five cubes carry this duplication today:

| Cube | Status |
|---|---|
| `osworld-cube` | on cube-harness `main` |
| `webarena-verified` | on cube-harness `main` |
| `swebench-verified-cube` | in cube-harness PR #323 |
| `swebench-live-cube` | in cube-harness PR #323 |
| `terminalbench-cube` | in cube-harness PR #323 |

The (3) override in particular copies the resource-provisioning `for` loop from
`BenchmarkConfig.make()` and is a maintenance hazard: any future change to base
provisioning won't propagate to the five overrides. Five copy-pastes of the
same loop is what motivated this proposal.

## Solution

Promote `infra` into the base contract:

1. `Benchmark.__init__(self, config, infra: InfraConfig | None = None)` —
   accepts and stashes `self._infra: InfraConfig | None`.
2. `BenchmarkConfig.make(infra)` — passes `infra=infra` when constructing
   `benchmark_class`.
3. `CompositeBenchmark.__init__` — accepts `infra=None` to match the new base
   signature; doesn't consume it (each sub-benchmark already received its own
   infra via its own `make(infra)`).

That's the entire change in cube-standard: a few lines plus docstring updates.

After this lands, cubes that need infra in `runtime_context` write only the
cube-specific `_setup()` body:

```python
class FooBenchmark(Benchmark):
    def _setup(self) -> None:
        if self._infra is not None:
            self._infra.cleanup_stale()
            self._runtime_context["infra"] = self._infra

    def close(self) -> None:
        ...
```

No `__init__` override. No `make()` override. The duplicated provisioning loop
disappears.

## Backwards compatibility

**Fully backwards compatible.**

- `Benchmark` subclasses with no `__init__` override: unchanged — they inherit
  the new signature, infra defaults to `None`, behaviour identical to today.
- `Benchmark` subclasses already declaring `__init__(self, config, infra=None)`
  (the five SWE/Desktop cubes above): signature is now compatible with the
  base. The override becomes redundant but not breaking — cubes can drop it as
  a follow-up.
- `Benchmark` subclasses with custom `__init__(self, config)` (e.g. miniwob):
  must add `infra=None` to their signature and forward to `super().__init__()`.
  In-tree, `CompositeBenchmark` is the only such case and is updated in this
  PR.
- Direct `Benchmark` constructions in tests
  (`tests/test_benchmark_composite.py:238: CompositeBenchmark(config=suite)`)
  keep working — `infra` defaults to `None`.

**One subtle behaviour to call out for cube authors.** Several existing cubes'
`make(infra)` overrides default `infra or LocalInfraConfig()` if no infra is
passed. The base `make()` does **not** apply this default — it forwards
whatever the caller passed (including `None`). A cube that drops its `make()`
override and relied on this defaulting needs either:

- callers to pass `LocalInfraConfig()` explicitly, **or**
- a small `make()` that resolves the default and calls `super().make(infra=resolved)`.

This is documented in the migration notes below and in the spec.

## Non-goals

- Auto-publishing `runtime_context["infra"] = self._infra` from base `setup()`.
  Considered and rejected: `webarena-verified` publishes a service handle (not
  the bare `InfraConfig`) into `runtime_context`, so a base auto-publish would
  conflict with cube-specific shapes. Each cube's `_setup()` decides what to
  publish.
- Removing `BenchmarkConfig.make()` overrides in cube-harness in this PR. That
  is a follow-up cube-harness PR; it depends on this change landing first.

## Migration

**This PR (cube-standard):**

- `cube.benchmark`: `Benchmark.__init__` accepts `infra: InfraConfig | None = None`,
  stashes `self._infra`. `BenchmarkConfig.make` passes `infra=infra` when
  constructing `benchmark_class`. `CompositeBenchmark.__init__` accepts and
  forwards `infra=None`.
- `openspec/specs/benchmark/spec.md`: updated `Benchmark.__init__` signature,
  noted `self._infra` and the forwarding rule on `make`.
- No template / examples changes required — none override `__init__`.

**Follow-up PRs in cube-harness (out of scope here):**

- `osworld-cube`, `webarena-verified`: drop the `__init__(config, infra=None)`
  override, drop the `make(infra)` override (or shrink to just the
  default-to-`LocalInfraConfig()` shim if the cube wants that defaulting).
  Keep the cube-specific `_setup()` body.
- The three SWE cubes in PR #323: same cleanup, can be folded into PR #323 or
  a follow-up commit on the same branch.
- `miniwob` (cube-harness): one-line update — add `infra=None` to its
  `__init__` and forward to `super()` (or remove the `__init__` if its server
  state can move to `_setup()`).

Approximate cube-harness LOC saved by the follow-up: ~140 across five cubes
(deletion of `__init__` overrides + `make()` overrides including the duplicated
provisioning loop).

## Out of scope

- Any change to `BenchmarkConfig`'s field surface.
- Any change to `_setup()` / `close()` / `spawn()` semantics.
- Any change to composite routing.
