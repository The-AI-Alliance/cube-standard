# Task permission requirements + infra capability handshake

**Status:** Proposed
**Date:** 2026-05-21
**Scope:** `cube.container.ContainerConfig`, `cube.resource.InfraConfig`,
`cube.benchmark.Benchmark.setup`
**Targets:** `dev`
**Related:** cube-harness session `2026-05-20_tbench2-infra-model-matrix`,
finding F7 (toolkit's non-root user vs tbench2's apt-installing tasks).

---

## Problem

Some cube tasks need infra permissions not every `InfraConfig` provides —
e.g. tbench2 tasks `apt-get install` packages and write to `/etc`, `/var`,
which needs a **root** container; EAI Toolkit forces **uid 13011** by
cluster policy. Today this mismatch surfaces as **silent low scores** (the
agent tries `apt-get`, gets `Permission denied`, fails the task at reward 0)
and can even burn the auto-retry budget re-running episodes that fail
identically every time.

The framework already has the matching mechanism: `InfraConfig.capabilities()
-> set[str]`, `ResourceConfig.requirements() -> set[str]`, and
`InfraConfig.can_serve(resource)` doing the set-inclusion test (tokens in use:
`kvm`, `docker`, `gpu:nvidia`, `network:egress`). This proposal extends it
with a permission token, a per-task compatibility check, and an
infra-owned policy for handling mismatches.

## Design

**1. One token.** Add `container:root` to the `capabilities()` vocabulary
("container processes run as uid 0"). A single token is deliberate: `apt`,
writes to `/etc`/`/var`, ports <1024, and `systemctl` are all root-gated and
correlate with "is the container root?" on both sides. The vocabulary stays
open (`set[str]`) so a finer token can be added later if a partial-root
infra ever needs it.

**2. Per-task `ContainerConfig.requires` (source of truth).** Each task
already owns a `ContainerConfig` (tbench2: 89 tasks, 89 distinct images):

```python
class ContainerConfig(TypedBaseModel):
    ...
    requires: set[str] = Field(default_factory=set)
    def requirements(self) -> set[str]:
        out = set(self.requires)
        if self.gpu:
            out.add("gpu:nvidia")
        return out
```

The blanket "all of tbench2 needs root" case is handled by the
metadata-generation script (`create_task_metadata.py`) stamping
`requires={"container:root"}` on every task — no per-task triage.
Heterogeneous benchmarks set `requires` per task.

**3. Per-task bool from the infra.** No benchmark-level aggregate (awkward to
build); the infra answers one task at a time, reusing `capabilities()`:

```python
class InfraConfig(TypedBaseModel, ABC):
    ...
    def can_serve_task(self, container_config: ContainerConfig) -> bool:
        return container_config.requirements().issubset(self.capabilities())
```

The harness loops this over tasks; "is the whole benchmark incompatible?"
falls out of the loop — nothing to keep in sync.

**4. Policy flag on `InfraConfig`.** The infra owns how to handle a mismatch:

```python
OnIncompatible = Literal["raise", "skip", "force"]

class InfraConfig(TypedBaseModel, ABC):
    on_incompatible: OnIncompatible = "raise"
```

`Benchmark.setup(infra)` reads it (no new setup parameter):

- **`"raise"`** (default) — if **any** task is incompatible, raise
  `IncompatibleInfraError` at setup, **before any episode is created**. No
  spend, no retries. tbench2 on toolkit refuses outright.
- **`"skip"`** — run the compatible subset; each incompatible task's episode
  is recorded `INVALID_CONFIG` (§5) and excluded from the accuracy
  denominator (`n_skipped`).
- **`"force"`** — launch everything anyway. The escape hatch for probing
  whether an infra/image change has obviated a stale requirement.

You pick the policy where you pick the infra:
`ToolkitInfraConfig(on_incompatible="skip")`.

**5. Status reuse — no new status.** cube-harness already has
`INVALID_CONFIG`: terminal and **non-retriable** (`RETRIABLE_STATUSES =
{FAILED, CANCELLED, STALE}` excludes it) — documented as "the identical
request will fail identically on retry". An incompatible task is exactly
that. `"raise"` aborts at setup (no episode). `"skip"` writes
`INVALID_CONFIG`, which the auto-retry loop leaves alone by construction. A
dedicated `INCOMPATIBLE` status is an optional future refinement for
reporting, not required for correct retry semantics.

## Migration

Backward-compatible by default: `ContainerConfig.requires` defaults to
`set()` (so `can_serve_task` is always True), and `on_incompatible` defaults
to `"raise"` but is a no-op until a task declares `requires`. Behaviour
changes only when a cube stamps `requires` AND a paired infra-side PR
publishes the token. Rollout (follow-up PRs, not this RFC):

1. This RFC: token + `ContainerConfig.requires` + `InfraConfig.can_serve_task`
   + `on_incompatible` + the `setup` loop + `INVALID_CONFIG` mapping.
2. cube-resources: root-capable infras (`Local`, `Daytona`, `AWS`, `Azure`)
   publish `container:root`; `Toolkit` does not.
3. cube-harness: tbench2 codegen stamps `requires={"container:root"}`.
4. cube-harness: read `infra.on_incompatible` in setup/launch; map skip-mode
   episodes to `INVALID_CONFIG`.

## Alternatives considered

- **Benchmark-level aggregate requirement** (a declared `requirements()`
  field, or a derived `aggregate_requirements()`). Rejected: a declared
  field duplicates the per-task `ContainerConfig` and drifts; an aggregate is
  awkward to build (materialise/union every task resource). The per-task
  `can_serve_task` bool + harness loop needs neither.
- **A new `INCOMPATIBLE` episode status as the v1 mechanism.** Rejected for
  v1: `INVALID_CONFIG` already has the exact non-retriable-terminal
  semantics. Clean later refinement, not a prerequisite.
- **Encode requirements in the Docker image** (`LABEL cube.requires=root`).
  Rejected: foreign system; couples requirement cadence to image rebuilds;
  violates "Python is the configuration".

## Open questions

1. Infra field only, or also a per-run override (`Experiment` field that
   wins over the infra default)? Lean: infra field only for v1.
2. `can_serve_task` takes a `ContainerConfig` (cube.container) but lives on
   `InfraConfig` (cube.resource) — confirm import direction is clean, or make
   it a free function / `ContainerConfig.is_served_by(infra)`. Implementation
   detail; doesn't change the design.
3. Confirm `INVALID_CONFIG` reuse is acceptable vs. adding `INCOMPATIBLE` now.
