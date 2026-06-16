# Task permission requirements + infra capability handshake

**Status:** Implemented
**Date:** 2026-05-22
**Scope:** `cube.resource` (`ResourceConfig`, `InfraConfig`), `cube.benchmark` (`make`),
the `cube-resources` infra packages.
**Targets:** `dev`
**Related:** cube-harness session `2026-05-20_tbench2-infra-model-matrix`, finding F7
(EAI Toolkit's non-root uid vs tbench2's apt-installing tasks). Builds on the
`ContainerConfig` → `ResourceConfig` unification (#201).

## Problem

Some tasks need infra permissions not every infra provides — tbench2 tasks `apt-get install`
and write to `/etc`, `/var` (need a **root** container); EAI Toolkit pins **uid 13011** by
cluster policy. Today the mismatch surfaces as **silent reward-0 failures** and can burn the
auto-retry budget re-running episodes that fail identically.

## Design (implemented)

The framework already has the handshake — `ResourceConfig.requirements()` vs
`InfraConfig.capabilities()`, matched by `InfraConfig.can_serve(resource)`. Since the
unification a task's `container_config` **is** a `ResourceConfig`, so the existing
**per-resource `can_serve` is the unit** — no new `can_serve_task`, no benchmark aggregate.

1. **Token.** Add `container:root` to the capability vocabulary (uid 0). Root-capable infras
   (Local, AWS, Azure, Daytona, Modal) publish it; Toolkit deliberately does not.
2. **Declaration.** `ResourceConfig.requires: set[str]` (base field), folded into
   `requirements()` by each subclass via `super().requirements()`. A task declares
   `requires={"container:root"}`.
3. **Policy.** `InfraConfig.on_incompatible: Literal["raise","force"] = "raise"`.
4. **Gate.** `BenchmarkConfig.make()` runs `can_serve` over each task's `container_config`
   and the benchmark's declared `resources` **before provisioning**:
   `"raise"` → `IncompatibleInfraError` (pre-episode, no spend) if any resource is
   incompatible; `"force"` → run everything. Metadata-only and launch-free.
   (A silent `"skip"` was deliberately not kept — see below.)

## Why this shape

- **Per-resource `can_serve`** (not a `can_serve_task`, not a benchmark-level aggregate) is
  the overridable unit. It composes with *composite* resources and with a future **meta-infra**
  that overrides `can_serve` to delegate per-resource to children — the gate is forward-
  compatible by construction.
- **`requires` on the base `ResourceConfig`** generalizes the escape hatch to any resource/token.

## Out of scope / follow-ups

- **cube-harness:** tbench2 (and other root-needing cubes) codegen stamps
  `requires={"container:root"}`.
- **Per-task-raise mode:** a future `"per-task-raise"` may let the benchmark proceed while
  each incompatible task raises at episode start (recorded as a terminal per-task error). A
  silent `"skip"` is intentionally absent — silently dropping tasks is the failure mode this
  gate exists to remove.
- **Meta-infra + composite resources:** a separate effort. This gate already supports it.
