# Resource / Infra convergence — lessons from the 3×4 migration

**Status:** Draft
**Date:** April 2026
**Depends on:** `deprecate-container-backend` (PR #115), `feat/daytona-infra-config` (PR #116)

---

## Background

While migrating three cubes (terminal-bench, SWE-bench Verified, SWE-bench Live)
across four InfraConfig backends (Local, Daytona, Toolkit/EAI, Modal), a handful
of schema-level friction points surfaced that are worth formalising in
`openspec/specs/` so the next wave of cubes + infras doesn't rediscover them.

Each item below is scoped as a narrow, backwards-compatible spec addition.

---

## 1. Standard `cube.launch` label convention

**Status:** shipped in `LocalInfraConfig`, not yet specced.

**Problem.** `LocalInfraConfig` originally identified "its" containers via a
`docker ps -q` before/after snapshot. Under concurrent launches (pytest-xdist,
Ray workers), worker A saw worker B's containers and treated them as its own,
blowing up with `got N containers` RuntimeErrors.

**Fix applied.** `LocalInfraConfig` now exports `CUBE_LAUNCH_ID=<run_id>` into
the launch-script environment, and `cube.task_infra.build_docker_run_script`
emits `docker run --label "cube.launch=$CUBE_LAUNCH_ID"`. Launch identifies
its own containers via `--filter label=cube.launch=<run_id>`.

**Proposed spec change.** Add an invariant to `resource/spec.md`:

> Infras that shell-out to bash to run a `DockerServiceConfig.launch_script`
> MUST export `CUBE_LAUNCH_ID=<run_id>` into the script's environment. Scripts
> that use `cube.task_infra.build_docker_run_script` get label-based tracking
> automatically; hand-rolled scripts SHOULD opt in via
> `--label cube.launch=$CUBE_LAUNCH_ID` so parallel launches don't collide.

---

## 2. `ResourceHandle.container` promoted to base class

**Problem.** Three independent handle types (`LocalDockerServiceHandle`,
`DaytonaResourceHandle`, `ToolkitResourceHandle`, `ModalResourceHandle`) each
implement a `.container` property ad-hoc. Cube code depends on it via duck
typing — `self._container = handle.container`. A typo in any handle goes
undetected until test time.

**Proposed spec change.** In `resource/spec.md`, `ResourceHandle` gets:

```python
@property
def container(self) -> "Container | None":
    """The single ``cube.container.Container`` this handle exposes, or None for
    multi-container or non-container resources (VMs, DockerServiceConfig stacks,
    L2 shared services). Single-container L3 handles MUST override to return a
    live wrapper; the tool layer reads it in ``Task.model_post_init``.
    """
    return None
```

Cube code becomes typed rather than duck-typed.

---

## 3. Infra capability tokens for network policy

**Problem.** `terminalbench-cube`'s evaluator runs a `test.sh` that
`curl`s `https://astral.sh/...` to bootstrap `uv` + `pytest`. On backends with
restricted outbound (Daytona default, EAI cluster network), this fails silently
(reward=0). The cube has no way to assert "I need unrestricted outbound" beyond
implicitly trying and failing.

**Proposed spec change.** Extend `InfraConfig.capabilities()` standard tokens
(in `resource/spec.md`) with a network tier:

| Token | Meaning |
|---|---|
| `network:public-internet` | Outbound HTTPS to arbitrary hosts works. |
| `network:egress-allowlist` | Outbound works but restricted to the infra's allow-list. |
| `network:none` | Air-gapped; no outbound permitted. |

Cubes that require it declare `requirements()` → `{"network:public-internet"}`.
`Benchmark.setup()` fails fast with a clear message if the infra doesn't
advertise it, instead of producing a mysterious `reward=0`.

---

## 4. (Parked) Persistent-session Exec for CLI-slow backends

**Observed.** `ToolkitInfraConfig.exec()` per-command overhead is ~1s
baseline — acceptable — but the `eai` CLI occasionally hangs indefinitely
on specific commands (large payloads, likely cluster-side congestion),
leaving defunct `(eai)` processes behind even after `subprocess.run` timeout
fires.

**Mitigated.** The `_run_eai` helper now puts the CLI in its own process
group and SIGKILLs the group on timeout. This removes the defunct-process
symptom but doesn't fix the underlying hang.

**Deferred.** A proper fix is a persistent-session exec mode — `eai job port-forward`
+ ssh into the running job, keep one pooled shell, dispatch per-command over it.
Order-of-magnitude faster and sidesteps the hang. Scoped as a future change;
not proposed in this document.

---

## What's NOT proposed

- Removing the `before/after docker ps` fallback in `LocalInfraConfig` — kept
  for hand-rolled launch scripts that don't use `build_docker_run_script`.
- Promoting the `resource.requirements()` / `infra.capabilities()` check from
  "recommended" to "mandatory" in the spec — some cubes genuinely don't know
  their requirements upfront (e.g. image content determines them).

---

## Sequencing

1. Land `deprecate-container-backend` (PR #115) — no code impact of this proposal.
2. Land `feat/daytona-infra-config` (PR #116) — ships items 1 + 4 (mitigation).
3. Separate follow-up PR applies items 1 (spec text), 2 (base class), 3 (new tokens)
   once this proposal reaches consensus.
