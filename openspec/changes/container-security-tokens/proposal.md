# Container security capability tokens

**Status:** Proposed + implemented in this PR
**Date:** 2026-06-13
**Scope:** `cube.resource` (token vocabulary + apply contract), `cube.task_infra`, `cube.task`,
`cube.infra_local`.
**Targets:** `dev`
**Related:** extends `task-permission-requirements` (the `container:root` handshake).

## Implementation (this PR)

- `task_infra.build_docker_run_script(..., requires)` maps gate+apply tokens to `docker run`
  flags (`--privileged`, `--cgroupns=host`); `launch_task_container(..., requires)` threads
  them and re-declares them on the task `DockerServiceConfig`.
- `task.model_post_init` passes `container_config.requirements()`, so every standard-path
  cube gets it automatically just by declaring `requires` on its `ContainerConfig`.
- `LocalInfraConfig.capabilities()` advertises both tokens when Docker is present (single
  tenant); shared infra (Toolkit) advertises neither — unchanged. Cloud infras (AWS/Daytona/
  Modal/Azure) are left unadvertised pending per-backend verification, so `can_serve`
  safely rejects requiring tasks there.
- Spec (`resource/spec.md`) + docstrings updated; unit tests in `tests/test_resource_lifecycle.py`.

## Problem

`task-permission-requirements` gave us the capability handshake — a task declares
`ResourceConfig.requires`, an infra advertises `capabilities()`, `can_serve` gates the pair
before spend — and shipped one container token: `container:root`.

Crucially, **`requires` is read only by the gate** (`can_serve`); no `launch()` reads it.
`container:root` works as a *gate-only* token because uid 0 is Docker's default — the token
just excludes infras that pin a non-root uid; nothing has to be *applied*. So a security
flag that is **opt-in per container** cannot be expressed at all today: adding it to
`requires` would pass `can_serve` and then launch *without* the flag — a silent capability
downgrade, the exact failure `task-permission-requirements` exists to prevent.

That blocks the Security/CTF category outright. Concretely, Cybench (the category's primary
eval) needs `--privileged` **and** `--cgroupns=host` to run its challenge harness; NYU CTF,
BountyBench, and CyberGym likewise need `--privileged`. On the structured
`ContainerConfig` path these flags have no expression: `build_docker_run_script`
(`task_infra.py`) is a fixed template with no flag input. (A cube *can* smuggle flags into a
`DockerServiceConfig.launch_script` raw-bash string — but only on `Local`; Daytona/Modal/AWS
build from `docker_images[0]` and ignore it, so it is not a portable answer.)

**Demand basis:** a CUBE-fit survey of 14 candidate benchmarks (forward-looking, not a
recurring closed request). No prior issues/PRs for these tokens.

## Design

Two additions, both riding the existing handshake. **No new fields, no new methods, no
breaking change.**

1. **Tokens.** Add `container:privileged` and `container:cgroupns-host` to the standard
   vocabulary, declared via the existing `ResourceConfig.requires`.
2. **Apply contract (the one genuinely new behavior).** A token can now play one of two
   roles, made explicit in the spec: *gate-only* (`container:root` — reflects infra default;
   `can_serve` only excludes infras that lack it) or *gate+apply* (these two — the infra MUST
   translate a present token into the launch flag). The apply point is
   `build_docker_run_script` — the single-container template reached via
   `launch_task_container` (which builds a `DockerServiceConfig(scope="task")`). It gains the
   resource's `requirements()` as an argument (an additive signature change) and emits
   `--privileged` / `--cgroupns=host` when the token is present; each cloud infra's `launch()`
   maps the token to its SDK's equivalent. Additive only — **no new field**.
3. **Advertise.** An infra publishes a token only where backend *and tenancy* allow:
   single-tenant `Local` (Docker present) and dedicated-VM infras may; shared multi-tenant
   `Toolkit` must not — same trust model as `container:root`. `can_serve` then gates safely.

## Why this shape

- **Rides the accepted precedent.** Declaration (`requires`), gate (`can_serve`), policy
  (`on_incompatible`) are reused unchanged; we extend a `set[str]` vocabulary designed to
  stay open. The apply contract is the minimum needed to make opt-in flags safe.
- **Safety is automatic.** An infra that shouldn't grant the capability doesn't advertise it;
  tasks needing it fail the gate pre-spend rather than silently launching unprivileged or
  dangerously over-privileged.
- **General.** Unblocks the whole Security category (5 benchmarks), anchored by Cybench.

## On the two-token split (a deliberate call)

`container:privileged` is the hard-block — keep. `container:cgroupns-host` is kept **because
Cybench concretely needs it on top of `--privileged`**, and `--cgroupns=host` is orthogonal
to `--privileged` (the latter does not set the cgroup namespace mode), so privileged-only
would leave the category's P0 anchor blocked — the RFC's headline goal. This meets the lean
bar ("add a token when a surveyed cube proves it needs it"), not speculation. **Open
question for review:** if implementation confirms `--privileged` is sufficient for Cybench in
practice, collapse to the single `container:privileged` token.

## Out of scope (deliberately not bundled)

- **`network:host` / multiple resources per task.** TheAgentCompany's "reach co-located
  services" is better modeled as a *separate tracked resource* (agent container + service
  container), not `--network=host`. That is a larger change (singular
  `TaskMetadata.container_config` → composite + cleanup-tagging) and a separate RFC;
  `--network=host` now would be a workaround to retract later.
- **A finer `cap_add` matrix** (`cap:net_admin`, …). Add narrower tokens only when a concrete
  cube needs less than full privilege.
