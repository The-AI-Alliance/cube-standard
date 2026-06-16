# Deltas — Container security capability tokens

Applied to `openspec/specs/resource/spec.md`.

## ADDED — `resource/spec.md`: two capability tokens + the apply role

Extend the `capabilities()` / `requirements()` vocabulary with:

- `container:privileged` — container runs with `--privileged`.
- `container:cgroupns-host` — container shares the host cgroup namespace (`--cgroupns=host`).

Declared via the existing `ResourceConfig.requires`; no new fields. The token-list note
gains one sentence distinguishing the two roles a token can play:

> Tokens are **gate-only** when the capability is the infra's default (`container:root` —
> `can_serve` merely excludes infras that lack it), or **gate+apply** when it is opt-in
> per container (`container:privileged`, `container:cgroupns-host` — an infra advertising
> the token MUST translate it to the launch flag).

Example: `ContainerConfig(image="cybench/...", requires={"container:privileged", "container:cgroupns-host"})`.

## MODIFIED — `resource/spec.md`: apply invariant

One invariant added: an infra that advertises a gate+apply token MUST apply the
corresponding launch flag; one that does not advertise it MUST reject a requiring resource
via the existing `can_serve` gate (no silent downgrade). Application is keyed off
`resource.requirements()`, threaded as a new argument into `build_docker_run_script` (the
single-container template reached via `launch_task_container`; additive signature change) and
mapped to the SDK equivalent in each infra's `launch()` — **no new field or method**;
`can_serve`, `on_incompatible`, and `BenchmarkConfig.make()` are unchanged.

## Capability declarations (infra drivers)

- `Local` (Docker present): MAY advertise both (single-tenant).
- `AWS` (dedicated VM per run): MAY advertise both.
- `Daytona` / `Modal` / `Azure`: per-backend, pending verification that the managed backend
  permits `--privileged` (they build from `docker_images[0]`, not raw run flags).
- `Toolkit`: MUST NOT advertise either (shared multi-tenant) — same rationale as
  `container:root`.
