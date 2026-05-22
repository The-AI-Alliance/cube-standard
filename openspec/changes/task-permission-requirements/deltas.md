# Deltas — Task permission requirements + infra capability handshake

Applied to `openspec/specs/resource/spec.md` and `openspec/specs/benchmark/spec.md`.

## ADDED — `resource/spec.md`: `container:root` capability token

One token added to the `capabilities()` / `requirements()` vocabulary: `container:root`
(container processes run as uid 0). Single token by design — `apt`, `/etc`+`/var` writes,
ports <1024, and `systemctl` all correlate with "is the container root?". Vocabulary stays
open (`set[str]`); finer tokens can be added later.

## MODIFIED — `resource/spec.md`: `ResourceConfig.requires`

```python
class ResourceConfig(TypedBaseModel):
    ...
    requires: set[str] = Field(default_factory=set)   # NEW — explicit extra tokens
    def requirements(self) -> set[str]:
        return set(self.requires)                      # base; subclasses super()-union
```

Subclasses fold `requires` via `super().requirements()`:
`VMResourceConfig` → `… | ({"kvm"} if requires_kvm else set())`;
`DockerServiceConfig` → `… | {"docker"}`;
`ContainerConfig` → `… | {"docker"}` (+ `"gpu:nvidia"` if `gpu`).

`requires` is the source of truth a cube stamps (e.g. tbench2 codegen sets
`{"container:root"}` on every task). Backward-compatible: default empty adds nothing.

## ADDED — `resource/spec.md`: `InfraConfig.on_incompatible` + `IncompatibleInfraError`

```python
class InfraConfig(...):
    on_incompatible: Literal["raise", "skip", "force"] = "raise"
```

`can_serve(resource)` (existing — `requirements() <= capabilities()`) is the per-resource
unit; **no new method**. `IncompatibleInfraError(RuntimeError)` is raised by the gate.

Capability declarations: `Local` (when docker present), `AWS`, `Azure`, `Daytona`, `Modal`
publish `container:root`; `Toolkit` does not (pins a non-root uid).

## MODIFIED — `benchmark/spec.md`: `BenchmarkConfig.make()` gate

`make(infra)` runs the capability gate **before provisioning** when `infra` is set and
`on_incompatible != "force"`: loop `can_serve` over each task's `container_config` and the
benchmark's `resources`, then apply the policy:

- `"raise"` — any incompatible resource → `IncompatibleInfraError` (pre-episode, no spend).
- `"skip"` — narrow the task view to the compatible subset (`subset_from_list`); a shared
  benchmark-scoped incompatible resource still raises.
- `"force"` — skip the gate entirely.

## CONSUMED (cube-harness follow-up, not in this change)

- **skip mode** episodes map to the existing terminal, non-retriable `INVALID_CONFIG`.
- tbench2 (and other root-needing cubes) codegen stamps `requires={"container:root"}`.

## Migration

Backward-compatible by default: `requires` defaults to empty (so `requirements()` is
unchanged for existing resources) and `on_incompatible` is a no-op until a task declares a
token the infra lacks. Behaviour changes only when a cube stamps `requires` AND a paired
infra lacks the token.
