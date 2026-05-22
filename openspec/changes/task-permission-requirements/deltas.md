# Deltas — Benchmark/task permission requirements + infra capability handshake

## ADDED — `openspec/specs/resource/spec.md`: capability token vocabulary

The `InfraConfig.capabilities()` docstring is extended with one
permission-shaped token:

| Token | Meaning |
|---|---|
| `container:root` | Container processes run as uid 0 |

A single token is intentional: `apt-get`, writes to `/etc` / `/var`,
binding ports <1024, and `systemctl` are all root-gated and correlate
near-perfectly with "is the container root?" on both the infra and image
sides. The vocabulary stays open (`set[str]`), so a finer token (e.g.
`container:cap-net-bind` for a future rootless-with-capabilities infra)
can be added later without breaking existing declarations.

Adding a token is non-breaking. Infras default to **not** publishing one;
benchmarks/tasks default to **not** requiring one. The pre-existing
vocabulary (`kvm`, `docker`, `vm`, `gpu:nvidia`, `network:egress`) is
unchanged.

## ADDED — `openspec/specs/resource/spec.md`: `InfraConfig.can_serve_task()` (per-task bool) + `on_incompatible`

The compatibility primitive is a **per-task bool** the infra returns; the
harness loops over tasks. No benchmark-level aggregate (awkward to build).

```python
OnIncompatible = Literal["raise", "skip", "force"]

class InfraConfig(TypedBaseModel, ABC):
    ...
    on_incompatible: OnIncompatible = "raise"
    """Policy when a task needs a capability this infra lacks. 'raise':
    abort setup if ANY task is incompatible. 'skip': run the compatible
    subset, mark the rest INVALID_CONFIG. 'force': attempt every task
    anyway (override for probing stale requirements)."""

    def can_serve_task(self, container_config: ContainerConfig) -> bool:
        """True iff this infra provides every capability the task's container
        requires. Cheap, metadata-only — no resource built, no launch.
        Mirrors the existing can_serve(resource) set-inclusion test."""
        return container_config.requirements().issubset(self.capabilities())
```

The **source of truth is the per-task `ContainerConfig.requires`** (below).
"Is the whole benchmark incompatible?" is answered by the harness looping
`can_serve_task` over the tasks — there is no aggregate method to maintain.
The blanket "all of tbench2 needs root" case is realised by the
metadata-generation script stamping `requires={"container:root"}` on every
task's `ContainerConfig`.

## MODIFIED — `openspec/specs/container/spec.md`: `ContainerConfig`

**Before**

```python
class ContainerConfig(TypedBaseModel):
    image: str
    ram_gb: float = 4.0
    cpu_cores: float = 2.0
    gpu: bool = False
    disk_gb: float = 10.0
    ports: list[int] | None = None
```

**After**

```python
class ContainerConfig(TypedBaseModel):
    image: str
    ram_gb: float = 4.0
    cpu_cores: float = 2.0
    gpu: bool = False
    disk_gb: float = 10.0
    ports: list[int] | None = None
    requires: set[str] = Field(default_factory=set)
    """Permission tokens this task's container needs from its infra. The
    source of truth for requirements. A blanket-per-benchmark need (e.g.
    'all of tbench2 needs root') is realised by the cube's metadata-
    generation script stamping this on every task; heterogeneous benchmarks
    set it per task. Empty (default) = runs on any infra."""

    def requirements(self) -> set[str]:
        out = set(self.requires)
        if self.gpu:
            out.add("gpu:nvidia")
        return out
```

Mirrors the existing `VMResourceConfig.requires_kvm: bool` →
`requirements() -> {"kvm"}` projection.

## MODIFIED — `openspec/specs/benchmark/spec.md`: `Benchmark.setup` gate (per-task loop, infra-owned policy)

`Benchmark.setup` reads `infra.on_incompatible` and loops the per-task bool
`infra.can_serve_task(...)`. No new `setup` parameter, no benchmark-level
aggregate:

```python
def setup(self, infra: InfraConfig | None) -> None:
    if infra is not None and infra.on_incompatible != "force":
        incompatible = [
            SkippedTask(task_id, sorted(meta.container_config.requirements() - infra.capabilities()))
            for task_id, meta in self.tasks.items()
            if meta.container_config is not None and not infra.can_serve_task(meta.container_config)
        ]
        if incompatible and infra.on_incompatible == "raise":
            raise IncompatibleInfraError(
                f"{type(infra).__name__} cannot run {len(incompatible)}/{len(self.tasks)} "
                f"tasks of {self.config.metadata.name}: "
                f"{[(s.task_id, s.missing_capabilities) for s in incompatible[:5]]}. "
                f"Set infra.on_incompatible='skip' to run the compatible subset, "
                f"or 'force' to attempt anyway."
            )
        skipped_ids = {s.task_id for s in incompatible}    # skip mode
        self._runnable_tasks = [t for t in self.tasks if t not in skipped_ids]
        self._skipped_tasks = incompatible
    else:
        self._runnable_tasks, self._skipped_tasks = list(self.tasks), []
    # ... existing resource-provisioning setup
```

- **`"raise"`** (default): if the per-task loop finds **any** incompatible
  task, abort setup with `IncompatibleInfraError`. No episodes created →
  cube-harness surfaces a setup/system failure → nothing to retry.
- **`"skip"`**: incompatible tasks are excluded from `_runnable_tasks` and
  recorded in `_skipped_tasks`; the experiment runs the compatible subset.
- **`"force"`**: loop skipped entirely; every task runs (per-task launch may
  still fail naturally, but the harness no longer pre-empts it). A warning is
  logged once.

When `infra is None` (debug / non-container runs) the gate is a no-op and
the whole task list passes through — matching today's behaviour.

(Optional, not required by this RFC: `cube.task_infra.launch_task_container`
may additionally thread `container_config.requirements()` into the
`DockerServiceConfig` it builds, so the existing `can_serve(resource)` also
enforces at launch time — defense in depth. The setup-time `can_serve_task`
loop is the primary gate.)

## ADDED — `openspec/specs/benchmark/spec.md`: `IncompatibleInfraError`, `SkippedTask`, `CompatibilityReport`

```python
class IncompatibleInfraError(Exception):
    """Raised by Benchmark.setup when infra.on_incompatible == 'raise' and the
    per-task can_serve_task loop finds ≥1 task whose requirements the infra's
    capabilities don't satisfy. A setup-time, pre-episode failure: the
    experiment aborts before launching work, so no retry budget is consumed."""

@dataclass
class SkippedTask:
    task_id: str
    missing_capabilities: list[str]   # sorted, deterministic
```

```python
@dataclass
class CompatibilityReport:
    benchmark: str
    infra: str
    compatible: bool                       # benchmark-level
    benchmark_missing: list[str]
    n_total: int
    n_runnable: int                        # under skip mode
    skipped_tasks: list[SkippedTask]
```

`compatibility_report(cls, infra)` reads `cls.tasks_metadata()` only (cheap;
no archive load) for `cube list --infra=<name>` and CI gates.

## CONSUMED (not modified here) — cube-harness episode-status mapping

This is a cube-standard spec change; the cube-harness side consumes it.
Documented here for reviewer context (cube-harness follow-up PR):

- **skip mode**: each skipped task's episode is written with the existing
  terminal, non-retriable status **`INVALID_CONFIG`**
  (`cube_harness.episode_status`). This is correct by construction:
  `RETRIABLE_STATUSES = {FAILED, CANCELLED, STALE}` excludes
  `INVALID_CONFIG`, so the auto-retry loop never re-runs a skipped task —
  matching the documented semantic "the identical request will fail
  identically on retry, so replaying only burns the retry budget."
- **raise mode**: `IncompatibleInfraError` propagates out of
  `Benchmark.setup` before the episode loop starts; cube-harness reports it
  as a run-level setup failure (no per-episode status).
- An optional dedicated `INCOMPATIBLE` status (cleaner reporting) is left as
  a future cube-harness refinement; not required for correct retry
  semantics.

## Migration impact

| Audience | Change | When |
|---|---|---|
| **Cubes that stamp no `requires`** | None. `ContainerConfig.requires` defaults `set()` → `can_serve_task` returns True for every task → loop is a no-op. | Immediate. |
| **Existing `InfraConfig` subclasses** | None. They inherit `on_incompatible="raise"` + the default `can_serve_task`; both no-op until a task declares `requires`. | Immediate. |
| **`Benchmark.setup` callers** | No signature change — `setup` reads `infra.on_incompatible` internally. Existing call sites unaffected. | This PR (spec). |
| **Cubes adopting requirements (tbench2)** | Stamp `ContainerConfig.requires` (via the codegen script). Gated on infras lacking the token. | Per-cube follow-up. |
| **Root-capable infras** | Publish `container:root` in `capabilities()`. | Paired with the first adopting cube. |
| **Constrained infras (toolkit)** | Optionally set `on_incompatible` default (e.g. `"raise"`); otherwise inherit the base default. | Optional. |
| **cube-harness runner** | Read `infra.on_incompatible` in `setup`/launch; map skip-mode episodes to `INVALID_CONFIG`. | cube-harness follow-up. |

No silent behaviour change: the gate only fires for benchmarks whose tasks
stamp `requires`.
