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

## ADDED — `openspec/specs/benchmark/spec.md`: `BenchmarkConfig.aggregate_requirements()` (derived)

```python
class BenchmarkConfig(TypedBaseModel, ABC):
    ...
    def aggregate_requirements(self) -> set[str]:
        """Union of every task's container_config.requirements(). DERIVED
        from the per-task resources — there is no separately-declared
        benchmark-level requirement to drift out of sync. Reads task
        metadata only (no archive load, no container launch); the infra
        introspects this at setup to decide whole-benchmark compatibility."""
        out: set[str] = set()
        for meta in self.tasks_metadata().values():
            if meta.container_config is not None:
                out |= meta.container_config.requirements()
        return out
```

The **source of truth is the per-task `ContainerConfig.requires`** (below).
`aggregate_requirements()` is a read-only projection over the tasks, used
for setup-time whole-benchmark gating and the static compatibility report.
The blanket "all of tbench2 needs root" case is realised by the
metadata-generation script stamping `requires={"container:root"}` on every
task's `ContainerConfig` — not by a second benchmark-level declaration.

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

## MODIFIED — `openspec/specs/resource/spec.md` + `cube.task_infra`: thread `requires` into the per-task resource

`DockerServiceConfig` gains a `requires: set[str] = Field(default_factory=set)`
field; its existing `requirements()` returns it (currently returns `set()`).
`cube.task_infra.launch_task_container` threads the task's tokens through:

```python
resource = DockerServiceConfig(
    name=name, scope="task", docker_images=[image],
    requires=container_config.requirements(),   # NEW (was implicitly empty)
    launch_script=build_docker_run_script(container_name, image, ram_gb, cpu_cores),
)
```

This makes the **already-existing** `InfraConfig.can_serve(resource)`
(= `resource.requirements().issubset(infra.capabilities())`) answer per-task
compatibility with no new primitive. `launch_task_container` (or `setup` in
skip mode) consults `can_serve` before `launch`.

## MODIFIED — `openspec/specs/benchmark/spec.md`: `Benchmark.setup` gate

`Benchmark.setup` gains an `on_incompatible` parameter and an explicit
capability gate that fires **before any episode is created**:

```python
OnIncompatible = Literal["raise", "skip", "force"]

def setup(self, infra: InfraConfig | None, *, on_incompatible: OnIncompatible = "raise") -> None:
    if infra is not None and on_incompatible != "force":
        # Whole-benchmark gate: derived union over the per-task resources.
        bench_missing = self.config.aggregate_requirements() - infra.capabilities()
        if bench_missing and on_incompatible == "raise":
            raise IncompatibleInfraError(
                f"{type(infra).__name__} cannot run {self.config.metadata.name}: "
                f"missing required capabilities {sorted(bench_missing)}. "
                f"Use on_incompatible='skip' to run the compatible subset, or "
                f"'force' to attempt anyway."
            )
        # skip mode: check each task's own resource via the existing can_serve.
        runnable, skipped = [], []
        for task_id, meta in self.tasks.items():
            req = meta.container_config.requirements() if meta.container_config else set()
            missing = req - infra.capabilities()
            (skipped if missing else runnable).append(
                SkippedTask(task_id, sorted(missing)) if missing else task_id
            )
        self._runnable_tasks, self._skipped_tasks = runnable, skipped
    else:
        self._runnable_tasks, self._skipped_tasks = list(self.tasks), []
    # ... existing resource-provisioning setup
```

- **`"raise"`** (default): whole-benchmark incompatibility (derived from the
  per-task resources) aborts setup with `IncompatibleInfraError`. No episodes
  created → cube-harness surfaces a setup/system failure → nothing to retry.
- **`"skip"`**: incompatible tasks are excluded from `_runnable_tasks` and
  recorded in `_skipped_tasks`; the experiment runs the compatible subset.
- **`"force"`**: gate skipped entirely; every task runs (per-task launch may
  still fail, but the harness no longer pre-empts it). A warning is logged.

When `infra is None` (debug / non-container runs) the gate is a no-op and
the whole task list passes through — matching today's behaviour.

## ADDED — `openspec/specs/benchmark/spec.md`: `IncompatibleInfraError`, `SkippedTask`, `CompatibilityReport`

```python
class IncompatibleInfraError(Exception):
    """Raised by Benchmark.setup(on_incompatible='raise') when the benchmark's
    aggregate_requirements() (derived from the per-task resources) are not
    satisfied by the target infra's capabilities(). A setup-time, pre-episode
    failure: the experiment aborts before launching work, so no retry budget
    is consumed."""

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
| **Cubes that stamp no `requires`** | None. `ContainerConfig.requires` defaults `set()` → `aggregate_requirements()` empty → gate is a no-op. | Immediate. |
| **Existing `InfraConfig` subclasses** | None. New token unrequested until a cube opts in. | Immediate. |
| **`DockerServiceConfig` consumers** | New `requires` field defaults `set()`; existing `requirements()` returns it. No change unless populated. | This PR (spec). |
| **`Benchmark.setup` callers** | New keyword-only `on_incompatible="raise"`, no-op when `aggregate_requirements()` is empty. Existing call sites unaffected. | This PR (spec); cube-harness wires the knob. |
| **Cubes adopting requirements (tbench2)** | Stamp `ContainerConfig.requires` (via the codegen script). Gated on infras lacking the token. | Per-cube follow-up. |
| **Root-capable infras** | Publish `container:root` in `capabilities()`. | Paired with the first adopting cube. |
| **cube-harness runner** | Forward `on_incompatible` from `Experiment`/`run_*`; map skip-mode episodes to `INVALID_CONFIG`. | cube-harness follow-up. |

No silent behaviour change: the gate only fires for benchmarks whose tasks
stamp `requires`.
