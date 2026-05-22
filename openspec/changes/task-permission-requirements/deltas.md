# Deltas — Benchmark/task permission requirements + infra capability handshake

## ADDED — `openspec/specs/resource/spec.md`: capability token vocabulary

The `InfraConfig.capabilities()` docstring is extended with the standard
permission-shaped token vocabulary:

| Token | Meaning |
|---|---|
| `container:root` | Container processes run as uid 0 |
| `container:apt` | `apt-get install` (and equivalent) work — implies root + reachable Debian/Ubuntu mirrors |
| `container:privileged-ports` | A process inside the container may `bind()` ports <1024 |
| `container:systemd` | `systemctl` / `service` work (PID 1 is systemd-compatible) |

Adding a token is non-breaking. Infras default to **not** publishing one;
benchmarks/tasks default to **not** requiring one. The pre-existing
vocabulary (`kvm`, `docker`, `vm`, `gpu:nvidia`, `network:egress`) is
unchanged.

## ADDED — `openspec/specs/benchmark/spec.md`: `BenchmarkConfig.requirements()`

```python
class BenchmarkConfig(TypedBaseModel, ABC):
    ...
    def requirements(self) -> set[str]:
        """Permission tokens EVERY task in this benchmark needs from its
        infra. Default empty = runs on any infra. The primary, blanket
        mechanism: a benchmark whose tasks broadly need root declares it
        once here rather than annotating each task. Composes (union) with
        any per-task ContainerConfig.requires."""
        return set()
```

This is the **primary** requirement-declaration surface. The effective
requirement for a single task is
`benchmark.requirements() | task.metadata.container_config.requirements()`.

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
    """Optional per-task permission tokens, for heterogeneous benchmarks
    where requirements differ across tasks. Most cubes leave this empty
    and declare blanket needs via BenchmarkConfig.requirements()."""

    def requirements(self) -> set[str]:
        out = set(self.requires)
        if self.gpu:
            out.add("gpu:nvidia")
        return out
```

Mirrors the existing `VMResourceConfig.requires_kvm: bool` →
`requirements() -> {"kvm"}` projection.

## MODIFIED — `openspec/specs/benchmark/spec.md`: `Benchmark.setup` gate

`Benchmark.setup` gains an `on_incompatible` parameter and an explicit
capability gate that fires **before any episode is created**:

```python
OnIncompatible = Literal["raise", "skip", "force"]

def setup(self, infra: InfraConfig | None, *, on_incompatible: OnIncompatible = "raise") -> None:
    if infra is not None and on_incompatible != "force":
        bench_missing = self.config.requirements() - infra.capabilities()
        if bench_missing and on_incompatible == "raise":
            raise IncompatibleInfraError(
                f"{type(infra).__name__} cannot run {self.config.metadata.name}: "
                f"missing required capabilities {sorted(bench_missing)}. "
                f"Use on_incompatible='skip' to run the compatible subset, or "
                f"'force' to attempt anyway."
            )
        # skip mode: partition tasks; benchmark-level miss skips ALL tasks
        runnable, skipped = [], []
        for task_id, meta in self.tasks.items():
            cc = meta.container_config
            task_req = self.config.requirements() | (cc.requirements() if cc else set())
            missing = task_req - infra.capabilities()
            (skipped if missing else runnable).append(
                SkippedTask(task_id, sorted(missing)) if missing else task_id
            )
        self._runnable_tasks, self._skipped_tasks = runnable, skipped
    else:
        self._runnable_tasks, self._skipped_tasks = list(self.tasks), []
    # ... existing resource-provisioning setup
```

- **`"raise"`** (default): benchmark-level incompatibility aborts setup
  with `IncompatibleInfraError`. No episodes created → cube-harness
  surfaces a setup/system failure → nothing to retry.
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
    requirements() are not satisfied by the target infra's capabilities().
    A setup-time, pre-episode failure: the experiment aborts before launching
    work, so no retry budget is consumed."""

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
| **Cubes that declare no requirements** | None. `requirements()` defaults `set()`; gate is a no-op. | Immediate. |
| **Existing `InfraConfig` subclasses** | None. New tokens unrequested until a cube opts in. | Immediate. |
| **`Benchmark.setup` callers** | New keyword-only `on_incompatible="raise"` with a no-op default when `requirements()` is empty. Existing call sites unaffected. | This PR (spec); cube-harness wires the knob. |
| **Cubes adopting requirements (tbench2)** | Implement `BenchmarkConfig.requirements()`. Gated on infras lacking the tokens. | Per-cube follow-up. |
| **Root-capable infras** | Publish `container:root` / `container:apt` / `container:privileged-ports` in `capabilities()`. | Paired with the first adopting cube. |
| **cube-harness runner** | Forward `on_incompatible` from `Experiment`/`run_*`; map skip-mode episodes to `INVALID_CONFIG`. | cube-harness follow-up. |

No silent behaviour change: the gate only fires for benchmarks that
explicitly declare requirements.
