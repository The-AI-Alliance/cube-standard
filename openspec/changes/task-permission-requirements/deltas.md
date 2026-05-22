# Deltas — Task permission requirements + infra capability handshake

## ADDED — `resource/spec.md`: `container:root` capability token

Extend the `InfraConfig.capabilities()` vocabulary with one token:

| Token | Meaning |
|---|---|
| `container:root` | Container processes run as uid 0 |

Single token by design: `apt`, `/etc`+`/var` writes, ports <1024, and
`systemctl` are all root-gated and correlate with "is the container root?".
Vocabulary stays open (`set[str]`); a finer token can be added later for a
partial-root infra. Pre-existing tokens (`kvm`, `docker`, `gpu:nvidia`,
`network:egress`) unchanged. Adding a token is non-breaking.

## MODIFIED — `container/spec.md`: `ContainerConfig` gains `requires`

```python
class ContainerConfig(TypedBaseModel):
    image: str
    ram_gb: float = 4.0
    cpu_cores: float = 2.0
    gpu: bool = False
    disk_gb: float = 10.0
    ports: list[int] | None = None
    requires: set[str] = Field(default_factory=set)   # NEW
    """Capability tokens this task's container needs, e.g. {"container:root"}.
    Empty (default) = runs on any infra."""
```

`requires` is the source of truth. A blanket per-benchmark need is stamped
on every task by the cube's metadata-generation script; heterogeneous
benchmarks set it per task.

## ADDED — `resource/spec.md`: `InfraConfig.can_serve_task()` + `on_incompatible`

```python
OnIncompatible = Literal["raise", "skip", "force"]

class InfraConfig(TypedBaseModel, ABC):
    ...
    on_incompatible: OnIncompatible = "raise"
    """Policy when a task needs a capability this infra lacks. 'raise': abort
    setup if ANY task is incompatible. 'skip': run the compatible subset,
    mark the rest INVALID_CONFIG. 'force': attempt every task anyway."""

    def can_serve_task(self, container_config: ContainerConfig) -> bool:
        """Per-task compatibility bool — set-inclusion of the task's requires
        against this infra's capabilities. Mirrors the existing
        can_serve(resource); cheap, metadata-only, no launch."""
        return container_config.requires.issubset(self.capabilities())
```

The harness loops `can_serve_task` over tasks. No benchmark-level aggregate.

## MODIFIED — `benchmark/spec.md`: `Benchmark.setup` reads `infra.on_incompatible`

`setup` keeps its signature; it reads the infra's policy and loops the
per-task bool **before any episode is created**:

```python
def setup(self, infra: InfraConfig | None) -> None:
    if infra is not None and infra.on_incompatible != "force":
        incompatible = [
            SkippedTask(tid, sorted(m.container_config.requires - infra.capabilities()))
            for tid, m in self.tasks.items()
            if m.container_config and not infra.can_serve_task(m.container_config)
        ]
        if incompatible and infra.on_incompatible == "raise":
            raise IncompatibleInfraError(
                f"{type(infra).__name__} cannot run {len(incompatible)}/{len(self.tasks)} "
                f"tasks of {self.config.metadata.name}: "
                f"{[(s.task_id, s.missing_capabilities) for s in incompatible[:5]]}"
            )
        skip = {s.task_id for s in incompatible}          # skip mode
        self._runnable_tasks = [t for t in self.tasks if t not in skip]
        self._skipped_tasks = incompatible
    else:
        self._runnable_tasks, self._skipped_tasks = list(self.tasks), []
    # ... existing resource-provisioning setup
```

`infra is None` (debug runs) → no-op, whole list passes through.

## ADDED — `benchmark/spec.md`: `IncompatibleInfraError`, `SkippedTask`

```python
class IncompatibleInfraError(Exception):
    """Raised by Benchmark.setup when infra.on_incompatible == 'raise' and the
    per-task can_serve_task loop finds ≥1 incompatible task. Setup-time,
    pre-episode: the experiment aborts before launching, so no retry budget
    is consumed."""

@dataclass
class SkippedTask:
    task_id: str
    missing_capabilities: list[str]   # sorted
```

`BenchmarkSummary` gains `n_skipped: int` and `skipped_tasks: list[SkippedTask]`;
accuracy is computed over completed (runnable) tasks, not over the skipped set.

## CONSUMED (cube-harness follow-up, not modified here)

- **skip mode** writes each skipped episode as the existing terminal,
  non-retriable `INVALID_CONFIG` (`cube_harness.episode_status`); the
  auto-retry loop excludes it by construction.
- **raise mode**: `IncompatibleInfraError` propagates out of `setup` before
  the episode loop — reported as a run-level setup failure.
- An optional dedicated `INCOMPATIBLE` status is a future refinement; not
  required for correct retry semantics.

## Migration

| Audience | Change |
|---|---|
| Cubes with no `requires` | None — `can_serve_task` always True; loop is a no-op. |
| Existing `InfraConfig` subclasses | None — inherit `on_incompatible="raise"` + default `can_serve_task`, both no-op until a task declares `requires`. |
| `Benchmark.setup` callers | No signature change. |
| tbench2 | Codegen stamps `requires={"container:root"}` on every task. |
| Root-capable infras | Publish `container:root` in `capabilities()`. |
| cube-harness runner | Read `infra.on_incompatible`; map skip-mode episodes to `INVALID_CONFIG`. |

No silent behaviour change — the gate fires only for tasks that stamp
`requires`.
