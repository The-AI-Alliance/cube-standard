# Deltas — Task permission requirements + infra capability handshake

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
    """Permission tokens this container needs from its host infra. See
    `cube.resource.InfraConfig.capabilities` for the authoritative
    vocabulary. Common tokens: `container:root`, `container:apt`,
    `container:privileged-ports`, `container:systemd`. Empty (default)
    = no special requirements; container runs on every infra."""

    def requirements(self) -> set[str]:
        """Projection used by `InfraConfig.can_serve` and `Benchmark._setup`.
        Composes typed booleans (`gpu`) with free-form `requires` tokens."""
        out = set(self.requires)
        if self.gpu:
            out.add("gpu:nvidia")
        return out
```

The `requirements()` projection mirrors the existing pattern on
`VMResourceConfig.requires_kvm: bool` → `requirements() -> {"kvm"}`.
`ContainerConfig.requirements()` is the canonical task-level answer to
"what does this container need from its infra".

## ADDED — `openspec/specs/resource/spec.md`: capability token vocabulary

The `InfraConfig.capabilities()` docstring is extended with the standard
permission-shaped token vocabulary:

| Token | Meaning |
|---|---|
| `container:root` | Container processes run as uid 0 |
| `container:apt` | `apt-get install` (and equivalent) work — implies root + reachable Debian/Ubuntu mirrors |
| `container:privileged-ports` | A process inside the container may `bind()` ports <1024 |
| `container:systemd` | `systemctl` / `service` work (PID 1 is systemd or a systemd-compatible init) |

Adding a token to the vocabulary is non-breaking. Infras default to
**not** publishing a token; tasks default to **not** requiring it. A
task is gated only when the task explicitly requires a token AND the
target infra does not publish it.

The pre-existing vocabulary (`kvm`, `docker`, `vm`, `gpu:nvidia`,
`network:egress`) is unchanged.

## MODIFIED — `openspec/specs/benchmark/spec.md`: `Benchmark._setup`

`Benchmark._setup(infra)` (called by `Benchmark.setup`) gains an explicit
task-compatibility filter:

```python
def _setup(self, infra: InfraConfig | None) -> None:
    if infra is None:
        self._runnable_tasks = list(self.tasks)
        self._skipped_tasks = []
        return
    runnable, skipped = [], []
    for task_id, task_meta in self.tasks.items():
        cc = task_meta.container_config
        if cc is None:
            runnable.append(task_id)
            continue
        missing = cc.requirements() - infra.capabilities()
        if missing:
            skipped.append(SkippedTask(task_id=task_id, missing_capabilities=sorted(missing)))
        else:
            runnable.append(task_id)
    if skipped and self.config.on_incompatible_task == "raise":
        raise IncompatibleTaskError(
            f"{len(skipped)} task(s) require capabilities not provided by "
            f"{type(infra).__name__}: {[(s.task_id, s.missing_capabilities) for s in skipped]}"
        )
    self._runnable_tasks = runnable
    self._skipped_tasks = skipped
```

When `infra is None` (debug / non-container runs), no filtering is
applied — the whole task list passes through, matching today's behaviour.

`Experiment` and `run_with_ray` / `run_sequentially` consume
`benchmark._runnable_tasks` instead of `benchmark.tasks` after setup,
and surface `benchmark._skipped_tasks` in the summary.

## ADDED — `openspec/specs/benchmark/spec.md`: `BenchmarkConfig.on_incompatible_task`

New optional field on `BenchmarkConfig`:

```python
class BenchmarkConfig(TypedBaseModel, ABC):
    ...
    on_incompatible_task: Literal["skip", "raise"] = "skip"
```

Default `"skip"` preserves "run what you can" semantics. `"raise"`
short-circuits the experiment for strict same-task comparison runs.

## ADDED — `openspec/specs/benchmark/spec.md`: `SkippedTask` and `IncompatibleTaskError`

```python
@dataclass
class SkippedTask:
    task_id: str
    missing_capabilities: list[str]   # sorted, deterministic

class IncompatibleTaskError(Exception):
    """Raised when `on_incompatible_task='raise'` and one or more tasks'
    requirements are not satisfied by the target infra's capabilities."""
```

`BenchmarkSummary` (in the storage / summary layer) gains:

```python
class BenchmarkSummary:
    ...
    n_skipped: int = 0
    skipped_tasks: list[SkippedTask] = field(default_factory=list)
```

Accuracy and other aggregates are computed over `n_completed`
(runnable tasks that finished), not over `n_skipped + n_completed`.
This makes cross-infra comparisons of the *attempted* overlap fair by
construction.

## ADDED — `openspec/specs/benchmark/spec.md`: `BenchmarkConfig.compatibility_report`

```python
@classmethod
def compatibility_report(cls, infra: InfraConfig) -> CompatibilityReport:
    """Static preview of which tasks would be skipped on `infra`, without
    launching anything. Used by `cube list --infra=<name>` and CI gates."""
```

```python
@dataclass
class CompatibilityReport:
    benchmark: str
    infra: str
    n_total: int
    n_runnable: int
    n_skipped: int
    skipped_tasks: list[SkippedTask]
    missing_capability_counts: dict[str, int]   # which tokens block the most tasks
```

Reads `cls.tasks_metadata()` only (cheap; no archive load). Surfaces:

> `terminalbench2-cube on toolkit:`
> `  runnable: 84/89`
> `  skipped: 5 (container:root × 5; container:apt × 5)`
> `  Use a root-capable infra (daytona, aws, azure, local) to run these.`

## Migration impact

| Audience | Change | When |
|---|---|---|
| **Cubes that don't set `container_config.requires`** | No change. Default `set()` means tasks pass the filter on every infra. | Immediate. |
| **Existing `InfraConfig` subclasses** | No code change required. New tokens are not requested by any task until a cube opts in. | Immediate. |
| **Cubes adopting requirements** | Set `container_config.requires` in `TaskMetadata`. Tasks become opt-in gated on infras that don't publish the matching tokens. | Per-cube follow-up PRs. |
| **Root-capable infras** | Publish `container:root` / `container:apt` / `container:privileged-ports` in `capabilities()` so newly-annotated tasks still run there. | Paired with the first cube that opts in. |
| **`Experiment` / runners** | Switch from `benchmark.tasks` to `benchmark.runnable_tasks` for episode iteration. Read `benchmark.skipped_tasks` for summary reporting. Two-line change. | This PR. |

No silent behaviour change. The compatibility filter only fires for
tasks that explicitly request tokens.
