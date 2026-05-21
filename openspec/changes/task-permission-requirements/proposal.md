# Task permission requirements + infra capability handshake (extension)

**Status:** Proposed
**Date:** 2026-05-21
**Scope:** `cube.container.ContainerConfig`, `cube.resource.InfraConfig`,
`cube.benchmark.Benchmark._setup` (filter behaviour)
**Targets:** `dev`
**Related:** Surfaced by cube-harness session `2026-05-20_tbench2-infra-model-matrix`
([REPORT](https://github.com/The-AI-Alliance/cube-harness/blob/dev/...))
finding F7 ("toolkit's non-root user vs tbench2's apt-installing tasks").

---

## Problem

Some cube tasks need infra permissions that not every `InfraConfig`
provides:

- `terminalbench2-cube/nginx-request-logging` and `install-windows-3.11`
  ask the agent to `apt-get install` packages and write to `/etc/`,
  `/var/log/` — only achievable inside a container running as **root**.
- `cube-infra-toolkit` (EAI Toolkit) hard-enforces **uid 13011** in every
  container by cluster policy; `apt-get` is unreachable.

Today this mismatch surfaces as **silent low scores**: the agent runs,
tries `apt-get`, gets `Permission denied`, eventually emits `final_step`
with a wrong solution, the evaluator scores reward=0. The cube/infra
combination is *partially incompatible* but the contract has no way to
say so. A user looking at `tbench2 on toolkit ⇒ 0.17` cannot tell
whether the gap is a model-quality issue, a cube-installation issue, or
a fundamental infra mismatch.

Empirical evidence from the surfacing session:

| | daytona (root) | toolkit (uid 13011) |
|---|---|---|
| haiku × 25 tasks | 9/23 = **0.39** | 4/23 = **0.17** |
| sonnet × 12 tasks | 3/6 = **0.50** | 1/9 = **0.11** |

Stronger model lifts daytona but does not lift toolkit — because the
toolkit-impossible subset is impossible regardless of reasoning.

Heuristic classification of all 89 tbench2 tasks against root-needing
operations (apt, `/etc/*`, `/var/*`, ports <1024, `systemctl`): only
**~5–7 tasks** truly require root; the remaining 82+ would run fine on
non-root infras if the framework gated only the incompatible tasks
instead of silently mis-scoring them.

The framework **already has** the right mechanism. `InfraConfig` exposes
`capabilities() -> set[str]`. `ResourceConfig` exposes
`requirements() -> set[str]`. `InfraConfig.can_serve(resource)` does
`requirements.issubset(capabilities)`. The pattern is in place for
hardware-level requirements (`"kvm"`, `"docker"`, `"gpu:nvidia"`,
`"network:egress"`). This proposal extends it to **permission-shaped
tokens** and wires the check into `Benchmark._setup` at task-selection
time, so incompatible tasks are **declined fast** with a clear reason
instead of failing silently.

## Proposal

Three small additions, all on top of existing types:

### 1. Standard permission token vocabulary

Add to `cube.resource.InfraConfig.capabilities` docstring (the
authoritative token list):

| Token | Meaning |
|---|---|
| `"container:root"` | Container processes run as uid 0 |
| `"container:apt"` | `apt-get install` works (implies root + Debian-mirror egress) |
| `"container:privileged-ports"` | Process can `bind()` ports <1024 |
| `"container:systemd"` | `systemctl` / `service` work |

These are *additional* to the existing vocabulary (`"kvm"`, `"docker"`,
`"gpu:nvidia"`, `"network:egress"`). Adding a token is non-breaking:
infras default to absence; only infras that explicitly claim a token
satisfy a task that requires it.

### 2. `ContainerConfig.requirements()`

Add a typed `requires` field to `ContainerConfig`, mirroring how
`VMResourceConfig.requires_kvm: bool` projects into `requirements() ->
{"kvm"}`:

```python
class ContainerConfig(TypedBaseModel):
    image: str
    ram_gb: float = 4.0
    cpu_cores: float = 2.0
    gpu: bool = False
    disk_gb: float = 10.0
    ports: list[int] | None = None
    requires: set[str] = Field(default_factory=set)   # NEW

    def requirements(self) -> set[str]:
        """Permission tokens this container needs. Composes with the gpu/kvm
        booleans already on this class; cubes can also set free-form tokens
        via `requires`."""
        out = set(self.requires)
        if self.gpu:
            out.add("gpu:nvidia")
        return out
```

This is the *task-level* projection: `TaskMetadata.container_config.requirements()`
becomes the canonical answer to "what does this task need from its infra".

### 3. Benchmark `_setup` task filter

The base `Benchmark._setup(infra)` iterates declared tasks and partitions
them:

```python
for task_id, task_meta in self.tasks.items():
    cc = task_meta.container_config
    if cc is None:
        runnable.append(task_id)
        continue
    missing = cc.requirements() - infra.capabilities()
    if missing:
        skipped.append((task_id, sorted(missing)))
    else:
        runnable.append(task_id)
```

Two modes (controlled by `BenchmarkConfig.on_incompatible_task: Literal["skip", "raise"] = "skip"`):

- **`"skip"`** (default) — incompatible tasks are recorded in
  `BenchmarkSummary.skipped_tasks: list[SkippedTask]`. Accuracy is
  reported over `len(runnable)` tasks, not over `len(all_tasks)`, with
  `n_skipped` and the reason set surfaced in summary output. The
  experiment continues; cross-infra comparisons of the *attempted*
  task overlap remain meaningful.
- **`"raise"`** — first incompatibility raises `IncompatibleTaskError`
  at setup, before any episode runs. For users who want strict
  same-task comparisons.

`BenchmarkConfig` may opt to expose a compatibility preview without
launching anything:

```python
@classmethod
def compatibility_report(cls, infra: InfraConfig) -> CompatibilityReport: ...
```

Returns a structured report (count runnable / skipped, distribution of
missing tokens) so tooling (`cube list --infra=toolkit`) can show
"tbench2: 84/89 runnable on toolkit; 5 need `container:root`" without
spinning up containers.

## Migration impact

Backward compatibility is the default:

- Existing `ContainerConfig` instances default to `requires = set()`. No
  task gets newly gated by this PR alone.
- Existing `InfraConfig.capabilities()` implementations don't declare
  the new tokens. As long as no task requires them, nothing changes.
- Behaviour change happens **only** when a cube starts declaring
  `container:root` (or similar) AND a paired infra-side PR teaches
  root-capable infras to publish the token.

Recommended rollout order (handled in follow-up PRs, **not** this RFC):

1. Land this RFC (vocabulary + type extensions + filter behaviour).
2. cube-resources side: `LocalInfraConfig`, `DaytonaInfraConfig`,
   `AWSInfraConfig`, `AzureInfraConfig` publish
   `"container:root", "container:apt", "container:privileged-ports"`
   (root-capable infras). `ToolkitInfraConfig` publishes none of these.
3. cube-harness side: `terminalbench2-cube` annotates the ~5 root-required
   tasks via `container_config.requires = {"container:root", "container:apt"}`.
4. Tooling: `cube list` and CI dashboards display the compatibility matrix.

After (3), tbench2 on toolkit declines 5 tasks fast with reason, scores
the rest, and reports `0.X over 84 tasks attempted` instead of `0.17 over
all 89`. The metric becomes legible.

## Alternatives considered

- **Mark the whole benchmark as root-required, not per task.** Rejected:
  too coarse for tbench2 (84 of 89 tasks are fine non-root). Forces a
  binary cube/infra compatibility that hides the runnable subset.
- **Encode requirements in the image** (e.g., `LABEL cube.requires=root`).
  Rejected: image-side is a foreign system to cube-standard; couples
  the requirement-declaration cadence to image rebuilds. Python-side
  is the configuration (per the constitution, Pillar II).
- **Boolean fields per requirement** (`requires_root: bool`,
  `requires_apt: bool`, …). Rejected: doesn't grow gracefully. The
  existing `set[str]` token vocabulary already proved this point — new
  capabilities arrive without breaking older serialisations.
- **Tool-layer wrapping (the simulate-non-root flag for fairness
  benchmarking).** Adjacent but orthogonal — covered by a separate
  proposal once this contract lands.

## Open questions

1. Where do the tokens *live*? Two options:
   (a) free-form strings, documented in `resource/spec.md`. Cheap.
   (b) `Literal` type or `Enum`. Stronger typing but every new token
   becomes a contract change. **Lean toward (a)** for v1; can promote
   to `Literal` later without breaking existing declarations.
2. Should `Benchmark._setup` distinguish `infra is None` (debug-run, no
   container launches) from compatibility check? Probably yes — when
   `infra is None`, skip the filter (no enforcement makes sense without
   a target infra to compare against).
3. Should the skipped-tasks summary include a hint at which infras
   *would* serve them (`"these tasks need container:root; available
   on: daytona, aws, azure, local"`)? Useful but requires the framework
   to know about other infras at setup time. Defer to tooling.
