# Benchmark/task permission requirements + infra capability handshake

**Status:** Proposed (rev 2 — incorporates review feedback on status reuse,
benchmark-level requirements, and the three-mode policy)
**Date:** 2026-05-21
**Scope:** `cube.container.ContainerConfig`, `cube.benchmark.BenchmarkConfig`,
`cube.resource.InfraConfig`, `cube.benchmark.Benchmark.setup` (gate behaviour)
**Targets:** `dev`
**Related:** Surfaced by cube-harness session `2026-05-20_tbench2-infra-model-matrix`
finding F7 ("toolkit's non-root user vs tbench2's apt-installing tasks").
cube-harness consumes via `Benchmark.setup` + the episode-status taxonomy.

---

## Problem

Some cube benchmarks need infra permissions that not every `InfraConfig`
provides:

- `terminalbench2-cube` tasks ask the agent to `apt-get install` packages
  and write to `/etc/`, `/var/log/` — achievable only inside a container
  running as **root**.
- `cube-infra-toolkit` (EAI Toolkit) hard-enforces **uid 13011** in every
  container by cluster policy; `apt-get` is unreachable.

Today this mismatch surfaces as **silent low scores**. The agent runs,
tries `apt-get`, gets `Permission denied`, eventually emits `final_step`
with a wrong solution, the evaluator scores reward=0. A user looking at
`tbench2 on toolkit ⇒ 0.17` cannot tell whether that's a model-quality
issue, a cube-installation issue, or a fundamental infra mismatch. Worse,
under cube-harness's auto-retry loop these episodes can be classified
`FAILED` (retriable) and **burn the retry budget re-running an episode
that will fail identically every time**.

Empirical evidence (haiku × 25 tasks): daytona 0.39 vs toolkit 0.17; the
gap is mostly silent partial failure, not model capability — stronger
models (sonnet) lift daytona but not toolkit.

The framework **already has** the matching mechanism. `InfraConfig` exposes
`capabilities() -> set[str]`; `ResourceConfig` exposes
`requirements() -> set[str]`; `InfraConfig.can_serve(resource)` does
`requirements.issubset(capabilities)`. The token vocabulary in use today is
hardware-shaped (`kvm`, `docker`, `gpu:nvidia`, `network:egress`). This
proposal (a) extends the vocabulary with permission tokens, (b) lets a
**benchmark** (not just a per-task container) declare requirements, and (c)
gates at `Benchmark.setup` time with a clear three-mode policy.

## Design

### 1. Permission token vocabulary (additive)

Add one token to the `InfraConfig.capabilities()` documented vocabulary:

| Token | Meaning |
|---|---|
| `container:root` | Container processes run as uid 0 |

A single `container:root` token is deliberate. In practice the
finer-grained needs — `apt-get install`, writing to `/etc` and `/var`,
binding ports <1024, `systemctl` — are **all root-gated**, and they
correlate near-perfectly on both sides: an infra either grants root (and
thus all of them) or it doesn't (Daytona/local/AWS/Azure run as root;
EAI Toolkit forces uid 13011 and so denies the whole set). Splitting into
`container:apt` / `container:systemd` / `container:privileged-ports`
would be speculative granularity with no current consumer that needs the
distinction. The token vocabulary is open (`set[str]`), so a future infra
that offers a *partial* root surface — e.g. rootless Podman with
`CAP_NET_BIND_SERVICE` but no apt — can introduce a finer token then,
without breaking anything.

Adding a token is non-breaking: infras default to *not* publishing it;
benchmarks/tasks default to *not* requiring it.

### 2. Requirements ride on the per-task resource; the benchmark exposes a derived aggregate

Requirements live where the resource lives — on each task's
`ContainerConfig` — and flow through the **existing** resource machinery.
Each tbench2 task already instantiates its own `ContainerConfig`
(89 tasks, 89 distinct images, no shared template) which `launch_task_container`
turns into a per-task `DockerServiceConfig`. We thread `requires` through
that build so `DockerServiceConfig.requirements()` returns the tokens and
the **already-existing** `InfraConfig.can_serve(resource)` answers
per-task compatibility — no new check primitive:

```python
class ContainerConfig(TypedBaseModel):
    ...
    requires: set[str] = Field(default_factory=set)
    def requirements(self) -> set[str]:
        out = set(self.requires)
        if self.gpu:
            out.add("gpu:nvidia")
        return out
```

```python
# cube.task_infra.launch_task_container — thread the tokens into the resource
resource = DockerServiceConfig(
    name=name, scope="task", docker_images=[image],
    requires=container_config.requirements(),   # NEW: was implicitly empty
    launch_script=build_docker_run_script(...),
)
```

**Blanket use case stays cheap.** "All of tbench2 needs root" is achieved
not by per-task investigation but by the metadata-generation script
(`scripts/create_task_metadata.py`) stamping `requires={"container:root"}`
on every task's `ContainerConfig` at generation time. One line in the
codegen, all 89 tasks covered, zero manual triage. Heterogeneous benchmarks
that genuinely differ per task set `requires` per task instead.

**Benchmark-level compatibility is derived, not separately declared.** A
benchmark exposes an introspection helper that unions the per-task
requirements so an infra can answer "is the *whole* benchmark
incompatible?" without launching anything:

```python
class BenchmarkConfig(TypedBaseModel, ABC):
    ...
    def aggregate_requirements(self) -> set[str]:
        """Union of every task's container_config.requirements(). Reads
        task metadata only (no archive load, no container launch). The
        infra introspects this at setup to decide whole-benchmark
        compatibility. Derived — there is no separately-declared
        benchmark-level requirement to drift out of sync with the tasks."""
        out: set[str] = set()
        for meta in self.tasks_metadata().values():
            if meta.container_config is not None:
                out |= meta.container_config.requirements()
        return out
```

For tbench2 this returns `{"container:root"}` (every task stamped); for a
benchmark with no root-needing tasks it returns `set()`.

### 3. The gate: `Benchmark.setup(infra, on_incompatible=...)` — three modes

```python
OnIncompatible = Literal["raise", "skip", "force"]
```

Resolved at experiment-setup time (default `"raise"`):

- **`"raise"`** (default). At `setup`, introspect the per-task resources
  via `aggregate_requirements()` and compute
  `missing = benchmark.aggregate_requirements() - infra.capabilities()`.
  If non-empty, **raise `IncompatibleInfraError` immediately, before any
  episode is created**. tbench2 on toolkit stops at setup with:
  `"ToolkitInfraConfig cannot run terminalbench2-cube: missing
  {container:root}."` No episodes, no retries, no spend.

- **`"skip"`**. Run the compatible subset. Each task is checked
  individually via its own resource (`infra.can_serve(task_resource)`); a
  task whose requirement isn't met is **not launched**, its episode is
  recorded with a terminal, **non-retriable** status (§4), and it is
  excluded from the accuracy denominator. Reported as `n_skipped`. This is
  where per-task granularity earns its keep: a heterogeneous benchmark runs
  its root-free tasks and skips the rest.

- **`"force"`** (try-anyway override). Skip the gate entirely; launch every
  task regardless. The escape hatch for probing whether a new image policy /
  infra change has obviated a stale requirement — "did toolkit start
  allowing root? run with `force` and see." Logs a warning so the override
  is visible in the trace.

Where the mode lives: a field on the experiment-run path (cube-harness
`Experiment` / `run_*`), forwarded into `Benchmark.setup`. cube-standard
defines the enum + the `setup` parameter; cube-harness wires the CLI/recipe
knob. A given infra MAY also pin a stricter default, but the per-run setting
wins.

### 4. Status mapping — reuse the existing non-retriable terminal bucket

**No new episode status is required.** cube-harness already has
`INVALID_CONFIG`: a terminal, non-retriable status whose documented meaning
is *"a permanent error… the identical request will fail identically on
retry, so replaying only burns the retry budget."* `RETRIABLE_STATUSES =
{FAILED, CANCELLED, STALE}` deliberately excludes it.

An incompatible task is exactly that shape: it will fail identically every
retry. So:

- **`"raise"` mode**: the failure happens at `Benchmark.setup`, before
  episodes exist. It surfaces as a **setup/system failure** in cube-harness
  — the experiment aborts; nothing to retry.
- **`"skip"` mode**: each skipped task's episode is written with status
  `INVALID_CONFIG` (terminal, non-retriable). The auto-retry loop leaves it
  alone by construction; the summary counts it under `n_skipped` /
  `n_invalid_config` rather than `n_failed`, keeping the accuracy
  denominator honest.

A dedicated `INCOMPATIBLE` / `SKIPPED` status is an **optional future
refinement** for cleaner dashboards (it would distinguish "infra can't
serve" from "bad model name"). It touches the `Status` `Literal`,
`TERMINAL_STATUSES`, `STATUS_ICONS`, and exhaustive matches in
cube-harness — modest but non-trivial. Recommendation: **ship v1 reusing
`INVALID_CONFIG`**; add `INCOMPATIBLE` later only if the reporting
distinction proves valuable. Either way, the retry semantics are correct
from day one because both are non-retriable terminal.

### 5. Static compatibility preview

```python
@classmethod
def compatibility_report(cls, infra: InfraConfig) -> CompatibilityReport: ...
```

Reads metadata only (no container launches) — built on the same
`aggregate_requirements()` introspection — so `cube list --infra=toolkit`
and CI gates can show "tbench2: incompatible on toolkit (needs
container:root)" before anyone spends a dollar.

## Migration impact

Backward compatibility is the default:

- `ContainerConfig.requires` defaults to `set()`, so
  `aggregate_requirements()` returns `set()` for every existing cube. No
  benchmark is newly gated by this PR alone.
- Existing `InfraConfig.capabilities()` implementations are unchanged.
- `Benchmark.setup` gains an `on_incompatible="raise"` parameter that is a
  **no-op when `aggregate_requirements()` is empty** — existing cubes see
  no behaviour change.
- `DockerServiceConfig` gains a `requires` field defaulting to `set()`;
  the `requirements()` it already exposes simply returns it.

Behaviour changes only when a cube stamps `ContainerConfig.requires` AND a
paired infra-side PR teaches root-capable infras to publish the token.
Recommended rollout (follow-up PRs, not this RFC):

1. Land this RFC (token + `ContainerConfig.requires` + `DockerServiceConfig`
   threading + `aggregate_requirements()` + `setup` gate + status mapping).
2. cube-resources: `LocalInfraConfig`, `DaytonaInfraConfig`, `AWSInfraConfig`,
   `AzureInfraConfig` publish `container:root`. `ToolkitInfraConfig` does not.
3. cube-harness: `terminalbench2-cube`'s `scripts/create_task_metadata.py`
   stamps `requires={"container:root"}` on every task's `ContainerConfig`.
   tbench2 on toolkit now fails fast at setup with a clear message.
4. cube-harness: wire the `on_incompatible` knob into `Experiment` / `run_*`
   and the episode-status `INVALID_CONFIG` mapping for skip mode.
5. Tooling: `cube list --infra=<name>` and CI dashboards display the
   compatibility matrix.

## Alternatives considered

- **A separately-declared benchmark-level `requirements()` field (rev 2).**
  Rejected: it duplicates information that already lives on the per-task
  `ContainerConfig` and can drift out of sync with the tasks. The benchmark
  view is *derived* (`aggregate_requirements()`) from the per-task
  resources instead — single source of truth, and the infra introspects
  the real per-task resources. The blanket "all tasks need root" case is
  handled by the codegen stamping each task, not by a second declaration.
- **A brand-new `INCOMPATIBLE` episode status as the v1 mechanism.**
  Rejected for v1: `INVALID_CONFIG` already carries the exact
  non-retriable-terminal semantics. New status is a clean later refinement,
  not a prerequisite.
- **Two modes only (raise / skip).** Rejected: the "force / try-anyway"
  mode is operationally important for validating that an infra/image
  policy change has made an old requirement obsolete without editing the
  cube.
- **Encode requirements in the Docker image** (`LABEL cube.requires=root`).
  Rejected: foreign system; couples requirement cadence to image rebuilds;
  violates "Python is the configuration".

## Open questions

1. Where exactly does the `on_incompatible` knob live in cube-harness —
   `Experiment` field, `run_*` arg, or both? (cube-standard only defines
   the `setup` parameter + enum; cube-harness owns the surface.)
2. Should `"force"` mode downgrade the requirement check to a logged
   warning at *every* task launch, or just once at setup? (Lean: once at
   setup + a per-task trace breadcrumb.)
3. Confirm `INVALID_CONFIG` reuse is acceptable to cube-harness owners vs.
   adding `INCOMPATIBLE` now. (This RFC recommends reuse; flagging for
   explicit sign-off since it slightly overloads the status's meaning.)
