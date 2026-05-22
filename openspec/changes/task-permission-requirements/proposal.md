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

### 2. Requirements ride on the per-task `ContainerConfig`; the infra answers a per-task bool

Requirements live where the resource lives — on each task's
`ContainerConfig`. Each tbench2 task already instantiates its own
`ContainerConfig` (89 tasks, 89 distinct images, no shared template), so
this is a natural home:

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

**Blanket use case stays cheap.** "All of tbench2 needs root" is achieved
not by per-task investigation but by the metadata-generation script
(`scripts/create_task_metadata.py`) stamping `requires={"container:root"}`
on every task's `ContainerConfig` at generation time. One line in the
codegen, all 89 tasks covered, zero manual triage. Heterogeneous benchmarks
set `requires` per task instead.

**The compatibility primitive is a per-task bool the infra returns.** Rather
than a benchmark-level aggregate (awkward to build — the benchmark would
have to materialise or union every task's resource), the infra answers one
task at a time, reusing its existing `capabilities()`:

```python
class InfraConfig(TypedBaseModel, ABC):
    ...
    def can_serve_task(self, container_config: ContainerConfig) -> bool:
        """True iff this infra provides every capability the task's container
        requires. Cheap, metadata-only — no resource built, no launch."""
        return container_config.requirements().issubset(self.capabilities())
```

This mirrors the existing `InfraConfig.can_serve(resource)` (which checks a
`ResourceConfig`); `can_serve_task` is the same set-inclusion test against a
task's `ContainerConfig`. The harness loops over tasks and collects the
per-task bool — there is **no aggregate method to keep in sync**, and
"is the whole benchmark incompatible?" is simply "did every task come back
False?".

### 3. The mode is a flag on the `InfraConfig`

The policy for handling incompatible tasks belongs to the infra — it is the
component that knows its own constraints. Add a field:

```python
OnIncompatible = Literal["raise", "skip", "force"]

class InfraConfig(TypedBaseModel, ABC):
    on_incompatible: OnIncompatible = "raise"
```

At `Benchmark.setup(infra)` / before launch, the harness walks the tasks,
calls `infra.can_serve_task(task.container_config)` for each, and applies
`infra.on_incompatible`:

- **`"raise"`** (default). If **any** task is incompatible, **raise
  `IncompatibleInfraError` immediately, before any episode is created**.
  tbench2 on toolkit stops at setup with:
  `"ToolkitInfraConfig cannot run 5/89 tasks of terminalbench2-cube
  (missing container:root): nginx-request-logging, …"`. No episodes, no
  retries, no spend. (This is the whole-benchmark fail-fast, achieved by
  the per-task loop — no aggregate needed.)

- **`"skip"`**. Run the compatible subset. Each incompatible task is **not
  launched**; its episode is recorded with a terminal, **non-retriable**
  status (§4) and excluded from the accuracy denominator. Reported as
  `n_skipped`. Where per-task granularity earns its keep: a heterogeneous
  benchmark runs its root-free tasks and skips the rest.

- **`"force"`** (try-anyway override). Skip the check; launch every task
  regardless. The escape hatch for probing whether a new image policy /
  infra change has obviated a stale requirement — "did toolkit start
  allowing root? set `force` and see." Logs a warning so the override is
  visible in the trace.

Putting the flag on `InfraConfig` means a user picks the policy where they
pick the infra — e.g. `ToolkitInfraConfig(on_incompatible="skip")` to run
the compatible subset on toolkit, or the default `"raise"` to refuse the
combination outright. cube-standard owns the field + enum; cube-harness
reads `infra.on_incompatible` in `setup`/launch and maps skip-mode episodes
to `INVALID_CONFIG`.

### 4. Status mapping — reuse the existing non-retriable terminal bucket

**No new episode status is required.** cube-harness already has
`INVALID_CONFIG`: a terminal, non-retriable status whose documented meaning
is *"a permanent error… the identical request will fail identically on
retry, so replaying only burns the retry budget."* `RETRIABLE_STATUSES =
{FAILED, CANCELLED, STALE}` deliberately excludes it.

An incompatible task is exactly that shape: it will fail identically every
retry. So:

- **`"raise"` mode**: the failure happens at `Benchmark.setup` (the per-task
  loop hit ≥1 incompatible task), before episodes exist. It surfaces as a
  **setup/system failure** in cube-harness — the experiment aborts; nothing
  to retry.
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

Reads metadata only (no container launches) — the same per-task
`infra.can_serve_task(meta.container_config)` loop — so `cube list
--infra=toolkit` and CI gates can show "tbench2: 84/89 runnable on toolkit;
5 need container:root" before anyone spends a dollar.

## Migration impact

Backward compatibility is the default:

- `ContainerConfig.requires` defaults to `set()`, so
  `can_serve_task` returns `True` for every task of every existing cube. No
  benchmark is newly gated by this PR alone.
- Existing `InfraConfig.capabilities()` implementations are unchanged.
- `InfraConfig` gains `on_incompatible: OnIncompatible = "raise"` and a
  `can_serve_task()` default method — both no-ops when no task declares
  `requires`.

Behaviour changes only when a cube stamps `ContainerConfig.requires` AND a
paired infra-side PR teaches root-capable infras to publish the token.
Recommended rollout (follow-up PRs, not this RFC):

1. Land this RFC (token + `ContainerConfig.requires` + `InfraConfig.can_serve_task`
   + `InfraConfig.on_incompatible` + the setup/launch loop + status mapping).
2. cube-resources: `LocalInfraConfig`, `DaytonaInfraConfig`, `AWSInfraConfig`,
   `AzureInfraConfig` publish `container:root`. `ToolkitInfraConfig` does not.
3. cube-harness: `terminalbench2-cube`'s `scripts/create_task_metadata.py`
   stamps `requires={"container:root"}` on every task's `ContainerConfig`.
   tbench2 on toolkit now fails fast at setup with a clear message.
4. cube-harness: read `infra.on_incompatible` in `setup`/launch; map
   skip-mode episodes to `INVALID_CONFIG`.
5. Tooling: `cube list --infra=<name>` and CI dashboards display the
   compatibility matrix.

## Alternatives considered

- **A benchmark-level aggregate requirement (rev 4 `aggregate_requirements()`,
  or a separately-declared rev-2 `requirements()` field).** Rejected: the
  aggregate is awkward to implement (the benchmark would have to materialise
  or union every task's resource) and a separately-declared field duplicates
  what already lives on the per-task `ContainerConfig`. Instead the infra
  answers a **per-task bool** (`can_serve_task`) and the harness loops;
  "is the whole benchmark incompatible?" falls out of the loop with no
  aggregate to maintain.
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

1. `on_incompatible` lives on `InfraConfig` (a config field, serialised with
   the infra). Should a per-run override also exist (e.g. an `Experiment`
   field that wins over the infra default), or is the infra field the single
   source? (Lean: infra field only for v1 — simplest; add an override later
   if a need appears.)
2. Should `"force"` mode log a warning once at setup or per forced task?
   (Lean: once at setup + a per-task trace breadcrumb.)
3. Confirm `INVALID_CONFIG` reuse is acceptable to cube-harness owners vs.
   adding `INCOMPATIBLE` now. (This RFC recommends reuse; flagging for
   explicit sign-off since it slightly overloads the status's meaning.)
4. `can_serve_task` takes a `ContainerConfig` (cube.container), but
   `InfraConfig` lives in cube.resource. Confirm the import direction is
   clean, or whether the check should live as a free function /
   `ContainerConfig.is_served_by(infra)` to avoid a resource→container
   dependency. (Implementation detail; doesn't change the design.)
