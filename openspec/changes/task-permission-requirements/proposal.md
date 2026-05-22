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

### 2. Requirements at the benchmark level (primary) and task level (optional)

The **main expected use case** is blanket: "all of tbench2 needs root".
Investigating which individual tasks truly need root is too costly and
fragile, so a benchmark declares its requirement once:

```python
class BenchmarkConfig(TypedBaseModel, ABC):
    ...
    def requirements(self) -> set[str]:
        """Permission tokens EVERY task in this benchmark needs from its
        infra. Default empty = runs anywhere. Cubes with heterogeneous
        needs may instead declare per-task via ContainerConfig.requires
        and leave this empty; the effective requirement for a task is the
        union of the two."""
        return set()
```

```python
# cubes/terminalbench2-cube
class TerminalBench2BenchmarkConfig(BenchmarkConfig):
    def requirements(self) -> set[str]:
        return {"container:root"}
```

A per-task override stays available for genuinely heterogeneous
benchmarks (kept from rev 1):

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

Effective requirement for a task = `benchmark.requirements() ∪
task.container_config.requirements()`.

### 3. The gate: `Benchmark.setup(infra, on_incompatible=...)` — three modes

```python
OnIncompatible = Literal["raise", "skip", "force"]
```

Resolved at experiment-setup time (default `"raise"`):

- **`"raise"`** (default — the main use case). At `setup`, compute the
  benchmark-level `missing = benchmark.requirements() - infra.capabilities()`.
  If non-empty, **raise `IncompatibleInfraError` immediately, before any
  episode is created**. tbench2 on toolkit stops at setup with:
  `"ToolkitInfraConfig cannot run terminalbench2-cube: missing
  {container:root}."` No episodes, no retries, no spend.

- **`"skip"`**. Run the compatible subset. A task whose effective
  requirement isn't met is **not launched**; its episode is recorded with
  a terminal, **non-retriable** status (see §4) and excluded from the
  accuracy denominator. Reported as `n_skipped` in the summary. Useful
  when a benchmark is heterogeneous (per-task requirements) and you want
  the runnable overlap.

- **`"force"`** (try-anyway override). Ignore the check entirely; launch
  every task regardless. The escape hatch for probing whether a new image
  policy / infra change has obviated a stale requirement — e.g. "did
  toolkit start allowing root? run tbench2 with `force` and see." Logs a
  warning per forced task so the override is visible in the trace.

Where the mode lives: a field on the experiment-run path (cube-harness
`Experiment` / `run_*`), forwarded into `Benchmark.setup`. cube-standard
defines the enum + the `setup` parameter; cube-harness wires the CLI/recipe
knob. A given infra MAY also pin a default (e.g. `ToolkitInfraConfig` could
default callers to `"raise"`), but the per-run setting wins.

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

### 5. Static compatibility preview (unchanged from rev 1)

```python
@classmethod
def compatibility_report(cls, infra: InfraConfig) -> CompatibilityReport: ...
```

Reads metadata only (no container launches) so `cube list --infra=toolkit`
and CI gates can show "tbench2: incompatible on toolkit (needs
container:root)" before anyone spends a dollar.

## Migration impact

Backward compatibility is the default:

- `BenchmarkConfig.requirements()` defaults to `set()`; `ContainerConfig.requires`
  defaults to `set()`. No benchmark is newly gated by this PR alone.
- Existing `InfraConfig.capabilities()` implementations are unchanged.
- `Benchmark.setup` gains an `on_incompatible="raise"` parameter with a
  default that is a **no-op when `requirements()` is empty** — so existing
  cubes see no behaviour change.

Behaviour changes only when a cube declares `requirements()` AND a paired
infra-side PR teaches root-capable infras to publish the matching tokens.
Recommended rollout (follow-up PRs, not this RFC):

1. Land this RFC (vocabulary + types + gate behaviour + status mapping).
2. cube-resources: `LocalInfraConfig`, `DaytonaInfraConfig`, `AWSInfraConfig`,
   `AzureInfraConfig` publish `container:root`. `ToolkitInfraConfig` does not.
3. cube-harness: `terminalbench2-cube` declares
   `requirements() = {"container:root"}`. tbench2 on toolkit now fails fast
   at setup with a clear message.
4. cube-harness: wire the `on_incompatible` knob into `Experiment` / `run_*`
   and the episode-status `INVALID_CONFIG` mapping for skip mode.
5. Tooling: `cube list --infra=<name>` and CI dashboards display the
   compatibility matrix.

## Alternatives considered

- **Per-task-only requirements (rev 1's primary framing).** Rejected as the
  *primary* mechanism — the main use case is blanket-per-benchmark, and
  per-task investigation is too costly. Kept as an optional override for
  heterogeneous benchmarks.
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
