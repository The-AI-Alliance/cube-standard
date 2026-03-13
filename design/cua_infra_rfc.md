# RFC: CUA Benchmark Infrastructure — Survey, Abstraction Critique, and Recommended Changes

**Date**: 2026-03-12
**Status**: RFC

---

## Overview

This document analyzes 7 recent CUA benchmarks (OSWorld-G, WindowsAgentArena, macOSWorld, SCUBA,
TheAgentCompany, OpenCUA, ScreenSuite) against the current cube-standard abstractions, identifies
gaps, and recommends a minimal set of changes. It should be read alongside [vm_backend.md](vm_backend.md),
which covers the full `VMBackend` implementation plan and cloud provider matrix.

---

## Part 1: What the Benchmarks Actually Need

### The Real Infrastructure Matrix

| Benchmark | Desktop VM? | Evaluator | Reset | Action Interface | Backend Variety |
|---|---|---|---|---|---|
| **OSWorld / OSWorld-G** | Yes (Ubuntu) | Separate container | Container restart | PyAutoGUI via Flask | Local QEMU, AWS, GCP |
| **WindowsAgentArena** | Yes (Win11) | In-VM Flask server | Snapshot/restart | PyAutoGUI + UIA | Local QEMU, Azure ML |
| **macOSWorld** | Yes (macOS) | Task-specific | VMware snapshot / AWS | VNC → HTTP proxy | AWS mac2.metal, VMware |
| **SCUBA** | Yes (Linux) + Cloud | Cloud API | GCP restart + API reset | PyAutoGUI / Browser | GCP + Salesforce org |
| **TheAgentCompany** | No | In-task script | `init.sh` container restart | Browser + Shell | Local Docker Compose |
| **OpenCUA** | Yes (multi-OS) | AgentNet offline | Container restart | PyAutoGUI | Local QEMU |
| **ScreenSuite** | Mixed (see below) | Benchmark-specific | Per-mode (see below) | Mixed | Docker (OSWorld mode), HF datasets (offline) |

### ScreenSuite Deep-Dive

ScreenSuite is **not a single benchmark** — it is a meta-benchmark suite wrapping 17 configs across
13 datasets in three execution modes. The infrastructure needs vary dramatically by mode:

#### Mode 1: Live execution (OSWorld, AndroidWorld, GAIA, Mind2Web-Live, BrowseComp)

- OSWorld: Docker + QEMU (`happysixd/osworld-docker:latest`), requires `/dev/kvm`, 4 CPU / 4 GB RAM
- AndroidWorld: Android emulator via `android_world` package, one emulator per thread
- Web (GAIA, Mind2Web-Live, BrowseComp): Real browser sessions, URL-match or LLM-judge evaluation
- Reset: Per-process/thread isolation; OSWorld uses file-based resume (`result.txt` presence)

#### Mode 2: Single-step execution (MMind2Web, AndroidControl, Showdown Clicks)

- Predefined initial screenshots + target state; agent outputs one action
- No live VM/container needed — dataset-driven

#### Mode 3: Offline perception (ScreenSpot v1/v2/Pro, ScreenQA, WebSrc, VisualWebBench)

- Pure inference on static HuggingFace datasets
- No VM, no container, no reset — just image + LLM → evaluate

#### Consequence for cube-standard mapping

- ScreenSuite's OSWorld sub-benchmark maps identically to Pattern A/B below
- The offline perception modes need only a `ToolConfig` that wraps an LLM call — no infrastructure
- The web modes need a `BrowserToolConfig` — same as MiniWob/WorkArena

### Four Distinct Infrastructure Patterns

#### Pattern A: VM-only (OSWorld, WindowsArena, macOSWorld, OpenCUA)

- One VM per task. VM contains both the desktop and the Flask control server.
- Reset = restore VM to snapshot (container restart for QEMU, native snapshot for VMware/AWS).
- Evaluator runs *inside* the VM (via Flask endpoint) or is offline (grounding benchmarks).

#### Pattern B: VM + Evaluator Container (OSWorld with evaluator)

- VM contains the desktop. A **separate** Docker container runs the evaluator.
- Evaluator container connects to the VM via the VM's HTTP endpoint.
- Two independent lifecycles: VM restarts per task; container may stay warm across tasks.

#### Pattern C: Containers only (TheAgentCompany, web benchmarks)

- No VM. Docker containers simulate services (GitLab, RocketChat, etc.).
- Agent accesses via browser or shell. Evaluation is deterministic (file checks, API state).
- Some containers are *shared* across all tasks (the "server stack"); others are *per-task*.

#### Pattern D: No live infrastructure (ScreenSuite offline/single-step modes, OSWorld-G grounding)

- Static datasets on HuggingFace. Agent receives image(s), emits prediction, score computed offline.
- No reset, no container, no VM. `ToolConfig` wraps dataset iteration + metric computation.

---

## Part 2: Current Abstractions — What Works and What Doesn't

### What Works Well

1. **`ToolConfig.make(container=None)`** — clean factory with optional container injection. Correct level.
2. **`ContainerConfig / ContainerBackend / Container`** (what/how/handle) — clean pattern, well-implemented.
3. **`@tool_action` + auto-discovery** — elegant; no boilerplate.
4. **`RuntimeContext = dict[str, Any]`** — flexible escape hatch for shared infrastructure (TheAgentCompany).
5. **`Task.model_post_init`** launching container then calling `tool_config.make(container)` — simple, correct flow.

### What's Broken or Missing

#### Problem 1: `ComputerConfig` doesn't extend `ToolConfig` and `make()` ignores `container`

```python
# Current — ComputerConfig is a TypedBaseModel, not a ToolConfig
class ComputerConfig(TypedBaseModel):
    def make(self) -> "Computer":           # no container arg!
        vm = self.vm_backend.launch(self.vm_config)
        return Computer(config=self, vm=vm)
```

`Task.model_post_init` calls `self.tool_config.make(container=self._container)` but
`ComputerConfig.make()` accepts no arguments, so the evaluator container (launched by Task)
is silently dropped and never reaches the Computer tool. The VM and evaluator container
can never communicate.

#### Problem 2: `VMConfig.snapshot_name` default is in the wrong place

`VMConfig.snapshot_name = "init_state"` is benchmark-level config, but:
- Different backends use different snapshot naming conventions.
- Per-task snapshot overrides have no clean typed path (workaround: `extra_info["snapshot"]`).
- The "default snapshot" is a backend concern, not a benchmark concern.

#### Problem 3: Benchmark warns falsely for VM-only benchmarks

```python
if self.container_backend is None:
    logger.warning("Benchmark initialization did not define a container backend.")
```

VM-only benchmarks (macOSWorld, WindowsArena) legitimately have no `container_backend`.
This warning is noise that makes valid configurations look broken.

#### Problem 4: No typed path for shared services

TheAgentCompany needs 175 containers running before any task starts. These shared services
go into `RuntimeContext` (an untyped `dict`). There's no typed contract for a task to
declare "I need services X, Y, Z to be alive," making harness validation impossible.

---

## Part 3: Design Options

### Option 1: Minimal Patch (KISS) ✅ Recommended

Fix only the immediate breakage: make `ComputerConfig` a proper `ToolConfig` and thread
the evaluator container through.

```python
# cube-computer-tool/computer.py
class ComputerConfig(ToolConfig):              # was TypedBaseModel
    vm_backend: VMBackend
    vm_config: VMConfig = Field(default_factory=VMConfig)
    require_a11y_tree: bool = True
    require_terminal: bool = False
    observe_after_action: bool = True

    def make(self, container: Container | None = None) -> "Computer":
        vm = self.vm_backend.launch(self.vm_config)
        return Computer(config=self, vm=vm, evaluator_container=container)

class Computer(Tool):
    def __init__(
        self,
        config: ComputerConfig,
        vm: VM,
        evaluator_container: Container | None = None,
    ) -> None:
        self._config = config
        self._vm = vm
        self._evaluator_container = evaluator_container
        ...
```

**Two files. ~16 lines. No new abstractions.**

### Option 2: Typed Services Registry

Replace the untyped `RuntimeContext` dict with a typed `Services` object:

```python
class Services(TypedBaseModel):
    containers: dict[str, Container] = {}
    vm: VM | None = None
    extra: dict[str, Any] = {}          # escape hatch preserved
```

**Pro**: Type-checked service lookup. Benchmarks declare dependencies explicitly.
**Con**: Breaks existing `TaskConfig.make(runtime_context, container_backend)` signature (RFC needed).
**Verdict**: Defer. `RuntimeContext = dict[str, Any]` is adequate for now.

### Option 3: `TaskInfraRequirements` in `TaskMetadata`

```python
class TaskInfraRequirements(TypedBaseModel):
    snapshot_name: str | None = None        # per-task snapshot override
    required_services: list[str] = []       # service names from Benchmark._services

class TaskMetadata(TypedBaseModel):
    ...
    infra: TaskInfraRequirements = Field(default_factory=TaskInfraRequirements)
```

**Pro**: Harness can validate requirements before starting. Clean snapshot override path.
**Con**: Adds infrastructure concerns into metadata (mixing layers). More boilerplate.
**Verdict**: Defer. `extra_info["snapshot"]` convention works; document it instead.

### Option 4: Full `InfraProvider` Protocol

Define `InfraProvider` as a typed abstraction replacing both `container_backend` and
`runtime_context` in `TaskConfig.make()`:

```python
class InfraProvider(Protocol):
    def get_vm(self, task_id: str) -> VM | None: ...
    def get_container(self, name: str) -> Container | None: ...
    def get_extra(self, key: str) -> Any: ...
```

**Pro**: Clean, typed, enables VM pooling later.
**Con**: Large breaking change. RFC-required. Over-engineered for current needs.
**Verdict**: Defer until VM pooling is a real requirement.

---

## Part 4: Recommended Changes

### Change 1: Fix `ComputerConfig` (Priority: High)

**File**: `cube-standard/cube-tools/cube-computer-tool/src/cube_computer_tool/computer.py`

- Change `ComputerConfig(TypedBaseModel)` → `ComputerConfig(ToolConfig)`
- Change `make(self)` → `make(self, container: Container | None = None)`
- Add `evaluator_container: Container | None = None` to `Computer.__init__`

This fixes the OSWorld evaluator container path without any new abstractions.

### Change 2: Soften Benchmark Warnings (Priority: Medium)

**File**: `cube-standard/src/cube/benchmark.py`

```python
# Before (false positive for VM-only benchmarks):
if self.container_backend is None:
    logger.warning("Benchmark initialization did not define a container backend.")
if self.default_tool_config is None:
    logger.warning("Benchmark initialization did not define a default tool config.")

# After (only warn if nothing is configured at all):
if self.container_backend is None and self.default_tool_config is None:
    logger.warning("Benchmark setup defined no container backend and no default tool config.")
```

### Change 3: Make `VMConfig.snapshot_name` Optional (Priority: Low)

**File**: `cube-standard/src/cube/vm.py`

```python
class VMConfig(TypedBaseModel):
    snapshot_name: str | None = None    # None = use backend's default
    screen_size: tuple[int, int] = (1920, 1080)
```

Backends own their default snapshot names. Tasks can set `snapshot_name` when they need
a specific snapshot (OSWorld tasks with per-task setup), otherwise the backend default applies.

---

## Part 5: What to Defer (and Why)

| Feature | Why Defer |
|---|---|
| Typed `Services` replacing `RuntimeContext` | `dict[str, Any]` works. Typing it is nice-to-have, not blocking. |
| `InfraProvider` protocol | No benchmark needs VM pooling today. |
| `TaskInfraRequirements` in `TaskMetadata` | `extra_info["snapshot"]` convention is adequate. Document it. |
| VNC / ADB / non-PyAutoGUI input backends | No cube benchmark currently uses non-PyAutoGUI desktop control. |
| VM Pool Manager | Future optimization for high-throughput RL data generation. |
| Cloud VM backends (AWS, Azure, GCP) | Implement in `cube-computer-tool`, not `cube-standard`. |

---

## Part 6: Does "What vs How" Hold?

`vm_backend.md` claims the core design principle is:

```
VMConfig  = WHAT the task needs   (benchmark-owned)
VMBackend = HOW to provision it   (harness-owned)
```

This mirrors the container model:

```
ContainerConfig  = WHAT (benchmark-owned, lives in TaskMetadata)
ContainerBackend = HOW  (harness-owned, set at Benchmark construction)
```

**The container case: the separation is genuine.**

The benchmark author writes `ContainerConfig(image="evaluator:latest", ram_gb=2)` into
`TaskMetadata`. The harness user sets `container_backend=LocalDockerBackend()` (or
`ModalBackend()`, or `KubernetesBackend()`). These are independent objects at different
levels. Swapping the backend requires zero changes to benchmark code.

**The VM case: the separation is weaker — and that's correct.**

A VM is not a *task-level* resource — it is a *tool-level* resource. Every task in OSWorld
uses the same Ubuntu VM type. The VM exists to serve the `Computer` tool, not the task.
Compare:

- A task's **evaluator container** varies per task family → naturally owned by `TaskMetadata`
- The **desktop VM** is the same for all tasks in a benchmark → naturally owned by `ToolConfig`

So `VMBackend` living *inside* `ComputerConfig` (a `ToolConfig`) is **not a design flaw** —
it correctly reflects the ownership tier. The harness user builds `ComputerConfig(vm_backend=...)`
and passes it as `Benchmark.default_tool_config`. Swapping local for cloud is just swapping
the `vm_backend` field:

```python
# Local development:
bench = OSWorldBenchmark(
    default_tool_config=ComputerConfig(vm_backend=LocalQEMUVMBackend(...)),
)

# Cloud scale:
bench = OSWorldBenchmark(
    default_tool_config=ComputerConfig(vm_backend=AWSQEMUVMBackend(...)),
)
```

Benchmark code is unchanged. The separation holds — it just lives at a different boundary
than the container case.

**The asymmetry that actually matters:**

| Resource | Owned by | Lives in | Swappable without benchmark changes? |
|---|---|---|---|
| Evaluator container | Benchmark (task-level) | `TaskMetadata.container_config` | Yes — swap `container_backend` on `Benchmark` |
| Desktop VM | Harness (tool-level) | `ComputerConfig.vm_backend` | Yes — swap field in `default_tool_config` |
| Browser | Harness (tool-level) | `BrowserToolConfig` fields | Yes — swap `default_tool_config` |
| Shared services | Benchmark (benchmark-level) | `_runtime_context` | Yes — `_setup()` creates them |

The key insight: **VMs are closer to browsers than to evaluator containers.** Both VMs and
browsers are the *environment the tool runs in* — not something a task declares as a requirement.

### What Change 1 does for this separation

Without Change 1, `ComputerConfig` is a `TypedBaseModel` — it cannot be passed as
`Benchmark.default_tool_config` because it doesn't satisfy the `ToolConfig` contract.
The harness user has no standard way to inject the VM backend at the benchmark level.

With Change 1, `ComputerConfig` becomes a proper `ToolConfig`:

```python
bench = OSWorldBenchmark(
    container_backend=LocalDockerBackend(),          # for evaluator container
    default_tool_config=ComputerConfig(              # for the desktop VM
        vm_backend=LocalQEMUVMBackend(...),
    ),
)
```

Both backends are now injected at the same level, through the same pattern, by the harness
user — and neither requires touching benchmark code to swap.

### Benchmark coverage

| Pattern | Benchmark | Mapping |
|---|---|---|
| A: VM-only | OSWorld, WindowsArena, macOSWorld | `VMBackend` inside `ComputerConfig` (`default_tool_config`). VM launches in `make()`. |
| B: VM + evaluator container | OSWorld with evaluator | Same as A. Evaluator container from `container_backend` passed via `Task → make(container)`. **(Requires Change 1)** |
| C: Containers only | TheAgentCompany | `ContainerBackend` on `Benchmark`. Shared services in `_runtime_context`. |
| C: Web browser | ScreenSuite (live web), MiniWob | `BrowserToolConfig` as `default_tool_config`. No VM needed. |
| D: No infrastructure | ScreenSuite (offline), OSWorld-G grounding | `ToolConfig.make()` returns a dataset-backed tool. No container, no VM. |

All four patterns are supported with the same `Benchmark` API. No new abstractions needed.

---

## Summary

| Issue | Severity | Fix | Est. Lines |
|---|---|---|---|
| `ComputerConfig` doesn't extend `ToolConfig`, ignores `container` arg | **High** | Change 1 | ~16 |
| False-positive warnings for VM-only benchmarks | **Medium** | Change 2 | ~5 |
| `VMConfig.snapshot_name` default in wrong place | **Low** | Change 3 | ~3 |
| Untyped `RuntimeContext` for shared services | Low | Defer | — |
| Non-PyAutoGUI input backends | Future | Defer | — |

**Total immediate work: ~24 lines across 3 files. No new abstractions. No new files.**
