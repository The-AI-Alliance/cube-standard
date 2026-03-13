# Migration Plan: VM Lifecycle → Task-Owned (Parallel to Container)

**Date**: 2026-03-12
**Status**: Draft
**Prerequisite**: RFC required — this changes `ToolConfig.make()` signature (core API)

---

## Goal

Move VM lifecycle ownership from `ComputerConfig.make()` into `Task.model_post_init`,
mirroring how containers are already handled. After this migration:

- `ComputerConfig` contains zero VM lifecycle logic — it is pure config
- `Task` owns both VM and container creation symmetrically
- `ToolConfig.make(vm, container)` receives fully-resolved resources
- Swapping local ↔ cloud VM requires changing only the `Benchmark` constructor

---

## What breaks

### Tier 1: Core API change (all `ToolConfig` implementors)

**`ToolConfig.make()` signature changes** from:
```python
def make(self, container: Container | None = None) -> AbstractTool
```
to:
```python
def make(self, vm: VM | None = None, container: Container | None = None) -> AbstractTool
```

Every class that implements `ToolConfig.make()` must be updated, even if it ignores `vm`.

Files affected:
- `cube-standard/src/cube/tool.py` — base class signature
- `cube-standard/src/cube/tools/browser.py` — `BrowserToolConfig.make()`
- `cube-standard/cube-tools/cube-browser-tool/src/cube_browser_tool/playwright_tool.py` — `PlaywrightToolConfig.make()`
- `cube-standard/cube-tools/cube-computer-tool/src/cube_computer_tool/computer.py` — `ComputerConfig.make()`
- `cube-standard/examples/counter-cube/src/counter_cube/tool.py` — example `ToolConfig.make()`
- `cube-standard/examples/counter-cube/src/counter_cube/pluggable_tool.py` — pluggable variant
- `cube-standard/src/cube/_template/new_cube_package/src/cube_package/tool.py` — template
- `cubes/osworld-cube/src/osworld_cube/computer.py` — `ComputerConfig.make()`
- `cubes/arithmetic-cube/src/arithmetic_cube/tool.py` — `ToolConfig.make()`
- Any downstream cube packages not in this repo

### Tier 2: `Task.model_post_init` (one file, new vm_backend field)

**`Task`** gains a `vm_backend` field and must launch the VM before calling `make()`.

File: `cube-standard/src/cube/task.py`

```python
# Before
def model_post_init(self, __context):
    if self.container_backend and self.metadata.container_config:
        self._container = self.container_backend.launch(self.metadata.container_config)
    self._tool = self.tool_config.make(container=self._container)

# After
def model_post_init(self, __context):
    if self.container_backend and self.metadata.container_config:
        self._container = self.container_backend.launch(self.metadata.container_config)
    if self.vm_backend and self.metadata.vm_config:
        self._vm = self.vm_backend.launch(self.metadata.vm_config)
    self._tool = self.tool_config.make(vm=self._vm, container=self._container)
```

New field on `Task`:
```python
vm_backend: VMBackend | None = Field(default=None)
_vm: VM | None = PrivateAttr(default=None)
```

New field on `TaskMetadata` (already has `container_config`, add `vm_config`):
```python
vm_config: VMConfig | None = Field(default=None)
```

### Tier 3: `ComputerConfig` loses `vm_backend` and `vm_config` fields

**`cube_computer_tool.ComputerConfig`** currently holds `vm_backend` and `vm_config`.
After migration these move to `Task` / `TaskMetadata`. `ComputerConfig` becomes:

```python
class ComputerConfig(ToolConfig):
    require_a11y_tree: bool = True
    require_terminal: bool = False
    observe_after_action: bool = True
    os_type: str = "ubuntu"            # kept — tool-level config, not VM config

    def make(self, vm: VM | None = None, container: Container | None = None) -> Computer:
        assert vm is not None, "ComputerConfig requires a VM — pass vm_backend to Task/Benchmark"
        return Computer(config=self, vm=vm, evaluator_container=container)
```

`osworld_cube.computer.ComputerConfig` (the thin wrapper) is similarly simplified.

### Tier 4: `Benchmark` gains `vm_backend` field, passes it to `TaskConfig.make()`

**`Benchmark`** currently has `container_backend`. Add `vm_backend` parallel field:

```python
class Benchmark(TypedBaseModel, ABC):
    container_backend: ContainerBackend | None = None
    vm_backend: VMBackend | None = None          # new
    default_tool_config: ToolConfig | None = None
```

**`Benchmark.spawn()`** and `TaskConfig.make()` pass `vm_backend` through:

```python
# Benchmark.spawn()
task = task_config.make(
    runtime_context=self._runtime_context,
    container_backend=self.container_backend,
    vm_backend=self.vm_backend,            # new
)
```

**`TaskConfig.make()`** signature:
```python
@abstractmethod
def make(
    self,
    runtime_context: RuntimeContext | None = None,
    container_backend: ContainerBackend | None = None,
    vm_backend: VMBackend | None = None,           # new
) -> Task: ...
```

### Tier 5: `OSWorldBenchmark` and `OSWorldTaskConfig`

**`OSWorldBenchmark`** currently puts `vm_backend` inside `ComputerConfig`:
```python
# Before
bench = OSWorldBenchmark(
    default_tool_config=ComputerConfig(
        vm_backend=LocalQEMUVMBackend(...),
    )
)
```

After migration:
```python
# After
bench = OSWorldBenchmark(
    vm_backend=LocalQEMUVMBackend(...),      # harness-owned
    default_tool_config=ComputerConfig(),    # pure config, no backend
)
```

`OSWorldTaskConfig.make()` passes `vm_backend` through to `OSWorldTask`.
`OSWorldTask` (which is `Task`) launches the VM in `model_post_init`.

OSWorld tasks need `vm_config` in `TaskMetadata` for per-task snapshot names.
Currently this lives in `metadata.extra_info["snapshot"]` and is read inside
`setup_task()`. After migration it can be promoted to `metadata.vm_config`:

```python
# TaskMetadata for an OSWorld task
TaskMetadata(
    id="chrome_1234",
    vm_config=VMConfig(snapshot_name="init_state"),   # was extra_info["snapshot"]
    container_config=ContainerConfig(...),             # evaluator container (unchanged)
    extra_info={...},
)
```

This is a one-line change per task in `OSWorldBenchmark._load_task_metadata_from_repo()`.

### Tier 6: `Benchmark.setup()` warning logic (Change 2 from cua_infra_rfc.md)

The false-positive warning for VM-only benchmarks (`container_backend is None`) should
be updated as part of this migration since `vm_backend` is now a valid substitute:

```python
# After
if self.container_backend is None and self.vm_backend is None and self.default_tool_config is None:
    logger.warning("Benchmark setup defined no container backend, no VM backend, and no default tool config.")
```

---

## What does NOT break

- `cube.container` — untouched
- `BrowserToolConfig.make()` — adds `vm: VM | None = None` kwarg, ignores it. One line.
- All test infrastructure (mock tools) — add `vm=None` to their `make()` signatures
- OSWorld task logic (`task.py`, `evaluate()`, `reset()`) — unchanged, `self._computer` still works
- Serialization — `vm_backend` on `Benchmark` serializes like `container_backend` already does

---

## Migration Steps

### Step 1 — Add `vm_backend` to `Task` and `TaskMetadata` (non-breaking addition)

- Add `vm_backend: VMBackend | None = None` to `Task`
- Add `vm_config: VMConfig | None = None` to `TaskMetadata`
- Add `_vm: VM | None = PrivateAttr(default=None)` to `Task`
- `model_post_init` launches VM if both `vm_backend` and `metadata.vm_config` are set
- `ToolConfig.make()` signature stays `(container=None)` at this step

**Breakage**: none. `vm_backend=None` is the default.

### Step 2 — Add `vm_backend` to `Benchmark` and thread it through `spawn()` / `TaskConfig.make()`

- Add `vm_backend: VMBackend | None = None` to `Benchmark`
- Update `Benchmark.spawn()` to pass `vm_backend`
- Update `TaskConfig.make()` abstract signature to include `vm_backend=None`
- Update all `TaskConfig.make()` implementations to accept and pass `vm_backend`

**Breakage**: any external `TaskConfig.make()` implementations without `vm_backend` kwarg.
Mitigated by: `vm_backend=None` default → old implementations still work, just get a deprecation warning.

### Step 3 — Change `ToolConfig.make()` to `make(vm=None, container=None)`

- Update `ToolConfig.make()` abstract signature
- Update all implementations: browser tool, computer tool, examples, template, arithmetic cube
- `Task.model_post_init` now passes both: `self.tool_config.make(vm=self._vm, container=self._container)`

**Breakage**: any external `ToolConfig.make(container)` positional call breaks. All calls must be keyword args. Grep for `.make(` in downstream repos before releasing.

### Step 4 — Strip `vm_backend` / `vm_config` from `ComputerConfig`

- Remove `vm_backend: VMBackend` and `vm_config: VMConfig` from `ComputerConfig`
- `ComputerConfig.make(vm, container)` asserts `vm is not None`
- Update `OSWorldBenchmark` to pass `vm_backend=LocalQEMUVMBackend(...)` at Benchmark level

**Breakage**: anyone constructing `ComputerConfig(vm_backend=...)` directly. This is the primary user-facing breaking change. A deprecation shim can ease transition:
```python
# Temporary shim in ComputerConfig (remove after one release)
def __init__(self, vm_backend=None, **kwargs):
    if vm_backend is not None:
        warnings.warn("vm_backend on ComputerConfig is deprecated; pass it to Benchmark instead", DeprecationWarning)
    super().__init__(**kwargs)
```

### Step 5 — Promote `extra_info["snapshot"]` to `metadata.vm_config.snapshot_name` in OSWorld

- Update `OSWorldBenchmark._load_task_metadata_from_repo()` and `_load_task_metadata_from_file()`
- `setup_task()` reads snapshot from `self._config.vm_config.snapshot_name` instead of `task_config["snapshot"]`

**Breakage**: none externally, but OSWorld task JSON format is no longer the source of truth for snapshot names.

---

## Summary table

| Step | Files changed | External breakage | Reversible? |
|------|--------------|-------------------|-------------|
| 1 | `task.py`, `container.py` | None | Yes |
| 2 | `benchmark.py`, all `TaskConfig.make()` impls | Downstream `TaskConfig` subclasses | Yes (default=None) |
| 3 | `tool.py`, 8+ `ToolConfig.make()` impls | Positional `.make(container)` call sites | No (keyword only) |
| 4 | `computer.py` (both), `benchmark.py` (osworld) | `ComputerConfig(vm_backend=...)` users | With shim |
| 5 | `benchmark.py` (osworld) | None | Yes |

**Recommended release strategy**: Steps 1–2 in one PR (additive), Steps 3–5 in a second PR (breaking, semver minor bump).
