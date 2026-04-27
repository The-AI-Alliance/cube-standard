# TODO checklist — per-layer guidance

Use this at the start of each layer in phase 4. TODOs mirror the current scaffold in `src/cube/_template/new_cube_package/`.

## Layer 1 — `tool.py`

### About the `CubeEnv` placeholder

`CubeEnv` in the template is a **usage example**, not a required class. It exists to show explicitly that the `Tool` holds a reference to some environment state. Your real "env" will be a browser page, a VM handle, an API client, a database connection, or whatever fits your benchmark. You are free to:

- **Remove `CubeEnv` entirely** and keep the state inline on your `Tool` subclass — often the clearest choice when you're reusing a shared tool like `cube-browser-tool` or `cube-computer-tool`.
- **Keep the `Env` pattern** if your tool has non-trivial state that benefits from separation.
- **Repurpose it** — rename and reshape it to match your benchmark's domain.

Whatever you pick, keep task-specific logic OUT of the tool; put it in `Task.evaluate()`.

### TODOs (in file order)

**`CubeEnv.__init__`** — `# TODO: initialise your environment state here.`

**`CubeEnv.reset`** — `# TODO: reset mutable state.`
> Called by `Tool.reset()` which is called by `Task.reset()`. Bring the env back to episode-start without reconstructing.

**`CubeToolConfig`** — `# TODO: add config fields, e.g. enable_some_action: bool`
> Only serializable fields that actually affect behavior.

**`example_action`** — `# TODO: implement action logic.` / `# TODO: add more @tool_action methods.`
> Each method is one agent-facing action. The docstring becomes the action description shown to the LLM — write it as if explaining the action to a new collaborator. Include a NumPy-style `Parameters` block; every param needs a non-empty description.

### Layer 1 checklist
- [ ] Every agent-facing method has `@tool_action`.
- [ ] `execute_action()` never raises — return a `StepError` via normal return.
- [ ] Tool config fields are JSON-serializable.
- [ ] If reusing a shared tool, import its config and pass it in — do not reimplement actions.

## Layer 2 — `task.py`

### TODOs

**`Task.reset`** — `# TODO: build a meaningful opening observation.`
> The opening obs must state the goal clearly enough that the agent knows what to do. Include grounding (URL, file paths, fixtures). Call `self.tool.reset()` FIRST.

**`Task.evaluate`** — `# TODO: inspect obs and self.tool to determine the reward of the current state.`
> Pure function. Read state from `self.tool` (or `self.tool._env`); do NOT mutate. Return `(reward ∈ [0.0, 1.0], info: dict)`.

**`Task.finished`** — `# TODO: return True when the goal is achieved.`
> Optional. Default False. Enables early termination so an episode doesn't run out `max_steps` after success.

### `TaskConfig.make()` note

The template's `CubeTaskConfig.make(runtime_context, container_backend)` currently carries `container_backend` because the current spec requires it. Do not strip this. Once harness #300 ships, the template will be updated.

### Custom `TaskMetadata` subclass

Feel free to define a custom subclass with domain-specific fields (repo, instruction, snapshot, container image, …). Pattern from `swebench-live-cube`:

```python
from cube.task import TaskMetadata

class MyTaskMetadata(TaskMetadata):
    repo: str
    base_commit: str
    splits: list[str]
    log_parser: str
```

**Keep it lightweight.** Target ~1 KB per task average (~1 MB for 1000 tasks, ~2 MB for 2000 tasks). Anything heavier goes on a typed `TaskExecutionInfo` subclass via the install pattern below.

### Heavy per-task data: typed `TaskExecutionInfo` + `BenchmarkConfig.install()`

If a task needs heavy data (problem statements, binaries, evaluation code, patches, archives), declare a `TaskExecutionInfo` subclass for it and populate `Task.execution_info` lazily on workers. Pattern:

1. Declare the typed shape next to your `TaskConfig`:
   ```python
   from cube.task import TaskExecutionInfo

   class MyExecutionInfo(TaskExecutionInfo):
       problem_statement: str
       patch: str
       test_patch: str
       fail_to_pass: list[str]
       pass_to_pass: list[str]
   ```
2. `BenchmarkConfig.install()` writes one JSON per task to
   `cls.task_config_class.task_execution_cache_dir() / f"{task_id}.json"`
   (HF download, repo clone, archive extraction, …). Operators run
   `cube install <bench>` once per worker environment (Dockerfile / init
   container / shared volume) — **the cache is NOT committed.**
3. `TaskConfig.make()` hydrates `Task.execution_info` from the cache:
   ```python
   class MyTaskConfig(TaskConfig):
       @classmethod
       def verify_installed(cls) -> None:
           if not list(cls.task_execution_cache_dir().iterdir()):
               raise RuntimeError("Run `cube install <bench>` first.")

       def make(self, runtime_context=None, container_backend=None):
           type(self).verify_installed()
           exec_info = MyExecutionInfo.model_validate(
               self.load_task_execution_info(self.task_id)
           )
           return MyTask(
               metadata=self.metadata,
               execution_info=exec_info,
               tool_config=self.tool_config or MyToolConfig(),
               runtime_context=runtime_context,
               container_backend=container_backend,
           )
   ```
4. The task reads typed fields directly: `self.execution_info.problem_statement`, `self.execution_info.patch`, etc. — autocomplete, validation, no string keys.

Reference: `cube-harness/cubes/swebench-live-cube/src/swebench_live_cube/`.

### Layer 2 checklist
- [ ] `reset()` calls `self.tool.reset()`.
- [ ] `evaluate()` is pure — grep for writes to `self.tool` / `self.tool._env`.
- [ ] Opening observation contains the goal text.
- [ ] `close()` releases task-scoped resources (and calls `super().close()`).
- [ ] If using a custom `TaskMetadata` subclass, only public lightweight fields — no heavy blobs.

## Layer 3 — `benchmark.py`

### TODOs

**`BenchmarkMetadata.description`** — `"TODO: describe what this benchmark tests"`
> One paragraph. Problem, domain, what the reward means.

**`TaskMetadata` entries** — `# TODO: add one TaskMetadata per task.`, `abstract_description="TODO: ..."`, `recommended_max_steps=...`
> Every task_id referenced in `debug.py` needs an entry. `abstract_description` is one sentence; agents see it.

### Option A vs Option B

- **Option A — inline ClassVars**: declare `benchmark_metadata` and `task_metadata` as ClassVars in `benchmark.py`. Good for small, hand-written task sets.
- **Option B — JSON files**: delete the ClassVar declarations; framework auto-loads from `benchmark_metadata.json` + `task_metadata.json` next to the module.

**Most cubes use Option B.** Don't mix — pick one and commit.

### Option B requires a `scripts/create_task_metadata.py` dev script

If the user picks Option B, **write this script before filling `benchmark.py`**. The script lives at the cube root in `scripts/` and is NOT shipped with the package (exclude from `pyproject.toml` packaging).

Requirements:
- Idempotent — by default skip if `task_metadata.json` already exists; `--force` to regenerate.
- Committed to the repo for reproducibility. Anyone can regenerate the metadata if the upstream source changes.
- Fetches from the real source (HF dataset, a cloned upstream repo, CSV, DB).
- **Auto-downloads bulk data** (datasets, archives, fixtures) into `benchmark.cache_dir()` — typically `~/.cube/<benchmark-id>/` — rather than committing it under `src/<pkg>/data/` or similar. In-tree data bloats wheels and hides the regeneration path.
- Writes lightweight public fields only — any heavy per-task data goes into the `install()`-populated cache, not the JSON.

Templates to mirror:
- `cube-harness/cubes/osworld-cube/scripts/create_task_metadata.py` (clones an upstream repo, walks JSON test sets)
- `cube-harness/cubes/swebench-live-cube/scripts/create_task_metadata.py` (downloads HF splits, writes lightweight fields only, heavy data deferred to `install()`)

Flow:
1. Co-design the script with the user.
2. Run it to produce `src/<pkg>/task_metadata.json`.
3. Only then fill `benchmark.py` — the framework auto-loads the JSON on first access.

### Multiple splits → `named_subsets`

If your benchmark has splits (train/val/test, or L1/L2/L3), keep **one** `task_metadata` set with all tasks and declare the splits on `BenchmarkMetadata.named_subsets`:

```python
benchmark_metadata = BenchmarkMetadata(
    ...,
    named_subsets={
        "train": lambda t: "train" in t.splits,
        "test":  lambda t: "test"  in t.splits,
    },
)
```

Users get `benchmark.subset_from_name("test")`. Do NOT ship separate metadata files per split.

### Idempotency is mandatory

`install()`, `_setup()`, and `close()` MUST be idempotent:
- `install()` — second call with outputs already present ⇒ no-op (unless `force=True`).
- `_setup()` — calling twice on the same instance ⇒ safe (guard with `if self._already_setup: return`).
- `close()` — calling twice ⇒ safe. The compliance suite calls `task.close()` twice as a check.

### Layer 3 checklist
- [ ] `benchmark_metadata` has non-empty name, version, description (no TODO strings).
- [ ] Every `task_metadata` key matches its `TaskMetadata.id`.
- [ ] `_setup()` populates `self._runtime_context` (or is a no-op for self-contained benchmarks).
- [ ] `close()` tears down whatever `_setup()` created, idempotently.
- [ ] If Option B: `scripts/create_task_metadata.py` exists, is idempotent, and has been run.
- [ ] If splits: declared as `named_subsets`, not separate files.
- [ ] If heavy per-task data: declared on a typed `TaskExecutionInfo` subclass, written to the per-task cache by `install()`, hydrated on workers in `TaskConfig.make()` via `cls.load_task_execution_info(self.task_id)`.

## Layer 4 — `debug.py`

### TODOs

**`_TASK_ACTIONS`** — `# TODO: add one entry per task defined in CubeBenchmarkConfig.task_metadata.`
> Hard-code the action sequence that solves each task. The debug agent replays these. Every sequence must reach `reward == 1.0`.

### Layer 4 checklist
- [ ] One `_TASK_ACTIONS` entry per task in the debug subset.
- [ ] `get_debug_benchmark()` takes **no arguments** and returns a `BenchmarkConfig` (full set, or `config.subset_from_list([...])` for a tiny debug-only set).
- [ ] `make_debug_agent(task_id)` returns a callable; raises on unknown task_id.

## Layer 5 — `pyproject.toml`

- [ ] `project.name` matches the benchmark id.
- [ ] `project.description` is a real sentence.
- [ ] `project.authors` has real names + emails.
- [ ] `[project.entry-points."cube.benchmarks"]` key = benchmark id, value = `"<module>.benchmark:<ClassName>"`.
- [ ] `scripts/` directory excluded from package build (Option B only).

Verify with `cube list` — the benchmark should appear.

## Layer 6 — `tests/`

Template smoke tests validate:
- metadata non-empty, no `TODO` strings left
- `task_metadata` keys match `TaskMetadata.id`
- every debug task reaches `reward == 1.0`

Run `pytest tests/` before `cube test`.
