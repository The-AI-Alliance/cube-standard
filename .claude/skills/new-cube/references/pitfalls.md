# Common pitfalls

Preempt these during interview (phase 2) and implementation (phase 4).

## Top 5 — most frequent bugs

### 1. Forgetting `@tool_action`
Method runs, tests pass, but the agent never sees the action. Symptom: "my agent is doing nothing." Every agent-facing method needs the decorator.

### 2. Putting task_metadata inside `TaskConfig`
Wrong:
```python
class MyTaskConfig(TaskConfig):
    metadata: TaskMetadata   # ← DO NOT DO THIS
```
Right:
```python
# benchmark.py
class MyBenchmark(Benchmark):
    task_metadata: ClassVar[dict[str, TaskMetadata]] = {"task-1": TaskMetadata(...)}

# task.py
class MyTaskConfig(TaskConfig):
    def make(self, runtime_context=None, container_backend=None):
        from my_cube.benchmark import MyBenchmark   # inside make(), not at module top
        metadata = MyBenchmark.task_metadata[self.task_id]
        ...
```
`TaskConfig` is the serialization payload; keep it lean. Metadata stays on the benchmark ClassVar.

### 3. `Task.evaluate()` mutating state
`evaluate()` runs many times per episode (after each step) and must be pure. If it clicks, writes, or mutates, you get non-reproducible runs.

### 4. `Task.reset()` not calling `self.tool.reset()`
Episodes carry state across runs. Reproducibility check in `cube test` will fail.

### 5. Debug agent not reaching `reward == 1.0`
The compliance suite is strict: every debug task must reach full reward. Off-by-one in the action sequence ⇒ failure.

## Subtler traps

### Module-top import of the Benchmark class in task.py
Circular import: `benchmark.py` imports `TaskConfig`; `task.py` imports `Benchmark` ⇒ import loop. Put the import **inside** `TaskConfig.make()`.

### Heavy data inlined into task_metadata
Keep the shipped JSON lean (< a few KB). If you need MB of per-task data, populate it via `Benchmark.install()` into an execution cache, then attach to `metadata.extra_info` lazily in `TaskConfig.make()`. See `cube-harness/cubes/swebench-live-cube`.

### Using `extra_info` for light structured fields
If you're reaching for `extra_info` to hold per-task metadata (repo, base_commit, instruction, splits, log_parser, container image, …), promote those to first-class fields on a custom `TaskMetadata` subclass instead. `extra_info` is reserved for heavy runtime data populated lazily via `install()`, not light structured metadata. `extra_info` should be empty in the shipped `task_metadata.json`.

### Bulk data committed under src/<pkg>/
Datasets, archives, fixtures, etc. should be auto-downloaded by `scripts/create_task_metadata.py` into `benchmark.cache_dir()` (typically `~/.cube/<benchmark-id>/`), not committed under `src/<pkg>/data/` or similar. In-tree data bloats wheels, complicates distribution, and hides the regeneration path.

### Shipping one metadata file per split
Don't. One `task_metadata.json` with all tasks; declare splits via `BenchmarkMetadata.named_subsets`.

### Non-idempotent `install()` / `_setup()` / `close()`
Second call must be safe. Especially `close()` — the compliance suite calls it twice on tasks.

### `_runtime_context` writes after setup
Populated in `Benchmark._setup()`. Treat it as read-only thereafter. Parallel runs see inconsistent state if tasks write to it.

### Rebuilding a tool from scratch when a shared one exists
Web-based? `cube-browser-tool` probably covers 90%. Don't reimplement `browser_click` / `browser_type` / `browser_goto`.

### Inlining credentials on a Config
`ToolConfig`, `TaskConfig`, `InfraConfig` are serialized across processes. Never put API keys or passwords on them. Resolve from env vars inside `make()`.

### ClassVar mutation via `subset_from_*`
Historical bug: `Benchmark.subset_from_list` used to mutate shared ClassVar state. Fixed via deepcopy. Don't reach into ClassVars yourself. #111 will move these onto `BenchmarkConfig`.

### Container leaks on construction failure
`Task.model_post_init` launches the tool/container eagerly. If construction fails afterward, containers may leak. Wrap extra setup in try/finally.

## User-intent signals to flag early (phase 2)

| Signal | Say |
|--------|-----|
| "each task needs its own Docker image" | Per-task Docker not yet supported. Tracked in cube-standard #111 and cube-harness #300. Can you start with a single shared image? |
| "streaming audio / video" | Streaming actions and observations are not in the current protocol. Flag as extension work. |
| "multiple agents cooperating" | Multi-agent is on the `core-extensions` RFC roadmap but not landed. Can you model as a single agent with a richer action set? |
| "async tool actions" | `AsyncTool` exists but coverage is partial. Check `openspec/changes/core-extensions/` before committing. |
