# Common pitfalls

Preempt these during interview (phase 2) and implementation (phase 4).

## Top 5 — most frequent bugs

### 1. Forgetting `@tool_action`
Method runs, tests pass, but the agent never sees the action. Symptom: "my agent is doing nothing." Every agent-facing method needs the decorator.

### 2. `Task.evaluate()` mutating state
`evaluate()` runs many times per episode (after each step) and must be pure. If it clicks, writes, or mutates, you get non-reproducible runs.

### 3. `Task.reset()` not calling `self.tool.reset()`
Episodes carry state across runs. Reproducibility check in `cube test` will fail.

### 5. Debug agent not reaching `reward == 1.0`
The compliance suite is strict: every debug task must reach full reward. Off-by-one in the action sequence ⇒ failure.

## Subtler traps

### Heavy data inlined into task_metadata
Keep the shipped JSON lean (~1 byte per task average — ~1 MB for 1000 tasks, ~2 MB for 2000 tasks). If you need MB of per-task data, populate the cache via `BenchmarkConfig.install()`, then merge into `metadata.extra_info` inside `BenchmarkConfig.get_task_configs()` (override it to call `self.load_task_execution_info(task_id)` and stamp into each emitted `TaskConfig`). Workers receive fully-populated `TaskConfig`s and never touch disk. See `cube-harness/cubes/swebench-live-cube`.

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

### Reaching into ClassVars directly for subsetting
Use `config.subset_from_list([...])` — returns `self.model_copy(update={"task_ids": [...]})`. Don't mutate `task_metadata` directly or try to shadow ClassVars on an instance.

### Container leaks on construction failure
`Task.model_post_init` launches the tool/container eagerly. If construction fails afterward, containers may leak. Wrap extra setup in try/finally.

## User-intent signals to flag early (phase 2)

| Signal | Say |
|--------|-----|
| "each task needs its own Docker image" | Supported — set `container_config: ContainerConfig` on each `TaskMetadata` with the task-specific image, RAM, and CPU. `infra` flows from `BenchmarkConfig.make(infra)` into `runtime_context["infra"]`; `Task.model_post_init` picks it up and calls `launch_task_container()` automatically. Cubes override `_build_tool()` for any image-specific setup after the container is up. |
| "streaming audio / video" | Streaming actions and observations are not in the current protocol. Flag as extension work. |
| "multiple agents cooperating" | Multi-agent is on the `core-extensions` RFC roadmap but not landed. Can you model as a single agent with a richer action set? |
| "async tool actions" | `AsyncTool` exists but coverage is partial. Check `openspec/changes/core-extensions/` before committing. |
