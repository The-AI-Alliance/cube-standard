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
Keep the shipped JSON lean (~1 KB per task average — ~1 MB for 1000 tasks, ~2 MB for 2000 tasks). If you need more than that, declare a typed `TaskExecutionInfo` subclass for it, populate the on-disk cache one-time per worker environment via `BenchmarkConfig.install()` (writing one JSON per task to `cls.task_config_class.task_execution_cache_dir() / f"{task_id}.json"`), and hydrate `Task.execution_info` inside `TaskConfig.make()` by validating `cls.load_task_execution_info(self.task_id)` against the subclass. See `cube-harness/cubes/swebench-live-cube` for the canonical pattern; operators run `cube install <bench>` (Dockerfile / init container / shared volume) so workers never download lazily.

### Using stringly-typed dicts for per-task fields
For per-task metadata (repo, base_commit, instruction, splits, log_parser, container image, …), declare a `TaskMetadata` subclass with named typed fields. The base `TaskMetadata` accepts only the framework-defined fields; everything else goes on a subclass.

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
