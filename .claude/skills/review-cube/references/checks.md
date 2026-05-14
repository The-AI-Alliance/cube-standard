# Review checklist

Walk every applicable section. Each bullet is either a **Blocking** check (B) or a **Suggestion** (S-75 / S-50 / S-25). Skip a section only if it isn't in scope (e.g., no YAML present).

## 1. Registry YAML

Applies when the input is a YAML path, a cube-registry PR, or when a `cube-registry-entry.yaml` is present in the cube package dir.

**Required fields present (B):**
`id`, `name`, `version`, `description`, `package`, `authors`, `legal`, `paper`, `getting_started_url`, `tags`, `max_concurrent_tasks`, `parallelization_mode`.

(Note: `supported_infra` is optional for cubes that run locally or in Docker — required only for VM-based cubes.)

**Value validation (B):**
- `id` matches `^[a-z0-9]+(-[a-z0-9]+)*$` (kebab-case).
- `version` exactly matches the project version at `dev_install_url` (or the published PyPI version of `package`).
- `legal.wrapper_license` is a valid SPDX identifier.
- `tags` ⊆ `{web, coding, os, gui, mobile, science, math, multi-agent, desktop, multimodal, nlp, reasoning, robotics, games, swe, docker, terminal, browser}` — enforced by cube-registry's JSON Schema, so violations block at CI.
- `supported_infra` (if present) ⊆ `{aws, azure, gcp, local}` — enforced by JSON Schema. Field is optional for cubes that run locally or in Docker without cloud infra; only flag missing-but-needed if the cube clearly requires cloud VMs.
- `parallelization_mode` ∈ `{sequential, task-parallel, benchmark-parallel}` — enforced.
- `legal.notices[].type` (if present) ∈ `{third_party_data, software_registration, live_website_clone, attribution}` — enforced.
- `dev_install_url` starts with `git+https://github.com/` or `git+https://gitlab.com/`.
- `legal.benchmark_license.source_url` returns HTTP 200 (short timeout; if network unavailable, downgrade to S-75 with explanation).
- CI-derived fields NOT filled: `status`, `task_count`, `has_debug_task`, `has_debug_agent`, `resources`, `features.*`, `stress_results_url`.

**Suggestions:**
- S-75: `legal.notices[]` missing but the package reads like a wrapper around third-party data or a live site (heuristic: README or description mentions "dataset from X", a public benchmark name, or a live platform).
- S-50: `authors[].github` missing for any author.
- S-25: `description` is a single short phrase with no context about what the benchmark tests.

## 2. Cube code — static

**Terminology:**
- **Option A (inline ClassVars)** — `benchmark_metadata` and `task_metadata` are declared directly as `ClassVar`s on the `BenchmarkConfig` subclass.
- **Option B (file-loaded)** — ClassVars are omitted; the framework auto-loads `benchmark_metadata.json` and/or `task_metadata.json` from the same directory as the benchmark module.

Detect which the cube uses by grepping the benchmark module for the ClassVar assignments; if absent, the cube is Option B and the JSON files must exist.

**Blocking:**
- No class inheriting from `BenchmarkConfig` found in the package source.
- No `[project.entry-points."cube.benchmarks"]` table in `pyproject.toml`, or the target import path doesn't resolve to the `BenchmarkConfig` subclass.
- No `@tool_action` decorators found in the tool module(s). (Exception: the benchmark is a thin subclass of a shared tool like `cube-browser-tool` and inherits actions — verify the imported tool class has `@tool_action` methods.)
- Option B chosen but `task_metadata.json` missing (or `benchmark_metadata.json` missing when also needed).
- Option A chosen but required ClassVars (`benchmark_metadata`, `task_metadata`, `task_config_class`, `benchmark_class`) missing.
- `task_metadata.json` average size per task exceeds **1000 bytes**: `size_bytes / num_tasks > 1000` (target is ~1 KB/task, so ~1 MB for 1000 tasks). Recommend declaring a typed `TaskExecutionInfo` subclass for the heavy fields, populating the on-disk cache via `BenchmarkConfig.install()`, and hydrating `Task.execution_info` lazily inside `TaskConfig.make()` (see `cube-harness/cubes/swebench-live-cube`).
- Any `TaskMetadata` subclass declares a field literally named `extra_info` (or any `dict[str, Any]` bag-of-anything field). For per-task fields, declare named typed fields on the subclass.
- Heavy per-task data (problem statements, patches, archives, evaluator scripts, …) appears as fields on a `TaskMetadata` subclass instead of on a typed `TaskExecutionInfo` subclass surfaced via `Task.execution_info`. Heavy data must live on `TaskExecutionInfo`, not `TaskMetadata`.

**Suggestions:**
- S-75: Tool code in `tool.py` reimplements a surface that an existing `cube-tools/*` package already provides (browser navigation, mouse/keyboard, web search, …) instead of importing/subclassing it. Per [`openspec/specs/tool/spec.md` § Packaging conventions](../../../../openspec/specs/tool/spec.md#packaging-conventions), generalist tools live in `cube-standard`; cubes consume or subclass them. If the cube needs a new generalist primitive, raise it for upstream rather than embedding it here.
- S-75: `Task.reset()` has no visible call to `self.tool.reset()`.
- S-75: `Task.evaluate()` appears to mutate state (writes to `self.tool._env.*`, calls tool action methods, writes to `self._runtime_context`). `evaluate()` must be pure.
- S-75: `_setup()` / `install()` / `close()` appear non-idempotent (no early-return guard, no `if self._already_setup: return`, destructive ops unconditionally).
- S-75: Option B with `task_metadata.json` committed but no metadata generator script at the repo root (`scripts/create_task_metadata.py` or `scripts/generate_task_metadata.py`). Without one, the metadata can't be regenerated if the upstream source changes.
- S-75: Bulk data files committed inside the package source (e.g. `src/<pkg>/data/`, `src/<pkg>/assets/`). Heavy data should be auto-downloaded by the generator script into `benchmark.cache_dir()` (typically `~/.cube/<benchmark-id>/`), not shipped in-tree — committing bloats wheels and makes the regeneration path opaque.
- S-50: `tool` property override or `isinstance(self.tool, FooTool)` asserts — drop via `class FooTask(Task[FooMeta, FooTool])`. See [task/spec.md](../../../../openspec/specs/task/spec.md).
- S-50: If Option B, the generator script (`scripts/create_task_metadata.py` or `scripts/generate_task_metadata.py`) exists but has no `--force` flag or no idempotency guard.
- S-50: Metadata-generation logic is inlined in `benchmark.py` (or another package module) rather than living in `scripts/*.py` at the repo root. Move it to the script so regeneration is explicit and reproducible.
- S-50: Multiple "split"-like fields or values appear in `TaskMetadata` but `BenchmarkMetadata.named_subsets` isn't declared.
- S-25: TODO placeholders left in `benchmark_metadata.description`, `TaskMetadata.abstract_description`, or anywhere in the source.
- S-25: Template boilerplate left in `tool.py` (`example_action` still present unchanged).
- S-25: `README.md` missing from the cube package root.

## 3. Compliance suite — dynamic

**Always run** (no confirmation): `pytest tests/` and `cube test <benchmark-id>`.

**Blocking:**
- `pytest tests/` exits non-zero.
- `cube test` fails any of: `test_debug_tasks_exist`, `test_debug_agent_exists`, `test_full_episode` (reward < 1.0 on any debug task), `test_tools_list`, `test_close_idempotent`, `test_benchmark_metadata`.
- Installation failed (recorded from step 2).

**Suggestions:**
- S-75 (advisory): `cube test`'s `test_reset_reproducibility` fails. Some benchmarks are legitimately stochastic (the opening observation depends on a random variable) and can't always pass this test. **Review the failure manually**: if the non-determinism is expected and acceptable, document it in the cube's README; otherwise fix the `reset()` determinism.
- S-50: `cube test` reports unusually high p99 latency relative to the benchmark's declared infra model (only if an obvious threshold is exceeded — don't manufacture one).
- S-50: `tests/` contains only the template smoke tests — no cube-specific unit tests.
- S-50: `debug.py` `_TASK_ACTIONS` looks like unmodified template boilerplate (e.g. a single placeholder action sequence that doesn't exercise the real benchmark tasks). Debug tasks must represent the actual benchmark dynamics.

## 4. Hygiene

**Blocking:**
- Committed credentials: API keys, passwords, `.env` with secrets, private key blocks. Grep for common patterns (`AKIA`, `sk-ant-`, `xoxb-`, `-----BEGIN`, `api_key = "..."` with a long non-placeholder value).

**Suggestions:**
- S-50: `pyproject.toml` description is the scaffold default (`"A CUBE benchmark package"`) or empty.
- S-50: No `README.md` at the repo / package root.
- S-25: `authors` list empty in `pyproject.toml`.
- S-25: No SPDX `LICENSE` file at the repo root, even though `legal.wrapper_license` is declared in the YAML.

## Finding format

Every finding is a dict / bullet with:
- **title** — one line
- **severity** — B / S-75 / S-50 / S-25
- **where** — file:line or spec section
- **why** — one or two sentences
- **fix** — suggested action (optional for S-25)

When in doubt on severity, prefer the lower bucket.
