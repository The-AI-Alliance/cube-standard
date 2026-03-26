# Contributing to CUBE Standard

## Repo layout

```
src/cube/
  core.py        # Action, Content, Observation, EnvironmentOutput, TypedBaseModel
  tool.py        # AbstractTool, Tool, @tool_action, ToolConfig
  task.py        # Task, TaskMetadata, TaskConfig, STOP_ACTION
  benchmark.py   # Benchmark, BenchmarkMetadata — ClassVar-driven registry
  testing.py     # run_debug_suite(), assert_debug_tasks_reward_one()
  cli.py         # `cube init` / `cube test` entry points
  server.py      # FastAPI wrapper around a Task (REST ↔ cube protocol)
  containers.py  # ContainerBackend / Container abstractions
  _template/     # Scaffolded by `cube init` — keep in sync with core
examples/counter-cube/   # Canonical reference implementation — read this first
```

## Setup

```sh
uv sync --all-extras          # install dev deps
pre-commit install --hook-type pre-commit --hook-type commit-msg
```

```sh
make lint    # ruff check + format (auto-fix)
make test    # pytest tests/
```

All commits need a DCO sign-off: `git commit -s -m "..."`.
See `README.md` for license and community details.

## The five layers

Read the module docstrings — they are the authoritative spec:

| Layer | File | What to implement |
|-------|------|-------------------|
| **Core** | [`core.py`](src/cube/core.py) | Nothing (shared types only) |
| **Tool** | [`tool.py`](src/cube/tool.py) | Subclass `Tool`, decorate actions with `@tool_action`, implement `ToolConfig.make()` |
| **Task** | [`task.py`](src/cube/task.py) | `reset()` and `evaluate()` are the two abstract methods; `finished()` / `filter_actions()` are optional hooks |
| **Benchmark** | [`benchmark.py`](src/cube/benchmark.py) | Define three `ClassVar`s (`benchmark_metadata`, `task_metadata`, `task_config_class`); implement `_setup()` / `close()` |
| **Debug** | [`testing.py`](src/cube/testing.py) | Add a `debug.py` that exposes `get_debug_benchmark()` and `make_debug_agent()` |

## Writing a new cube package

```sh
cube init my-env       # scaffold from _template/new_cube_package
cd my-env && uv sync
cube test my_env       # must reach reward == 1.0 on every debug task
```

Work through the `TODO` comments in this order:

1. [`tool.py`](src/cube/_template/new_cube_package/src/cube_package/tool.py) — define `CubeEnv`, add `@tool_action` methods
2. [`task.py`](src/cube/_template/new_cube_package/src/cube_package/task.py) — implement `reset()` and `evaluate()`
3. [`benchmark.py`](src/cube/_template/new_cube_package/src/cube_package/benchmark.py) — fill metadata (inline or via CSV/JSON)
4. [`debug.py`](src/cube/_template/new_cube_package/src/cube_package/debug.py) — write deterministic action sequences for each task

The counter-cube example covers all four layers concretely:
- Tool with conditional `action_set` filtering → [`examples/counter-cube/src/counter_cube/tool.py`](examples/counter-cube/src/counter_cube/tool.py)
- Partial-progress reward and early termination → [`examples/counter-cube/src/counter_cube/task.py`](examples/counter-cube/src/counter_cube/task.py)
- Hardcoded debug sequences → [`examples/counter-cube/src/counter_cube/debug.py`](examples/counter-cube/src/counter_cube/debug.py)

## Metadata: inline vs file-based

The `benchmark.py` template explains both options in its docstring. Short version:

- **Inline `ClassVar`** — fine for small benchmarks; define `benchmark_metadata` and `task_metadata` directly in the class body.
- **CSV / JSON files** — drop `benchmark_metadata.csv` and `task_metadata.csv` next to `benchmark.py`; the framework auto-loads them (see [`benchmark.py:__init_subclass__`](src/cube/benchmark.py)).

## Key invariants

- Every `@tool_action` method must return something `Content.from_data()` can wrap (str, dict, PIL Image, …). See [`core.py`](src/cube/core.py).
- `evaluate()` must return `(reward: float, info: dict)` where `reward == 1.0` means solved.
- `TaskConfig` must be JSON-serializable (it travels over the network to workers).
- `debug.py` sequences must be deterministic and must reach `reward == 1.0` — `cube test` enforces this.

## Changing the template

The template at `src/cube/_template/new_cube_package/` is normal Python — edit it directly. `cube init` does a straight `shutil.copytree`. Keep the template's `TODO` comments accurate; they are the contributor's primary guide.

## Contributing a Benchmark

If you have a benchmark you'd like to wrap as a CUBE — or want to flag one as a good candidate — [fill out this short form](https://docs.google.com/forms/d/e/1FAIpQLSddMFyRXZJPpD0I2K27OEmIPUpj57w--u2NuMscrjNlkqy8rQ/viewform). It helps us track interest, assess schema fit, and pair you with the right support. No commitment required at this stage.

## Contribution philosophy

We are in an era where AI coding agents generate more code than any team can carefully review. High-volume, low-quality submissions slow everyone down — so we ask contributors to hold themselves to a higher bar before opening a PR, RFC, or bug report.

**Use a coding agent to review your own work first.** Before submitting, run your changes through a capable coding agent (e.g. Claude Code, Cursor) and iterate until it finds nothing material to flag. The goal is not to outsource judgment — it is to arrive at the conversation with humans already having done the obvious passes.

**Communicate through concise, iterated markdown.** The best PRs and RFCs arrive as a tight markdown document — clear problem statement, specific proposal, relevant context — refined through a few rounds of human-agent iteration before anyone else reads it. A well-drafted document respects reviewers' time and gets better feedback faster.

**We loosely follow the [Open Spec](https://open-spec.org) methodology.** Design intent lives in `design/` as versioned specs. Implementation follows the spec; the spec is updated when reality diverges. When in doubt, write the spec first.

## RFC Process

Large changes to the core protocol — new abstract methods, breaking type changes, new layers — go through an RFC before implementation.

**When an RFC is needed:**
- Adding or changing an abstract method on `Tool`, `Task`, or `Benchmark`
- Changing the `Observation` / `Action` / `EnvironmentOutput` data model
- New optional protocol extensions (streaming, async, multi-agent, multi-dim reward)
- Anything that forces existing cube packages to change

**When it is NOT needed:**
- Bug fixes
- New examples or benchmarks
- Documentation improvements
- Additive changes that don't touch existing interfaces

**Process:**

1. **Discuss** — Open a [GitHub Discussion](https://github.com/The-AI-Alliance/cube-standard/discussions) or issue tagged `RFC` to gauge interest.
2. **Draft** — Copy the template below into `design/rfc_<short_name>.md` and open a PR. Prefix the PR title with `RFC:`.
3. **Review** — Collect feedback for at least one week. The PR author is responsible for iterating on the draft.
4. **Merge** — A maintainer merges when there is rough consensus (no blocking objections from core contributors).
5. **Implement** — Follow-up PRs implement the RFC. Link them back to the RFC PR.

**RFC template** (`design/rfc_<short_name>.md`):

```markdown
# RFC: <Title>

## Summary
One paragraph: what changes and why.

## Motivation
What problem does this solve? What is the cost of not doing it?

## Design
Concrete API / interface sketch. Include before/after code snippets where helpful.

## Alternatives considered
What else was explored and why it was rejected.

## Open questions
Unresolved issues that reviewers should weigh in on.
```

Existing RFCs live in [`design/`](design/).

## Known gaps / TODOs

Tracked inline in source:

- `core.py` — `AudioContent` and `VideoContent` rendering ([`core.py`](src/cube/core.py))
- `task.py` — truncation logic in `step()`, `get_status()` contract ([`task.py`](src/cube/task.py))
- `benchmark.py` — `BenchmarkMetadata` field list (homepage, citation, …) ([`benchmark.py`](src/cube/benchmark.py))
- `__init__.py` — cache management helpers ([`__init__.py`](src/cube/__init__.py))
