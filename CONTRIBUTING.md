# Contributing to CUBE Standard

> **Building a new CUBE benchmark?** Start with the [Authoring a CUBE guide](https://the-ai-alliance.github.io/cube-standard/authoring-a-cube) — three starting paths, implementation order, validation, and submission. Come back here for framework invariants and the RFC process when you need them. This file covers contributing to the cube-standard **framework itself** (adding features, changing specs, modifying the template).

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
  resource.py    # InfraConfig / ResourceConfig / ResourceHandle — provisioning
  container.py   # ContainerConfig (task requirement) / Container (live handle)
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

Full walkthrough lives in the [Authoring a CUBE guide](https://the-ai-alliance.github.io/cube-standard/authoring-a-cube) — three starting paths (`/new-cube` skill, copy counter-cube, `cube init`), implementation order, validation with `cube test` and `/review-cube`, and submission via `cube registry add --submit`.

Short form: `cube init my-env` → fill TODOs in `tool.py` → `task.py` → `benchmark.py` → `debug.py` (in that order) → `cube test my_env` must reach `reward == 1.0` on every debug task.

The counter-cube example covers all layers concretely:
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

**We follow the [OpenSpec](https://github.com/Fission-AI/OpenSpec) methodology** for managing contracts between layers. OpenSpec is a lightweight, spec-driven approach that keeps AI coding agents and human contributors aligned on the same contracts — without heavyweight processes. Each layer has a living spec in `openspec/specs/<layer>/spec.md` that defines its public API, invariants, and gotchas.

The three habits:

1. **Read** the spec for any layer you're about to touch.
2. **Sync** the spec after merging — run `/update-openspec` in Claude Code.
3. **Propose** before breaking — write a short delta in `openspec/changes/<name>/` so the team sees contract changes before code lands.

Full workflow, delta format, and examples: [`openspec/README.md`](openspec/README.md).

## RFC / Change Proposal Process

Large changes to the core protocol — new abstract methods, breaking type changes, new layers — go through a change proposal before implementation.

**When a proposal is needed:**
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
2. **Draft** — Create `openspec/changes/<name>/proposal.md` and `deltas.md`, open a PR prefixed `RFC:`. See [`openspec/README.md`](openspec/README.md) for the delta format.
3. **Review** — Collect feedback for at least one week. The PR author iterates on the draft.
4. **Merge** — A maintainer merges when there is rough consensus (no blocking objections from core contributors).
5. **Implement** — Follow-up PRs implement the RFC. Link them back to the RFC PR.
6. **Archive** — Move `openspec/changes/<name>/` to `openspec/changes/archive/YYYY-MM-DD-<name>/` and apply deltas to the main spec.

Active proposals live in [`openspec/changes/`](openspec/changes/).

## Releases & dev versioning

Releases are tag-driven and per-package: pushing a `cube-standard/v*`,
`cube-tools/*/v*`, or `cube-resources/*/v*` tag triggers
[`release.yml`](.github/workflows/release.yml), which builds and publishes
that package to PyPI.

**The `dev` branch always carries the *next* unreleased version**, never the
last published one. Concretely: immediately after a release, bump the `dev`
`version` to the next pre-release (e.g. publish `0.1.0rc8` → bump `dev` to
`0.1.0rc9`).

Why this matters: cube-harness and the cubes pin `cube-standard>=<rcN>`. If
`dev` keeps the *published* version string while diverging (e.g. `dev` adds
`ConfigRegistry` but stays `0.1.0rc8`), `uv` treats the dev build and the
PyPI wheel as the same version and silently swaps the dev install for the
PyPI one during cross-repo CI — producing confusing `ImportError`s for
symbols that "exist on dev." A distinct dev version makes the two artifacts
unambiguous and lets cross-repo installs resolve correctly without
`--force-reinstall` workarounds. See cube-standard #167 for the full
rationale and the companion ordered-release-pipeline work.

## Known gaps / TODOs

Tracked inline in source:

- `core.py` — `AudioContent` and `VideoContent` rendering ([`core.py`](src/cube/core.py))
- `task.py` — truncation logic in `step()`, `get_status()` contract ([`task.py`](src/cube/task.py))
- `benchmark.py` — `BenchmarkMetadata` field list (homepage, citation, …) ([`benchmark.py`](src/cube/benchmark.py))
- `__init__.py` — cache management helpers ([`__init__.py`](src/cube/__init__.py))
