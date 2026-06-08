# Contributing to CUBE Standard

> **Building a new CUBE benchmark?** Start with the [Authoring a CUBE guide](https://the-ai-alliance.github.io/cube-standard/authoring-a-cube) — three starting paths, implementation order, validation, and submission. Come back here for framework invariants and the RFC process when you need them. This file covers contributing to the cube-standard **framework itself** (adding features, changing specs, modifying the template).

> **Before you propose a framework change, read the [Design Philosophy](https://the-ai-alliance.github.io/cube-standard/design-philosophy).** CUBE is a small, shared contract that many cubes depend on; most "change the framework" needs are better served by a subclass, harness code, or a smaller in-schema change. The philosophy page explains how we evaluate that — and the `/gatekeep-rfc` skill lets you check your own draft against it before anyone else reads it.

## The contribution workflow (end to end)

For framework changes, the path from idea to merged PR:

0. **Understand the broader picture first.** Read the [Design Philosophy](https://the-ai-alliance.github.io/cube-standard/design-philosophy) and the [ROADMAP](ROADMAP.md), then the spec for the layer you'll touch (`openspec/specs/<layer>/spec.md`). Most proposed API changes have a smaller form that fits the existing schema — find it first.

1. **Branch off `dev`, merge back to `dev`.** Never `main`. *Tip:* if you drive the work with a coding agent, use a `git worktree` per branch so parallel agents don't collide.

2. **Plan with an OpenSpec RFC — and converge it *locally* first.** Write `openspec/changes/<name>/proposal.md` + `deltas.md` (Problem · Proposed solution · Alternatives; an optional mermaid diagram if it clarifies). Keep it concise — coding agents draft long; tighten before others read it.
   - **Run `/gatekeep-rfc` early and often, on your own machine** — on a one-paragraph sketch, not just a finished draft. It reads your idea the way a maintainer will, separates the real need from the mechanism, and points you at the smallest version (often something you can do entirely in your own package). It's cheap; use it as a *loop*, reshaping as you go.
   - **Avoid the expensive cycle:** *don't* polish a full proposal, open a PR, and only then discover it needs a different shape. Reshaping a sketch costs nothing; reshaping a finished PR wastes your work and clogs the queue. Open the PR once the direction has already converged locally.
   - **Iterate with a coding agent** (Claude Code, etc.) on the converged proposal until it flags nothing material.
   - **Open the RFC PR** (prefix the title `RFC:`). A reviewer (or a community gatekeeper) may run `/gatekeep-rfc` again to route it; you then iterate with the team. **No approval gate blocks you from continuing** — you can start coding and put the implementation in the same PR.

3. **Code it.** Implement against the RFC; keep the diff lean.

4. **Verify with the right tests.** CI runs **unit + integration** automatically — fast, binary, but limited in scope. **Smoke tests** (`scripts/smoke/*.py`) cover end-to-end scopes CI can't reach (real infra, API keys, minutes-long runs). The coding agent decides which existing smokes to run and which new one to add; see the smoke-test conventions in [CLAUDE.md](CLAUDE.md#testing-and-linting).

5. **Self-review with `/code-review`** (the Anthropic code-review skill, run as a sub-agent) and address what it finds.

6. **Submit code in the same PR as the RFC.** One PR carries the proposal, the deltas, and the implementation.

7. **Human code review** merges when there's rough consensus. On merge, archive the change (step 6 of the RFC process below).

The rest of this file is the reference detail behind these steps.

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
- Small additive changes that don't touch existing interfaces — a lean living-spec edit is enough. (Additive still isn't free: the [Design Philosophy](https://the-ai-alliance.github.io/cube-standard/design-philosophy) leanness bar still applies — keep it general and minimal, or leave it out.)

**Process:** (this is the RFC detail behind "[The contribution workflow](#the-contribution-workflow-end-to-end)" above; the proposal and its implementation live in **one PR**)

1. **Draft** — Create `openspec/changes/<name>/proposal.md` and `deltas.md` (Problem · Proposed solution · Alternatives — see [`openspec/README.md`](openspec/README.md)). Run `/gatekeep-rfc` locally and iteratively *while drafting* — converge on the smallest in-schema form **before** opening the PR, not after. Optionally open a [GitHub Discussion](https://github.com/The-AI-Alliance/cube-standard/discussions) first for big or contentious ideas.
2. **Open the PR** prefixed `RFC:`. A reviewer (or community gatekeeper) runs `/gatekeep-rfc` to route it; you iterate with the team. No approval gate blocks you from continuing.
3. **Implement in the same PR** — push the code alongside the proposal; verify with smokes (workflow step 4) and self-review with `/code-review`.
4. **Merge** — a maintainer merges when there's rough consensus (no blocking objections from core contributors).
5. **Archive** — move `openspec/changes/<name>/` to `openspec/changes/archive/YYYY-MM-DD-<name>/` and apply deltas to the main spec.

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
