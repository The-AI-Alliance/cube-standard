# Contributing to CUBE Standard

> **Building a new CUBE benchmark?** Start with the [Authoring a CUBE guide](https://the-ai-alliance.github.io/cube-standard/authoring-a-cube) — three starting paths, implementation order, validation, and submission. This file is for contributing to the cube-standard **framework itself** (features, specs, the scaffold template).

> **Before you propose a framework change, read the [Design Philosophy](https://the-ai-alliance.github.io/cube-standard/design-philosophy).** CUBE is a small, shared contract that many cubes depend on; most "change the framework" needs are better served by a subclass, harness code, or a smaller in-schema change. The `/gatekeep-rfc` skill lets you check your own draft against it before anyone else reads it.

## The contribution workflow (end to end)

For framework changes, the path from idea to merged PR:

0. **Understand the broader picture first.** Read the [Design Philosophy](https://the-ai-alliance.github.io/cube-standard/design-philosophy) and the [ROADMAP](ROADMAP.md), then the spec for the layer you'll touch (`openspec/specs/<layer>/spec.md`). Most proposed API changes have a smaller form that fits the existing schema — find it first.

1. **Branch off `dev`, merge back to `dev`.** Never `main`. *Tip:* if you drive the work with a coding agent, use a `git worktree` per branch so parallel agents don't collide.

2. **Plan with an OpenSpec RFC — converge it *locally* first.** Write `openspec/changes/<name>/proposal.md` + `deltas.md` (Problem · Proposed solution · Alternatives — format in [`openspec/README.md`](openspec/README.md)). Run `/gatekeep-rfc` early and often on your own machine — on a rough sketch, not just a finished draft — and reshape as you go. Open the PR only once the direction has converged; *don't* polish a full proposal and discover after submitting that it needs a different shape (that wastes your work and clogs the queue).

3. **Open the RFC PR** (title prefixed `RFC:`). **No approval gate blocks you** — start coding and put the implementation in the *same* PR. A reviewer (or community gatekeeper) may run `/gatekeep-rfc` to route it; you iterate with the team.

4. **Code it, verify it.** Keep the diff lean. CI runs **unit + integration** (fast, binary, limited scope); for end-to-end paths CI can't reach (real infra, API keys), run or add a **smoke** (`scripts/smoke/*.py`) — see [CLAUDE.md](CLAUDE.md#testing-and-linting). Self-review with `/code-review` before asking for human eyes.

5. **Merge & archive.** A maintainer merges on rough consensus. On merge, move `openspec/changes/<name>/` to `openspec/changes/archive/YYYY-MM-DD-<name>/` and apply the deltas to the layer spec.

**When does a change need an RFC?** A proposal is for changes to the shared contract — a new or changed abstract method on `Tool`/`Task`/`Benchmark`, a change to the `Observation`/`Action`/`EnvironmentOutput` model, a new protocol extension, or anything that forces existing cubes to change. It is **not** needed for bug fixes, new examples/benchmarks, docs, or small additive edits — those are a lean living-spec edit. (Additive still isn't free: the [leanness bar](https://the-ai-alliance.github.io/cube-standard/design-philosophy) applies — keep it general and minimal, or leave it out.) Active proposals live in [`openspec/changes/`](openspec/changes/).

## Setup

```sh
uv sync --all-extras    # dev deps
pre-commit install --hook-type pre-commit --hook-type commit-msg
make lint               # ruff check + format — run both; CI checks both
make test               # pytest tests/
```

Every commit needs a DCO sign-off (`git commit -s`). See [README.md](README.md) for license and community details.

## Where things live

```
src/cube/
  core.py tool.py task.py benchmark.py   # layers 1–4: the protocol contracts
  testing.py                              # debug suite (run_debug_suite, …)
  server.py cli.py                        # JSON-RPC server, `cube` CLI
  resource.py container.py                # provisioning + single-container abstraction
  _template/                              # scaffold copied by `cube init` (keep in sync)
examples/counter-cube/                    # canonical reference cube — read this first
```

Each layer's contract is a living spec in [`openspec/specs/<layer>/spec.md`](openspec/specs/) — the authoritative public API, invariants, and gotchas. **Read it before touching a layer**; the module docstrings mirror it. (A fuller package map for coding agents lives in [CLAUDE.md](CLAUDE.md).)

## Working with OpenSpec

We use the lightweight [OpenSpec](https://github.com/Fission-AI/OpenSpec) methodology: each layer's contract lives in a living spec, and contract changes land as a delta in `openspec/changes/` before the code does. Three habits — **read** the spec before touching a layer, **propose** before breaking, **sync** the spec after merging (`/update-openspec`). Format and examples: [`openspec/README.md`](openspec/README.md).

Two culture notes that make review fast, in an era where agents generate more code than any team can carefully read:

- **Self-review with a coding agent first** — iterate until it flags nothing material, so you arrive with the obvious passes already done.
- **Communicate in concise, iterated markdown** — a tight problem-statement-plus-proposal respects reviewers' time and gets better feedback.

## Changing the cube template

The scaffold at `src/cube/_template/new_cube_package/` is normal Python — edit it directly; `cube init` does a straight `shutil.copytree`. Keep its `TODO` comments accurate: they are the contributor's primary guide.

## Releases & dev versioning

Releases are tag-driven and per-package: pushing a `cube-standard/v*`, `cube-tools/*/v*`, or `cube-resources/*/v*` tag triggers [`release.yml`](.github/workflows/release.yml) to build and publish that package to PyPI.

**The `dev` branch always carries the *next* unreleased version**, never the last published one — immediately after a release, bump `dev` to the next pre-release (publish `0.1.0rc8` → bump `dev` to `0.1.0rc9`). This matters because cubes and cube-harness pin `cube-standard>=<rcN>`: if `dev` kept a published version string while diverging, `uv` would treat the dev build and the PyPI wheel as the same version and silently swap them during cross-repo CI. See [#167](https://github.com/The-AI-Alliance/cube-standard/pull/167) for the full rationale.

## Authoring a benchmark

Wrapping a benchmark as a CUBE is a different — and usually easier — path: you implement four classes and submit one YAML, without touching the framework. The [Authoring a CUBE guide](https://the-ai-alliance.github.io/cube-standard/authoring-a-cube) is the full walkthrough (the `/new-cube` and `/review-cube` skills, implementation order, validation with `cube test`, submission). Have one to contribute or a candidate to flag? [Fill out this short form](https://docs.google.com/forms/d/e/1FAIpQLSddMFyRXZJPpD0I2K27OEmIPUpj57w--u2NuMscrjNlkqy8rQ/viewform) — no commitment required.
