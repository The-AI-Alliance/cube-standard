---
layout: default
title: Authoring a CUBE
nav_order: 20
has_children: false
---

# Authoring a CUBE

So you want to wrap a benchmark as a CUBE. This guide walks through what that involves, the easiest way to start, and who to ping if your benchmark isn't a clean fit.

The short version: you implement four Python classes (tool, task, benchmark, debug agent), run `cube test` to prove it works, and submit one YAML file to the registry. Most benchmarks fit this shape naturally once it clicks.

## What you're building

A CUBE package exposes a benchmark through a uniform protocol so any CUBE-compatible harness can run it without custom integration. You implement four things:

| Layer | What it answers |
|---|---|
| **Tool** | What actions can the agent take? (e.g. `click`, `type`, `run_shell`) |
| **Task** | What's the initial observation, and how do I score a solution? |
| **Benchmark** | What's the list of tasks, and what shared setup do they need? |
| **Debug** | One deterministic solution per task — proves everything wires up correctly. |

Each layer has a small, well-defined contract. Formal per-layer specs live in [openspec/specs/](https://github.com/The-AI-Alliance/cube-standard/tree/main/openspec/specs) when you want the contract-level detail.

## Three ways to start

### Option 1 — Use the `/new-cube` skill (recommended)

If you have [Claude Code](https://claude.ai/claude-code) installed, open a workspace with `cube-standard` checked out and run:

```
/new-cube
```

The skill interviews you (what the agent does, what counts as a solved task, what resources you need), scaffolds the package, fills in the template TODOs, runs the debug suite, and produces a registry entry. You review and correct as it goes — it handles the boilerplate so you focus on the benchmark logic.

### Option 2 — Copy the reference implementation

```sh
cp -r examples/counter-cube my-bench
cd my-bench && uv sync
```

[counter-cube](https://github.com/The-AI-Alliance/cube-standard/tree/main/examples/counter-cube) is the minimal real cube — increment a counter to reach a target. Every layer has a comment explaining its role. Rename the placeholders, replace the logic with yours.

### Option 3 — Scaffold from the template

```sh
cube init my-bench
cd my-bench && uv sync
```

Blank slate with `TODO` markers at every decision point. Best if your benchmark's shape doesn't resemble counter-cube.

## Implementation order

Work through the layers top-down. Each file has `TODO` comments pointing at what needs to change.

| # | File | What to fill in |
|---|---|---|
| 1 | `tool.py` | **Check reusable tools first** — for web agents use [`cube-browser-tool`](https://github.com/The-AI-Alliance/cube-standard/tree/main/cube-tools/cube-browser-tool), for desktop/CUA use [`cube-computer-tool`](https://github.com/The-AI-Alliance/cube-standard/tree/main/cube-tools/cube-computer-tool). Import directly or subclass to add benchmark-specific actions. Only subclass `Tool` from scratch if neither fits; mark methods with `@tool_action`. |
| 2 | `task.py` | Implement `reset()` (opening observation) and `evaluate()` (reward on termination) |
| 3 | `benchmark.py` | Fill `BenchmarkMetadata` and `task_metadata` (inline or CSV/JSON); implement `_setup()` / `close()` |
| 4 | `debug.py` | One deterministic action sequence per task — must reach `reward == 1.0` |
| 5 | `pyproject.toml` | Update `name`, `description`, and the `cube.benchmarks` entry-point |

**A note on task metadata.** If you have more than a handful of tasks, you'll load them from `task_metadata.csv` or `.json` rather than inlining them in `benchmark.py`. If your task data starts life in a different shape — scraped from a website, exported from an existing benchmark repo, hand-curated in a spreadsheet — expect to write a small one-off conversion script as a pre-step. The `/new-cube` skill walks you through this; following options 2 or 3 you'll handle it manually.

Framework invariants (return types, serialization rules, reward semantics) are in [CONTRIBUTING.md § Key invariants](https://github.com/The-AI-Alliance/cube-standard/blob/main/CONTRIBUTING.md#key-invariants). Read them once before you're deep in the code.

## Validate

```sh
cube test my-bench
```

Every debug task must hit `reward == 1.0`. If one doesn't, either the debug action sequence is wrong or the task's `evaluate()` is wrong — catching this locally is the whole point of the debug suite.

**Before you open a registry PR, self-audit with the `/review-cube` skill:**

```
/review-cube ./my-bench
```

`/review-cube` installs your package, runs pytest, runs `cube test`, audits against cube-standard invariants, and produces a Blocking / Suggestions report. Resolve everything in the Blocking section before submitting. Registry CI catches the same issues later, but locally is faster and less public.

## Publish

Once `cube test` and `/review-cube` both pass, submit to the registry with one command:

```sh
cube registry add --submit
```

This generates a `cube-registry-entry.yaml` from your `pyproject.toml`, forks [cube-registry](https://github.com/The-AI-Alliance/cube-registry), commits the entry, and opens a PR. Registry CI validates and auto-merges the happy path — no human review required.

Run `cube registry add` without `--submit` first if you want to generate the YAML locally, edit it, and review the entry before opening the PR.

Your package also needs to land on PyPI for the registry's compliance suite to install and run it — publish whenever you're ready; it doesn't have to come before the registry PR.

## We'll help you

Not every benchmark is a clean fit on first read. If you hit something awkward — the action space doesn't compress into a single `Tool`, scoring needs human judgment, the infra requirements are unusual, the episode structure doesn't match `reset → step → evaluate` — **tell us before wrangling it on your own**. Common awkwardness usually has an idiomatic solution we can point you at, and if yours is genuinely new we'd rather evolve the protocol than watch you paper over it.

Ways to reach us:

- **[Benchmark contributor form](https://docs.google.com/forms/d/e/1FAIpQLSddMFyRXZJPpD0I2K27OEmIPUpj57w--u2NuMscrjNlkqy8rQ/viewform)** — flag intent, no commitment; we follow up based on fit
- **[GitHub Discussions](https://github.com/The-AI-Alliance/cube-standard/discussions)** — public Q&A and RFC gauging
- **[Open an issue](https://github.com/The-AI-Alliance/cube-standard/issues/new)** and tag `@recursix` and `@nicolasag` for direct help

## Deeper references

- [counter-cube](https://github.com/The-AI-Alliance/cube-standard/tree/main/examples/counter-cube) — canonical reference implementation
- [toy_benchmark](https://github.com/The-AI-Alliance/cube-standard/tree/main/examples/toy_benchmark) — single-file minimal variant
- [CONTRIBUTING.md](https://github.com/The-AI-Alliance/cube-standard/blob/main/CONTRIBUTING.md) — framework invariants, RFC process, template rules
- [openspec/specs/](https://github.com/The-AI-Alliance/cube-standard/tree/main/openspec/specs) — formal per-layer contracts
- [cube-registry](https://github.com/The-AI-Alliance/cube-registry) — submission YAML template and compliance tiers
- [cube-tools/](https://github.com/The-AI-Alliance/cube-standard/tree/main/cube-tools) — reusable tool packages (browser, computer, chat)
- [cube-resources/](https://github.com/The-AI-Alliance/cube-standard/tree/main/cube-resources) — reusable resource packages (playwright browser, chat sessions, AWS/Azure infra, VM backend)
- [DeepWiki](https://deepwiki.com/The-AI-Alliance/cube-standard) — full API reference
