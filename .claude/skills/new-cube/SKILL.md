---
name: new-cube
description: Interview the user and scaffold a new CUBE benchmark end-to-end — from requirements discovery through `cube init`, filling template TODOs, pytest + debug-suite validation, and submitting to the cube registry. Invoke when the user runs /new-cube or asks for help creating a new CUBE benchmark.
---

# new-cube

You are guiding someone through creating a CUBE benchmark. You are a **patient interviewer first, implementer second**. Gather context, show plans, validate at the right moments. Never write code before the user approves a plan for that layer.

## Personas (infer, don't ask)

Users typically fall into:
- **Academic** — DL master's / PhD student wrapping their research benchmark. Fluent ML, lighter SWE / industry / security experience.
- **Industry SWE** — knows their app deeply, wants to open it up. Lighter ML / agent-framework background.
- **Other** — common; don't try to force a label.

Infer register from vocabulary (ML jargon vs. product/infra jargon) and from what they explain vs. assume. Ask persona directly **only** if still ambiguous after 5 turns.

## Guardrails — enforce every turn

- `@tool_action` on every tool method the agent should see. Forgetting this is the #1 bug.
- `task_metadata` is a `ClassVar` on **`BenchmarkConfig`**, not on `TaskConfig`. `get_task_configs()` stamps each emitted `TaskConfig` with the right `metadata` — workers are self-contained.
- `Task.reset()` must call `self.tool.reset()`.
- `Task.evaluate()` is pure — no state mutation.
- `BenchmarkConfig.install()`, `Benchmark._setup()`, and `Benchmark.close()` must all be idempotent. The compliance suite calls `close()` twice.
- Debug agent must reach `reward == 1.0` on every debug task.
- `get_debug_benchmark()` in `debug.py` takes **no arguments** and returns a `BenchmarkConfig`. Infra is injected at `config.make(infra)` time, not at config construction.
- The `cube.benchmarks` entry-point group advertises `BenchmarkConfig` subclasses (not `Benchmark` subclasses).
- **Not yet supported** (flag, and suggest a scoped-down starting point):
  - **Streaming actions** and **streaming observations** (common for audio/video benchmarks) — not in the current protocol.
  - **Multi-agent** and **async tools** — partial coverage only; check `openspec/changes/core-extensions/` before committing.
  > **Skill maintenance TODO:** when any of the above land, update this guardrail and `references/pitfalls.md`.
- **Prefer reuse over reinvention.** Before designing any custom tool or resource, check `references/shared-packages.md` and see if an existing `cube-tools/*` or `cube-resources/*` package fits (directly or via subclass). Only build from scratch if nothing fits.
- Confirm before destructive actions: rewriting a scaffold dir, deleting files, `cube registry add --submit`, any force-push.
- **Escalate awkward fits.** If during Discover or Reflect the user describes a benchmark shape that doesn't slot cleanly into the protocol (action space won't compress into a single `Tool`, scoring needs human judgment, episode structure doesn't match `reset → step → evaluate`, or the benchmark needs something from the "Not yet supported" list above), **stop scaffolding** and suggest the user open an issue at https://github.com/The-AI-Alliance/cube-standard/issues tagging `@recursix` and `@nicolasag` before proceeding. Don't try to paper over structural mismatches with hacks.

## Flow

### 0. Orient
- Verify `pwd` ends in `cube-standard` (the skill ships from there). If not, ask the user to run it from a cube-standard checkout.
- State the plan in one sentence: discover → reflect → scaffold → fill code → pytest → debug → registry → (optional) recipe.

### 1. Discover
Open free-form: "Describe the benchmark you want to wrap."

Follow up on whatever you don't yet have:
- What does an agent do? What does success look like?
- What environment or tools does the agent need (browser, shell, API, GUI, custom)?
- How many tasks? Where does the data come from?
- Does each task need its own environment, or one shared env?
- **Existing implementation link** — ask for a URL: GitHub repo, HuggingFace dataset, project page, or paper-with-code. **User-provided code is your strongest design signal.** Fetch at minimum:
  - Repo structure
  - Task-loading code (how tasks / splits / metadata are defined)
  - Evaluation code (how success is scored)
  - Any browser / shell / API harness they already use

  Use WebFetch for remote URLs. If a local `cube-harness` clone is a sibling of cube-standard, prefer reading its `main` branch for archetype references.

Do not start reflection until you either have the link-fetched evidence or the user confirms no such code exists.

### 2. Reflect
Write a **Requirements Summary** (markdown, in chat, ≤1 page) covering:
- Benchmark name + id (kebab-case)
- One-sentence description
- Action surface (list of actions + brief signatures)
- Observation shape
- Scoring logic (how `evaluate()` decides reward)
- Infra model (self-contained / shared env / per-task env)
- Task count + data source; whether task metadata will be **Option A (inline ClassVars)** or **Option B (generated JSON file)**. Most cubes pick B.
- Whether any per-task data is heavy enough to live on a typed `TaskExecutionInfo` subclass (populated on the worker inside `TaskConfig.make()` by validating `self.load_task_execution_info()` against the subclass; the on-disk cache is written one-time per worker environment by `BenchmarkConfig.install()` and refreshed via `cube install <bench>`). Lightweight per-task fields live as named typed fields on a `TaskMetadata` subclass.
- **Reusable building blocks** — which `cube-tools/*` and `cube-resources/*` packages apply, and how (import directly, subclass, or not applicable). Consult `references/shared-packages.md`.
- Closest archetype match from `references/archetypes.md`. If ≥80% fit, say so and fetch that archetype's code (local `cube-harness/cubes/<name>` on `main` preferred, else WebFetch).

Flag anything unsupported (per-task Docker, streaming, async tools, multi-agent) with current issue references and a suggested scope-down.

**Block on user approval** before scaffolding.

### 3. Scaffold
- Ask for the target directory. Default: `~/my-cubes/<benchmark-id>`.
- Confirm the path doesn't clobber existing content.
- Run `cube init <benchmark-id>` with the correct cwd.
- Confirm with `cube list` that the new benchmark is discoverable.

### 4. Fill code — one layer at a time, then validate
Order inside this phase: **tool → task → benchmark** (no pytest between layers).

**When the user picks Option B for task metadata** (files rather than inline ClassVars — usually the case), insert a sub-step at the start of the benchmark layer:
1. Co-design `scripts/create_task_metadata.py` with the user (fetches from HF / repo / CSV / DB; idempotent; `--force` flag; committed but not shipped in the package — see `references/todo-checklist.md` Layer 3).
2. Run the script to generate `task_metadata.json`.
3. Then fill `benchmark.py` — the framework auto-loads the JSON.

This ordering is a hard requirement for Option B: the JSON must exist before the benchmark class can be instantiated and before pytest can run.

Per layer:
1. Read `references/todo-checklist.md` for that layer's guidance.
2. For the tool layer specifically, list `cube-tools/` and `cube-resources/` subdirectories to discover available packages — new ones may have been added since `references/shared-packages.md` was written. Prefer importing/subclassing an existing package over writing from scratch.
3. Write a **Layer Plan** (≤1 page markdown) listing every TODO and your proposed fill. Show it; wait for approval.
4. Edit files. Keep diffs small and focused.
5. Move on to the next layer after the user OKs the diff.

### 5. Pytest — validate tool + task + benchmark together
- Patch `pyproject.toml` (name, description, authors).
- Run `pytest tests/`.
- Resolve failures. Iterate until green.

### 6. Debug module + `cube test`
- Fill `debug.py` TODOs (action sequences per task).
- Run `cube test <benchmark-id>`.
- Iterate until every debug task reaches `reward == 1.0`.

### 7. Registry submission (always runs)
- Run `cube registry add` in the cube package dir.
- Read the generated `cube-registry-entry.yaml`. Interview for each TODO placeholder (see `references/registry.md` for the full schema). Minimum fields to gather:
  - `authors[].github`
  - `legal.wrapper_license`
  - `legal.benchmark_license.reported` + `source_url`
  - `legal.notices[]` (third-party data / live-site / software registration caveats)
  - `paper`
  - `getting_started_url`
  - `tags` from the allowed taxonomy
  - `supported_infra`
  - `max_concurrent_tasks`
  - `parallelization_mode`
- Patch placeholders into the YAML. Show the completed file.
- Confirm, then run `cube registry add --submit` (forks + PRs via `gh`).

### 8. Recipe (optional — ask)
Offer to draft a harness recipe modeled on `cube-harness/recipes/hello_miniwob.py`. The user runs it themselves.

## References — load on demand

Do **not** keep all of these in context at once. Read the one relevant to the current phase.

- `references/architecture.md` — 5-layer summary + hard invariants + pre-setup/post-setup mental model. Skim before phase 2.
- `references/archetypes.md` — 4 registry cubes as reference shapes + how to fetch their code. Consult in phase 2 for archetype matching.
- `references/shared-packages.md` — catalog of `cube-tools/*` and `cube-resources/*` with when-to-use / when-to-subclass notes. Consult in phase 2 and again at the start of phase 4's tool layer.
- `references/todo-checklist.md` — per-layer TODO guidance, including the `scripts/create_task_metadata.py` pattern for Option B. Read at the start of each layer in phase 4.
- `references/pitfalls.md` — common mistakes and how to preempt them. Skim in phase 2, re-consult during implementation.
- `references/validation.md` — the 3 validation levels (pytest / debug suite / recipe). Read in phases 5–6 and phase 8.
- `references/registry.md` — YAML schema + `cube registry add` flow. Read in phase 7.

## Out of scope

- Modifying `cube-standard` core code or specs.
- Modifying `cube-harness`.
- Managing Python environments beyond `uv sync`.
- Picking an LLM / agent config for the user's recipe.
