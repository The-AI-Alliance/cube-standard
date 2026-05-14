# CLAUDE.md — cube-standard

You are working in **cube-standard**, the protocol and base classes that benchmarks and
harnesses implement. This file is your map; it is deliberately short. Read the relevant
spec in `openspec/specs/` before modifying any layer.

## What this repo is

CUBE Standard defines the contract: how benchmarks expose tasks, how tools expose
actions, how resources are provisioned. It does NOT run agents, record trajectories,
or coordinate experiments — that lives in **cube-harness**.

## The 5-layer architecture

| Layer | Module | Spec | What it does |
|-------|--------|------|--------------|
| 1. Core types | `cube.core` | [core/spec.md](openspec/specs/core/spec.md) | `Action`, `Observation`, `Content`, `EnvironmentOutput`, `TypedBaseModel` |
| 2. Tool | `cube.tool` | [tool/spec.md](openspec/specs/tool/spec.md) | `Tool`, `AsyncTool`, `@tool_action`, `ToolConfig`, `Toolbox` |
| 3. Task | `cube.task` | [task/spec.md](openspec/specs/task/spec.md) | `Task`, `TaskMetadata`, `TaskConfig`, gym-style `reset/step/evaluate` |
| 4. Benchmark | `cube.benchmark` | [benchmark/spec.md](openspec/specs/benchmark/spec.md) | `Benchmark`, `BenchmarkMetadata`, class-level registry |
| 5. Testing | `cube.testing` | [testing/spec.md](openspec/specs/testing/spec.md) | `run_debug_suite`, `assert_debug_tasks_reward_one` |

Cross-cutting:
- **Resource lifecycle** — [resource/spec.md](openspec/specs/resource/spec.md) (L1 provisioned images, L2 benchmark-scoped, L3 task-scoped)
- **Container** — [container/spec.md](openspec/specs/container/spec.md) (single-container abstraction for tasks)
- **Server** — [server/spec.md](openspec/specs/server/spec.md) (JSON-RPC 2.0, MCP-compatible)
- **CLI** — [cli/spec.md](openspec/specs/cli/spec.md) (`cube init`, `cube list`, `cube test`, `cube registry add`)

## Engineering principles

- **Read the spec first.** Before touching any layer, read its spec in `openspec/specs/`. Specs are the authoritative design intent — but they can be stale or wrong; flag discrepancies rather than silently working around them.
- **Fix in the right place.** A quick local experiment to understand a problem is fine. But the committed fix must address the root cause in the correct layer — not a workaround scoped to a single call site or context.
- **Understand before fixing.** Many bad fixes come from acting too fast. Make sure you understand the broader design before proposing a change. A fix that misses the bigger picture is worse than no fix.
- **Lean diffs.** Make the minimal change that solves the problem. Avoid verbose additions, unnecessary abstractions, and duplicated logic that already exists elsewhere. If existing code can be reused or consolidated, do it. A hard-to-review diff is a liability.
- **Think long-term.** Every change should age well. Ask whether today's shortcut becomes tomorrow's debt — and whether the design could evolve cleanly if requirements change.

## Code review

**Sign your commits.** Every commit needs a `Signed-off-by` line (`git commit -s`). DCO is enforced by CI — unsigned commits will be blocked.

PRs are reviewed with `/code-review` ([plugin docs](https://github.com/anthropics/claude-code/blob/main/plugins/code-review/README.md)), which audits changes against these guidelines. Write PRs as if a reviewer will check each principle above against the diff.

## Workflow for code changes

1. **Find the relevant spec** — which layer? Start there.
2. **Read the spec's "Invariants" and "Gotchas" sections** — these are the traps.
3. **Check for an active change** in `openspec/changes/` — someone may already be working on this.
4. **For substantive changes to a spec's contract**, write a delta spec in `openspec/changes/<name>/deltas.md` first (ADDED / MODIFIED / REMOVED requirements) before coding.
5. **For completed changes**, move the folder to `openspec/changes/archive/YYYY-MM-DD-<name>/` and apply deltas to the main spec.

## Package layout

```
src/cube/                       Core framework
├── core.py tool.py task.py     Layers 1–3
├── benchmark.py                Layer 4
├── testing.py                  Debug suite
├── server.py                   JSON-RPC / FastAPI
├── cli.py                      `cube` command
├── resource.py                 L1/L2/L3 resource lifecycle
├── container.py                Single-container abstraction
├── backends/                   Docker, Modal, Daytona, Toolkit backends
├── tools/                      Generalist tool ABCs + dep-free concrete impls (browser ABC, terminal)
├── resources/                  BrowserSession, ChatSession protocols
├── integrations/nemogym.py     NemoGym interop
└── _template/                  Scaffold used by `cube init`

cube-resources/                 Optional resource packages (playwright, chat, infra-*)
cube-tools/                     Optional concrete tool packages — one per heavy dep (browser, computer, chat, web)
examples/                       counter-cube (reference), toy_benchmark
tests/                          Unit + integration + backends
```

## Tools architecture

ABCs live in `src/cube/tools/`. Concrete impls live in `cube-tools/cube-<name>-tool/`
when they pull a non-trivial dep; otherwise alongside the ABC. **Tool implementations
never live in cube-harness.** Full rule: [tool/spec.md § Packaging conventions](openspec/specs/tool/spec.md#packaging-conventions).

## Key conventions

- **Serializable configs** subclass `TypedBaseModel` — polymorphic via injected `_type` field.
- **ClassVar registries** on `BenchmarkConfig`: `benchmark_metadata`, `task_metadata`, `task_config_class`, `benchmark_class` are class-level, not constructor params. Auto-loaded from files next to the module (metadata only).
- **Config → Factory** pattern: `XyzConfig.make()` returns a live `Xyz`. Config is serialized across process boundaries; live object never is.
- **`TaskConfig` is the serialization boundary** — workers get a `TaskConfig` and call `.make()` locally. Task objects never cross processes.
- **Credentials** are resolved from env vars at runtime. Never fields on `InfraConfig` or `ContainerBackend` (would be serialized).

## Design docs / RFCs

Active proposals: `openspec/changes/`. Archived: `openspec/changes/archive/`.

## Testing

`make lint` and `make test`. For benchmark debug suite: `cube test <benchmark-name>`.

## What lives elsewhere

- **cube-harness** — runs experiments, agents, trajectories, XRay viewer
- **cube-registry** — metadata registry for published benchmarks (`cube registry add`)
