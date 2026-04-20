# OpenSpec — cube-standard

Machine-friendly specifications for coding agents. Each spec describes a capability:
its contract, invariants, and constraints. Read these before modifying code in the
corresponding layer.

## Structure

```
openspec/
├── specs/         # Living contracts, one dir per capability
│   ├── core/      # Action, Observation, Content, EnvironmentOutput
│   ├── tool/      # Tool, AsyncTool, ToolConfig, @tool_action
│   ├── task/      # Task, TaskMetadata, TaskConfig
│   ├── benchmark/ # Benchmark, BenchmarkMetadata, class-level registry
│   ├── resource/  # ResourceConfig, InfraConfig, ResourceHandle (L1/L2/L3)
│   ├── container/ # ContainerConfig, Container, ContainerBackend
│   ├── server/    # FastAPI benchmark server protocol
│   ├── cli/       # `cube init`, `cube list`, `cube test`
│   └── testing/   # run_debug_suite, assert_debug_tasks_reward_one
└── changes/       # Active RFCs / proposals (moved from design/)
```

## For humans

Narrative guides, tutorials, and architecture overviews live in [../docs/](../docs/).
Docs cross-reference specs — see any tutorial's "contract" section for links.

## Writing style

Specs are terse. They define WHAT code must do. They do NOT explain WHY (that's for
design/changes) or HOW to use the library (that's for docs). Each spec covers:

- **Purpose** — one sentence
- **Public API** — types, methods, signatures
- **Invariants** — what must always hold
- **Contracts** — what implementers must guarantee
- **Gotchas** — non-obvious constraints
