# design/ — reference deep-dives

**For authoritative specs, see [`../openspec/specs/`](../openspec/specs/).**
**For active proposals, see [`../openspec/changes/`](../openspec/changes/).**

This directory contains long-form design documents that provide context and rationale
beyond what openspec specs capture. They are reference material — not the contract.
When the code and a design doc disagree, trust the code and update the doc (or the spec).

## Contents

| File | Purpose | Linked spec |
|------|---------|-------------|
| [architecture-diagram.md](../architecture-diagram.md) (repo root) | Visual overview of the 5-layer architecture | — |
| [browser-tool.md](browser-tool.md) | Deep dive on the browser-tool abstraction and how tasks use it | [tool](../openspec/specs/tool/spec.md), [resource](../openspec/specs/resource/spec.md) |
| [docker_wrapper.md](docker_wrapper.md) | Docker containerization design for container backends | [container](../openspec/specs/container/spec.md) |
| [environment-abstraction.md](environment-abstraction.md) | Why Task unifies env dynamics + evaluation | [task](../openspec/specs/task/spec.md) |
| [resource_lifecycle.md](resource_lifecycle.md) | Full design of L1/L2/L3 resource lifecycle | [resource](../openspec/specs/resource/spec.md) |
| [stress_test_specs.md](stress_test_specs.md) | CUBE compliance/latency suite | [testing](../openspec/specs/testing/spec.md) |
| [user_experience.md](user_experience.md) | CLI design and debugging workflow | [cli](../openspec/specs/cli/spec.md) |
| [vm_backend.md](vm_backend.md) | Unified VM abstraction (in flight) | [resource](../openspec/specs/resource/spec.md) |

## What was removed

The following docs were deleted (content preserved in git history):

- `main_specs.md` — superseded by `openspec/specs/`
- `browser-tool-abc-vs-protocol.md` — decision made (uses Protocol)
- `nemogym_lessons_for_cube.md` — lessons learned, not a living spec
- `json-rpc.md` — conversation transcript, not a spec
- `rfc_core_extensions.md` → moved to [`../openspec/changes/core-extensions/`](../openspec/changes/core-extensions/)
- `json-rpc-plan.md` → moved to [`../openspec/changes/json-rpc-streaming/`](../openspec/changes/json-rpc-streaming/)
