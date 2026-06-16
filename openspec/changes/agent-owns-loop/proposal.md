# RFC: Agent Owns the Loop — cube-standard companion

**Status:** DRAFT
**Author:** Alexandre Lacoste
**Reviewer:** TBD
**Date:** 2026-05-13

**Primary RFC:** `cube-harness/openspec/changes/agent-owns-loop/proposal.md`.
This companion records the (small) cube-standard surface area touched.

---

## Problem

The cube-harness "agent owns the loop" RFC introduces a `MonitoredToolbox` that
wraps tools to capture trajectory events. That wrapper lives in cube-harness
because it captures harness-side state (storage, summary, trajectory). But the
contract between an external agent and a cube task — over JSON-RPC, via
`cube.server` — is already defined here, and we should make sure the new
harness model and the existing RPC layer agree on the boundary.

## Scope

### In

- Add an invariant to `tool/spec.md` clarifying that **trajectory monitoring is
  not a cube-standard concern**. Tools expose `execute_action`; capture and
  persistence belong to whatever runtime is driving them (cube-harness today,
  potentially others tomorrow).
- Add a note to `server/spec.md` clarifying that the existing JSON-RPC
  `tools/call` / `cube/step` endpoints remain the canonical external surface
  for agents that don't run in-process. cube-harness's
  `cube_harness/mcp/server.py` is duplicative and will be retired in a
  follow-up — no action needed here.
- Confirm that `Toolbox` / `AsyncToolbox` are stable enough to be the
  composition target for harness-side `MonitoredToolbox`. No API change.
- Declare an optional `Task.primitive_toolbox() -> AsyncToolbox | None`
  method in `task/spec.md`. Returns a Pi-style primitive toolset
  (`read`/`write`/`edit`/`bash`) for cubes with a shell-accessible sandbox;
  returns `None` by default. **Protocol declaration only in Phase 1;
  concrete implementations are Phase 2.** This seam lets agents declare
  whether they want the rich per-task action set (MCP-style) or the
  primitive toolset (Pi-style) without changing the agent contract.

### Out

- Any change to `cube.server`, `cube.client`, JSON-RPC method shapes,
  WebSocket / streaming (already covered by `json-rpc-streaming`).
- Adding monitoring hooks inside `cube.tool.Tool` — explicitly rejected; that
  would mix concerns.
- Renaming or moving any class. Strictly an invariant clarification.

## Design

Two small textual edits, both to existing specs. See `deltas.md`.

## Migration

None — this is a clarification of existing invariants. Existing code already
respects this separation; the harness side is what's changing.

## Risks

- The "MonitoredToolbox doesn't belong here" position bakes in a separation
  that may need revisiting if/when we add multi-agent or fully-remote agent
  scenarios. Those are Phase 2+ concerns and will get their own RFCs (likely
  building on `json-rpc-streaming`). We're not painting into a corner because
  the harness-side wrapper composes with cube-standard tools, not inherits
  from them.
