# Task Artifacts — typed side-channel outputs from tasks and tools

**Status:** Proposed
**Date:** 2026-05-14
**Scope:** `cube.core`, `cube.task`, `cube.tool`
**Targets:** `main`
**Related:** Unblocks cube-harness episode artifact export (Playwright traces,
HAR files, screenshots) and OTel artifact attachment.

---

## Problem

Tasks and tools produce side-channel outputs that aren't part of the
observation stream: Playwright traces, HAR network logs, screenshots,
container logs. Today there's no standard way to collect these after an
episode completes. cube-harness has ad-hoc code that reaches into tool
internals to extract traces — tightly coupled to specific tool
implementations and not generalizable.

## Solution

Three additions:

1. **`Artifact` + blob types in `cube.core`** — a typed envelope for
   side-channel data. Each artifact has an `id`, a `mime` type, and a
   `blob` that is one of: `BinaryArtifactBlob` (in-memory bytes),
   `FileArtifactBlob` (local path), `RemoteFileArtifactBlob` (URL),
   `TextArtifactBlob` (plain text). The blob union is discriminated on
   `type` for clean serialization.

2. **`Tool.artifacts() -> list[Artifact]`** — abstract method on
   `AbstractTool`. Called after `close()` to collect any artifacts the
   tool produced during the episode. Default on `Tool` returns `[]`.
   `Toolbox` fans out to all contained tools and concatenates results.

3. **`Task.artifacts() -> list[Artifact]`** — collects tool artifacts
   plus any task-specific artifacts. Two methods:
   - `task_artifacts()` — override point for task-specific artifacts
     (default `[]`).
   - `artifacts()` — returns `task_artifacts() + tool.artifacts()`.
     Not intended to be overridden.

The harness calls `task.artifacts()` after the episode loop completes
(after `close()`) and handles export (write to disk, upload to GCS,
attach to OTel spans) — that's a harness concern, not a cube-standard
concern.

## Backwards compatibility

**Fully backwards compatible for `Tool` and `AsyncTool` subclasses.**
The concrete bases provide a default `artifacts()` returning `[]`.

**Breaking for direct `AbstractTool`/`AbstractAsyncTool` subclasses**
that don't go through `Tool`/`AsyncTool`. These must add an
`artifacts()` implementation. In cube-harness, `ToolWithTelemetry` and
`AsyncToolWithTelemetry` are the known cases — they should delegate to
the wrapped tool's `artifacts()`.

**Fully backwards compatible for existing Task subclasses.** Both
`task_artifacts()` and `artifacts()` have default implementations.

## Non-goals

- Defining how artifacts are exported/stored — that's the harness's job.
- Async artifact collection — artifacts are collected after `close()`,
  not during the hot loop. Sync is fine.
- Artifact streaming during an episode — out of scope. Artifacts are
  batch-collected at episode end.

## Migration

**This PR (cube-standard):**

- `cube.core`: add `BinaryArtifactBlob`, `FileArtifactBlob`,
  `RemoteFileArtifactBlob`, `TextArtifactBlob`, `ArtifactBlob`
  (discriminated union), `Artifact`.
- `cube.tool`: add `AbstractTool.artifacts()` abstract method.
  `Tool.artifacts()` returns `[]`. `Toolbox.artifacts()` fans out.
- `cube.task`: add `Task.task_artifacts()` (default `[]`) and
  `Task.artifacts()` (combines task + tool artifacts).
- Update specs: `core/spec.md`, `tool/spec.md`, `task/spec.md`.

**Follow-up PRs:**

- cube-standard `feat/playwright-trace-artifacts`: `PlaywrightSession`
  tracing + `SyncPlaywrightTool.artifacts()` returning the trace ZIP.
- cube-harness: episode artifact export calling `task.artifacts()` and
  routing to storage/OTel.
