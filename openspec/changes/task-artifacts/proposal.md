# Task Artifacts — side-channel outputs from tasks and tools

**Status:** Proposed
**Date:** 2026-05-14
**Scope:** `cube.core`, `cube.task`, `cube.tool`
**Targets:** `dev`
**Related:** Unblocks cube-harness episode artifact export (Playwright traces,
HAR files, screenshots) and OTel artifact attachment.

---

## Problem

Tasks and tools produce side-channel outputs that aren't part of the observation
stream: Playwright traces, HAR network logs, screenshots, container logs. There's
no standard way to collect these after an episode completes, so the harness would
have to reach into tool internals — coupled to specific implementations and not
generalizable.

## Solution

Reuse the existing `cube.core.Content` union as the artifact payload type instead
of inventing a parallel envelope. Screenshots → `ImageContent`, logs →
`TextContent`, HAR → `StructuredContent`. The only thing `Content` can't express
is *out-of-line* data (a path/URL instead of inlined bytes), so we add one
subclass:

1. **`cube.core.FileContent`** — `Content` whose `data` is a location (local path
   or remote URL, `is_remote` flag), with an optional `mime` hint. For large
   artifacts that must not be inlined into trajectory JSON or pickled across
   workers. Uses `Content`'s existing polymorphic serialization — no new
   discriminated union.

2. **`Tool.artifacts() -> list[Content]`** — collected after `close()`; never
   enters the LLM observation stream. **Concrete default returning `[]`** on
   `AbstractTool`/`AbstractAsyncTool` (not abstract — most tools have none, so
   nothing is forced to implement it). `Toolbox` fans out and concatenates.

3. **`Task.artifacts() -> list[Content]`** — the task's *own* artifacts (override
   point, default `[]`). Symmetric with `Tool.artifacts()`: each object reports only
   its own outputs. Tool artifacts stay reachable via `task.tool.artifacts()`.

The harness, after the episode loop (after `close()`), collects
`task.artifacts() + task.tool.artifacts()` and handles export (write to disk, upload,
attach to OTel spans) — a harness concern.

## Why reuse `Content`

`Artifact` + a 4-variant `ArtifactBlob` union would duplicate types we already
have (`TextArtifactBlob`≈`TextContent`, binary≈the `bytes` already on
`Audio`/`VideoContent`). The genuine delta over `Content` is out-of-line storage
(path/URL) plus an explicit `mime` — both folded into one `FileContent`. Bonus:
existing `to_markdown()` means screenshots/logs render in XRay for free.

## Backwards compatibility

Fully backwards compatible. `artifacts()` has a concrete default on the abstract
tool bases, so every existing tool (direct subclass or via `Tool`/`AsyncTool`)
is unaffected; `Task` defaults return `[]`. No new required methods.

## Non-goals

- How artifacts are exported/stored — harness's job.
- Async collection — collected after `close()`, not on the hot loop. Sync is fine.
- Streaming during an episode — batch-collected at episode end.

## Migration

**This PR (cube-standard):**

- `cube.core`: add `FileContent`.
- `cube.tool`: add `AbstractTool.artifacts()` / `AbstractAsyncTool.artifacts()`
  (concrete default `[]`); `Toolbox`/`AsyncToolbox` fan out.
- `cube.task`: add `Task.artifacts()` (task's own artifacts, default `[]`).
- Update specs: `core/spec.md`, `tool/spec.md`, `task/spec.md`.

**Follow-up PRs:**

- cube-standard `feat/playwright-trace-artifacts`: `PlaywrightSession` tracing +
  `SyncPlaywrightTool.artifacts()` returning a `FileContent` for the trace ZIP.
- cube-harness: episode artifact export calling `task.artifacts()` and routing to
  storage/OTel.
