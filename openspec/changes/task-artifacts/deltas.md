# Deltas — Task Artifacts

## ADDED — `cube.core`: `FileContent` (out-of-line `Content`)

```python
class FileContent(Content):
    data: str            # local filesystem path or remote URL
    mime: str | None = None
    is_remote: bool = False
```

Artifacts reuse the existing `Content` union rather than a new envelope:
screenshots → `ImageContent`, logs → `TextContent`, HAR → `StructuredContent`.
The only gap is *out-of-line* data, so we add one `Content` subclass whose
`data` is a location, not the inlined bytes — for large artifacts (Playwright
traces, recordings) that must not be embedded in trajectory JSON or pickled
across workers. `FileContent` participates in `Content`'s existing polymorphic
(`_type`) serialization; no discriminated union is introduced. `to_markdown()`
renders a link (so XRay shows it); `to_llm_message()` raises — it is a
side-channel reference, never an LLM-facing observation.

## ADDED — `cube.tool`: `AbstractTool.artifacts()` / `AbstractAsyncTool.artifacts()`

```python
class AbstractTool(ABC):
    def artifacts(self) -> list[Content]:
        return []

class AbstractAsyncTool(ABC):
    def artifacts(self) -> list[Content]:
        return []
```

Called after `close()` to collect side-channel outputs; never enters the LLM
observation stream. Must not raise. **Concrete default returning `[]`** (not
abstract) — no subclass is forced to implement it; tools that produce artifacts
override it. `Toolbox.artifacts()` / `AsyncToolbox.artifacts()` concatenate
results from all contained tools.

## ADDED — `cube.task`: `Task.artifacts()`

```python
class Task:
    def artifacts(self) -> list[Content]:
        return []
```

Returns the task's *own* artifacts (override to provide some). Consistent with
`Tool.artifacts()` — each object reports only its own outputs. Tool artifacts are
reached separately via `self.tool.artifacts()`; the harness combines both at episode
end (`task.artifacts() + task.tool.artifacts()`).

## Migration impact

**Fully backwards compatible.** `artifacts()` has a concrete default on
`AbstractTool`/`AbstractAsyncTool`, so existing tools (direct or via
`Tool`/`AsyncTool`) need no change. `Task` defaults return `[]`. No new
required methods anywhere.
