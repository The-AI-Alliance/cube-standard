# Deprecated — cube-standard

Code and artifacts flagged for future cleanup. Each item is actionable — remove,
replace, rewrite, or consolidate. Nothing here is currently in progress; the list
exists so we can pick up simplification work when timing allows.

---

## Dead / placeholder code

### `AudioContent` and `VideoContent` are placeholder classes
[src/cube/core.py:331-370](src/cube/core.py#L331-L370)

Both classes raise `NotImplementedError` for `to_markdown()` and `to_llm_message()`.
No code path in the framework ever constructs them.

```python
class AudioContent(Content):
    data: bytes
    duration_seconds: float | None = None

    def to_markdown(self) -> str:
        raise NotImplementedError("Markdown rendering is not supported for audio content.")

    def to_llm_message(self) -> dict:
        raise NotImplementedError("Audio is not supported by LLM message format.")
```

**Action:** delete both classes. If audio/video support ever materializes, do it as a
concrete openspec change with a real rendering strategy (transcript? thumbnail?).

### `ResourceConfig.source_hash` is declared unused
[src/cube/resource.py:127-128](src/cube/resource.py#L127-L128)

```python
source_hash: str | None = None
"""Content hash for informational purposes; not used for deduplication in v1."""
```

**Action:** remove the field. If v2 dedup is planned, reintroduce with the actual
dedup logic in the same PR.

### Legacy tuple format in `aggregate_profiling`
[src/cube/testing.py](src/cube/testing.py) — search for `(start_ts, end_ts)` tuple handling

The function accepts a legacy `(start_ts, end_ts)` tuple alongside the current
`float | dict` shape. Benchmark authors migrated to the new format.

**Action:** drop the tuple branch; require `float` or `dict`.

### Double `task.close()` in debug episode
[src/cube/testing.py:173-178](src/cube/testing.py#L173-L178)

`run_debug_episode` calls `close()` twice to verify idempotency — once outside the
try and once inside a try/except that writes `close_idempotent_ok`.

**Action:** call `close()` once in `finally`. Move idempotency into a dedicated
unit test in `tests/`.

---

## Legacy parameters / migration debt

### `container_backend` is being replaced by the infra/resource pattern
- Forwarded through [src/cube/task.py:116-118](src/cube/task.py#L116-L118) (constructor field)
- Passed in [src/cube/benchmark.py:570-597](src/cube/benchmark.py#L570-L597) (`spawn()`)
- Explicitly dropped by [src/cube/server.py:133-151](src/cube/server.py#L133-L151) (`_spawn_task_subprocess`, with comment)

`server.py` already comments *"container_backend is intentionally not forwarded —
it is a legacy parameter being replaced by the infra / resource pattern."*

**Action:** remove the parameter from `Task`, `TaskConfig.make`, and `Benchmark`
in a single breaking change. Downstream callers in cube-harness
([episode.py:48,113](https://github.com/The-AI-Alliance/cube-harness/blob/main/src/cube_harness/episode.py#L48-L113)
and [experiment.py:87](https://github.com/The-AI-Alliance/cube-harness/blob/main/src/cube_harness/experiment.py#L87))
must be updated in the same PR.

### `make_benchmark_rpc_server` uses a thread instead of a subprocess
[src/cube/server.py:381-416](src/cube/server.py#L381-L416)

```python
# Uses a **thread** (not a subprocess) because Benchmark instances are not
# guaranteed to be picklable ...
# TODO: once BenchmarkConfig lands ...
```

**Action:** once `BenchmarkConfig` lands (track in
[openspec/changes/core-extensions](openspec/changes/core-extensions/)), switch to
`multiprocessing.Process` mirroring `make_task_rpc_server` for true process
isolation.

### `Task.step` never sets `truncated=True`
[src/cube/task.py:256-257](src/cube/task.py#L256-L257)

```python
# TODO: Add truncation logic based on step limits or time limits
```

**Action:** decide whether truncation is a Task concern or a harness concern. If
Task-level, implement. If harness-level, remove the unused `truncated` field from
`EnvironmentOutput` (or document that it's harness-owned).

---

## Untyped dicts that should be models

### `RuntimeContext` is a bare dict
[src/cube/task.py:39-45](src/cube/task.py#L39-L45)

```python
RuntimeContext = dict[str, Any]
"""
example:
    {"container_id": "abc123", "vm_address": "http://12.34.56.78", "ssh_session": session}
"""
```

Every benchmark stores its own shape; no validation; documented keys rot.

**Action:** introduce a `RuntimeContext(TypedBaseModel)` with typed slots for the
common infra references (server URLs, DB connections, handles). Benchmarks that
need custom fields subclass.

---

## Duplication to consolidate

### `BrowserSession` / `AsyncBrowserSession` are near-duplicates
[src/cube/resources/browser_session.py](src/cube/resources/browser_session.py) (and matching configs)

Two abstracts + two config classes differ only in `async def`. Pattern will
repeat for every new session type.

**Action:** consolidate on one protocol with an async mode flag, or generate one
from the other with a decorator. Same treatment for `ChatSession` siblings if
they appear.

---

## Template drift risk

### `cube init` template ships inside the package
[src/cube/_template/new_cube_package/](src/cube/_template/new_cube_package/) + [src/cube/cli.py:105](src/cube/cli.py#L105)

Any API change in `cube.task`, `cube.benchmark`, `cube.tool` must be reflected in
the template or `cube init` emits stale scaffolding.

**Action:** add a lint-in-CI that imports the template files against the current
package to catch drift. Or run `cube test` against the scaffold as part of CI.
