# Deltas: Agent Owns the Loop (cube-standard companion)

Applies to:

- `openspec/specs/tool/spec.md`
- `openspec/specs/server/spec.md`

See primary RFC: `cube-harness/openspec/changes/agent-owns-loop/`.

---

## MODIFIED — `openspec/specs/tool/spec.md`

### New invariant: monitoring is not a cube-standard concern

Add to the Invariants section:

> Trajectory capture, persistence, and per-call instrumentation are NOT
> responsibilities of `Tool` / `AsyncTool` / `Toolbox`. Runtimes that drive
> tools (cube-harness, future remote runners) attach monitoring by composing
> wrappers around tools — they do not subclass `Tool` to add side-effects.
> Adding storage, summary, or trajectory hooks inside a `Tool` subclass is a
> review-blocking design error.

Rationale: cube-harness's new `MonitoredTool` composes around any
`cube.tool.Tool` or `AsyncTool`. Keeping `Tool` free of monitoring concerns
means the same task can be driven by the in-process harness, by a future
remote runner over `cube.server`, or by any third-party runtime, without
duplicating capture logic inside the tool implementation.

No code change.

### New optional method `async_execute_action` on `AbstractTool` / `AbstractAsyncTool`

```python
class AbstractTool(ABC):
    @abstractmethod
    def execute_action(self, action: Action) -> Any: ...

    async def async_execute_action(self, action: Action) -> Any:
        """Async-shaped facade. Default: call sync `execute_action`
        directly on the current thread (no `to_thread` hop). Tools with
        truly async I/O subclass `AbstractAsyncTool` and let their async
        `execute_action` become the canonical implementation."""
        return self.execute_action(action)


class AbstractAsyncTool(ABC):
    @abstractmethod
    async def execute_action(self, action: Action) -> Any: ...

    async def async_execute_action(self, action: Action) -> Any:
        """Mirror — default delegates to async `execute_action`."""
        return await self.execute_action(action)
```

Rationale: gives every tool — sync or async — a uniform async
call-site. Lets a single container (`AsyncToolbox`) host BOTH sync and
async leaves: dispatch always goes through `async_execute_action`, which
runs sync leaves synchronously (no thread hop) and awaits async leaves
normally. Eliminates the dual `MonitoredTool` / `AsyncMonitoredTool`
class split on the harness side (the harness wraps any inner with one
`MonitoredTool` class and exposes both call shapes).

### `AsyncToolbox` accepts mixed sync + async leaves

```python
class AsyncToolbox(AsyncTool):
    def __init__(self, tools: list[AbstractTool | AbstractAsyncTool]):
        ...

    async def execute_action(self, action: Action) -> Observation | StepError:
        return await self._action_name_to_tool[action.name].async_execute_action(action)
```

`reset()` / `close()` similarly tolerate both — if the leaf method
returns a coroutine, it's awaited; otherwise the sync return is
accepted directly.

Backward compatibility: pure-async toolboxes (the existing usage —
e.g. AsyncBrowserTool) work unchanged because their `async_execute_action`
default already awaits `execute_action`. Existing pure-sync `Toolbox`
behavior is unchanged.

---

## MODIFIED — `openspec/specs/server/spec.md`

### New note in Public API

Add at the end of the Public API section:

> The JSON-RPC endpoints `tools/call` and `cube/step` are the canonical surface
> for external agents that do not run in the same process as the task. A
> harness driving an agent in-process is free to compose monitoring wrappers
> around the task's `Toolbox` (see cube-harness `MonitoredTool`); those
> wrappers are not part of this contract. Remote-agent monitoring, when added,
> will attach on the harness side of the connection and is out of scope for
> the server protocol itself.
>
> **CLI-agent connectors.** cube-harness's planned Phase-2 connectors for
> CLI agents (Codex CLI, Goose, Pi, …) work by launching the binary inside
> the cube's sandbox and pointing it at this JSON-RPC server (over its native
> MCP-compatible wire format). The harness is responsible for: (a) starting
> a per-task server instance scoped to the episode, (b) injecting its URL
> into the subprocess (env var / config file / CLI flag — depending on the
> agent), and (c) shutting the server down on episode finalization. The
> server protocol itself requires no changes to support this — it is the
> per-task `make_task_jsonrpc_app(task)` / `make_task_rpc_server` already
> shipped in Phase 1.

No code change.

---

## ADDED — `openspec/specs/task/spec.md`

### Optional `Task.primitive_toolbox()` method

```python
# Task is generic over (TTMetadata, TTool) per cube-standard:2dfcdeb;
# omitted here for brevity.
class Task(TypedBaseModel, Generic[TTMetadata, TTool], ABC):
    # ... existing methods ...

    def primitive_toolbox(self) -> AsyncToolbox | None:
        """Optional Pi-style primitive toolset for shell-accessible cubes.

        Returns a toolbox exposing the four generic primitives
        (`read`, `write`, `edit`, `bash`) operating on the task's sandbox,
        or `None` if the task has no shell. Default returns `None`.

        Agents that want a primitive-only surface (e.g. PiStyleAgent,
        PiCliAgent) call this; agents that want the rich per-task action
        set use `task.toolbox` (today's behavior) instead. Both shapes
        compose with cube-harness's `MonitoredToolbox` identically.
        """
        return None
```

### Invariants

- Default implementation returns `None`. Shell-based cubes
  (TerminalBench, SWEBench, OSWorld, …) override to return a populated
  toolbox. Browser / API cubes leave it `None`.
- The primitive toolbox and `task.toolbox` (rich action set) are
  independent surfaces; a task may expose both. Agents pick one.
- The four primitive tools (`read`, `write`, `edit`, `bash`) ship from
  a future `cube-tools/cube-shell-tools/` package — **not part of this
  RFC**. Phase 1 only declares the method.

### Phase 2 deliverables (out of scope here, listed for context)

- `cube-tools/cube-shell-tools/` package shipping `ReadTool`, `WriteTool`,
  `EditTool`, `BashTool` parametrized by a `Container`.
- TerminalBench / SWEBench / OSWorld override `primitive_toolbox()` to
  return one of these.
- cube-harness ships a `PiStyleAgent` reference using the primitive
  toolbox, and a `PiCliAgent` that spawns the real Pi CLI as a subprocess.

## REMOVED — none
