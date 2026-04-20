# RFC: CUBE Core Extensions
## Streaming Observations · Async Core · Multi-Agent Schema · Multi-Dimensional Reward

**Status:** Draft
**Date:** March 2026
**Scope:** `cube/core.py`, `cube/task.py`, `cube/tool.py`

---

## Overview

This RFC proposes four targeted extensions to the CUBE core schema. Each is a
backward-compatible addition: no existing abstract method signatures change and
all existing single-agent, synchronous, scalar-reward benchmarks continue to
work without modification.

| # | Feature | Changed types |
|---|---|---|
| 1 | Observation streaming | `Task`, `AbstractTool`, RPC layer |
| 2 | Async core | `Task`, `AbstractTool`, `ToolConfig` |
| 3 | Multi-agent schema | `Action`, `EnvironmentOutput`, `Task` |
| 4 | Multi-dimensional reward | `EnvironmentOutput`, `Task` |

---

## RFC 1 — Observation Streaming

### Motivation

`Task.step()` currently blocks until the tool finishes executing and returns a
complete `Observation`. This model fits tool-calling agents well but cannot
represent:

- **Sensor streams**: a robot arm emitting joint states at 100 Hz while a
  long-running action executes.
- **Video observations**: a camera feed returning 30 frames per second, where
  the harness may want to process frames as they arrive.
- **Incremental text**: an LLM-powered sub-agent streaming its response token
  by token back into the outer loop.
- **Partial results**: a code execution environment streaming `stdout` before
  the process terminates.

### Current state

`Task.step()` ([task.py:203](../src/cube/task.py#L203)) accumulates `Content`
objects synchronously and returns a single `EnvironmentOutput`. There is no
mechanism for the task or tool to emit intermediate content.

`Tool.execute_action()` ([tool.py]) returns `Observation | StepError` — a
complete value, never a generator.

### Proposed API

#### 1.1 `AbstractTool.stream_action()`

```python
class AbstractTool(ABC):
    # Existing — unchanged
    def execute_action(self, action: Action) -> Observation | StepError: ...

    # New — optional streaming variant
    async def stream_action(
        self, action: Action
    ) -> AsyncGenerator[Content, None]:
        """
        Yield Content items as they become available during action execution.

        Default implementation wraps execute_action() and yields the full
        Observation contents at once, so tools that do not implement streaming
        remain compatible.
        """
        result = self.execute_action(action)
        if isinstance(result, Observation):
            for content in result.contents:
                yield content
        else:
            raise result.to_exception()
```

#### 1.2 `Task.stream_step()`

```python
class Task(TypedBaseModel, ABC):
    # Existing step() — unchanged

    async def stream_step(
        self, action: Action
    ) -> AsyncGenerator[Content | EnvironmentOutput, None]:
        """
        Stream Content items produced during action execution, then yield a
        terminal EnvironmentOutput as the final item.

        Yields:
            Content: intermediate observations as they arrive.
            EnvironmentOutput: exactly one, as the last item, containing the
                full accumulated Observation and final reward/done flags.

        The harness MUST consume all items and treat the last one as the step
        result. Stopping early leaves the task in an undefined state.
        """
        accumulated = Observation()
        async for content in self.tool.stream_action(action):
            accumulated += Observation(contents=[content])
            yield content

        done = self.finished(accumulated)
        reward, info = self.evaluate(accumulated) if (done or self.validate_per_step) else (0.0, {})
        obs = self.obs_postprocess(accumulated)
        yield EnvironmentOutput(obs=obs, reward=reward, done=done, info=info)
```

#### 1.3 `ObservationStream` sentinel type

To make it unambiguous that a caller is receiving a stream vs. a snapshot,
introduce a lightweight type alias:

```python
from typing import AsyncGenerator

ObservationStream = AsyncGenerator[Content | EnvironmentOutput, None]
```

#### 1.4 RPC layer

When a task is spawned as a FastAPI server (`benchmark.spawn()`), the streaming
endpoint maps to a Server-Sent Events (SSE) response:

```
POST /cube/stream_step   →   text/event-stream (SSE)
```

Each `Content` is emitted as a `data:` event with JSON payload. The terminal
`EnvironmentOutput` is emitted with `event: done`.

### Backward compatibility

`step()` is unchanged. Harnesses that do not need streaming call `step()` as
before. `stream_step()` has a working default implementation that wraps
`execute_action()`, so existing `Tool` subclasses gain streaming for free.

### Open questions

- Should intermediate `Content` items carry a sequence number or timestamp so
  the harness can detect dropped frames?
- Should `stream_step()` support cancellation (i.e., the harness sends a cancel
  signal mid-stream and the tool aborts)?
- For RPC mode, SSE is simpler but WebSocket enables bidirectional signalling.
  Should the RPC layer support both?

---

## RFC 2 — Async Core

### Motivation

All of `Task.reset()`, `Task.step()`, `Task.evaluate()`, and
`Tool.execute_action()` are synchronous. This means:

1. **A harness running multiple episodes in the same process must use threads**
   to get concurrency. Threads have GIL overhead and are harder to reason about
   than `asyncio` tasks.
2. **Container-backed tasks** block the process on `httpx` calls to the
   container. Long-running physics steps (e.g., MuJoCo at 1000 Hz substeps)
   block the entire harness.
3. **`stream_step()` from RFC 1 is already async.** A synchronous `reset()` is
   inconsistent in an otherwise async task lifecycle.
4. **Timeout handling** is awkward with sync code — it requires
   `concurrent.futures` wrappers rather than `asyncio.wait_for()`.

### Current state

`AbstractTool` has an `async_execute_action()` method ([tool.py]) that wraps
the sync version via `asyncio.get_event_loop().run_in_executor()`. This is a
threadpool escape hatch, not true async. `Task` has no async methods.

### Proposed API

#### 2.1 Optional async overrides on `Task`

All async methods have synchronous default implementations that delegate to
their sync counterparts via `asyncio.to_thread()`. Benchmarks that are
inherently async (e.g., HTTP-backed tools, container-backed tasks) override
the async version directly. Existing synchronous benchmarks do not need to
change.

```python
class Task(TypedBaseModel, ABC):

    # ── Existing sync API (unchanged, still abstract) ─────────────────────

    @abstractmethod
    def reset(self) -> tuple[Observation, dict]: ...

    @abstractmethod
    def evaluate(self, obs: Observation) -> tuple[float, dict]: ...

    def step(self, action: Action | list[Action]) -> EnvironmentOutput: ...

    # ── New async API (optional overrides) ────────────────────────────────

    async def async_reset(self) -> tuple[Observation, dict]:
        """
        Async variant of reset(). Default: runs sync reset() in a thread.
        Override when the task natively supports async (e.g. aiohttp to container).
        """
        return await asyncio.to_thread(self.reset)

    async def async_step(
        self, action: Action | list[Action]
    ) -> EnvironmentOutput:
        """
        Async variant of step(). Default: runs sync step() in a thread.
        """
        return await asyncio.to_thread(self.step, action)

    async def async_evaluate(
        self, obs: Observation
    ) -> tuple[float, dict]:
        """
        Async variant of evaluate(). Default: runs sync evaluate() in a thread.
        """
        return await asyncio.to_thread(self.evaluate, obs)

    async def async_close(self) -> None:
        """
        Async variant of close(). Default: runs sync close() in a thread.
        """
        await asyncio.to_thread(self.close)
```

#### 2.2 `AbstractTool` — promote async to first-class

```python
class AbstractTool(ABC):

    # Keep sync as default-implemented (not abstract) for backward compat
    def execute_action(self, action: Action) -> Observation | StepError:
        """
        Sync execution. Default: runs async_execute_action() in a new event loop.
        Subclasses may override either or both variants.
        """
        return asyncio.run(self.async_execute_action(action))

    @abstractmethod
    async def async_execute_action(
        self, action: Action
    ) -> Observation | StepError:
        """
        Async execution. This becomes the primary abstract method.
        Tools that are inherently sync override execute_action() directly
        and the sync-wrapping default above handles the async call.
        """
        ...
```

> **Note on the abstraction flip:** Currently `execute_action()` is abstract
> and `async_execute_action()` is the derived wrapper. This RFC inverts that:
> `async_execute_action()` becomes abstract, and `execute_action()` becomes the
> sync wrapper. This is a **breaking change for existing Tool subclasses** and
> must be introduced with a deprecation period (see Compatibility below).

#### 2.3 `ToolConfig.make()` — no change

`ToolConfig.make()` returns an `AbstractTool` instance. Since the async API is
an interface on the instance, not the factory, no change is needed here.

#### 2.4 FastAPI server — `async def` handlers

The RPC layer already uses `async def` route handlers. With async tasks, the
handler for `/cube/step` can `await task.async_step(action)` directly without a
threadpool.

### Compatibility

The inversion of `execute_action()` / `async_execute_action()` is the only
breaking change. Migration path:

1. **v0 (current):** `execute_action()` is abstract; `async_execute_action()`
   wraps it.
2. **v1 (transition):** Both are non-abstract with mutual delegation and a
   deprecation warning when the sync-wrapping path is taken.
3. **v2 (target):** `async_execute_action()` is abstract; `execute_action()`
   wraps it.

### Open questions

- Should `Task` carry an `is_async: bool` class variable so harnesses can
  choose whether to call the sync or async path without try/except?
- `asyncio.to_thread()` requires Python 3.9+. Is that already the minimum
  supported version?
- Should `async_reset()` / `async_step()` be advertised on the RPC API
  (`/cube/async_step`) or should the RPC layer always use the async path
  transparently?

---

## RFC 3 — Multi-Agent Schema

### Motivation

The current schema models a single agent interacting with a single environment.
Emerging benchmarks require:

- **Collaborative manipulation**: two robot arms working on a shared object
  (PARTNR, RoboSuite multi-agent, BEHAVIOR-Team).
- **Competitive games**: adversarial agents in a shared world (MuJoCo Soccer,
  Hide-and-Seek).
- **Mixed initiative**: one agent controls locomotion, another controls a
  high-level planner (hierarchical multi-agent RL).
- **Independent but correlated**: multiple agents in the same scene whose
  actions affect each other's observations (multi-robot navigation).

The current `step(action: Action | list[Action])` accepts a list, but it is
interpreted as **sequential actions from one agent**, not **simultaneous actions
from multiple agents**. There is no concept of agent identity anywhere in the
schema.

### Current state

- `Action` ([core.py:106](../src/cube/core.py#L106)): no `agent_id`.
- `EnvironmentOutput` ([core.py:371](../src/cube/core.py#L371)): single `obs`,
  single `reward` — no per-agent structure.
- `Task.action_set` ([task.py:152](../src/cube/task.py#L152)): returns one flat
  list — no per-agent action sets.

### Proposed API

#### 3.1 `agent_id` on `Action`

```python
class Action(TypedBaseModel):
    id: str | None = None           # tool call ID (unchanged)
    name: str                        # unchanged
    arguments: dict[str, Any] = Field(default_factory=dict)  # unchanged

    # New
    agent_id: str = "default"
    """
    Identifier of the agent emitting this action.
    Default "default" preserves single-agent semantics.
    """
```

This is a **non-breaking addition**: all existing `Action` instances default to
`agent_id="default"`.

#### 3.2 `MultiAgentEnvironmentOutput`

Rather than changing `EnvironmentOutput` (which is used throughout), introduce
a subclass:

```python
class MultiAgentEnvironmentOutput(EnvironmentOutput):
    """
    Environment output for multi-agent episodes.

    The top-level obs/reward/done/truncated/info fields carry the
    global environment state. Per-agent data lives in `agents`.
    """

    agents: dict[str, EnvironmentOutput] = Field(default_factory=dict)
    """
    Per-agent outputs, keyed by agent_id.
    Each entry contains the observation, reward, done, and info
    visible to / attributed to that specific agent.

    Agents absent from this dict receive no individual observation
    on this step (e.g., they acted this step but the env has not
    yet responded to them).
    """
```

The global `EnvironmentOutput` fields are the joint environment state:
- `obs`: shared / global observation (e.g., overhead camera all agents see).
- `reward`: joint reward (sum or mean — task-defined).
- `done`: episode terminated for all agents.

Per-agent `EnvironmentOutput` inside `agents`:
- `obs`: agent-local observation (e.g., ego-centric camera for agent `"arm_0"`).
- `reward`: individual reward (e.g., shaped reward for that agent's subtask).
- `done`: whether _this agent_ is done (supports asynchronous episode
  termination per agent).

#### 3.3 `MultiAgentTask`

```python
class MultiAgentTask(Task, ABC):
    """
    Task subclass for benchmarks with more than one agent.

    Adds:
    - agent_ids: declared set of agents in this task.
    - per_agent_action_set: returns action schemas scoped per agent.
    - multi_step: simultaneous step for all agents.
    - multi_evaluate: per-agent and joint reward.

    Single-agent Task.step() / Task.evaluate() are still present for
    compatibility; their defaults delegate to multi_step / multi_evaluate.
    """

    agent_ids: list[str] = Field(
        default_factory=list,
        description="Ordered list of agent identifiers present in this task.",
    )

    @property
    def per_agent_action_set(self) -> dict[str, list[ActionSchema]]:
        """
        Returns available actions per agent.
        Default: all agents share the same action_set from the tool.
        Override to give different agents different capabilities.
        """
        return {agent_id: self.action_set for agent_id in self.agent_ids}

    @abstractmethod
    def multi_step(
        self, actions: dict[str, Action | list[Action]]
    ) -> MultiAgentEnvironmentOutput:
        """
        Execute one simultaneous step for all agents.

        Args:
            actions: mapping from agent_id to the action(s) for that agent.
                     Agents not present in the dict are assumed to issue a
                     no-op action.

        Returns:
            MultiAgentEnvironmentOutput with per-agent observations and rewards.
        """
        ...

    @abstractmethod
    def multi_evaluate(
        self, obs: dict[str, Observation]
    ) -> tuple[float, dict[str, float], dict]:
        """
        Evaluate state for all agents simultaneously.

        Returns:
            joint_reward (float): global reward for the episode.
            per_agent_rewards (dict[str, float]): individual rewards.
            info (dict): additional metadata.
        """
        ...

    # ── Compatibility bridges ─────────────────────────────────────────────

    def step(self, action: Action | list[Action]) -> EnvironmentOutput:
        """
        Single-agent compatibility shim.
        Routes single-agent action through multi_step() using agent_id="default".
        """
        actions = [action] if isinstance(action, Action) else action
        result = self.multi_step({"default": actions})
        # Return the global env output (single-agent harnesses ignore agents dict)
        return result

    def evaluate(self, obs: Observation) -> tuple[float, dict]:
        joint_reward, _, info = self.multi_evaluate({"default": obs})
        return joint_reward, info
```

#### 3.4 RPC endpoints

```
POST /cube/multi_step        body: {actions: {agent_id: Action | Action[]}}
                             returns: MultiAgentEnvironmentOutput

GET  /cube/agent_ids         returns: {agent_ids: string[]}
GET  /cube/per_agent_tools   returns: {agent_id: ActionSchema[]}
```

### Backward compatibility

- `Action.agent_id` defaults to `"default"` — no existing serialized data
  breaks.
- `MultiAgentTask` is a separate subclass. Existing `Task` subclasses are
  unaffected.
- Harnesses that call `task.step()` on a `MultiAgentTask` get the global
  `EnvironmentOutput` via the compatibility shim.

### Open questions

- Should `agent_ids` be declared on `TaskMetadata` (for pre-episode
  configuration) or only on `Task` (runtime)?
- How should asynchronous agent termination interact with `done`? If agent A
  finishes but agent B has not, should the harness keep calling `multi_step()`
  with only agent B's actions?
- Should `per_agent_action_set` be exposed on `TaskMetadata` to allow harnesses
  to configure model routing before task instantiation?

---

## RFC 4 — Multi-Dimensional Reward

### Motivation

`EnvironmentOutput.reward` is a single `float`. This is insufficient for:

- **Reward shaping research**: logging and analysing individual components
  (goal progress, efficiency, safety) separately.
- **RL training diagnostics**: knowing which reward term is driving or
  suppressing learning.
- **Multi-objective RL**: Pareto-optimal policy search where objectives must
  remain separate.
- **Evaluation reporting**: publishing a benchmark result as a breakdown
  (`{correctness: 0.8, efficiency: 0.6, safety: 1.0}`) rather than a single
  opaque scalar.
- **NeMo Gym interoperability**: NeMo Gym's `reward_profiles` expects named
  reward dimensions (see interop report §5.3).

### Current state

`EnvironmentOutput.reward: float` ([core.py:387](../src/cube/core.py#L387)).
`Task.evaluate()` returns `tuple[float, dict]`. The `dict` already carries
arbitrary info, and some benchmarks informally put reward components there, but
there is no standard key or type.

### Proposed API

#### 4.1 `reward_breakdown` field on `EnvironmentOutput`

```python
class EnvironmentOutput(TypedBaseModel):
    obs: Observation
    reward: float = 0.0              # unchanged — scalar aggregate
    done: bool = False
    truncated: bool = False
    info: dict = Field(default_factory=dict)
    error: StepError | None = None

    # New
    reward_breakdown: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Named reward components that sum to (or otherwise compose) `reward`. "
            "Keys are benchmark-defined component names (e.g. 'goal_progress', "
            "'efficiency', 'safety'). Empty dict means the task does not provide "
            "a breakdown. Harnesses MUST NOT require this field to be non-empty."
        ),
    )
```

`reward` remains the authoritative scalar. `reward_breakdown` is informational.
Tasks that do not implement breakdown leave it empty — existing harnesses see no
change.

#### 4.2 Extended `Task.evaluate()` signature — opt-in

Changing the abstract `evaluate()` signature would break all existing
subclasses. Instead, introduce an optional companion method:

```python
class Task(TypedBaseModel, ABC):

    @abstractmethod
    def evaluate(self, obs: Observation) -> tuple[float, dict]:
        """
        (unchanged)
        Returns (scalar_reward, info).
        """
        ...

    def evaluate_detailed(
        self, obs: Observation
    ) -> tuple[float, dict[str, float], dict]:
        """
        (Optional override) Returns (scalar_reward, reward_breakdown, info).

        Default: calls evaluate() and returns an empty breakdown.
        Benchmarks that want to expose components override this method.
        The scalar_reward MUST equal or be derivable from reward_breakdown.
        """
        reward, info = self.evaluate(obs)
        return reward, {}, info
```

`Task.step()` is updated to call `evaluate_detailed()` and populate
`reward_breakdown` on `EnvironmentOutput`:

```python
def step(self, action: Action | list[Action]) -> EnvironmentOutput:
    ...
    if done or self.validate_per_step:
        reward, reward_breakdown, info = self.evaluate_detailed(obs)
    ...
    return EnvironmentOutput(
        obs=obs,
        reward=reward,
        reward_breakdown=reward_breakdown,
        done=done,
        info=info,
        error=error,
    )
```

#### 4.3 Convention: `reward_breakdown` must be consistent with `reward`

The spec requires that the components in `reward_breakdown` are semantically
consistent with the scalar `reward`. The most common patterns:

| Pattern | Convention |
|---|---|
| Weighted sum | `reward = sum(w_i * v_i)`, breakdown lists `v_i` (not `w_i * v_i`) |
| Minimum gate | `reward = min(v_i)`, breakdown lists each `v_i` |
| Single component | `reward_breakdown = {"reward": reward}` (trivial) |

The exact aggregation is task-defined. Harnesses must not assume any particular
aggregation.

#### 4.4 Shared rollout format alignment

The `reward_breakdown` field aligns directly with §7.3 of the CUBE × NeMo Gym
Interoperability Report's proposed shared rollout JSON format:

```json
{
  "reward": 0.85,
  "reward_breakdown": {"correctness": 1.0, "efficiency": 0.8, "safety": 0.75}
}
```

NeMo Gym harnesses consuming CUBE trajectories can read `reward_breakdown`
directly as `reward_profiles`.

### Backward compatibility

- `reward_breakdown` defaults to `{}` — no existing serialized data breaks.
- `evaluate()` signature is unchanged — all existing subclasses compile.
- `evaluate_detailed()` has a working default — no subclass is required to
  implement it.

### Open questions

- Should `reward_breakdown` keys be validated against a task-declared schema
  (e.g., a `reward_components: list[str]` on `TaskMetadata`) to prevent
  inconsistent key names across episodes?
- Should the harness aggregate `reward_breakdown` across steps (cumulative
  component returns) automatically, or is that always the harness's
  responsibility?
- For multi-agent tasks (RFC 3), should `MultiAgentEnvironmentOutput.agents`
  carry per-agent `reward_breakdown` as well?

---

## Summary of Schema Changes

### `core.py`

| Type | Change | Breaking? |
|---|---|---|
| `Action` | Add `agent_id: str = "default"` | No |
| `EnvironmentOutput` | Add `reward_breakdown: dict[str, float] = {}` | No |
| `MultiAgentEnvironmentOutput` | New subclass | No |
| `ObservationStream` | New type alias | No |

### `task.py`

| Type | Change | Breaking? |
|---|---|---|
| `Task` | Add `async_reset/step/evaluate/close()` | No |
| `Task` | Add `stream_step()` | No |
| `Task` | Add `evaluate_detailed()` | No |
| `Task.step()` | Calls `evaluate_detailed()`, populates `reward_breakdown` | No |
| `MultiAgentTask` | New subclass | No |

### `tool.py`

| Type | Change | Breaking? |
|---|---|---|
| `AbstractTool` | Add `stream_action()` with default | No |
| `AbstractTool` | Flip abstract from `execute_action` to `async_execute_action` | **Yes — v2 only** |

---

## Appendix: Interaction Between RFCs

The four extensions compose cleanly:

- **Async + Streaming**: `stream_step()` (RFC 1) is defined as `async` (RFC 2).
  This is not accidental — streaming is inherently async.
- **Multi-agent + Async**: `multi_step()` (RFC 3) should also have an
  `async_multi_step()` variant (RFC 2 pattern).
- **Multi-agent + Multi-dim reward**: `multi_evaluate()` (RFC 3) already
  returns `(joint_reward, per_agent_rewards, info)`, which maps naturally to
  `reward_breakdown` (RFC 4) where keys are agent IDs.
- **Streaming + Multi-agent**: `stream_step()` can yield `Content` tagged with
  `agent_id` if needed. This is left as a future extension.
