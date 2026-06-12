"""
Task management for CUBE.

This module defines the Task base class, TaskMetadata, and TaskConfig for
implementing and configuring individual benchmark tasks. By design, Task
unifies gym-like environment dynamics (reset/step/close) and task-specific
logic (evaluate/_filter_actions/obs_postprocess) in a single class, so that
benchmark authors have one coherent place to define both what the agent can
do and how it is evaluated.

Abstract classes:
    Task — subclasses must implement:
        reset() -> (Observation, dict)        set up initial state, return first obs
        evaluate(obs: Observation) -> (float, dict)   score the current state
    TaskConfig — subclasses must implement:
        make(...) -> Task     instantiate the Task from serialized config data
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, Generic, List, Literal, Tuple

from pydantic import ConfigDict, Field, PrivateAttr, SerializeAsAny
from typing_extensions import TypeVar

from cube import get_cache_dir
from cube.container import Container
from cube.core import (
    STOP_ACTION as STOP_ACTION,  # re-exported: `from cube.task import STOP_ACTION` keeps working
)
from cube.core import (
    Action,
    ActionSchema,
    AgentStop,
    Content,
    EnvironmentOutput,
    Observation,
    StepError,
    StructuredContent,
    TypedBaseModel,
)
from cube.resource import ContainerConfig, ResourceHandle
from cube.tool import AbstractTool, ToolConfig

# Type parameters for ``Task``. ``TTMetadata`` narrows ``self.metadata``; ``TTool``
# narrows ``self.tool`` so cubes that bind to a specific tool surface (e.g.
# ``TerminalTool``, ``BrowserTool``) can drop ``isinstance`` asserts and
# per-cube property overrides. Defaults keep ``Task[Meta]`` working as before
# — ``TTool`` resolves to ``AbstractTool``. ``typing_extensions.TypeVar`` is
# used (not the stdlib) because ``default=`` on a ``TypeVar`` is PEP 696,
# which the stdlib added in Python 3.13; we still support 3.12.
TTMetadata = TypeVar("TTMetadata", bound="TaskMetadata")
TTool = TypeVar("TTool", bound=AbstractTool, default=AbstractTool)

RuntimeContext = dict[str, Any]
"""
Type alias for shared infrastructure references created during benchmark.setup().

example:
    {"container_id": "abc123", "vm_address": "http://12.34.56.78", "ssh_session": session}
"""

logger = logging.getLogger(__name__)

# STOP_ACTION + AgentStop now live in cube.core (STOP is a real tool action — Tool.final_step
# — that raises AgentStop; there's no STOP special-casing here anymore).


class TaskMetadata(TypedBaseModel):
    """
    Lightweight, eager-loaded metadata describing a task.

    Lives in the wheel — ships next to the cube package and powers
    ``cube list``, registry listings, glob-based subsetting, and human
    inspection. Heavy per-task data (problem statements, patches, archives,
    evaluator scripts, …) does NOT belong here; put it on a
    ``TaskExecutionInfo`` subclass and surface it via ``Task.execution_info``.

    Cube authors needing per-task fields beyond the defaults subclass
    ``TaskMetadata`` with named, typed fields. Polymorphism is preserved
    through the ``TypedBaseModel`` ``_type`` discriminator.

    Used by:
    - Task: metadata attribute
    - API endpoint: cube/tasks (list of TaskMetadata in response)

    Attributes:
        id (str): Unique task identifier
        split (Literal["train", "val", "test"]): Split for the task (default: "test")
        abstract_description (str): Broad description of the task for searching and filtering only. The task objective is part of the first Observation returned by task.reset(). (default: "")
        recommended_max_steps (int | None): Recommended maximum number of steps to help harness prevent infinite running agents. Not a hard limit, the task can still run longer if needed. (default: None)
        container_config (ContainerConfig | None): Optional container configuration for this task (default: None, meaning no container needed).
    """

    id: str = Field(..., description="Unique task identifier")
    split: Literal["train", "val", "test"] = Field(default="test", description="Split for the task")
    abstract_description: str = Field(
        default="",
        description="Broad description of the task for searching and filtering only. The task objective is part of the first Observation returned by task.reset().",
    )
    recommended_max_steps: int | None = Field(
        default=None,
        description="Recommended maximum number of steps to help harness prevent infinite running agents. Not a hard limit, the task can still run longer if needed.",
    )
    container_config: ContainerConfig | None = Field(
        default=None,
        description="Optional container configuration for this task (defaults to None, meaning no container needed).",
    )


class TaskExecutionInfo(TypedBaseModel):
    """Heavy, lazy per-task execution data surfaced via ``Task.execution_info``.

    Cube authors subclass with typed named fields (problem statements,
    patches, archives, evaluator scripts, …). Polymorphic via the
    ``TypedBaseModel`` ``_type`` discriminator.

    Populated on the worker — typically inside ``TaskConfig.make()`` by
    validating ``self.load_task_execution_info()`` against the subclass.

    Cubes with no heavy data leave the slot ``None``; the base class is
    instantiable but carries no fields.
    """


class Task(TypedBaseModel, Generic[TTMetadata, TTool], ABC):
    """
    Represents a task that an agent must complete in an environment.

    On construction, the base class launches the container (if configured) and then
    calls tool_config.make(container) to create the tool, so the tool can connect
    to the container from the start.

    This class contains:
    1. task logic:
        + evaluate(obs?) -> (float, dict)         abstract — score the current state
        - _filter_actions(actions, role?) -> actions  optional whitelist/mask of advertised actions
        - obs_postprocess(obs, role?) -> obs      optional per-seat observation post-processing
        - finished(obs?) -> bool                  optional early-termination check
        - get_privileged_info() -> Content        optional privileged task info

    2. gym-like environment dynamics:
        + reset() -> (Observation, dict)           abstract — set up initial state, return first obs
        - step(action) -> EnvironmentOutput        execute action via tool, evaluate if done
        - close()                                  optional resource cleanup

    Type parameters:
        ``TTMetadata`` (bound ``TaskMetadata``) narrows ``self.metadata`` so
        cubes don't have to re-annotate the field.

        ``TTool`` (bound ``AbstractTool``, default ``AbstractTool``) narrows
        ``self.tool`` to a specific tool surface (e.g. ``TerminalTool``).
        Cubes that bind it drop the ``isinstance(self.tool, FooTool)`` asserts
        and per-cube property overrides — ``self.tool`` is the right type by
        construction. Omitting the parameter is equivalent to
        ``Task[Meta, AbstractTool]``; existing cubes keep working unchanged.

    Three equivalent forms at runtime:

        # Bare — ``self.metadata: TaskMetadata``, ``self.tool: AbstractTool``.
        class FooTask(Task): ...

        # Metadata only — ``self.tool`` stays ``AbstractTool``.
        class FooTask(Task[FooTaskMetadata]): ...

        # Both — ``self.tool: FooTool``, no property override needed.
        class FooTask(Task[FooTaskMetadata, FooTool]): ...
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Serializable fields — SerializeAsAny preserves subclass-specific fields
    # through JSON round-trip (Pydantic otherwise strips to the declared base type).
    metadata: SerializeAsAny[TTMetadata]
    # Same rationale for TaskExecutionInfo and ToolConfig below.
    execution_info: SerializeAsAny[TaskExecutionInfo] | None = Field(
        default=None,
        description=(
            "Heavy, lazy per-task execution data (problem statements, patches, archives, …). "
            "Populated inside ``TaskConfig.make()`` / ``Task.model_post_init`` / "
            "``Task.reset()``. Cubes with no heavy data leave this None."
        ),
    )
    tool_config: SerializeAsAny[ToolConfig] = Field(description="Tool configuration used to instantiate the tool.")
    runtime_context: RuntimeContext | None = Field(
        default=None, description="Optional shared infrastructure references from Benchmark._setup()."
    )
    validate_per_step: bool = Field(
        default=False,
        description="If True, evaluate() is called after every step instead of only when the task is done.",
    )

    # Non-serializable runtime state, set during model_post_init
    _tool: TTool | None = PrivateAttr(default=None)
    _container: Container | None = PrivateAttr(default=None)
    _resource_handle: ResourceHandle | None = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Called after Pydantic __init__. Launches the container (if configured), runs the
        eager world setup (``prepare_world``), then builds the task's own tool ``_tool``
        (the no-role / admin handle used by reset / evaluate / finished). Per-seat agent
        tools are made on demand by ``get_agent_view(role)``.
        """
        cc = self.metadata.container_config
        if cc is not None and self.runtime_context is not None and "infra" in self.runtime_context:
            from cube.task_infra import launch_task_container  # local import avoids circular dep

            self._resource_handle, self._container = launch_task_container(
                self.runtime_context,
                name=self.metadata.id,
                image=cc.image,
                ram_gb=cc.ram_gb,
                cpu_cores=cc.cpu_cores,
            )

        # The task's own no-role tool (single-agent's tool, or a shared-world admin tool).
        # A *multi-agent* task whose tools are strictly per-role opts out by raising
        # NotImplementedError in _make_tool(None); we leave _tool unset and `tool` raises a
        # clear error pointing to get_agent_view(role). For a single-agent task the no-role
        # tool is mandatory, so we DON'T mask a NotImplementedError there — it's a real bug
        # (e.g. an unimplemented abstract method) and should surface immediately.
        try:
            self._tool = self._make_tool()  # cube code uses self._tool
        except NotImplementedError:
            if self.agent_roles() == {None: 1}:
                raise
            self._tool = None

    def _make_tool(self, role: "str | None" = None) -> TTool:
        """Create a tool — a session over the world. **The single tool-lifecycle hook**:
        a cube does any once-per-task world prep (relocate a read-only dir, fix perms) AND
        builds the tool here. Default ignores ``role`` and just calls
        ``tool_config.make(container)``. A multi-agent cube overrides to bind the role (a
        role-specific tool, or thread ``role`` into a role-aware ``tool_config.make``).
        Called once for the task's own ``_tool`` (role=None), and per seat by
        ``get_agent_view``; each call returns a fresh session.
        """
        return self.tool_config.make(container=self._container)  # type: ignore[return-value]

    def agent_roles(self) -> "dict[str | None, int]":
        """The roster: role → seat count. Default ``{None: 1}`` (single-agent). A
        multi-agent cube overrides, e.g. ``{"buyer": 2, "seller": 1}``."""
        return {None: 1}

    def get_agent_view(self, role: "str | None" = None) -> "AgentView":
        """The agent-facing view for an agent. The base implements **only** the single-agent
        case (``role=None``, reusing the task's own tool). A multi-agent benchmark **must
        override** this to build each seat's view — assigning the seat index and per-role
        tool (``_make_tool(role)``) **internally**. The runtime never passes a seat: it calls
        this once per seat declared in ``agent_roles()``, and the benchmark hands out the
        right view (e.g. tracking an internal per-role counter).
        """
        if role is None:
            return AgentView(self, role=None, tool=self._tool)
        raise NotImplementedError(
            f"{type(self).__name__} declares role {role!r} in agent_roles() but does not override "
            "get_agent_view() to build that seat's view — multi-agent benchmarks must implement it."
        )

    @property
    def tool(self) -> TTool:
        """The task's own underlying tool — what the task itself uses (reset / evaluate /
        setup) and what cube-standard internals (server, nemogym, debug suite) drive.

        This is the raw environment tool, NOT an agent surface: agents never hold it; they
        get an :class:`AgentView` from :meth:`get_agent_view` (which wraps a tool with the
        per-agent identity + eval callback). Backed by ``_tool``, built in ``model_post_init``.

        Raises if the task has no no-role tool — a multi-agent task whose tools are strictly
        per-role (its ``_make_tool(None)`` raises ``NotImplementedError``). Such a task has no
        single ``tool``; drive each seat via ``get_agent_view(role)`` instead.
        """
        if self._tool is None:
            raise RuntimeError(
                f"{type(self).__name__} has no no-role tool (its _make_tool(None) raised "
                "NotImplementedError). Use get_agent_view(role) for each seat's tool."
            )
        return self._tool  # type: ignore[return-value]

    @property
    def container(self) -> Container | None:
        return self._container

    @property
    def id(self) -> str:
        return self.metadata.id

    def _filter_actions(self, actions: List[ActionSchema], role: "str | None" = None) -> List[ActionSchema]:
        """(Optional benchmark-dev hook) Restrict which tool actions are *advertised* to an
        agent — a whitelist / mask over the tool's ``action_set``. Default exposes everything.

        Recomputed on every ``action_set`` access, so a cube can vary it across an episode
        from task state (phase gating, legal-action masking). Applied to BOTH the agent view
        (:meth:`AgentView.action_set`) and the gym view (:meth:`action_set`), so the two never
        diverge. ``role`` is the seat's role (``None`` for the gym/single-agent view).

        Advisory: it shapes what the agent *sees*, not execute-time enforcement — a tool still
        runs whatever action reaches it. No cube needs hard rejection yet; if one does, the
        single chokepoint is :meth:`AgentView.execute_action`. (Per-role action *sets* are
        better expressed by ``_make_tool(role)`` returning a role-bound tool; use this hook
        for task-state-dependent masking the tool itself can't see.)
        """
        return actions

    @property
    def action_set(self) -> List[ActionSchema]:
        """The gym-view action set (litellm-compatible) — the task's own tool's actions after
        ``_filter_actions`` (role=None). Already includes ``final_step`` (every Tool exposes
        it). Mirrors what an :class:`AgentView` advertises, so the gym and agent paths never
        diverge."""
        return self._filter_actions(self.tool.action_set)

    @abstractmethod
    def reset(self) -> Tuple[Observation, Dict]:
        """
        Reset the task to its initial state.
        Must call self._tool.reset() to reset the tool as well.

        Returns:
            Tuple of (Observation, dict with additional task info)
        """
        pass

    def step(self, action: Action | List[Action]) -> EnvironmentOutput:
        """Execute an action (or a sequential batch) and return the next state.

        This is the **gym-compatibility view** — FINALIZED; do NOT override. Each action
        goes through ``self._tool.execute_action`` (the same dispatch the agent-facing
        ``AgentView`` uses). The batch is run, then ``finished`` / ``evaluate`` are applied
        once (per batch); a tool error is folded into the observation (non-terminal);
        ``final_step`` raises :class:`AgentStop`, caught here to set ``done=True``.

        Args:
            action: Agent action (or list of actions, run sequentially).
        Returns EnvironmentOutput containing:
            observation: next state
            reward: reward signal (0.0 is not available)
            terminated: Task completed successfully
            truncated: Task hit limit (time, steps)
            info: Additional metadata, always includes a "profiling" key with wall-clock
                  timings (in seconds) for each phase:
                    - "tool_execute": dict with "total", "avg_per_action", "n_actions"
                    - "evaluate": float (only present when evaluate() was called)
                    - "obs_postprocess": float
        """
        actions = [action] if isinstance(action, Action) else action
        done = False
        reward = 0.0
        info = {}
        obs = Observation()  # will populate list of content after each action
        error: StepError | None = None
        action_times: list[float] = []
        for action in actions:
            t0 = time.perf_counter()
            try:
                result = self._tool.execute_action(action)
            except AgentStop as stop:
                obs += stop.observation
                done = True
                break
            # A tool error is folded into the action's obs (non-terminal); surface the
            # structured error on EnvironmentOutput.error too (obs += merges only contents).
            if result.error is not None:
                error = result.error
            obs += result
            action_times.append(time.perf_counter() - t0)
        done = done or self.finished(obs)
        # TODO: Add truncation logic based on step limits or time limits
        profiling: dict[str, Any] = (
            {
                "tool_execute": {
                    "total": sum(action_times),
                    "avg_per_action": sum(action_times) / len(action_times),
                    "n_actions": len(action_times),
                }
            }
            if action_times
            else {}
        )
        if done or self.validate_per_step:
            t_eval_start = time.perf_counter()
            reward, info = self.evaluate(obs)
            profiling["evaluate"] = time.perf_counter() - t_eval_start
        t_post_start = time.perf_counter()
        obs = self.obs_postprocess(obs)
        profiling["obs_postprocess"] = time.perf_counter() - t_post_start
        info["profiling"] = profiling
        return EnvironmentOutput(obs=obs, reward=reward, done=done, info=info, error=error)

    def obs_postprocess(self, obs: Observation, role: "str | None" = None) -> Observation:
        """(Optional) Post-process an observation before it reaches the agent. Default does
        nothing. The seat's ``role`` is passed (``None`` for the gym/single-agent view) so a
        shared-world multi-agent task can shape per-role views off the one tool — the twin of
        :meth:`_filter_actions` (the two per-seat view-shaping hooks; ``role`` belongs on
        exactly these, not on the world-global ``evaluate`` / ``reset`` / ``finished``).
        """
        return obs

    @abstractmethod
    def evaluate(self, obs: Observation | None = None) -> Tuple[float, dict]:
        """Validate the current state of the task and return (reward, info).

        ``obs`` is optional because many tasks derive the score entirely from
        internal tool state (e.g. a counter value, a VM screenshot taken inside
        the tool) and do not need the last observation passed back.  Callers
        that rely on observation content should pass it explicitly; callers
        that don't can omit it: act, act, evaluate.
        """
        pass

    def get_privileged_info(self) -> Content:
        """
        (Optional) Return privileged information about the task such as:
        - solution: list[Action] = Solve the task using a pre-defined solution.
        - evaluation_function_source_code: str
        - environment internal state summaries
        """
        return StructuredContent(data={})  # empty content by default, override to provide something else

    def get_status(self) -> str:
        """
        (Optional) Return current task status.
        Consider looking into self.runtime_context and/or self.container.
        """
        # TODO: figure out if we want to provide some standard for this?
        return ""

    def finished(self, obs: Observation | None = None) -> bool:
        """(Optional) Check if the task is finished."""
        return False

    def close(self) -> None:
        """
        Cleanup task resources. Calls self._tool.close() automatically.
        Override to add task-specific cleanup, and call super().close() to
        ensure the tool is also cleaned up.

        Examples of additional task-specific cleanup:
        - Stop containers
        - Remove temp files
        - Close network connections
        """
        self._tool.close()
        if self._resource_handle is not None:
            self._resource_handle.close()
            self._resource_handle = None
            self._container = None
        elif self._container is not None:
            self._container.stop()
            self._container = None


class AgentView:
    """The agent-facing view of a :class:`Task` — the ONLY surface an agent holds.

    Obs in, action out, no lifecycle. An agent gets exactly two things: ``action_set``
    (what it may do *now*) and ``execute_action`` (do one thing, see the result). It never
    sees ``reset`` / ``evaluate`` / ``close`` / ``step`` — the runtime drives those on the
    Task. Obtained from ``task.get_agent_view(role)``: ``role=None`` (single-agent) over the
    task's own tool, or one per seat for a multi-agent task.

    Each tool carries its ``role`` (``None`` for single-agent; e.g. ``"buyer"``) and
    ``seat`` index — the role drives the per-seat ``action_set`` and the stable ``agent_id``.

    Not a :class:`Tool` (it exposes no actions of its own) — it is a *facet* of a Task.
    """

    def __init__(
        self, task: "Task", role: "str | None" = None, tool: "AbstractTool | None" = None, seat: int = 0
    ) -> None:
        self._task = task
        self.role = role
        self.seat = seat
        # The actor's tool/session — its own (role-bound) instance for a named role, or
        # the task's default tool for the no-role seat. Dispatch goes through THIS tool,
        # so "which agent acted" is implicit in which session ran the action.
        self._tool = tool if tool is not None else task._tool
        self._eval_callback: "Callable[[float, dict], None] | None" = None

    def set_eval_callback(self, callback: "Callable[[float, dict], None]") -> None:
        """Register a runtime callback invoked with ``(reward, info)`` whenever a per-step
        evaluation fires (i.e. when ``task.validate_per_step`` is set). This is how the
        runtime *recuperates* the per-step eval that ``execute_action`` triggers — without
        polling, and without the reward ever reaching the agent (the agent only sees obs).
        A ``validate_per_step`` task with no callback registered is a wiring bug: the per-step
        reward would be silently dropped, so ``execute_action`` raises instead (see
        :meth:`_maybe_evaluate`).
        """
        self._eval_callback = callback

    def _maybe_evaluate(self, obs: Observation) -> None:
        """Trigger the per-step evaluation in the agent path, mirroring what gym ``step``
        does for ``validate_per_step`` — but surfaced out-of-band via the callback (reward
        is not the agent's concern), not folded into the returned obs.

        If the task sets ``validate_per_step`` but no callback is registered, the per-step
        reward has nowhere to go — a silent drop. That's a runtime wiring bug, so we raise
        loudly rather than discard it."""
        if not self._task.validate_per_step:
            return
        if self._eval_callback is None:
            raise RuntimeError(
                f"{type(self._task).__name__} sets validate_per_step=True, but no eval callback is "
                f"registered on this AgentView ({self.agent_id}). The runtime must call "
                "set_eval_callback() so per-step rewards are recuperated (they never reach the "
                "agent). See AgentView.set_eval_callback."
            )
        reward, info = self._task.evaluate(obs)
        self._eval_callback(float(reward), dict(info))

    @property
    def agent_id(self) -> str:
        """Stable per-seat id: ``"agent"`` for the single default seat, else
        ``"{role}-{seat}"`` (e.g. ``"buyer-0"``)."""
        return "agent" if self.role is None else f"{self.role}-{self.seat}"

    @property
    def action_set(self) -> List[ActionSchema]:
        """The actions advertised to this seat *right now* — its tool's actions after the
        task's ``_filter_actions(role)``. Per-role action sets mostly fall out of each seat
        holding a different (role-bound) tool; the filter adds task-state-dependent masking.

        Recomputed on every access, so a cube *may* vary it over an episode (legal-action
        masking / phase gating / real-time observe-no-op). In practice this is **rare** —
        almost every cube returns a static set — so treat the dynamic capability as
        available-but-uncommon. (Most agents also snapshot the set at construction today;
        re-reading it per turn is a forward extension for cubes that need it.)
        """
        return self._task._filter_actions(self._tool.action_set, self.role)

    def execute_action(self, action: Action) -> Observation:
        """Run one action through THIS seat's tool and return its (post-processed)
        observation.

        Relays to the Task's per-action core — the *same* execution as the gym ``step``
        view — and returns the **observation only**: no reward, no ``done``. When
        ``task.validate_per_step`` is set it triggers the per-step ``evaluate`` here (like
        ``step`` does) and surfaces ``(reward, info)`` through the registered eval callback
        — out-of-band, never in the returned obs. ``finished`` (episode termination) stays
        the runtime's call. ``final_step`` raises :class:`AgentStop`.
        """
        obs = self._task.obs_postprocess(self._tool.execute_action(action), self.role)
        self._maybe_evaluate(obs)
        return obs

    async def async_execute_action(self, action: Action) -> Observation:
        """Async twin of ``execute_action`` — the parallel-tool-call call-site."""
        obs = self._task.obs_postprocess(await self._tool.async_execute_action(action), self.role)
        self._maybe_evaluate(obs)
        return obs


class TaskConfig[TTMetadata: TaskMetadata](ABC, TypedBaseModel):
    """Serializable task configuration — self-contained unit handed to workers.

    Carries everything needed to instantiate a Task, including its
    ``TaskMetadata``. Workers never import the owning ``BenchmarkConfig`` to
    look up metadata; the config arrives complete and ``make()`` just uses
    ``self.metadata`` directly. Cache helpers do require the owning
    ``BenchmarkConfig`` to have been imported (automatic for normal cube
    package layouts) — without it the cache path falls back to the
    top-level Python package name.

    ``task_id`` is derived from ``metadata.id`` (prefixed with
    ``sub_bench_name`` for composite-routed configs) so there is a single
    source of truth.

    ``sub_bench_name`` is an optional routing hint used by
    ``CompositeBenchmark.spawn`` to dispatch a task to its origin
    sub-benchmark. Standalone benchmarks leave it None.

    Type parameter ``TTMetadata`` (bound to ``TaskMetadata``) lets cubes
    statically narrow ``self.metadata`` to a ``TaskMetadata`` subclass without
    re-annotating the field. Two equivalent forms at runtime:

        # Unparametrised — ``self.metadata`` typed as ``TaskMetadata``.
        class FooTaskConfig(TaskConfig): ...

        # Parametrised — ``self.metadata`` typed as ``FooTaskMetadata``,
        # autocomplete and static checking work for subclass-specific fields.
        class FooTaskConfig(TaskConfig[FooTaskMetadata]): ...
    """

    # ``SerializeAsAny`` preserves subclass-specific fields through JSON
    # round-trip. Every cube subclasses TaskMetadata with extra
    # per-task data — without this annotation those fields get silently
    # stripped when the config crosses a process / network / storage boundary.
    metadata: SerializeAsAny[TTMetadata] = Field(
        ...,
        description=(
            "Full task metadata. Stamped onto the config by "
            "``BenchmarkConfig.get_task_configs()`` on the driver so ``make()`` "
            "has everything it needs without importing the owning BenchmarkConfig."
        ),
    )
    seed: int | None = None
    # Same rationale for ToolConfig — cubes declare ToolConfig subclasses with
    # their own fields.
    tool_config: SerializeAsAny[ToolConfig] | None = None
    sub_bench_name: str | None = Field(
        default=None,
        description=(
            "Optional routing hint set by ``CompositeBenchmarkConfig.get_task_configs()``. "
            "Names the sub-benchmark this task originated from; "
            "``CompositeBenchmark.spawn()`` uses it to route to the right sub-benchmark's "
            "runtime_context. None for standalone (non-composite) benchmarks."
        ),
    )

    @property
    def task_id(self) -> str:
        """Derived task identifier. Prefixed with ``sub_bench_name`` when set
        (composite routing), otherwise just ``metadata.id``."""
        if self.sub_bench_name is not None:
            return f"{self.sub_bench_name}/{self.metadata.id}"
        return self.metadata.id

    @abstractmethod
    def make(
        self,
        runtime_context: RuntimeContext | None = None,
    ) -> Task:
        """Instantiate a Task from this config. Called on a worker after deserialization.

        Args:
            runtime_context: Shared infrastructure references created by
                Benchmark._setup() (e.g. server URLs, database connections,
                the injected ``InfraConfig`` under the ``"infra"`` key).
                Passed from Benchmark.spawn().

        Example:
        >>> return MyTask(
        ...     metadata=self.metadata,
        ...     tool_config=self.tool_config or MyDefaultToolConfig(),
        ...     runtime_context=runtime_context,
        ... )

        Cubes with heavy execution data (problem statements, patches, …)
        subclass ``TaskExecutionInfo`` and populate ``Task.execution_info``
        in this method, typically by calling
        ``MyTaskExecutionInfo.model_validate(self.load_task_execution_info())``.
        By convention, implementations call ``self.verify_installed()`` at
        the top so misconfigured workers fail fast with an actionable error.
        """
        pass

    # ──────────────────────────────────────────────────────────────────────────
    # Per-task execution cache (worker-side)
    # ──────────────────────────────────────────────────────────────────────────

    # Set by ``BenchmarkConfig.__init_subclass__`` on each owning benchmark's
    # ``task_config_class`` to ``cls.cache_dir()`` so the default
    # task-execution cache lives directly under the benchmark's cache dir
    # without ``task.py`` importing ``benchmark.py``. ClassVar — not serialized.
    _benchmark_cache_dir: ClassVar[Path | None] = None

    @classmethod
    def task_execution_cache_dir(cls) -> Path:
        """Directory where heavy per-task execution data is cached on this worker.

        Default: ``BenchmarkConfig.cache_dir() / "tasks_execution_info"`` —
        i.e. ``~/.cube/<benchmark-name>/tasks_execution_info/`` once the owning
        ``BenchmarkConfig`` has stamped its cache dir.
        Falls back to ``~/.cube/<top-level-package-name>/tasks_execution_info/``
        when ``_benchmark_cache_dir`` is None.

        Override on subclasses that use a non-default cache layout (e.g. cubes
        that co-locate the cache with other on-disk state).
        """
        # ``__dict__.get`` (not attribute lookup) so derived subclasses without
        # their own owning BenchmarkConfig don't silently inherit the parent's
        # stamp via the MRO.
        cache_dir = cls.__dict__.get("_benchmark_cache_dir") or get_cache_dir(cls.__module__.split(".")[0])
        return cache_dir / "tasks_execution_info"

    def load_task_execution_info(self) -> dict[str, Any]:
        """Read the per-task execution-info dict written by ``BenchmarkConfig.install()``.

        Uses ``self.task_id`` to locate the file under
        ``type(self).task_execution_cache_dir()``. Cube authors typically wrap
        this in ``MyTaskExecutionInfo.model_validate(...)`` inside ``make()``
        to get a typed ``TaskExecutionInfo`` instance.

        Raises ``RuntimeError`` with an actionable message if the cache file
        is missing — signals that ``install()`` has not run on this worker.
        """
        cache_file = type(self).task_execution_cache_dir() / f"{self.task_id}.json"
        if not cache_file.exists():
            raise RuntimeError(
                f"No execution data for task_id={self.task_id!r} at {cache_file}. "
                f"Run `cube install <bench>` (or `<OwnerBenchmarkConfig>.install()`) "
                f"to populate the per-task execution cache on this worker."
            )
        return json.loads(cache_file.read_text())

    def verify_installed(self) -> None:
        """Optional fail-fast check that data this task relies on is locally available.

        Default: no-op. Cube authors override with a check appropriate to
        their cache, e.g.::

            cache_dir = type(self).task_execution_cache_dir()
            if not cache_dir.exists() or not any(cache_dir.iterdir()):
                raise RuntimeError("Run `cube install <bench>` first.")

        Convention: ``TaskConfig.make()`` calls ``self.verify_installed()`` at
        the top so misconfigured workers fail fast with an actionable error
        instead of timing out on a surprise download.
        """
