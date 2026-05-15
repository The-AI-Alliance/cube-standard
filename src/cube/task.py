"""
Task management for CUBE.

This module defines the Task base class, TaskMetadata, and TaskConfig for
implementing and configuring individual benchmark tasks. By design, Task
unifies gym-like environment dynamics (reset/step/close) and task-specific
logic (evaluate/filter_actions/obs_postprocess) in a single class, so that
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
from typing import Any, ClassVar, Dict, Generic, List, Literal, Tuple

from pydantic import ConfigDict, Field, PrivateAttr, SerializeAsAny
from typing_extensions import TypeVar

from cube import get_cache_dir
from cube.container import Container, ContainerConfig
from cube.core import (
    Action,
    ActionSchema,
    Content,
    EnvironmentOutput,
    Observation,
    StepError,
    StructuredContent,
    TypedBaseModel,
)
from cube.resource import ResourceHandle
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


STOP_ACTION = ActionSchema(
    name="final_step",
    description="Stop the task execution.",
    parameters={"type": "object", "properties": {}},
)


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
        - filter_actions(actions) -> actions      optional whitelist of tool actions
        - obs_postprocess(obs) -> obs             optional observation post-processing
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
    accept_agent_stop: bool = Field(
        default=True, description="Whether the task accepts the agent emitting STOP_ACTION."
    )

    # Non-serializable runtime state, set during model_post_init
    _tool: TTool | None = PrivateAttr(default=None)
    _container: Container | None = PrivateAttr(default=None)
    _resource_handle: ResourceHandle | None = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Called after Pydantic __init__. Launches container if configured, then creates tool."""
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

        self._build_tool()

    def _build_tool(self) -> None:
        """Create ``self._tool`` from ``self.tool_config``.

        Override in subclasses to run cube-specific setup (e.g. relocating a
        read-only working directory) before calling ``tool_config.make()``.
        ``self._container`` is already set when this is called.
        """
        # ToolConfig.make() returns AbstractTool; cubes that parameterize
        # Task[Meta, TFooTool] vouch that their tool_config produces a TFooTool.
        self._tool = self.tool_config.make(container=self._container)  # type: ignore[assignment]

    @property
    def tool(self) -> TTool:
        return self._tool  # type: ignore[return-value]

    @property
    def container(self) -> Container | None:
        return self._container

    @property
    def id(self) -> str:
        return self.metadata.id

    @property
    def action_set(self) -> List[ActionSchema]:
        """
        Returns tool.action_set filtered through self.filter_actions(), then with
        STOP_ACTION appended when self.accept_agent_stop is True (dedup by name).
        Cube authors should NOT add STOP_ACTION in filter_actions.

        Tool definitions in litellm-compatible format.

        Returns a JSON-serializable list of tool descriptors, each with:
        - type: "function"
        - function: {name, description, parameters (JSON Schema)}

        This format is compatible with litellm/OpenAI function calling.
        Agents use this to discover available actions without knowing
        tool implementations in advance.

        Example return value:
        [
            {
                "type": "function",
                "function": {
                    "name": "click",
                    "description": "Click on a web element",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "selector": {"type": "string", "description": "CSS selector"}
                        },
                        "required": ["selector"]
                    }
                }
            }
        ]
        """
        actions = self.filter_actions(self.tool.action_set)
        if self.accept_agent_stop and not any(a.name == STOP_ACTION.name for a in actions):
            actions = [*actions, STOP_ACTION]
        return actions

    def filter_actions(self, actions: list[ActionSchema]) -> list[ActionSchema]:
        """
        (Optional) Allows the task to whitelist a subset of the actions provided by the tool.
        By default keeps all tool actions. STOP_ACTION is appended automatically by
        action_set when accept_agent_stop=True — do not add it here.
        """
        return actions

    @abstractmethod
    def reset(self) -> Tuple[Observation, Dict]:
        """
        Reset the task to its initial state.
        Must call self.tool.reset() to reset the tool as well.

        Returns:
            Tuple of (Observation, dict with additional task info)
        """
        pass

    def step(self, action: Action | List[Action]) -> EnvironmentOutput:
        """
        Execute action, return next state.
        - check if agent action is a STOP_ACTION
        - if not, execute the action and get the observation (self.tool.execute_action(act))
        - check if the task is done (self.finished(obs))
        - if done or self.validate_per_step, evaluate state (self.evaluate(obs))
        - return EnvironmentOutput with obs, reward, info, error, ...

        Args:
            action: Agent action
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
            error: if there was an exception executing this step
        """
        actions = [action] if isinstance(action, Action) else action
        done = False
        reward = 0.0
        info = {}
        error = None
        obs = Observation()  # will populate list of content after each action
        action_times: list[float] = []
        for action in actions:
            if action.name == STOP_ACTION.name and self.accept_agent_stop:
                obs += Observation.from_text("Task finished by the agent.")
                done = True
                break
            t0 = time.perf_counter()
            result = self.tool.execute_action(action)
            action_times.append(time.perf_counter() - t0)
            if isinstance(result, Observation):
                obs += result
            elif isinstance(result, StepError):
                error = result
                done = True
                break
            else:
                raise ValueError(
                    f"Unknown result type from calling action '{action.name}' with args {action.arguments}: "
                    f"got {type(result).__name__}, expected Observation or StepError"
                )
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

    def obs_postprocess(self, obs: Observation) -> Observation:
        """
        (Optional) Post-processing of observation before returning it to the agent.
        By default does nothing.
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
        Cleanup task resources. Calls self.tool.close() automatically.
        Override to add task-specific cleanup, and call super().close() to
        ensure the tool is also cleaned up.

        Examples of additional task-specific cleanup:
        - Stop containers
        - Remove temp files
        - Close network connections
        """
        self.tool.close()
        if self._resource_handle is not None:
            self._resource_handle.close()
            self._resource_handle = None
            self._container = None
        elif self._container is not None:
            self._container.stop()
            self._container = None


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
