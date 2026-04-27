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

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Tuple

from pydantic import ConfigDict, Field, PrivateAttr, SerializeAsAny

from cube.container import Container, ContainerBackend, ContainerConfig
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

RuntimeContext = dict[str, Any]
"""
Type alias for shared infrastructure references created during benchmark.setup().

example:
    {"container_id": "abc123", "vm_address": "http://12.34.56.78", "ssh_session": session}
"""

logger = logging.getLogger(__name__)


STOP_ACTION = ActionSchema(name="final_step", description="Stop the task execution.")


class TaskMetadata(TypedBaseModel):
    """
    Metadata describing a task.

    Used by:
    - Task: metadata attribute
    - API endpoint: cube/tasks (list of TaskMetadata in response)

    Attributes:
        id (str): Unique task identifier
        split (Literal["train", "val", "test"]): Split for the task (default: "test")
        abstract_description (str): Broad description of the task for searching and filtering only. The task objective is part of the first Observation returned by task.reset(). (default: "")
        recommended_max_steps (int | None): Recommended maximum number of steps to help harness prevent infinite running agents. Not a hard limit, the task can still run longer if needed. (default: None)
        container_config (ContainerConfig | None): Optional container configuration for this task (default: None, meaning no container needed).
        extra_info (dict[str, Any]): Additional task metadata, eg: difficulty level, domain, etc. (default: empty dict)
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
    extra_info: dict[str, Any] = Field(
        default_factory=dict, description="Additional task metadata, eg: difficulty level, domain, etc."
    )


class Task(TypedBaseModel, ABC):
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
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Serializable fields
    metadata: TaskMetadata
    tool_config: ToolConfig = Field(description="Tool configuration used to instantiate the tool.")
    container_backend: ContainerBackend | None = Field(
        default=None, description="Optional backend used to launch a container during model_post_init."
    )
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
    _tool: AbstractTool | None = PrivateAttr(default=None)
    _container: Container | None = PrivateAttr(default=None)
    _resource_handle: ResourceHandle | None = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        """Called after Pydantic __init__. Launches container if configured, then creates tool."""
        cc = self.metadata.container_config
        if self.runtime_context is not None and "infra" in self.runtime_context:
            if cc is not None:
                from cube.task_infra import launch_task_container  # local import avoids circular dep

                self._resource_handle, self._container = launch_task_container(
                    self.runtime_context,
                    name=self.metadata.id,
                    image=cc.image,
                    ram_gb=cc.ram_gb,
                    cpu_cores=cc.cpu_cores,
                )
        elif self.container_backend is not None and cc is not None:
            self._container = self.container_backend.launch(cc)

        self._build_tool()

    def _build_tool(self) -> None:
        """Create ``self._tool`` from ``self.tool_config``.

        Override in subclasses to run cube-specific setup (e.g. relocating a
        read-only working directory) before calling ``tool_config.make()``.
        ``self._container`` is already set when this is called.
        """
        self._tool = self.tool_config.make(container=self._container)

    @property
    def tool(self) -> AbstractTool:
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
        Returns tool.action_set filtered if self.filter_actions() is implemented.
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
        return self.filter_actions(self.tool.action_set)

    def filter_actions(self, actions: list[ActionSchema]) -> list[ActionSchema]:
        """
        (Optional) Allows the task to whitelist subset of all the actions provided by the tool.
        By default filters nothing, keep all tool actions.
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


class TaskConfig(ABC, TypedBaseModel):
    """Serializable task configuration — self-contained unit handed to workers.

    Carries everything needed to instantiate a Task, including its
    ``TaskMetadata``. Workers never import the owning ``BenchmarkConfig`` to
    look up metadata; the config arrives complete and ``make()`` just uses
    ``self.metadata`` directly.

    ``task_id`` is derived from ``metadata.id`` (prefixed with
    ``sub_bench_name`` for composite-routed configs) so there is a single
    source of truth.

    ``sub_bench_name`` is an optional routing hint used by
    ``CompositeBenchmark.spawn`` to dispatch a task to its origin
    sub-benchmark. Standalone benchmarks leave it None.
    """

    # ``SerializeAsAny`` preserves subclass-specific fields through JSON
    # round-trip. Every cube subclasses TaskMetadata with extra
    # per-task data — without this annotation those fields get silently
    # stripped when the config crosses a process / network / storage boundary.
    metadata: SerializeAsAny[TaskMetadata] = Field(
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
        container_backend: ContainerBackend | None = None,
    ) -> Task:
        """Instantiate a Task from this config. Called on a worker after deserialization.

        Args:
            runtime_context: Shared infrastructure references created by
                Benchmark._setup() (e.g. server URLs, database connections).
                Passed from Benchmark.spawn().
            container_backend: HOW to run containers (local, Modal, ...) —
                read from the owning BenchmarkConfig by Benchmark.spawn().

        Example:
        >>> return MyTask(
        ...     metadata=self.metadata,
        ...     tool_config=self.tool_config or MyDefaultToolConfig(),
        ...     runtime_context=runtime_context,
        ...     container_backend=container_backend,
        ... )
        """
        pass
