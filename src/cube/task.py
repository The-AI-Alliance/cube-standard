"""
Task Session Management for CUBE.

This module provides the Task base class and TaskSession class which implements
the task-level API for managing individual task instances. It handles both MCP
protocol methods (tools/*, resources/*) and CUBE extensions (cube/*).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Tuple

from pydantic import Field

from cube.containers import Container, ContainerBackend, ContainerConfig
from cube.core import Action, ActionSchema, EnvironmentOutput, Observation, StepError, TypedBaseModel
from cube.tool import AbstractTool, ToolConfig

if TYPE_CHECKING:
    from cube.benchmark import RuntimeContext

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
        abstract_description (str): Broad description of the task for searching and filtering only. The task objective is part of the first Observation returned by task.setup(). (default: "")
        tags (list[str]): List of task tags (default: empty list)
        recommended_max_steps (int | None): Recommended maximum number of steps to help harness prevent infinite running agents. Not a hard limit, the task can still run longer if needed. (default: None)
        extra_info (dict[str, Any]): Additional task metadata, eg: difficulty level, domain, etc. (default: empty dict)
    """

    id: str = Field(..., description="Unique task identifier")
    split: Literal["train", "val", "test"] = Field(default="test", description="Split for the task")
    abstract_description: str = Field(
        default="",
        description="Broad description of the task for searching and filtering only. The task objective is part of the first Observation returned by task.setup().",
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


class Task(ABC):
    """
    Represents a task that an agent must complete in an environment.

    This class contains:
    1. an `AbstractTool` and an `action_set`:
        the action set is by default the action set defined by the tool.

    2. the task logic:
        - filter_actions() -- optional method to white-list a subset of the tool actions (default filters nothing).
        - obs_postprocess(obs) -> obs -- optional method to modify the observation before returning it (default does nothing).
        + evaluate(obs) -> reward, info -- abstract method to implement
        - get_priviledged_info() -- optional method to return golden trajectory, eval function code, ... (default returns None).
        - finished() -- optional method to check if the task is done

    3. the gym-like environment dynamics:
        + setup() -> Observation, info -- abstract method to implement
        - step(Action) -> Observation -- calls self.tool.execute_action
        - close() -- optional method to cleanup resources
    """

    metadata: TaskMetadata
    tool: AbstractTool  # access to the environment tool, initialized in TaskConfig.make()
    runtime_context: RuntimeContext | None = None
    container: Container | None = None  # access to the environment container, initialized in setup()
    validate_per_step: bool = False
    accept_agent_stop: bool = True  # Optional, wether the task accepts the agent emitting STOP_ACTION

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
    def setup(self) -> Tuple[Observation, Dict]:
        """
        Set up the task to its initial state.
        Should call self.tool.reset() to reset the tool as well

        Returns:
            Tuple of (Observation, dict with additional task info)
        """
        # TODO: consider separating reset from setup if setup does heavy initialization
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
            info: Additional metadata
            error: if there was an exception executing this step
        """
        actions = [action] if isinstance(action, Action) else action
        done = False
        reward = 0.0
        info = {}
        error = None
        obs = Observation()  # will populate list of content after each action
        for action in actions:
            if action.name == STOP_ACTION.name and self.accept_agent_stop:
                obs += Observation.from_text("Task finished by the agent.")
                done = True
                break
            result = self.tool.execute_action(action)
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
        if done or self.validate_per_step:
            reward, info = self.evaluate(obs)
        obs = self.obs_postprocess(obs)
        return EnvironmentOutput(obs=obs, reward=reward, done=done, info=info, error=error)

    def obs_postprocess(self, obs: Observation) -> Observation:
        """
        (Optional) Post-processing of observation before returning it to the agent.
        By default does nothing.
        """
        return obs

    @abstractmethod
    def evaluate(self, obs: Observation) -> Tuple[float, dict]:
        """Validate the current state of the task and return (reward, info)."""
        pass

    def get_priviledged_info(self) -> Any:
        """
        (Optional) Return privileged information about the task such as:
        - solution: list[Action] = Solve the task using a pre-defined solution.
        - evaluation_function_soruce_code: str
        - environment internal state summaries
        """
        return None

    def get_status(self) -> str:
        """
        (Optional) Return current task status.
        Consider looking into self.runtime_context and/or self.container.
        """
        # TODO: figure out if we want to provide some standard for this?
        return ""

    def finished(self, obs: Observation) -> bool:
        """(Optional) Check if the task is finished."""
        return False

    def close(self) -> None:
        """
        (Optional) Cleanup task resources.
        Examples:
        - Close browser / vm / container
        - Cleanup temp files
        - Reset state for next task
        """
        pass


class TaskConfig(ABC, TypedBaseModel):
    """
    Serializable task configuration (Pydantic BaseModel).

    Must be JSON-serializable to pass to workers.
    Holds the minimal data needed to instantiate a Task: task_id, seed, and tool_config.
    TaskMetadata is retrieved via task_id.
    """

    task_id: str
    seed: int | None = None
    tool_config: ToolConfig | None = None

    @abstractmethod
    def make(
        self,
        runtime_context: RuntimeContext | None = None,
        container_backend: ContainerBackend | None = None,
    ) -> Task:
        """
        Instantiate a Task from this config.

        Called on a worker after deserialization.

        Args:
            runtime_context: Shared infrastructure references created by Benchmark._setup()
                             (e.g. server URLs, database connections). Passed from Benchmark.spawn().
            container_backend: HOW to run containers (local, Modal, ...) created by user and passed to benchmark constructor, then passed from Benchmark.spawn().

        Example:
        >>> task_metadata = MyBenchmark.task_metadata_dict[self.task_id]
        >>> return MyTask(task_metadata, self.tool_config, runtime_context, container_backend)
        """
        pass
