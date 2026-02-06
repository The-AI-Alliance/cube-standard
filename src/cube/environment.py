from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from mcp.types import Tool as MCPTool

from cube.types import Action, EnvironmentOutput, Observation

if TYPE_CHECKING:
    from cube.task import Task

STOP_ACTION = MCPTool(
    name="final_step",
    description="Stop the task execution.",
    inputSchema={"type": "object", "properties": {}},
)


class AbstractEnvironment(ABC):
    """Abstract interface for environments that agents interact with."""

    def __init__(self, task: Task, *args, **kwargs) -> None:
        super().__init__()
        self.task: Task = task

    @abstractmethod
    def reset(self) -> EnvironmentOutput:
        """Set up the environment before starting a task."""
        pass

    @abstractmethod
    def get_actions(self) -> list[MCPTool]:
        """Returns list of actions supported by that environment."""
        pass

    @abstractmethod
    def step(self, action: Action) -> EnvironmentOutput:
        """Execute a single or multiple actions and return the observation."""
        pass

    def close(self) -> None:
        """Optional clean up environment resources."""
        pass


class EnvConfig:
    """Runtime configuration for the Environment."""

    def __init__(self, task: Task) -> None:
        self.task = task

    def make(self) -> "Environment":
        return Environment(self.task)


class Environment(AbstractEnvironment):
    """Environment that encapsulates a task for CUBE lifecycle management."""

    def __init__(self, task: Task):
        self.task = task

    def get_actions(self) -> list[MCPTool]:
        """Return available actions - delegated to task's MCP tools."""
        # TODO: Get this from MCP server's list_tools()
        return []

    def reset(self) -> EnvironmentOutput:
        """Prepare the task."""
        obs, info = self.task.setup(None)  # No tool parameter needed
        return EnvironmentOutput(obs=obs, info=info)

    def step(self, action: Action | list[Action]) -> EnvironmentOutput:
        """
        Execute actions via MCP server and validate task.

        Note: With MCP architecture, actions are executed via MCP tools.
        This method primarily handles validation and state tracking.
        """
        actions = [action] if isinstance(action, Action) else action
        terminated = False
        truncated = False
        reward = 0.0
        info = {}
        tool_results: list[Observation] = []

        for action in actions:
            if action.name == STOP_ACTION.name and self.task.accept_agent_stop():
                tool_results.append(Observation.from_text("Task finished by the agent."))
                terminated = True
                break
            # TODO: Call MCP server tool here
            # For now, create empty observation
            tool_results.append(Observation.from_text(f"Action {action.name} executed"))

        obs = Observation(contents=[c for o in tool_results for c in o.contents])
        terminated = terminated or self.task.finished()

        # TODO: Add truncation logic based on step limits or time limits
        # For now, truncated remains False. Benchmarks can set this via info dict
        # or by extending Task with a check_truncation() method

        if self.task.validate_per_step or terminated:
            reward, info = self.task.validate_task(obs)

        obs = self.task.obs_postprocess(obs)
        return EnvironmentOutput(
            obs=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def close(self):
        """Clean up resources used by the task."""
        self.task.teardown()
