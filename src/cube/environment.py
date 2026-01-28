from abc import ABC, abstractmethod

from mcp.types import Tool as MCPTool

from cube.types import Action, EnvironmentOutput, Observation
from cube.tool import AbstractTool, ToolConfig
from cube.task import Task

STOP_ACTION = MCPTool(
    name="final_step",
    description="Stop the task execution.",
    inputSchema={"type": "object", "properties": {}},
)


class AbstractEnvironment(ABC):
    """Abstract interface for environments that agents interact with."""

    def __init__(self, task: "Task", *args, **kwargs) -> None:
        super().__init__()
        self.task: "Task" = task

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

    def __init__(self, task: "Task", tool_config: ToolConfig) -> None:
        self.task = task
        self.tool_config = tool_config

    def make(self) -> "Environment":
        tool = self.tool_config.make()
        return Environment(self.task, tool)


class Environment(AbstractEnvironment):
    """Environment that encapsulates a tool for interaction and a task."""

    def __init__(self, task: Task, tool: AbstractTool):
        self.task = task
        self.tool = tool

    def get_actions(self) -> list[MCPTool]:
        return self.tool.get_actions()

    def reset(self) -> EnvironmentOutput:
        """Prepare tool and set up the task."""
        self.tool.reset()
        obs, info = self.task.setup(self.tool)
        return EnvironmentOutput(obs=obs, info=info)

    def step(self, action: Action | list[Action]) -> EnvironmentOutput:
        """Execute a single or multiple actions using the appropriate tools, combine observations."""
        actions = [action] if isinstance(action, Action) else action
        terminated = False
        truncated = False
        reward = 0.0
        info = {}
        tool_results: list[Observation] = []

        for action in actions:
            if action.name == STOP_ACTION.name and self.task.accept_agent_stop():
                tool_results.append(
                    Observation.from_text("Task finished by the agent.")
                )
                terminated = True
                break
            tool_results.append(self.tool.execute_action(action))

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
        """Clean up resources used by all tools and the task in the right order."""
        self.task.teardown()
        self.tool.close()
