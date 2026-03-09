"""Tool layer for cube_package.

A Tool exposes a set of actions the agent can call.  Mark each action method
with @tool_action — the decorator registers it for schema generation and
dispatch.

Typical customisation points
-----------------------------
- Add / remove @tool_action methods.
- Store environment state on a plain helper class (see CubeEnv below) and
  keep the Tool itself thin — it just wraps the env.
- Expose config knobs via CubeToolConfig so tasks can vary behaviour
  without subclassing.
- Override the action_set property to filter which actions are shown to the
  agent based on the current config.
"""

from cube.container import Container
from cube.tool import Tool, ToolConfig, tool_action


class CubeEnv:
    """Holds the mutable environment state.

    Keeping state here (instead of on the Tool) makes reset() trivial and
    prevents accidental state sharing between episodes.
    """

    def __init__(self) -> None:
        # TODO: initialise your environment state here.
        pass

    def reset(self) -> None:
        """Restore environment to its initial state."""
        # TODO: reset mutable state.
        pass


class CubeToolConfig(ToolConfig):
    """Serialisable configuration that controls which actions are available.

    Add boolean / int / str fields here to toggle optional actions or tune
    the environment.  CubeTaskConfig.make() picks these up from
    metadata.extra_info["tool_config"] (see task.py).
    """

    # TODO: add config fields, e.g.:
    # enable_some_action: bool = False

    def make(self, container: Container | None = None) -> "CubeTool":
        """Instantiate the tool from this config.

        `container` is None for local tools; pass it to the tool constructor
        when the tool connects to a containerised service.
        """
        return CubeTool(self)


class CubeTool(Tool):
    """Agent-facing tool that wraps CubeEnv.

    Convention: keep a reference to the env as self._env so that Task.evaluate()
    can read environment state directly (e.g. self.tool._env.some_field).
    """

    def __init__(self, config: CubeToolConfig) -> None:
        self._env = CubeEnv()
        self._config = config

    def reset(self) -> None:
        """Reset env at the start of each episode."""
        self._env.reset()

    # ------------------------------------------------------------------
    # Actions
    # Each @tool_action method becomes an entry in action_set and can be
    # called by the agent.  The docstring is shown to the agent as the
    # action description; type annotations become the parameter schema.
    # ------------------------------------------------------------------

    @tool_action
    def example_action(self, message: str) -> str:
        """Do something with message and return an observation string.

        Parameters
        ----------
        message : str
            Input from the agent.
        """
        # TODO: implement action logic.
        return f"Received: {message}"

    # TODO: add more @tool_action methods.
    # Optional: override action_set to filter based on self._config:
    # @property
    # def action_set(self) -> list[ActionSchema]:
    #     return [a for a in super().action_set if a.name != "disabled_action"]
