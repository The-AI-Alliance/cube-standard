"""Abstract computer tool protocols for desktop/VM-based benchmark tasks.

cube-standard declares the contracts; concrete implementations live in
cube-tools/cube-computer-tool/ (or any compatible harness).

    AbstractComputerTool — protocol for a desktop computer tool
    ComputerToolConfig   — protocol for configs that produce AbstractComputerTool

Desktop benchmark tasks (OSWorld, ScreenSpot, WindowsAgentArena, …) type their
tool as AbstractComputerTool and require no knowledge of the concrete VM backend.

Example usage in a benchmark task:

    from cube.tools.computer import AbstractComputerTool

    class OSWorldTask(Task):
        def setup(self, tool: AbstractComputerTool) -> Observation:
            return tool.setup_task(self.task_config)

        def evaluate(self, tool: AbstractComputerTool) -> float:
            return tool.evaluate_task()
"""

from typing import Protocol, runtime_checkable

from cube.core import Action, ActionSchema, Observation, StepError


@runtime_checkable
class AbstractComputerTool(Protocol):
    """Synchronous computer tool protocol for desktop/VM benchmark tasks.

    Defines the contract that any desktop computer tool must satisfy to be used
    with VM-based benchmarks (OSWorld, ScreenSpot, WindowsAgentArena, …).
    Concrete implementations live in cube-tools/cube-computer-tool/ or can be
    provided by any compatible harness.

    Two groups of methods:

    Tool lifecycle / action execution:
        execute_action()  — dispatch an agent action; returns obs or error
        action_set        — list of available actions (property)
        close()           — release VM/container resources

    Desktop-specific task lifecycle methods:
        setup_task()      — reset VM to task snapshot and run setup commands
        get_observation() — capture current desktop state (screenshot + a11y)
        evaluate_task()   — run task evaluator; returns reward in [0, 1]
    """

    # --- Action execution ---

    def execute_action(self, action: Action) -> Observation | StepError:
        """Dispatch a single agent action and return the result.

        Returns Observation on success, StepError if the action raised an exception.
        """
        ...

    @property
    def action_set(self) -> list[ActionSchema]:
        """List of actions this tool exposes to the agent."""
        ...

    # --- Tool lifecycle ---

    def close(self) -> None:
        """Release VM/container resources (stop container, clean up)."""
        ...

    # --- Desktop-specific task lifecycle ---

    def setup_task(self, task_config: dict, seed: int = 42) -> Observation:
        """Reset VM to task snapshot and run task-specific setup commands.

        Typically restores the VM to a clean snapshot, runs any task-specific
        configuration steps (install apps, create files, etc.), then waits for
        the VM to stabilize before returning the initial observation.

        Args:
            task_config: Task configuration dict with snapshot, config, evaluator fields.
            seed: Random seed for task setup.

        Returns:
            Initial observation after setup (screenshot + accessibility tree).
        """
        ...

    def get_observation(self) -> Observation:
        """Capture the current desktop state as an Observation.

        Typically includes a screenshot and accessibility tree. The exact contents
        depend on the tool configuration (require_a11y_tree, require_terminal, …).
        """
        ...

    def evaluate_task(self) -> float:
        """Run the task evaluator and return the reward.

        Returns:
            Reward value in [0.0, 1.0]. 1.0 means task succeeded, 0.0 failed.
        """
        ...


@runtime_checkable
class ComputerToolConfig(Protocol):
    """Structural protocol for configs that produce an AbstractComputerTool.

    Any object with a make() method returning an AbstractComputerTool satisfies
    this protocol. Used by benchmarks to type their tool_config field abstractly,
    allowing users to swap VM backends without touching benchmark code.

    Example:

        class OSWorldBenchmark(Benchmark):
            tool_config: ComputerToolConfig  # accepts any computer tool impl
    """

    def make(self) -> AbstractComputerTool:
        """Instantiate and return the computer tool."""
        ...
