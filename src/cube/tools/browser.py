"""Abstract browser tool protocols for web-based benchmark tasks.

cube-standard declares the contracts; concrete implementations live in
cube-tools/cube-browser-tool/ (or any compatible harness, e.g. cube-harness).

Two protocols are provided — one sync, one async — mirroring the Playwright
sync/async API split:

    AbstractBrowserTool      — synchronous; primary path for cube-standard tasks.
    AsyncAbstractBrowserTool — asynchronous; for high-throughput parallel collection.

Web benchmark tasks (MiniWob, WorkArena, WebArena, …) type their tool as
AbstractBrowserTool and require no knowledge of the concrete implementation.

Example usage in a benchmark task:

    from cube.tools.browser import AbstractBrowserTool

    class MiniWobTask(Task):
        def reset(self) -> tuple[Observation, dict]:
            self.tool.reset()
            self.tool.goto("https://...")
            goal = self.tool.evaluate_js("() => core.getUtterance()")
            return Observation.from_text(goal) + self.tool.page_obs(), {}
"""

from typing import Any, Protocol, runtime_checkable

from cube.core import Action, ActionSchema, Observation, StepError


@runtime_checkable
class AbstractBrowserTool(Protocol):
    """Synchronous browser tool protocol for web-based benchmark tasks.

    Defines the contract that any sync browser tool must satisfy to be used
    with web benchmarks (MiniWob, WorkArena, WebArena, …). Concrete
    implementations live in cube-tools/cube-browser-tool/ or can be provided
    by any compatible harness (e.g. cube-harness).

    Two groups of methods:

    Tool lifecycle / action execution (mirrors AbstractTool):
        reset()           — reset browser state before a new episode
        close()           — release browser resources
        execute_action()  — dispatch an agent action; returns obs or error
        action_set        — list of available actions (property)

    Browser-specific task-internal methods:
        goto()            — navigate to a URL
        evaluate_js()     — run JavaScript and return the result
        page_obs()        — capture the current page as an Observation
    """

    # --- Tool lifecycle ---

    def reset(self) -> None:
        """Reset browser state (new page, clear cookies) before a new episode."""
        ...

    def close(self) -> None:
        """Release browser resources (page, browser process, Playwright instance)."""
        ...

    # --- Action execution ---

    def execute_action(self, action: Action) -> Observation | StepError:
        """Dispatch a single agent action and return the result.

        Returns Observation on success, StepError if the action raised an exception.
        The returned Observation typically includes the post-action page state.
        """
        ...

    @property
    def action_set(self) -> list[ActionSchema]:
        """List of actions this tool exposes to the agent."""
        ...

    # --- Browser-specific task-internal methods ---

    def goto(self, url: str) -> None:
        """Navigate to a URL and wait for the page to load."""
        ...

    def evaluate_js(self, js: str) -> Any:
        """Evaluate JavaScript in the browser context and return the result.

        The JS string may be a function body or an arrow function expression.
        The return value is whatever the JS expression evaluates to (str, int,
        list, dict, bool, None).
        """
        ...

    def page_obs(self) -> Observation:
        """Capture the current page state as an Observation.

        The exact contents depend on the tool configuration (screenshot, HTML,
        accessibility tree, …). Tasks should not assume a specific content layout.
        """
        ...


@runtime_checkable
class AsyncAbstractBrowserTool(Protocol):
    """Asynchronous browser tool protocol for web-based benchmark tasks.

    Same contract as AbstractBrowserTool but all browser-interacting methods
    are coroutines. Intended for high-throughput parallel data collection where
    async Playwright is preferred over the sync API.

    The action_set property is sync in both variants — it is a pure metadata
    query that requires no I/O.

    See AbstractBrowserTool for method-level documentation.
    """

    # --- Tool lifecycle ---

    async def reset(self) -> None: ...

    async def close(self) -> None: ...

    # --- Action execution ---

    async def execute_action(self, action: Action) -> Observation | StepError: ...

    @property
    def action_set(self) -> list[ActionSchema]:
        """List of actions this tool exposes to the agent (sync in both variants)."""
        ...

    # --- Browser-specific task-internal methods ---

    async def goto(self, url: str) -> None: ...

    async def evaluate_js(self, js: str) -> Any: ...

    async def page_obs(self) -> Observation: ...
