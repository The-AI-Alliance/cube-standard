"""ComputerConfig and Computer: HTTP-based desktop tool for VM benchmarks.

Computer communicates directly with the Flask control server running inside
the VM container.  No desktop_env dependency.

Action execution works by converting action dicts to PyAutoGUI Python code
and POSTing it to the /execute endpoint (the same mechanism used internally
by OSWorld's PythonController).

Flask endpoints used:
    GET  /screenshot   → PNG bytes
    GET  /accessibility → {"AT": xml_string}
    GET  /terminal     → {"output": string}
    POST /execute      → run command/Python inside VM
    POST /setup/execute → alias for /execute (used during task setup)

Usage:

    from cube_computer_tool import ComputerConfig
    from cube_computer_tool.backends.local_qemu import LocalQEMUVMBackend
    from cube.vm import VMConfig

    config = ComputerConfig(
        vm_backend=LocalQEMUVMBackend(
            vm_image_path="/path/to/ubuntu.qcow2",
            docker_image="happysixd/osworld-docker",
        ),
    )
    tool = config.make()
    try:
        obs = tool.setup_task(task_config)
        # ... run agent loop ...
        reward = tool.evaluate_task()
    finally:
        tool.close()
"""

from __future__ import annotations

import json
import logging
import time
from io import BytesIO
from typing import List, Optional

import requests
from PIL import Image
from pydantic import Field

from cube.core import Action, ActionSchema, Content, Observation, StepError, TypedBaseModel
from cube.tool import Tool, tool_action
from cube.vm import VM, VMConfig
from cube_computer_tool.backends.local_qemu import LocalQEMUVMBackend

logger = logging.getLogger(__name__)

# PyAutoGUI Python prefix injected before every GUI action command
_PYAUTOGUI_PREFIX = "import pyautogui; import time; pyautogui.FAILSAFE = False; "


class ComputerConfig(TypedBaseModel):
    """Configuration for the Computer tool.

    Satisfies the ComputerToolConfig Protocol structurally (has make() method).

    Fields:
        vm_backend:          How to launch/manage the VM container.
        vm_config:           What the VM should look like (snapshot, screen size).
        require_a11y_tree:   Include accessibility tree in observations.
        require_terminal:    Include terminal output in observations.
        observe_after_action: Capture full observation after every action.
    """

    vm_backend: LocalQEMUVMBackend
    vm_config: VMConfig = Field(default_factory=VMConfig)
    require_a11y_tree: bool = True
    require_terminal: bool = False
    observe_after_action: bool = True

    def make(self) -> "Computer":
        """Launch the VM and return a ready Computer tool."""
        vm = self.vm_backend.launch(self.vm_config)
        return Computer(config=self, vm=vm)


class Computer(Tool):
    """Desktop computer tool that communicates via HTTP with the VM Flask server.

    Exposes a standard set of mouse and keyboard actions via @tool_action.
    Task lifecycle (setup_task, get_observation, evaluate_task) is handled
    separately from agent actions.

    Action set (all via @tool_action):
        Mouse:    move_to, click, mouse_down, mouse_up, right_click, double_click,
                  drag_to, scroll
        Keyboard: typing, press, key_down, key_up, hotkey
        Control:  wait, done, fail
    """

    def __init__(self, config: ComputerConfig, vm: VM) -> None:
        super().__init__()
        self._config = config
        self._vm = vm
        self._task_config: Optional[dict] = None
        self._is_done: bool = False

    # ------------------------------------------------------------------
    # Task lifecycle (not agent actions)
    # ------------------------------------------------------------------

    def setup_task(self, task_config: dict, seed: int = 42) -> Observation:
        """Reset VM to task snapshot, run setup commands, return initial obs.

        Restores a clean VM state by restarting the container (the qcow2 disk
        is mounted read-only, so the new container always starts from the base
        image).  Then sends the task's setup config commands to the VM and
        waits 60 seconds for the desktop to stabilize.

        Args:
            task_config: Task configuration dict. Supported keys:
                - ``snapshot``: snapshot name (ignored in docker backend)
                - ``config``:   list of setup action dicts sent to /setup/execute
            seed: Unused; accepted for API compatibility.

        Returns:
            Initial observation after setup.
        """
        logger.info("Setting up task: %s", task_config.get("id", "unknown"))
        logger.info("Instruction: %s", task_config.get("instruction", ""))

        self._vm.restore_snapshot(task_config.get("snapshot", self._config.vm_config.snapshot_name))
        self._is_done = False
        self._task_config = task_config

        # Run task-specific setup commands inside the VM
        for setup_action in task_config.get("config") or []:
            self._run_setup_action(setup_action)

        logger.info("Waiting 60s for VM to stabilize…")
        time.sleep(60)

        return self.get_observation()

    def get_observation(self) -> Observation:
        """Capture the current desktop state (screenshot + optional a11y tree)."""
        obs = Observation()

        # Screenshot: GET /screenshot → PNG bytes → PIL Image
        try:
            resp = requests.get(f"{self._vm.endpoint}/screenshot", timeout=30)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            obs.contents.append(Content.from_data(img, name="screenshot"))
        except Exception as exc:
            logger.warning("Failed to capture screenshot: %s", exc)

        # Accessibility tree: GET /accessibility → {"AT": xml_string}
        if self._config.require_a11y_tree:
            try:
                resp = requests.get(f"{self._vm.endpoint}/accessibility", timeout=30)
                resp.raise_for_status()
                at = resp.json().get("AT", "")
                if at:
                    obs.contents.append(Content.from_data(at, name="accessibility_tree"))
            except Exception as exc:
                logger.warning("Failed to get accessibility tree: %s", exc)

        # Terminal output: GET /terminal → {"output": string}
        if self._config.require_terminal:
            try:
                resp = requests.get(f"{self._vm.endpoint}/terminal", timeout=30)
                resp.raise_for_status()
                terminal = resp.json().get("output", "")
                if terminal:
                    obs.contents.append(Content.from_data(terminal, name="terminal"))
            except Exception as exc:
                logger.warning("Failed to get terminal output: %s", exc)

        return obs

    def evaluate_task(self) -> float:
        """Evaluate the current task.

        TODO: Implement full OSWorld evaluation by running evaluator functions
        from the task config against the current VM state.  For now returns 0.0
        as a placeholder.  Benchmark-specific evaluation logic (e.g. osworld-cube)
        should override or extend this.

        Returns:
            Reward in [0.0, 1.0].
        """
        logger.warning("evaluate_task() is not implemented — returning 0.0")
        return 0.0

    # ------------------------------------------------------------------
    # Action execution override (adds observe_after_action)
    # ------------------------------------------------------------------

    def execute_action(self, action: Action) -> Observation | StepError:
        result = super().execute_action(action)
        if self._config.observe_after_action and action.name not in ("done", "fail", "wait"):
            if not isinstance(result, StepError):
                result += self.get_observation()
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_setup_action(self, setup_action: dict) -> None:
        """Send a task setup action to POST /setup/execute."""
        payload = {"shell": True, "command": setup_action.get("command", "")}
        try:
            resp = requests.post(
                f"{self._vm.endpoint}/setup/execute",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("Setup action failed: %s — %s", setup_action, exc)

    def _execute_pyautogui(self, pyautogui_code: str) -> str:
        """Execute a PyAutoGUI command inside the VM via POST /execute.

        Args:
            pyautogui_code: Python code string (without the import prefix).

        Returns:
            "Success" or an error description.
        """
        full_code = _PYAUTOGUI_PREFIX + pyautogui_code
        command_list = ["python", "-c", full_code]
        payload = json.dumps({"command": command_list, "shell": False})

        try:
            resp = requests.post(
                f"{self._vm.endpoint}/execute",
                data=payload,
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            resp.raise_for_status()
            result = resp.json()
            if result.get("returncode", 0) != 0 and result.get("error"):
                return f"Error: {result['error'].strip()}"
            return "Success"
        except Exception as exc:
            return f"Error: {exc}"

    # ------------------------------------------------------------------
    # Mouse actions
    # ------------------------------------------------------------------

    @tool_action
    def move_to(self, x: int, y: int) -> str:
        """Move the cursor to the specified position."""
        return self._execute_pyautogui(f"pyautogui.moveTo({x}, {y})")

    @tool_action
    def click(
        self,
        x: Optional[int] = None,
        y: Optional[int] = None,
        button: str = "left",
        num_clicks: int = 1,
    ) -> str:
        """Click the mouse button at an optional position."""
        args = f"button='{button}', clicks={num_clicks}"
        if x is not None and y is not None:
            args = f"{x}, {y}, {args}"
        return self._execute_pyautogui(f"pyautogui.click({args})")

    @tool_action
    def mouse_down(self, button: str = "left") -> str:
        """Press the mouse button down (without releasing)."""
        return self._execute_pyautogui(f"pyautogui.mouseDown(button='{button}')")

    @tool_action
    def mouse_up(self, button: str = "left") -> str:
        """Release the mouse button."""
        return self._execute_pyautogui(f"pyautogui.mouseUp(button='{button}')")

    @tool_action
    def right_click(self, x: Optional[int] = None, y: Optional[int] = None) -> str:
        """Right-click at an optional position."""
        return self.click(x=x, y=y, button="right")

    @tool_action
    def double_click(self, x: Optional[int] = None, y: Optional[int] = None) -> str:
        """Double-click at an optional position."""
        return self.click(x=x, y=y, num_clicks=2)

    @tool_action
    def drag_to(self, x: int, y: int) -> str:
        """Drag the cursor to the specified position with the left button held."""
        return self._execute_pyautogui(f"pyautogui.dragTo({x}, {y}, button='left')")

    @tool_action
    def scroll(self, dx: int = 0, dy: int = 0) -> str:
        """Scroll the mouse wheel by (dx, dy) clicks."""
        # PyAutoGUI scroll() takes clicks (positive=up, negative=down)
        code = ""
        if dy != 0:
            code += f"pyautogui.scroll({dy}); "
        if dx != 0:
            code += f"pyautogui.hscroll({dx}); "
        return self._execute_pyautogui(code or "pass")

    # ------------------------------------------------------------------
    # Keyboard actions
    # ------------------------------------------------------------------

    @tool_action
    def typing(self, text: str) -> str:
        """Type the specified text character by character."""
        escaped = text.replace("\\", "\\\\").replace("'", "\\'")
        return self._execute_pyautogui(f"pyautogui.typewrite('{escaped}', interval=0.05)")

    @tool_action
    def press(self, key: str) -> str:
        """Press and release the specified key."""
        return self._execute_pyautogui(f"pyautogui.press('{key}')")

    @tool_action
    def key_down(self, key: str) -> str:
        """Press the specified key without releasing it."""
        return self._execute_pyautogui(f"pyautogui.keyDown('{key}')")

    @tool_action
    def key_up(self, key: str) -> str:
        """Release the specified key."""
        return self._execute_pyautogui(f"pyautogui.keyUp('{key}')")

    @tool_action
    def hotkey(self, keys: List[str]) -> str:
        """Press a key combination simultaneously (e.g. ['ctrl', 'c'])."""
        if isinstance(keys, str):
            keys = keys.split("+")
        keys_str = ", ".join(f"'{k}'" for k in keys)
        return self._execute_pyautogui(f"pyautogui.hotkey({keys_str})")

    # ------------------------------------------------------------------
    # Control actions
    # ------------------------------------------------------------------

    @tool_action
    def wait(self) -> str:
        """Do nothing for one step."""
        return "Success"

    @tool_action
    def done(self) -> str:
        """Signal that the task has been completed successfully."""
        self._is_done = True
        return "Task marked as done"

    @tool_action
    def fail(self) -> str:
        """Signal that the task cannot be completed (infeasible)."""
        self._is_done = True
        return "Task marked as failed"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Stop the VM container and release all resources."""
        logger.info("Closing computer tool — stopping VM")
        self._vm.stop()
