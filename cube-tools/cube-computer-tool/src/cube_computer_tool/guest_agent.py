"""HTTP client for the Flask guest agent running inside a desktop VM.

The guest agent server runs at port 5000 inside the VM and exposes endpoints
for screenshots, accessibility trees, command execution, and file I/O.

Originally ported from desktop_env.controllers.python.PythonController.
"""

import json
import logging
import random
import time
import traceback
from typing import Any

import requests

logger = logging.getLogger(__name__)

# fmt: off
_KEYBOARD_KEYS = [
    "\t", "\n", "\r", " ", "!", '"', "#", "$", "%", "&", "'", "(", ")", "*",
    "+", ",", "-", ".", "/", "0", "1", "2", "3", "4", "5", "6", "7", "8",
    "9", ":", ";", "<", "=", ">", "?", "@", "[", "\\", "]", "^", "_", "`",
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n",
    "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "{", "|",
    "}", "~", "accept", "add", "alt", "altleft", "altright", "apps",
    "backspace", "browserback", "browserfavorites", "browserforward",
    "browserhome", "browserrefresh", "browsersearch", "browserstop",
    "capslock", "clear", "convert", "ctrl", "ctrlleft", "ctrlright",
    "decimal", "del", "delete", "divide", "down", "end", "enter", "esc",
    "escape", "execute", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8",
    "f9", "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18",
    "f19", "f20", "final", "fn", "hanguel", "hangul", "hanja", "help",
    "home", "insert", "junja", "kana", "kanji", "launchapp1", "launchapp2",
    "launchmail", "launchmediaselect", "left", "modechange", "multiply",
    "nexttrack", "nonconvert", "num0", "num1", "num2", "num3", "num4",
    "num5", "num6", "num7", "num8", "num9", "numlock", "pagedown", "pageup",
    "pause", "pgdn", "pgup", "playpause", "prevtrack", "print", "printscreen",
    "prntscrn", "prtsc", "prtscr", "return", "right", "scrolllock", "select",
    "separator", "shift", "shiftleft", "shiftright", "sleep", "space", "stop",
    "subtract", "tab", "up", "volumedown", "volumemute", "volumeup", "win",
    "winleft", "winright", "yen", "command", "option", "optionleft",
    "optionright",
]
# fmt: on

_PYAUTOGUI_PREFIX = "import pyautogui; import time; pyautogui.FAILSAFE = False; {command}"

_RETRY_TIMES = 3
_RETRY_INTERVAL = 5


class GuestAgent:
    """HTTP client for the Flask agent server running inside a desktop VM.

    Parameters
    ----------
    host : str
        Hostname or IP of the server (typically "localhost" with port-forwarded QEMU).
    port : int
        Host port mapped to the guest's Flask server (default 5000).
    """

    def __init__(self, host: str = "localhost", port: int = 5000) -> None:
        self.host = host
        self.port = port
        self._base_url = f"http://{host}:{port}"

    # ------------------------------------------------------------------
    # Observation retrieval
    # ------------------------------------------------------------------

    def get_screenshot(self) -> bytes | None:
        """Return raw PNG/JPEG bytes of the current screen, or None on failure."""
        for attempt in range(_RETRY_TIMES):
            try:
                resp = requests.get(self._base_url + "/screenshot", timeout=10)
                if resp.status_code == 200 and self._is_valid_image(resp.headers.get("Content-Type", ""), resp.content):
                    return resp.content
                logger.error("Invalid screenshot response (attempt %d/%d)", attempt + 1, _RETRY_TIMES)
            except Exception as exc:
                logger.error("Screenshot error: %s", exc)
            time.sleep(_RETRY_INTERVAL)
        logger.error("Failed to get screenshot after %d attempts", _RETRY_TIMES)
        return None

    def get_accessibility_tree(self, backend: str = "uia") -> str | None:
        """Return the XML accessibility tree string, or None on failure.

        Parameters
        ----------
        backend : str
            Accessibility backend passed as ``?backend=`` query param.
            ``"uia"`` (default) returns semantic element types and child
            controls; ``"win32"`` is faster but only returns top-level
            window handles without child UI elements.
        """
        for _ in range(_RETRY_TIMES):
            try:
                resp = requests.get(self._base_url + "/accessibility", params={"backend": backend}, timeout=300)
                if resp.status_code == 200:
                    return resp.json()["AT"]
                logger.error("Accessibility tree error: %d", resp.status_code)
            except Exception as exc:
                logger.error("Accessibility tree error: %s", exc)
            time.sleep(_RETRY_INTERVAL)
        logger.error("Failed to get accessibility tree")
        return None

    def get_terminal_output(self) -> str | None:
        """Return the terminal output string, or None on failure."""
        for _ in range(_RETRY_TIMES):
            try:
                resp = requests.get(self._base_url + "/terminal")
                if resp.status_code == 200:
                    return resp.json()["output"]
                logger.error("Terminal output error: %d", resp.status_code)
            except Exception as exc:
                logger.error("Terminal output error: %s", exc)
            time.sleep(_RETRY_INTERVAL)
        logger.error("Failed to get terminal output")
        return None

    def get_file(self, file_path: str) -> bytes | None:
        """Download a file from the VM by path, or None on failure."""
        for _ in range(_RETRY_TIMES):
            try:
                resp = requests.post(self._base_url + "/file", data={"file_path": file_path})
                if resp.status_code == 200:
                    return resp.content
                logger.error("Get file error: %d", resp.status_code)
            except Exception as exc:
                logger.error("Get file error: %s", exc)
            time.sleep(_RETRY_INTERVAL)
        logger.error("Failed to get file: %s", file_path)
        return None

    # ------------------------------------------------------------------
    # Command execution
    # ------------------------------------------------------------------

    def execute_python_command(self, command: str) -> dict[str, Any] | None:
        """Execute a Python command via pyautogui prefix inside the VM."""
        command_list = ["python", "-c", _PYAUTOGUI_PREFIX.format(command=command)]
        payload = json.dumps({"command": command_list, "shell": False})

        for _ in range(_RETRY_TIMES):
            try:
                resp = requests.post(
                    self._base_url + "/execute",
                    headers={"Content-Type": "application/json"},
                    data=payload,
                    timeout=90,
                )
                if resp.status_code == 200:
                    return resp.json()
                logger.error("Execute python error: %d", resp.status_code)
            except requests.exceptions.ReadTimeout:
                break
            except Exception as exc:
                logger.error("Execute python error: %s", exc)
            time.sleep(_RETRY_INTERVAL)
        logger.error("Failed to execute python command")
        return None

    def run_python_script(self, script: str) -> dict[str, Any] | None:
        """Execute a Python script file inside the VM via /run_python."""
        payload = json.dumps({"code": script})

        for _ in range(_RETRY_TIMES):
            try:
                resp = requests.post(
                    self._base_url + "/run_python",
                    headers={"Content-Type": "application/json"},
                    data=payload,
                    timeout=90,
                )
                if resp.status_code == 200:
                    return resp.json()
                return {
                    "status": "error",
                    "message": "Request failed",
                    "output": None,
                    "error": resp.json().get("error"),
                }
            except requests.exceptions.ReadTimeout:
                break
            except Exception:
                logger.error("Run python script error: %s", traceback.format_exc())
            time.sleep(_RETRY_INTERVAL)
        return {"status": "error", "message": "Retry limit reached", "output": "", "error": "Retry limit reached."}

    def run_bash_script(self, script: str, timeout: int = 30, working_dir: str | None = None) -> dict[str, Any] | None:
        """Execute a bash script inside the VM via /run_bash_script."""
        payload = json.dumps({"script": script, "timeout": timeout, "working_dir": working_dir})

        for _ in range(_RETRY_TIMES):
            try:
                resp = requests.post(
                    self._base_url + "/run_bash_script",
                    headers={"Content-Type": "application/json"},
                    data=payload,
                    timeout=timeout + 100,
                )
                if resp.status_code == 200:
                    return resp.json()
                logger.error("Run bash script error: %d %s", resp.status_code, resp.text)
            except requests.exceptions.ReadTimeout:
                return {"status": "error", "output": "", "error": f"Timed out after {timeout}s", "returncode": -1}
            except Exception as exc:
                logger.error("Run bash script error: %s", exc)
            time.sleep(_RETRY_INTERVAL)
        return {"status": "error", "output": "", "error": f"Failed after {_RETRY_TIMES} retries", "returncode": -1}

    def execute_action(self, action: dict[str, Any]) -> None:
        """Dispatch a computer_13 action dict to the appropriate pyautogui command.

        Mirrors the dispatch table in desktop_env.controllers.python.PythonController.execute_action.
        """
        if action in ("WAIT", "FAIL", "DONE"):
            return

        action_type: str = action["action_type"]
        parameters: dict = action.get("parameters") or {k: v for k, v in action.items() if k != "action_type"}

        move_mode = random.choice(
            [
                "pyautogui.easeInQuad",
                "pyautogui.easeOutQuad",
                "pyautogui.easeInOutQuad",
                "pyautogui.easeInBounce",
                "pyautogui.easeInElastic",
            ]
        )
        duration = random.uniform(0.5, 1)

        if action_type == "MOVE_TO":
            if not parameters:
                self.execute_python_command("pyautogui.moveTo()")
            elif "x" in parameters and "y" in parameters:
                self.execute_python_command(
                    f"pyautogui.moveTo({parameters['x']}, {parameters['y']}, {duration}, {move_mode})"
                )
            else:
                raise ValueError(f"Unknown MOVE_TO parameters: {parameters}")

        elif action_type == "CLICK":
            if not parameters:
                self.execute_python_command("pyautogui.click()")
            elif "button" in parameters and "x" in parameters and "y" in parameters:
                btn, x, y = parameters["button"], parameters["x"], parameters["y"]
                nc = parameters.get("num_clicks")
                if nc:
                    self.execute_python_command(f"pyautogui.click(button='{btn}', x={x}, y={y}, clicks={nc})")
                else:
                    self.execute_python_command(f"pyautogui.click(button='{btn}', x={x}, y={y})")
            elif "button" in parameters:
                btn = parameters["button"]
                nc = parameters.get("num_clicks")
                if nc:
                    self.execute_python_command(f"pyautogui.click(button='{btn}', clicks={nc})")
                else:
                    self.execute_python_command(f"pyautogui.click(button='{btn}')")
            elif "x" in parameters and "y" in parameters:
                x, y = parameters["x"], parameters["y"]
                nc = parameters.get("num_clicks")
                if nc:
                    self.execute_python_command(f"pyautogui.click(x={x}, y={y}, clicks={nc})")
                else:
                    self.execute_python_command(f"pyautogui.click(x={x}, y={y})")
            else:
                raise ValueError(f"Unknown CLICK parameters: {parameters}")

        elif action_type == "MOUSE_DOWN":
            btn = parameters.get("button", "left")
            self.execute_python_command(f"pyautogui.mouseDown(button='{btn}')")

        elif action_type == "MOUSE_UP":
            btn = parameters.get("button", "left")
            self.execute_python_command(f"pyautogui.mouseUp(button='{btn}')")

        elif action_type == "RIGHT_CLICK":
            if "x" in parameters and "y" in parameters:
                self.execute_python_command(f"pyautogui.rightClick(x={parameters['x']}, y={parameters['y']})")
            else:
                self.execute_python_command("pyautogui.rightClick()")

        elif action_type == "DOUBLE_CLICK":
            if "x" in parameters and "y" in parameters:
                self.execute_python_command(f"pyautogui.doubleClick(x={parameters['x']}, y={parameters['y']})")
            else:
                self.execute_python_command("pyautogui.doubleClick()")

        elif action_type == "DRAG_TO":
            self.execute_python_command(
                f"pyautogui.dragTo({parameters['x']}, {parameters['y']}, duration=1.0, button='left', mouseDownUp=True)"
            )

        elif action_type == "SCROLL":
            dx = parameters.get("dx")
            dy = parameters.get("dy")
            if dx is not None:
                self.execute_python_command(f"pyautogui.hscroll({dx})")
            if dy is not None:
                self.execute_python_command(f"pyautogui.vscroll({dy})")

        elif action_type == "TYPING":
            text = parameters["text"]
            self.execute_python_command("pyautogui.typewrite({:})".format(repr(text)))

        elif action_type == "PRESS":
            key = parameters["key"]
            if key.lower() not in _KEYBOARD_KEYS:
                raise ValueError(f"Key must be one of the known keyboard keys, got: {key!r}")
            self.execute_python_command(f"pyautogui.press('{key}')")

        elif action_type == "KEY_DOWN":
            key = parameters["key"]
            if key.lower() not in _KEYBOARD_KEYS:
                raise ValueError(f"Key must be one of the known keyboard keys, got: {key!r}")
            self.execute_python_command(f"pyautogui.keyDown('{key}')")

        elif action_type == "KEY_UP":
            key = parameters["key"]
            if key.lower() not in _KEYBOARD_KEYS:
                raise ValueError(f"Key must be one of the known keyboard keys, got: {key!r}")
            self.execute_python_command(f"pyautogui.keyUp('{key}')")

        elif action_type == "HOTKEY":
            keys: list[str] = parameters["keys"]
            if not isinstance(keys, list):
                raise ValueError("HOTKEY keys must be a list")
            for k in keys:
                if k.lower() not in _KEYBOARD_KEYS:
                    raise ValueError(f"Key must be one of the known keyboard keys, got: {k!r}")
            self.execute_python_command("pyautogui.hotkey('" + "', '".join(keys) + "')")

        elif action_type in ("WAIT", "FAIL", "DONE"):
            pass

        else:
            raise ValueError(f"Unknown action type: {action_type!r}")

    # ------------------------------------------------------------------
    # VM info
    # ------------------------------------------------------------------

    def get_vm_platform(self) -> str:
        """Return the platform string (e.g. 'Linux', 'Windows')."""
        result = self.execute_python_command("import platform; print(platform.system())")
        if result and result.get("output"):
            return result["output"].strip()
        return ""

    def get_vm_screen_size(self) -> dict[str, Any] | None:
        """Return the VM screen size dict."""
        for _ in range(_RETRY_TIMES):
            try:
                resp = requests.post(self._base_url + "/screen_size")
                if resp.status_code == 200:
                    return resp.json()
            except Exception as exc:
                logger.error("Screen size error: %s", exc)
            time.sleep(_RETRY_INTERVAL)
        return None

    def get_vm_window_size(self, app_class_name: str) -> dict[str, Any] | None:
        """Return the window size for an application by class name."""
        for _ in range(_RETRY_TIMES):
            try:
                resp = requests.post(self._base_url + "/window_size", data={"app_class_name": app_class_name})
                if resp.status_code == 200:
                    return resp.json()
            except Exception as exc:
                logger.error("Window size error: %s", exc)
            time.sleep(_RETRY_INTERVAL)
        return None

    def get_vm_desktop_path(self) -> str | None:
        """Return the desktop directory path inside the VM."""
        for _ in range(_RETRY_TIMES):
            try:
                resp = requests.post(self._base_url + "/desktop_path")
                if resp.status_code == 200:
                    return resp.json()["desktop_path"]
            except Exception as exc:
                logger.error("Desktop path error: %s", exc)
            time.sleep(_RETRY_INTERVAL)
        return None

    def get_vm_directory_tree(self, path: str) -> dict[str, Any] | None:
        """Return the directory tree for the given path inside the VM."""
        payload = json.dumps({"path": path})
        for _ in range(_RETRY_TIMES):
            try:
                resp = requests.post(
                    self._base_url + "/list_directory",
                    headers={"Content-Type": "application/json"},
                    data=payload,
                )
                if resp.status_code == 200:
                    return resp.json()["directory_tree"]
            except Exception as exc:
                logger.error("Directory tree error: %s", exc)
            time.sleep(_RETRY_INTERVAL)
        return None

    def get_vm_wallpaper(self) -> bytes | None:
        """Return the current desktop wallpaper image bytes."""
        for _ in range(_RETRY_TIMES):
            try:
                resp = requests.post(self._base_url + "/wallpaper")
                if resp.status_code == 200:
                    return resp.content
            except Exception as exc:
                logger.error("Wallpaper error: %s", exc)
            time.sleep(_RETRY_INTERVAL)
        return None

    # ------------------------------------------------------------------
    # VM state queries (used by WAA evaluation getters)
    # ------------------------------------------------------------------

    def get_vm_documents_path(self) -> str | None:
        """Return the Documents directory path inside the VM."""
        for _ in range(_RETRY_TIMES):
            try:
                resp = requests.post(self._base_url + "/documents_path", timeout=30)
                if resp.status_code == 200:
                    return resp.json()["documents_path"]
            except Exception as exc:
                logger.error("Documents path error: %s", exc)
            time.sleep(_RETRY_INTERVAL)
        return None

    def get_vm_folder_exists_in_path(self, folder_name: str, path: str) -> bool:
        """Return True if *folder_name* exists inside *path* on the VM."""
        payload = json.dumps({"folder_name": folder_name, "path": path})
        for _ in range(_RETRY_TIMES):
            try:
                resp = requests.post(
                    self._base_url + "/folder_exists",
                    headers={"Content-Type": "application/json"},
                    data=payload,
                    timeout=30,
                )
                if resp.status_code == 200:
                    return True
                return False
            except Exception as exc:
                logger.error("Folder exists error: %s", exc)
            time.sleep(_RETRY_INTERVAL)
        return False

    def get_vm_file_exists_in_path(self, file_name: str, path: str) -> bool:
        """Return True if *file_name* exists inside *path* on the VM."""
        payload = json.dumps({"file_name": file_name, "path": path})
        for _ in range(_RETRY_TIMES):
            try:
                resp = requests.post(
                    self._base_url + "/file_exists",
                    headers={"Content-Type": "application/json"},
                    data=payload,
                    timeout=30,
                )
                if resp.status_code == 200:
                    return True
                return False
            except Exception as exc:
                logger.error("File exists error: %s", exc)
            time.sleep(_RETRY_INTERVAL)
        return False

    def get_vm_are_files_sorted_by_modified_time(self, directory: str) -> bool:
        """Return True if files in *directory* are sorted by modified time."""
        payload = json.dumps({"directory": directory})
        for _ in range(_RETRY_TIMES):
            try:
                resp = requests.post(
                    self._base_url + "/are_files_sorted_by_modified_time",
                    headers={"Content-Type": "application/json"},
                    data=payload,
                    timeout=30,
                )
                if resp.status_code == 200:
                    return True
                return False
            except Exception as exc:
                logger.error("Files sorted check error: %s", exc)
            time.sleep(_RETRY_INTERVAL)
        return False

    def get_file_as_text(self, file_path: str) -> str | None:
        """Download a file from the VM and return its contents as text."""
        for _ in range(_RETRY_TIMES):
            try:
                resp = requests.post(self._base_url + "/file", data={"file_path": file_path}, timeout=30)
                if resp.status_code == 200:
                    return resp.text
            except Exception as exc:
                logger.error("Get file as text error: %s", exc)
            time.sleep(_RETRY_INTERVAL)
        return None

    def get_vm_file_explorer_is_details_view(self, path: str) -> bool:
        """Return True if File Explorer is in details view for *path*."""
        payload = json.dumps({"path": path})
        for _ in range(_RETRY_TIMES):
            try:
                resp = requests.post(
                    self._base_url + "/is_details_view",
                    headers={"Content-Type": "application/json"},
                    data=payload,
                    timeout=30,
                )
                if resp.status_code == 200:
                    return True
                return False
            except Exception as exc:
                logger.error("Details view check error: %s", exc)
            time.sleep(_RETRY_INTERVAL)
        return False

    def get_file_hidden_status(self, file_path: str) -> int:
        """Return 1 if the file is hidden (dotfile or Windows hidden attribute), 0 otherwise."""
        command = (
            f"import os; print(1 if os.path.basename(r'{file_path}').startswith('.') "
            f"or (os.name == 'nt' and bool(os.stat(r'{file_path}').st_file_attributes & 2)) else 0)"
        )
        response = self.execute_python_command(command)
        if response and response.get("status") == "success" and response.get("returncode") == 0:
            try:
                return int(response["output"].strip())
            except (ValueError, KeyError):
                return 0
        return 0

    def get_vm_is_directory_read_only_for_user(self, directory: str, user: str) -> bool:
        """Return True if *directory* is read-only for *user* on the VM."""
        payload = json.dumps({"directory": directory, "user": user})
        for _ in range(_RETRY_TIMES):
            try:
                resp = requests.post(
                    self._base_url + "/is_directory_read_only_for_user",
                    headers={"Content-Type": "application/json"},
                    data=payload,
                    timeout=30,
                )
                if resp.status_code == 200:
                    return True
                return False
            except Exception as exc:
                logger.error("Directory read-only check error: %s", exc)
            time.sleep(_RETRY_INTERVAL)
        return False

    def get_vm_are_all_images_tagged(self, directory: str, tag: str) -> bool:
        """Return True if all images in *directory* are tagged with *tag*."""
        payload = json.dumps({"directory": directory, "tag": tag})
        for _ in range(_RETRY_TIMES):
            try:
                resp = requests.post(
                    self._base_url + "/are_all_images_tagged",
                    headers={"Content-Type": "application/json"},
                    data=payload,
                    timeout=30,
                )
                if resp.status_code == 200:
                    return True
                return False
            except Exception as exc:
                logger.error("Images tagged check error: %s", exc)
            time.sleep(_RETRY_INTERVAL)
        return False

    def get_vm_library_folders(self, library_name: str) -> str | None:
        """Return the library folders output for *library_name*."""
        payload = json.dumps({"library_name": library_name})
        for _ in range(_RETRY_TIMES):
            try:
                resp = requests.post(
                    self._base_url + "/library_folders",
                    headers={"Content-Type": "application/json"},
                    data=payload,
                    timeout=30,
                )
                if resp.status_code == 200:
                    return resp.json()["output"]
            except Exception as exc:
                logger.error("Library folders error: %s", exc)
            time.sleep(_RETRY_INTERVAL)
        return None

    def get_vm_check_if_timer_started(self, hours: int, minutes: int, seconds: int) -> str:
        """Return 'True' if a timer with the given duration exists in the Clock app."""
        payload = json.dumps({"hours": hours, "minutes": minutes, "seconds": seconds})
        for _ in range(_RETRY_TIMES):
            try:
                resp = requests.post(
                    self._base_url + "/check_if_timer_started",
                    headers={"Content-Type": "application/json"},
                    data=payload,
                    timeout=30,
                )
                if resp.status_code == 200:
                    return "True"
                return "False"
            except Exception as exc:
                logger.error("Timer check error: %s", exc)
            time.sleep(_RETRY_INTERVAL)
        return "False"

    def get_vm_check_if_world_clock_exists(self, city: str, country: str) -> str:
        """Return 'True' if a world clock for *city*, *country* exists in the Clock app."""
        payload = json.dumps({"city": city, "country": country})
        for _ in range(_RETRY_TIMES):
            try:
                resp = requests.post(
                    self._base_url + "/check_if_world_clock_exists",
                    headers={"Content-Type": "application/json"},
                    data=payload,
                    timeout=30,
                )
                if resp.status_code == 200:
                    return "True"
                return "False"
            except Exception as exc:
                logger.error("World clock check error: %s", exc)
            time.sleep(_RETRY_INTERVAL)
        return "False"

    def execute_shell_command(self, command: str) -> dict[str, Any] | None:
        """Execute a shell command inside the VM via the /execute endpoint."""
        payload = json.dumps({"command": command})
        for _ in range(_RETRY_TIMES):
            try:
                resp = requests.post(
                    self._base_url + "/execute",
                    headers={"Content-Type": "application/json"},
                    data=payload,
                    timeout=90,
                )
                if resp.status_code == 200:
                    return resp.json()
            except Exception as exc:
                logger.error("Shell command error: %s", exc)
            time.sleep(_RETRY_INTERVAL)
        return None

    def get_registry_key(self, path: str, value: str) -> dict[str, Any] | None:
        """Read a Windows registry key via PowerShell."""
        command = f"powershell -Command \"Get-ItemPropertyValue -Path '{path}' -Name '{value}'\""
        return self.execute_shell_command(command)

    def get_all_installed_apps(self) -> dict[str, Any] | None:
        """Return JSON of all installed Windows apps via PowerShell."""
        command = 'powershell -Command "Get-AppxPackage | Select-Object Name,PackageFullName | ConvertTo-Json"'
        return self.execute_shell_command(command)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_valid_image(content_type: str, data: bytes | None) -> bool:
        if not isinstance(data, (bytes, bytearray)) or not data:
            return False
        if len(data) >= 8 and data[:8] == b"\x89PNG\r\n\x1a\n":
            return True
        if len(data) >= 3 and data[:3] == b"\xff\xd8\xff":
            return True
        if content_type and any(t in content_type for t in ("image/png", "image/jpeg", "image/jpg")):
            return True
        return False
