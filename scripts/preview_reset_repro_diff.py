#!/usr/bin/env python3
"""
Render sample **Reset reproducibility** panels using the same Rich layout as ``cube test``.

- **Default:** prints synthetic Examples A–C (dict, nested dict, opaque object; C uses ``repr()``).
- **With a path:** loads two ``EnvironmentOutput`` observations from a JSONL
  trajectory (same format as ``scripts/sample_trajectory.jsonl``).

Run from the repo root::

    uv run python scripts/preview_reset_repro_diff.py
    uv run python scripts/preview_reset_repro_diff.py scripts/sample_trajectory.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from rich.console import Console
from rich.rule import Rule

from cube.cli import _make_console, _print_reset_reproducibility_error_block
from cube.core import Observation
from cube.testing import format_observation_unified_diff

_RESET_MSG = "first observation differed between two resets"
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_JSONL = _REPO_ROOT / "scripts" / "sample_trajectory.jsonl"


def _terminal_width() -> int:
    try:
        return min(Console().size.width, 96)
    except Exception:
        return 96


def _load_two_obs_from_jsonl(path: Path) -> tuple[Observation, Observation]:
    obs_list: list[Observation] = []
    with path.open() as f:
        for line in f:
            step = json.loads(line)
            output = step.get("output", {})
            if output.get("_type", "").endswith("EnvironmentOutput"):
                obs_list.append(Observation.model_validate(output["obs"]))
                if len(obs_list) == 2:
                    break
    if len(obs_list) < 2:
        raise ValueError(f"Need at least 2 EnvironmentOutput entries in {path}, found {len(obs_list)}")
    return obs_list[0], obs_list[1]


def _emit(*, label: str, reset_diff: str, width: int) -> None:
    dash = _make_console(width=width)
    dash.print(Rule(f"[bold]{label}[/bold]", style="cyan"))
    _print_reset_reproducibility_error_block(
        dash,
        reset_ok=False,
        reset_msg=_RESET_MSG,
        reset_diff=reset_diff,
        panel_width=width,
    )


def main() -> None:
    w = _terminal_width()

    a1 = {"text": "Step 1", "task_id": "debug-1", "seed": 42}
    b1 = {"text": "Step 1 (variant)", "task_id": "debug-1", "seed": 42}
    _emit(
        label="Example A — dict / model_dump-style observation",
        reset_diff=format_observation_unified_diff(a1, b1),
        width=w,
    )

    a2 = {"screenshot": {"w": 80, "h": 60}, "hint": "ok"}
    b2 = {"screenshot": {"w": 80, "h": 61}, "hint": "ok"}
    _emit(
        label="Example B — nested dict observation",
        reset_diff=format_observation_unified_diff(a2, b2),
        width=w,
    )

    class _Opaque:
        def __init__(self, token: str) -> None:
            self._token = token

        def __str__(self) -> str:
            return f"OpaqueObs(token={self._token!r})"

    _emit(
        label="Example C — non-dict observation (repr / leaf comparison)",
        reset_diff=format_observation_unified_diff(_Opaque("alpha"), _Opaque("beta")),
        width=w,
    )


def main_from_jsonl(path: Path) -> None:
    obs_a, obs_b = _load_two_obs_from_jsonl(path)
    w = _terminal_width()
    _emit(
        label=f"From trajectory — {path}",
        reset_diff=format_observation_unified_diff(obs_a, obs_b),
        width=w,
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main_from_jsonl(Path(sys.argv[1]))
    else:
        main()
