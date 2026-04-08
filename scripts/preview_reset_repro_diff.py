#!/usr/bin/env python3
"""
Render sample **Reset reproducibility** warning panels (same layout as ``cube test``).

Use this to capture **PR screenshots** of the observation unified diff for different
payload shapes — without needing a benchmark that fails reset reproducibility.

Run from the repo root::

    uv run python scripts/preview_reset_repro_diff.py

Screenshot each labeled block (or one tall scroll) and attach to the pull request.
"""

from __future__ import annotations

import json
import sys

from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.text import Text

from cube.core import Observation
from cube.testing import format_observation_unified_diff


def _load_two_obs_from_jsonl(path: str) -> tuple[Observation, Observation]:
    obs_list = []
    with open(path) as f:
        for line in f:
            step = json.loads(line)
            output = step.get("output", {})
            if output.get("_type", "").endswith("EnvironmentOutput"):
                obs_list.append(Observation.model_validate(output["obs"]))
                if len(obs_list) == 2:
                    break
    if len(obs_list) < 2:
        raise ValueError(f"Need at least 2 EnvironmentOutput entries, found {len(obs_list)}")
    return obs_list[0], obs_list[1]


_RESET_MSG = "first observation differed between two resets"
_RESET_DIFF_DISPLAY_MAX = 24_000


def _warning_panel(*, label: str, reset_diff: str) -> None:
    console = Console(width=min(Console().size.width, 96))
    console.print(Rule(f"[bold]{label}[/bold]", style="cyan"))
    warn_bits: list = [
        Text.from_markup(
            "[warning]test_reset_reproducibility[/warning] "
            "[dim](first task, two fresh Task instances)[/dim]: "
            f"[bold]{_RESET_MSG}[/bold]. "
            "[dim]A mismatch is not always a bug (e.g. time-dependent observations).[/dim]"
        ),
    ]
    if reset_diff.strip():
        d = reset_diff
        if len(d) > _RESET_DIFF_DISPLAY_MAX:
            d = d[:_RESET_DIFF_DISPLAY_MAX] + "\n... [diff truncated]\n"
        warn_bits.append(Text.from_markup("[dim]Observation diff (unified — first reset vs second):[/dim]"))
        warn_bits.append(Syntax(d.rstrip("\n"), lexer="diff", word_wrap=True, line_numbers=False))
    console.print(
        Panel(
            Group(*warn_bits),
            title="[warning]Reset reproducibility[/warning]",
            border_style="yellow",
            padding=(0, 1),
        )
    )
    console.print()


def main() -> None:
    # 1) Typical Pydantic-style payload after model_dump(): dict with text + metadata
    a1 = {"text": "Step 1", "task_id": "debug-1", "seed": 42}
    b1 = {"text": "Step 1 (variant)", "task_id": "debug-1", "seed": 42}
    _warning_panel(
        label="Example A — dict / model_dump-style observation",
        reset_diff=format_observation_unified_diff(a1, b1),
    )

    # 2) Nested structure (e.g. multimodal-ish dict)
    a2 = {"screenshot": {"w": 80, "h": 60}, "hint": "ok"}
    b2 = {"screenshot": {"w": 80, "h": 61}, "hint": "ok"}
    _warning_panel(
        label="Example B — nested dict observation",
        reset_diff=format_observation_unified_diff(a2, b2),
    )

    # 3) No model_dump: compared via str() (e.g. opaque or legacy object)
    class _Opaque:
        def __init__(self, token: str) -> None:
            self._token = token

        def __str__(self) -> str:
            return f"OpaqueObs(token={self._token!r})"

    _warning_panel(
        label="Example C — non-dict observation (str() representation)",
        reset_diff=format_observation_unified_diff(_Opaque("alpha"), _Opaque("beta")),
    )


def main_from_jsonl(path: str) -> None:
    obs_a, obs_b = _load_two_obs_from_jsonl(path)
    _warning_panel(
        label=f"From trajectory — {path}",
        reset_diff=format_observation_unified_diff(obs_a, obs_b),
    )


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "scripts/sample_trajectory.jsonl"
    main_from_jsonl(path)
