"""Realistic observation mismatch cases (same shapes as ``scripts/preview_reset_repro_diff.py``)."""

from pathlib import Path

import pytest

from cube.core import Observation
from cube.testing import format_observation_diff

_REPO = Path(__file__).resolve().parent.parent
_SAMPLE_JSONL = _REPO / "scripts" / "sample_trajectory.jsonl"


def test_example_a_dict_style_diff_lists_text_path():
    a1 = {"text": "Step 1", "task_id": "debug-1", "seed": 42}
    b1 = {"text": "Step 1 (variant)", "task_id": "debug-1", "seed": 42}
    d = format_observation_diff(a1, b1)
    assert "Observation differences" in d
    assert "text" in d
    assert "Step 1" in d and "variant" in d


def test_example_b_nested_dict_diff():
    a2 = {"screenshot": {"w": 80, "h": 60}, "hint": "ok"}
    b2 = {"screenshot": {"w": 80, "h": 61}, "hint": "ok"}
    d = format_observation_diff(a2, b2)
    assert "screenshot" in d
    assert "first:" in d and "second:" in d


def test_example_c_opaque_objects_use_leaf_repr():
    class _Opaque:
        def __init__(self, token: str) -> None:
            self._token = token

        def __str__(self) -> str:
            return f"OpaqueObs(token={self._token!r})"

    d = format_observation_diff(_Opaque("alpha"), _Opaque("beta"))
    assert "<observation>" in d
    assert "first:" in d and "second:" in d
    # Mismatch uses repr() for arbitrary objects (not __str__).
    assert "_Opaque object" in d


def test_sample_trajectory_jsonl_first_two_environment_outputs_diff():
    """Loads bundled ``scripts/sample_trajectory.jsonl`` (two EnvironmentOutput lines)."""
    if not _SAMPLE_JSONL.is_file():
        pytest.skip("scripts/sample_trajectory.jsonl not present")

    import json

    obs_list: list[Observation] = []
    with _SAMPLE_JSONL.open() as f:
        for line in f:
            step = json.loads(line)
            output = step.get("output", {})
            if output.get("_type", "").endswith("EnvironmentOutput"):
                obs_list.append(Observation.model_validate(output["obs"]))
                if len(obs_list) == 2:
                    break
    assert len(obs_list) == 2
    d = format_observation_diff(obs_list[0], obs_list[1])
    assert "Observation differences" in d
    assert len(d) > 50
