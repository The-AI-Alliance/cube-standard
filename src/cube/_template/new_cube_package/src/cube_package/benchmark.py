"""Benchmark for cube_package.

BenchmarkConfig is the serialisable registry (ClassVar metadata, entry-point
target, unit of serialisation across process boundaries).  CubeBenchmark is
the paired runtime class (setup / teardown of shared infrastructure).

Metadata — choose ONE of these two approaches and delete the other:
-------------------------------------------------------------------

OPTION A — inline ClassVars (default, used below)
    Define benchmark_metadata and task_metadata directly in the class body.
    Good for small benchmarks where all metadata fits comfortably in Python.

OPTION B — auto-load from CSV / JSON files
    Delete (or don't define) benchmark_metadata and task_metadata from the
    class body.  The framework will automatically load them from files placed
    next to THIS file (benchmark.py):

        src/cube_package/benchmark_metadata.json   ← or .csv
        src/cube_package/task_metadata.json         ← or .csv

    Both files are included in this template — just fill them in and remove
    the inline ClassVar definitions below.

    See cube.benchmark.BenchmarkConfig for full format details.

_setup() / close() on CubeBenchmark are the right place to start/stop shared
infrastructure (Docker daemons, database servers, etc.).  Leave them as no-ops
if your tasks are self-contained.

install() / uninstall() are optional classmethods on BenchmarkConfig for
one-time setup such as downloading datasets or pulling Docker images.  The base
class provides no-op defaults.
"""

from typing import ClassVar

from cube.benchmark import Benchmark, BenchmarkConfig, BenchmarkMetadata
from cube.task import TaskConfig, TaskMetadata
from cube_package.task import CubeTaskConfig


class CubeBenchmark(Benchmark):
    """Runtime pair — start/stop shared infrastructure here."""

    def _setup(self) -> None:
        """Start shared infrastructure (containers, servers, etc.).

        Populate self._runtime_context with URLs / handles that tasks need.
        No-op if tasks are fully self-contained.
        """

    def close(self) -> None:
        """Stop any infrastructure started in _setup()."""


class CubeBenchmarkConfig(BenchmarkConfig):
    """Registry of cube_package tasks."""

    # ── OPTION A: inline metadata (delete these two ClassVars to switch to Option B) ──

    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="new-cube-package",
        version="0.1.0",
        description="TODO: describe what this benchmark tests",
        authors=[],  # e.g. ["Your Name"]
        license="",  # e.g. "Apache-2.0"
        requirements={},  # e.g. {"docker": True}
        num_tasks=1,  # update when you add more tasks
        tags=[],  # e.g. ["web", "navigation"]
        extra_info={},
    )

    task_metadata: ClassVar[dict[str, TaskMetadata]] = {
        # TODO: add one TaskMetadata per task.
        # Keys must match TaskMetadata.id exactly.
        "example-task": TaskMetadata(
            id="example-task",
            split="test",  # "train", "val", or "test"
            abstract_description="TODO: one-sentence description of what this task tests",
            recommended_max_steps=10,
            container_config=None,  # set if task needs a container
            extra_info={},  # reserved for heavy execution data too large for task_metadata.json (e.g. problem statements, patches); merge in via get_task_configs() + load_task_execution_info(). For lightweight per-task params, add typed fields to a TaskMetadata subclass instead.
        ),
    }

    task_config_class: ClassVar[type[TaskConfig]] = CubeTaskConfig
    benchmark_class: ClassVar[type[Benchmark]] = CubeBenchmark
