"""Benchmark for cube_package.

Benchmark is the top-level registry: it holds ClassVar metadata and vends
TaskConfig objects to the episode runner.

Metadata — choose ONE of these two approaches and delete the other:
-------------------------------------------------------------------

OPTION A — inline ClassVars (default, used below)
    Define benchmark_metadata and task_metadata directly in the class body.
    Good for small benchmarks where all metadata fits comfortably in Python.

OPTION B — auto-load from CSV / JSON files
    Delete (or don't define) benchmark_metadata and task_metadata from the
    class body.  The framework will automatically load them from files placed
    next to THIS file (benchmark.py):

        src/cube_package/benchmark_metadata.csv   ← or .json
        src/cube_package/task_metadata.csv         ← or .json

    Both files are included in this template — just fill them in and remove
    the inline ClassVar definitions below.

    CSV format for benchmark_metadata.csv:
        name,version,description,num_tasks,tags
        my-bench,0.1.0,My benchmark description,5,"[""tag1"",""tag2""]"

    CSV format for task_metadata.csv (one row per task):
        id,abstract_description,recommended_max_steps,extra_info
        task-1,Do something useful,10,"{""key"": ""value""}"

    JSON is also supported — see cube.benchmark.Benchmark for full details.

_setup() / close() are the right place to start/stop shared infrastructure
(Docker daemons, database servers, etc.).  Leave them as no-ops if your
tasks are self-contained.
"""

from typing import ClassVar

from cube.benchmark import Benchmark, BenchmarkMetadata
from cube.task import TaskConfig, TaskMetadata
from cube_package.task import CubeTaskConfig


class CubeBenchmark(Benchmark):
    """Registry of cube_package tasks."""

    # ── OPTION A: inline metadata (delete these two ClassVars to switch to Option B) ──

    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="new-cube-package",
        version="0.1.0",
        description="TODO: describe what this benchmark tests",
        # authors=["Your Name"],
        # license="Apache-2.0",
        num_tasks=1,  # update when you add more tasks
        tags=[],  # e.g. ["web", "navigation"]
    )

    task_metadata: ClassVar[dict[str, TaskMetadata]] = {
        # TODO: add one TaskMetadata per task.
        # Keys must match TaskMetadata.id exactly.
        "example-task": TaskMetadata(
            id="example-task",
            abstract_description="TODO: one-sentence description of what this task tests",
            recommended_max_steps=10,
            extra_info={
                # Arbitrary per-task parameters; read in Task / TaskConfig via
                # metadata.extra_info["key"].
                # "target": 42,
                # "tool_config": {"enable_some_action": True},
            },
        ),
    }

    # ── Always required ───────────────────────────────────────────────────────────

    task_config_class: ClassVar[type[TaskConfig]] = CubeTaskConfig

    def _setup(self) -> None:
        """Start shared infrastructure (containers, servers, etc.).

        Populate self._runtime_context with URLs / handles that tasks need.
        No-op if tasks are fully self-contained.
        """
        pass

    def close(self) -> None:
        """Stop any infrastructure started in _setup()."""
        pass
