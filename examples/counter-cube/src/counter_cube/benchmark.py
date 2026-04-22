"""Benchmark for counter-cube.

Top-level registry: holds ClassVar metadata and vends TaskConfig objects.
Required ClassVars on the BenchmarkConfig subclass: benchmark_metadata,
task_metadata, task_config_class, benchmark_class.
_setup() / close() on the paired Benchmark are no-ops here — real benchmarks
use them for containers / shared servers.
"""

from typing import ClassVar

from cube.benchmark import Benchmark, BenchmarkConfig, BenchmarkMetadata
from cube.task import TaskConfig, TaskMetadata

from counter_cube.task import CounterTaskConfig


class CounterBenchmark(Benchmark):
    """Runtime pair — counter-cube needs no shared infrastructure."""

    def _setup(self) -> None:
        """No shared infrastructure needed."""

    def close(self) -> None:
        """No resources to clean up."""


class CounterBenchmarkConfig(BenchmarkConfig):
    """Registry of counter tasks — minimal benchmark with no shared infrastructure."""

    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="counter-cube",
        version="0.1.0",
        description="Simplest possible benchmark — count to a target value",
        num_tasks=3,
        tags=["example"],
    )

    task_metadata: ClassVar[dict[str, TaskMetadata]] = {
        # Simplest task: increment 3 times, no extra actions.
        "count-to-3": TaskMetadata(
            id="count-to-3",
            abstract_description="Increment counter to reach value 3",
            recommended_max_steps=5,
            extra_info={"target": 3, "difficulty": "easy"},
        ),
        "count-to-3-with-decrement": TaskMetadata(
            id="count-to-3-with-decrement",
            abstract_description="Increment counter to reach value 3, with decrement available",
            recommended_max_steps=7,
            extra_info={"target": 3, "difficulty": "easy", "tool_config": {"enable_decrement": True}},
        ),
        "count-by-2": TaskMetadata(
            id="count-by-2",
            abstract_description="Reach 4 using an increment-by-2 tool",
            recommended_max_steps=4,
            extra_info={"target": 4, "difficulty": "easy", "tool_config": {"enable_increment_by": True}},
        ),
    }

    task_config_class: ClassVar[type[TaskConfig]] = CounterTaskConfig
    benchmark_class: ClassVar[type[Benchmark]] = CounterBenchmark
