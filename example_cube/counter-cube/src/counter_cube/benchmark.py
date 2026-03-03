"""Step 4b of 4 — Benchmark.

Benchmark is the top-level registry. Its two responsibilities:
  1. Hold metadata (as ClassVar) so it's available without instantiation.
  2. Vend TaskConfig objects via get_task_configs().

Three ClassVar attributes are required:
  benchmark_metadata  — BenchmarkMetadata describing the benchmark itself
  task_metadata       — dict[task_id, TaskMetadata] for all tasks
  task_config_class   — the TaskConfig subclass used to create tasks

For large benchmarks, metadata can be loaded from JSON/CSV files instead
of defined inline — see examples/toy_bench_auto_load_from_json/ for that
pattern. The inline approach shown here is easier to read at a glance.

_setup() and close() are no-ops here because this benchmark needs no
shared infrastructure (no VMs, no databases). Real cubes use _setup() to
start containers or connection pools and close() to tear them down.
"""

from typing import ClassVar

from cube.benchmark import Benchmark, BenchmarkMetadata
from cube.task import TaskConfig, TaskMetadata
from counter_cube.task import CounterTaskConfig




class CounterBenchmark(Benchmark):
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

    def _setup(self) -> None:
        """No shared infrastructure needed."""
        pass

    def close(self) -> None:
        """No resources to clean up."""
        pass
