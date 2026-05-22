"""Sidecar prompt overlay used by tests for ``load_benchmark_clarifications``.

Demonstrates reusing one clarification across several task ids (the regularizer).
"""

BENCHMARK_HINT = "Submit your final answer with final_step."

_SLIDER_TASKS = ["slider-1", "slider-2", "slider-3"]
TASK_CLARIFICATION = {tid: "After setting the values, click Submit." for tid in _SLIDER_TASKS}
