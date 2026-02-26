# osworld-cube: OSWorld benchmark ported to the CUBE protocol
#
# Minimal public surface for the simple agent loop:
#
#   from osworld_cube import OSWorldTask, ComputerConfig, OSWorldBenchmark
#
#   # Simple loop (no harness):
#   task = OSWorldTask(
#       metadata=TaskMetadata(id="task-uuid", abstract_description="Open calculator",
#           extra_info={"domain": "os", "snapshot": "init_state",
#                       "config": [], "evaluator": {}, "related_apps": []}),
#       tool_config=ComputerConfig(provider="docker"),
#   )
#   obs, info = task.reset()
#   while not done:
#       action = agent(obs, task.action_set)
#       env_out = task.step(action)      # inherited from cube.task.Task
#       obs, done = env_out.obs, env_out.done
#   task.close()
#
#   # Via benchmark (full run):
#   bench = OSWorldBenchmark(default_tool_config=ComputerConfig(provider="docker"))
#   bench.setup()
#   for task_config in bench.get_task_configs():
#       task = task_config.make()
#       ...

from osworld_cube.computer import Computer, ComputerConfig, VMProvider
from osworld_cube.task import OSWorldTask
from osworld_cube.benchmark import OSWorldBenchmark, OSWorldTaskConfig

__all__ = [
    "Computer",
    "ComputerConfig",
    "VMProvider",
    "OSWorldTask",
    "OSWorldBenchmark",
    "OSWorldTaskConfig",
]
