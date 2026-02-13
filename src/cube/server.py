import logging
import multiprocessing
from typing import Any, Dict, List, Tuple

import uvicorn
from fastapi import FastAPI

from cube.benchmark import Benchmark, BenchmarkMetadata
from cube.containers import ContainerBackend
from cube.core import Action, ActionSchema, EnvironmentOutput, Observation, StepError
from cube.task import Task, TaskConfig

# Type alias for server return value: (app, process, url)
ServerInfo = Tuple[FastAPI, multiprocessing.Process, str]

# TODO: figure out how to pass the relevant runtime info (host, port, credentials, etc.)

logger = logging.getLogger(__name__)


def make_benchmark_fastapi_app(benchmark: Benchmark) -> FastAPI:
    """
    Make a FastAPI app for a given benchmark (without spawning a server process).

    Exposes benchmark-level endpoints:
    - cube/info - Get benchmark metadata
    - cube/tasks - List available tasks
    - cube/spawn - Spawn new task server
    - cube/shutdown - Shutdown the benchmark

    Returns:
        FastAPI app
    """
    app = FastAPI(title=f"CUBE Benchmark Server - {benchmark.name}")

    @app.get("/cube/info")
    def cube_info() -> BenchmarkMetadata:
        return benchmark.metadata

    @app.get("/cube/tasks")
    def cube_tasks(task_id: str | None = None, offset: int = 0, limit: int = -1) -> list[TaskConfig]:
        return benchmark.get_task_configs(task_id=task_id, offset=offset, limit=limit)

    @app.post("/cube/spawn")
    def cube_spawn(task_id: str, container_backend: ContainerBackend | None = None) -> str:
        return benchmark.spawn(task_id=task_id, container_backend=container_backend)

    @app.post("/cube/shutdown")
    def cube_shutdown() -> None:
        return benchmark.close()

    return app


def make_benchmark_rpc_server(benchmark: Benchmark, host: str = "127.0.0.1", port: int = 8000) -> ServerInfo:
    """
    Make a JSON-RPC server for a given benchmark and spawn it in a separate process.

    Returns:
        ServerInfo: Tuple of (FastAPI app, server process, URL)
    """
    app = make_benchmark_fastapi_app(benchmark)

    def run_server() -> None:
        uvicorn.run(app, host=host, port=port)
        logger.info(f"Benchmark RPC server for benchmark {benchmark.name} started at http://{host}:{port}")

    server_process = multiprocessing.Process(target=run_server)
    server_process.start()

    url = f"http://{host}:{port}"
    return app, server_process, url


def make_task_fastapi_app(task: Task) -> FastAPI:
    """
    Make a FastAPI app for a given task (without spawning a server process).

    Exposes task-level endpoints:
    - /tools/list - List available tools
    - /tools/call - Call a tool
    - /resources/list - List available resources
    - /resources/read - Read a resource
    - /cube/evaluate - Evaluate an observation
    - /cube/step - Perform a step (tool call + evaluation)
    - /cube/reset - Reset a task
    - /cube/close - Close a task
    - /cube/status - Get task status
    - /cube/priviledged_info - Get task priviledged info

    Returns:
        FastAPI app
    """
    app = FastAPI(title=f"CUBE Task Server - {task.id}/{task.seed}")

    @app.get("/tools/list")
    def list_tools() -> List[ActionSchema]:
        return task.action_set

    @app.post("/tools/call")
    def call_tool(action: Action) -> Observation | StepError:
        return task.tool.execute_action(action)

    @app.get("/resources/list")
    def list_resources() -> List[str]:
        # TODO
        return []

    @app.post("/resources/read")
    def read_resource(resource_id: str) -> str:
        # TODO
        return ""

    @app.post("/cube/evaluate")
    def evaluate_task(obs: Observation) -> Tuple[float, dict]:
        return task.evaluate(obs)

    @app.post("/cube/step")
    def step_task(action: Action | List[Action]) -> EnvironmentOutput:
        """Combined tool call + evaluation."""
        return task.step(action)

    @app.post("/cube/reset")
    def reset_task() -> Tuple[Observation, Dict]:
        """Reset task to initial state."""
        return task.setup()

    @app.post("/cube/close")
    def close_task() -> None:
        """Close task."""
        task.close()

    @app.get("/cube/status")
    def get_status() -> str:
        """Get task status."""
        return task.get_status()

    @app.get("/cube/priviledged_info")
    def get_priviledged_info() -> Any:
        """Get task priviledged info."""
        return task.get_priviledged_info()

    return app


def make_task_rpc_server(task: Task, host: str = "127.0.0.1", port: int = 8000) -> ServerInfo:
    """
    Create a JSON-RPC server for a given task and spawn it in a separate process.

    Returns:
        ServerInfo: Tuple of (FastAPI app, server process, URL)
    """
    app = make_task_fastapi_app(task)

    def run_server() -> None:
        uvicorn.run(app, host=host, port=port)
        logger.info(f"Task RPC server for task {task.id}/{task.seed} started at http://{host}:{port}")

    task_process = multiprocessing.Process(target=run_server)
    task_process.start()

    url = f"http://{host}:{port}"
    return app, task_process, url
