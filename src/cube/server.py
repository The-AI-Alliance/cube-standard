"""
HTTP server utilities for CUBE.

This module provides FastAPI app factories and server launchers for serving
benchmarks and tasks over a REST API, exposing endpoints for task listing,
spawning, stepping, and evaluation.

Current limitation — local deployment only:
    The `make_benchmark_rpc_server` and `make_task_rpc_server` helpers accept
    only a (host, port) pair and launch a subprocess on the local machine via
    `multiprocessing.Process`. This works well for single-machine evaluations
    but is insufficient for distributed or cloud deployments.

    To support remote hosting (AWS, GCP, Azure, Modal, fly.io, …) a future
    abstraction should decouple app creation from app deployment. A natural
    design would mirror the existing ContainerBackend pattern:

        class ServerBackend(ABC):
            @abstractmethod
            def deploy(self, app: FastAPI) -> str:
                '''Deploy app and return its public URL.'''

    Concrete backends could then handle cloud-specific concerns (auth, TLS,
    environment variables, cold-start behaviour, scaling) while callers stay
    agnostic to the deployment target. The existing `make_*_fastapi_app`
    helpers already return a plain FastAPI object and are a good foundation
    for this split.
"""

import logging
import multiprocessing
from typing import Any, Dict, List, Tuple

import uvicorn
from fastapi import FastAPI

from cube.benchmark import Benchmark, BenchmarkMetadata
from cube.core import Action, ActionSchema, EnvironmentOutput, Observation, StepError
from cube.task import Task, TaskConfig, TaskMetadata

# Type alias for server return value: (app, process, url)
ServerInfo = Tuple[FastAPI, multiprocessing.Process, str]

# TODO: figure out how to pass the relevant task-level runtime context (host, port, credentials, etc.)

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
        return benchmark.benchmark_metadata

    @app.get("/cube/tasks")
    def cube_tasks(task_id: str | None = None, offset: int = 0, limit: int = -1) -> list[TaskMetadata]:
        """Get task metadata with optional filtering."""
        tasks = list(benchmark.task_metadata.values())

        # Apply filtering by task_id
        if task_id:
            tasks = [tm for tm in tasks if tm.id == task_id]

        # Apply offset and limit
        if limit == -1:
            tasks = tasks[offset:]
        else:
            tasks = tasks[offset : offset + limit]

        return tasks

    @app.post("/cube/spawn")
    def cube_spawn(task_config: TaskConfig) -> str:
        return benchmark.spawn(task_config)

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
    - /cube/privileged_info - Get task privileged info

    Returns:
        FastAPI app
    """
    app = FastAPI(title=f"CUBE Task Server - {task.id}")

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
        return task.reset()

    @app.post("/cube/close")
    def close_task() -> None:
        """Close task."""
        task.close()

    @app.get("/cube/status")
    def get_status() -> str:
        """Get task status."""
        return task.get_status()

    @app.get("/cube/privileged_info")
    def get_privileged_info() -> Any:
        """Get task privileged info."""
        return task.get_privileged_info()

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
        logger.info(f"Task RPC server for task {task.id} started at http://{host}:{port}")

    task_process = multiprocessing.Process(target=run_server)
    task_process.start()

    url = f"http://{host}:{port}"
    return app, task_process, url
