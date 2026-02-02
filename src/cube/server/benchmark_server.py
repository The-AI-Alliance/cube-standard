"""FastAPI server exposing benchmark-level APIs."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

if TYPE_CHECKING:
    from cube.benchmark import Benchmark
from cube.types import JSONRPCRequest, JSONRPCResponse, ShutdownRequest, SpawnRequest, StatusRequest, TaskRequest

logger = logging.getLogger(__name__)


def create_benchmark_server_app(benchmark: Benchmark) -> FastAPI:
    """
    Create FastAPI application for benchmark-level operations.

    Exposes benchmark-level endpoints:
    - cube/info - Get benchmark metadata
    - cube/tasks - List available tasks
    - cube/spawn - Spawn new task server
    - cube/status - Get task session status
    - cube/shutdown - Shutdown task sessions
    """
    app = FastAPI(
        title=f"CUBE Benchmark Server - {benchmark.metadata.name}",
        description=benchmark.metadata.description,
        version=benchmark.metadata.version
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/cube/info")
    async def cube_info(request: JSONRPCRequest) -> JSONRPCResponse:
        """Get benchmark metadata (CUBE cube/info)."""
        logger.debug(f"[ENTRY] cube/info - request_id={request.id}")
        try:
            metadata = benchmark.info()
            logger.debug(f"[EXIT] cube/info - request_id={request.id}, status=success")
            return JSONRPCResponse(
                result=metadata.model_dump(),
                id=request.id
            )
        except Exception as e:
            logger.debug(f"[EXIT] cube/info - request_id={request.id}, status=error, error={str(e)}")
            return JSONRPCResponse(
                error={"code": "INTERNAL_ERROR during cube/info", "message": str(e)},
                id=request.id
            )

    @app.post("/cube/tasks")
    async def cube_tasks(request: JSONRPCRequest) -> JSONRPCResponse:
        """List available tasks (CUBE cube/tasks)."""
        logger.debug(f"[ENTRY] cube/tasks - request_id={request.id}, params={request.params}")
        try:
            params = request.params or {}
            task_request = TaskRequest(**params)
            response = benchmark.list_tasks(task_request)

            logger.debug(f"[EXIT] cube/tasks - request_id={request.id}, status=success, task_count={len(response.tasks)}")
            return JSONRPCResponse(
                result=response.model_dump(),
                id=request.id
            )
        except Exception as e:
            logger.debug(f"[EXIT] cube/tasks - request_id={request.id}, status=error, error={str(e)}")
            return JSONRPCResponse(
                error={"code": "INTERNAL_ERROR during cube/tasks", "message": str(e)},
                id=request.id
            )

    @app.post("/cube/spawn")
    async def cube_spawn(request: JSONRPCRequest) -> JSONRPCResponse:
        """Spawn a new task server (CUBE cube/spawn)."""
        logger.debug(f"[ENTRY] cube/spawn - request_id={request.id}, params={request.params}")
        try:
            if not request.params:
                raise ValueError("Missing params for cube/spawn")

            spawn_request = SpawnRequest(**request.params)
            response = benchmark.spawn(spawn_request)

            logger.debug(f"[EXIT] cube/spawn - request_id={request.id}, status=success, session_id={response.session_id}")
            return JSONRPCResponse(
                result=response.model_dump(),
                id=request.id
            )
        except ValueError as e:
            logger.debug(f"[EXIT] cube/spawn - request_id={request.id}, status=error, error_type=INVALID_TASK, error={str(e)}")
            return JSONRPCResponse(
                error={"code": "INVALID_TASK during cube/spawn", "message": str(e)},
                id=request.id
            )
        except RuntimeError as e:
            logger.debug(f"[EXIT] cube/spawn - request_id={request.id}, status=error, error_type=RESOURCE_UNAVAILABLE, error={str(e)}")
            return JSONRPCResponse(
                error={"code": "RESOURCE_UNAVAILABLE during cube/spawn", "message": str(e)},
                id=request.id
            )
        except Exception as e:
            logger.debug(f"[EXIT] cube/spawn - request_id={request.id}, status=error, error_type=INTERNAL_ERROR, error={str(e)}")
            return JSONRPCResponse(
                error={"code": "INTERNAL_ERROR during cube/spawn", "message": str(e)},
                id=request.id
            )

    @app.post("/cube/status")
    async def cube_status(request: JSONRPCRequest) -> JSONRPCResponse:
        """Get task session status (CUBE cube/status)."""
        logger.debug(f"[ENTRY] cube/status - request_id={request.id}, params={request.params}")
        try:
            params = request.params or {}
            status_request = StatusRequest(**params)
            response = benchmark.get_task_status(status_request)

            logger.debug(f"[EXIT] cube/status - request_id={request.id}, status=success")
            return JSONRPCResponse(
                result=response.model_dump(),
                id=request.id
            )
        except Exception as e:
            logger.debug(f"[EXIT] cube/status - request_id={request.id}, status=error, error={str(e)}")
            return JSONRPCResponse(
                error={"code": "INTERNAL_ERROR during cube/status", "message": str(e)},
                id=request.id
            )

    @app.post("/cube/shutdown")
    async def cube_shutdown(request: JSONRPCRequest) -> JSONRPCResponse:
        """Shutdown task sessions (CUBE cube/shutdown)."""
        logger.debug(f"[ENTRY] cube/shutdown - request_id={request.id}, params={request.params}")
        try:
            params = request.params or {}
            shutdown_request = ShutdownRequest(**params)
            response = benchmark.shutdown(shutdown_request)

            logger.debug(f"[EXIT] cube/shutdown - request_id={request.id}, status=success")
            return JSONRPCResponse(
                result=response.model_dump(),
                id=request.id
            )
        except Exception as e:
            logger.debug(f"[EXIT] cube/shutdown - request_id={request.id}, status=error, error={str(e)}")
            return JSONRPCResponse(
                error={"code": "SHUTDOWN_FAILED", "message": str(e)},
                id=request.id
            )

    logger.info(f"Benchmark server for '{benchmark.metadata.name}' created.")
    return app