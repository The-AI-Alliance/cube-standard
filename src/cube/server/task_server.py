"""Manages the lifecycle of task server subprocesses, handles port allocation, tracks active sessions."""

from __future__ import annotations

import logging
import multiprocessing
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import uvicorn

from cube.task import Task, TaskSession
from cube.types import (
    SpawnRequest, SpawnResponse, StatusRequest, StatusResponse,
    ShutdownRequest, ShutdownResponse, TaskStatus, TaskStatusEnum
)
from cube.environment import EnvConfig

if TYPE_CHECKING:
    from cube.benchmark import Benchmark


logger = logging.getLogger(__name__)


class TaskServerProcess:
    """Represents a running task server subprocess."""
    def __init__(self, session: TaskSession, port: int, process: multiprocessing.Process):
        self.session = session
        self.port = port
        self.process = process
        self.created_at = datetime.now()
        self.step_count = 0


class SessionManager:
    """
    Manages spawned task server subprocesses.

    Responsibilities:
    - Allocate ports from available pool
    - Spawn task server subprocesses
    - Track active sessions
    - Report status
    - Shutdown and cleanup
    """

    def __init__(self, benchmark: Benchmark, available_ports: list[int], host: str = "localhost"):
        logger.debug(f"[ENTRY] SessionManager.__init__ - benchmark={benchmark.name}, host={host}, available_ports={available_ports}")
        self.benchmark = benchmark
        self.host = host
        self.available_ports = list(available_ports)
        self.used_ports: list[int] = []
        self.active_sessions: Dict[str, TaskServerProcess] = {}
        logger.debug(f"[EXIT] SessionManager.__init__ - benchmark {benchmark.name} initialized.")

    def spawn(self, request: SpawnRequest) -> SpawnResponse:
        """
        Spawn a new task server subprocess.

        Steps:
        1. Find task by ID from benchmark.load_tasks()
        2. Create EnvConfig and Environment
        3. Create TaskSession
        4. Allocate port from pool
        5. Start FastAPI server in subprocess
        6. Return URL and session_id
        """
        logger.debug(f"[ENTRY] SessionManager.spawn - task_id={request.task_id} seed={request.seed}")

        # Get port from pool
        if not self.available_ports:
            raise RuntimeError("No available ports for task server")
        port = self.available_ports.pop(0)
        self.used_ports.append(port)

        # Find task
        tasks = self.benchmark.load_tasks()
        task = next((t for t in tasks if t.id == request.task_id), None)
        if not task:
            raise ValueError(f"Task {request.task_id} not found")

        # Create environment
        env_config = EnvConfig(task=task, tool_config=self.benchmark.tool_config)
        env = env_config.make()
        env.reset()  # TODO: pass the seed

        # Create TaskSession
        session = TaskSession(task_id=request.task_id, env=env)

        # Create task server app (Phase 1: minimal placeholder server)
        app = create_task_server_app(request.task_id, request.seed)
        # TODO: Phase 2: app = create_task_server_app(session)  # Pass the entire session instead

        # Start task server in subprocess
        def run_server():
            uvicorn.run(app, host=self.host, port=port, log_level="info")
            logger.info(f"Task server for task '{request.task_id}/{request.seed}' started on {self.host}:{port}.")

        task_process = multiprocessing.Process(target=run_server)
        task_process.start()

        # Track session
        server_process = TaskServerProcess(session, port, task_process)
        self.active_sessions[session.session_id] = server_process

        # Update task status
        session.status = TaskStatusEnum.running

        response = SpawnResponse(
            url=f"http://{self.host}:{port}",
            session_id=session.session_id,
            other={"session": session}
        )
        logger.debug(f"[EXIT] SessionManager.spawn - task {request.task_id}/{request.seed} running on session_id={session.session_id}, url={response.url}")
        return response

    def get_status(self, request: StatusRequest) -> StatusResponse:
        """Get status of one or many task sessions."""
        logger.debug(f"[ENTRY] SessionManager.get_status")

        if request.session_id:
            # Single session status
            if request.session_id not in self.active_sessions:
                logger.debug(f"[EXIT] SessionManager.get_status - session not found, returning empty")
                return StatusResponse(tasks=[])

            server_proc = self.active_sessions[request.session_id]
            session = server_proc.session
            logger.debug(f"[EXIT] SessionManager.get_status - returning status for session_id={request.session_id}")
            return StatusResponse(tasks=[session.get_status()])

        # All sessions status
        all_statuses = []
        for session_id, server_proc in self.active_sessions.items():
            session = server_proc.session
            all_statuses.append(session.get_status())
        
        # Apply offset and limit
        if request.limit == -1:
            limited_statuses = all_statuses[request.offset:]
        else:
            limited_statuses = all_statuses[request.offset : request.offset + request.limit]

        # TODO: check request.filter for filtering results

        logger.debug(f"[EXIT] SessionManager.get_status - returning {len(limited_statuses)} statuses")
        return StatusResponse(tasks=limited_statuses)

    def _shutdown_one_process(self, server_proc: TaskServerProcess):
        """Shutdown a single task server subprocess."""
        logger.debug(f"[ENTRY] SessionManager._shutdown_one_process - session_id={server_proc.session_id}, port={server_proc.port}")
        server_proc.process.terminate()
        server_proc.process.join(timeout=5)
        # Return port to pool
        self.available_ports.append(server_proc.port)
        self.used_ports.remove(server_proc.port)
        logger.debug(f"[EXIT] SessionManager._shutdown_one_process - task {server_proc.task.task_id}/{server_proc.task.seed} process terminated on session_id={server_proc.session_id}, port={server_proc.port} returned to pool")

    def shutdown(self, request: ShutdownRequest) -> ShutdownResponse:
        """Shutdown one or all task server subprocesses."""
        logger.debug(f"[ENTRY] SessionManager.shutdown")
        cleaned = []

        if request.session_id:
            # Shutdown single session
            if request.session_id in self.active_sessions:
                server_proc = self.active_sessions[request.session_id]
                self._shutdown_one_process(server_proc)
                cleaned.append(request.session_id)
                del self.active_sessions[request.session_id]
        else:
            # Shutdown all sessions
            for session_id, server_proc in list(self.active_sessions.items()):
                self._shutdown_one_process(server_proc)
                cleaned.append(session_id)
            self.active_sessions = {}

        logger.debug(f"[EXIT] SessionManager.shutdown - cleaned={cleaned}")
        return ShutdownResponse(success=True, cleaned=cleaned)


def create_task_server_app(task_id: str, seed: int | None) -> FastAPI:
    app = FastAPI(
        title=f"CUBE Task Server - {task_id}",
        description=f"Task-level API placeholder for task {task_id}",
        version="1.0.0"
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {
            "status": "ok",
            "task_id": task_id,
            "seed": seed,
            "message": "Task server running (Phase 1 - no task endpoints yet)"
        }

    # TODO Phase 2: Add task-level endpoints here
    # - POST /tools/list (MCP)
    # - POST /tools/call (MCP)
    # - POST /resources/list (MCP)
    # - POST /resources/read (MCP)
    # - POST /cube/evaluation (CUBE)
    # - POST /cube/step (CUBE)
    # - POST /cube/reset (CUBE)
    # - POST /cube/close (CUBE)
    logger.info(f"Task server for task '{task_id}/{seed}' created.")
    return app
