"""Manages the lifecycle of task server subprocesses, handles port allocation, tracks active sessions."""

from __future__ import annotations

import logging
import multiprocessing
from typing import TYPE_CHECKING, Dict

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from cube.environment import EnvConfig
from cube.server.mcp_task_server import create_task_mcp_server
from cube.task import TaskSession
from cube.types import (
    MCPCallToolRequest,
    MCPReadResourceRequest,
    ResetRequest,
    ShutdownRequest,
    ShutdownResponse,
    SpawnRequest,
    SpawnResponse,
    StatusRequest,
    StatusResponse,
    StepRequest,
    TaskStatus,
    TaskStatusEnum,
)

if TYPE_CHECKING:
    from cube.benchmark import Benchmark


logger = logging.getLogger(__name__)


class TaskServerProcess:
    """Represents a running task server subprocess."""

    def __init__(self, session: TaskSession, port: int, process: multiprocessing.Process):
        self.session = session
        self.port = port
        self.process = process

    @property
    def session_id(self) -> str:
        return self.session.session_id

    @property
    def task_id(self) -> str:
        return self.session.task_id

    @property
    def seed(self) -> int | None:
        return self.session.env.task.seed

    def get_status(self) -> TaskStatus:
        return self.session.get_status()


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
        logger.debug(
            f"[ENTRY] SessionManager.__init__ - benchmark={benchmark.name}, host={host}, available_ports={available_ports}"
        )
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

        # Create in-memory MCP server (with tool_config if available)
        tool_config = getattr(self.benchmark, "tool_config", None)
        mcp_server = create_task_mcp_server(task, tool_config=tool_config)

        # Create environment (still needed for lifecycle)
        env_config = EnvConfig(task=task, tool_config=self.benchmark.tool_config)
        env = env_config.make()
        env.reset()  # TODO: pass the seed

        # Create TaskSession with MCP server reference
        session = TaskSession(task_id=request.task_id, env=env, mcp_server=mcp_server)

        # Create task server app
        app = create_task_server_app(session)

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
            url=f"http://{self.host}:{port}", session_id=session.session_id, other={"session": session}
        )
        logger.debug(
            f"[EXIT] SessionManager.spawn - task {request.task_id}/{request.seed} running on session_id={session.session_id}, url={response.url}"
        )
        return response

    def get_status(self, request: StatusRequest) -> StatusResponse:
        """Get status of one or many task sessions."""
        logger.debug("[ENTRY] SessionManager.get_status")

        if request.session_id:
            # Single session status
            if request.session_id not in self.active_sessions:
                logger.debug("[EXIT] SessionManager.get_status - session not found, returning empty")
                return StatusResponse(tasks=[])

            server_proc = self.active_sessions[request.session_id]
            logger.debug(f"[EXIT] SessionManager.get_status - returning status for session_id={request.session_id}")
            return StatusResponse(tasks=[server_proc.get_status()])

        # All sessions status
        all_statuses = []
        for session_id, server_proc in self.active_sessions.items():
            all_statuses.append(server_proc.get_status())

        # Apply offset and limit
        if request.limit == -1:
            limited_statuses = all_statuses[request.offset :]
        else:
            limited_statuses = all_statuses[request.offset : request.offset + request.limit]

        # TODO: check request.filter for filtering results

        logger.debug(f"[EXIT] SessionManager.get_status - returning {len(limited_statuses)} statuses")
        return StatusResponse(tasks=limited_statuses)

    def _shutdown_one_process(self, server_proc: TaskServerProcess):
        """Shutdown a single task server subprocess."""
        logger.debug(
            f"[ENTRY] SessionManager._shutdown_one_process - session_id={server_proc.session_id}, port={server_proc.port}"
        )
        server_proc.process.terminate()
        server_proc.process.join(timeout=5)
        # Return port to pool
        self.available_ports.append(server_proc.port)
        self.used_ports.remove(server_proc.port)
        logger.debug(
            f"[EXIT] SessionManager._shutdown_one_process - task {server_proc.task_id}/{server_proc.seed} process terminated on session_id={server_proc.session_id}, port={server_proc.port} returned to pool"
        )

    def shutdown(self, request: ShutdownRequest) -> ShutdownResponse:
        """Shutdown one or all task server subprocesses."""
        logger.debug("[ENTRY] SessionManager.shutdown")
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


def create_task_server_app(session: TaskSession) -> FastAPI:
    app = FastAPI(
        title=f"CUBE Task Server - {session.task_id}",
        description="Task-level API with MCP protocol and CUBE extensions",
        version="1.0.0",
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
            "task_id": session.task_id,
            "seed": session.seed,
            "session_id": session.session_id,
        }

    # MCP Protocol Endpoints (proxy to MCP server)
    @app.post("/tools/list")
    async def list_tools():
        """List available tools (MCP tools/list)."""
        result = await session.list_tools()
        return result

    @app.post("/tools/call")
    async def call_tool(request: MCPCallToolRequest):
        """Execute a tool (MCP tools/call)."""
        result = await session.call_tool(request)
        return result

    @app.post("/resources/list")
    async def list_resources():
        """List available resources (MCP resources/list)."""
        result = session.list_resources()
        return result

    @app.post("/resources/read")
    async def read_resource(request: MCPReadResourceRequest):
        """Read a resource (MCP resources/read)."""
        result = session.read_resource(str(request.params.uri))
        return result

    # CUBE Extension Endpoints
    @app.post("/cube/evaluate")
    async def evaluate_task():
        """Evaluate current task state."""
        result = session.evaluate()
        return result

    @app.post("/cube/step")
    async def step_task(request: StepRequest):
        """Combined tool call + evaluation."""
        result = await session.step(request)
        return result

    @app.post("/cube/reset")
    async def reset_task(request: ResetRequest):
        """Reset task to initial state."""
        result = session.reset(request)
        return result

    @app.post("/cube/close")
    async def close_task():
        """Close task session."""
        result = session.close()
        return result

    @app.post("/cube/status")
    async def get_status():
        """Get task session status."""
        result = session.get_status()
        return result

    logger.info(f"Task server for task '{session.task_id}/{session.seed}' created with MCP proxy endpoints.")
    return app
