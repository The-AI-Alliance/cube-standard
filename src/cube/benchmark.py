import logging
import multiprocessing
from abc import ABC, abstractmethod

import uvicorn
from pydantic import PrivateAttr

from cube.environment import EnvConfig
from cube.server import SessionManager, create_benchmark_server_app
from cube.task import Task, TaskSession
from cube.tool import ToolConfig
from cube.types import (
    BenchmarkMetadata,
    ShutdownRequest,
    ShutdownResponse,
    SpawnRequest,
    SpawnResponse,
    StatusRequest,
    StatusResponse,
    TaskListResponse,
    TaskRequest,
    TaskStatusEnum,
    TypedBaseModel,
)

logger = logging.getLogger(__name__)


class Benchmark(TypedBaseModel, ABC):
    """Represents a benchmark consisting of multiple tasks."""

    metadata: BenchmarkMetadata  # cube/info returns this
    tool_config: ToolConfig | None = None  # ToolConfig for MCP server creation
    _task_list: list[Task] = PrivateAttr(default_factory=list)  # cache loaded tasks
    _local_sessions: dict[str, TaskSession] = {}  # track task sessions in python mode
    _session_manager: SessionManager | None = None  # set in setup() when server_mode is True
    _server_process: multiprocessing.Process | None = None  # set in setup() when server_mode is True

    @property
    def name(self) -> str:
        return self.metadata.name

    @abstractmethod
    def setup_benchmark_resources(self, tool_config: ToolConfig | None, **kwargs) -> None:
        """
        Optional method to set up any benchmark-level resources.
        This can include downloading datasets, initializing databases, setting up Dockers,  etc.
        This is supposed to be implemented by Benchmark *creators*.
        """
        if tool_config is not None:
            self.tool_config = tool_config

    def setup(
        self,
        tool_config: ToolConfig | None = None,
        server_mode: bool = False,
        available_ports: list[int] = [],
        server_host: str = "localhost",
        server_port: int = 8000,
    ) -> str | None:
        """
        Check if server_mode is enabled, and set up the benchmark accordingly.
        This is the main entry point to set up the benchmark for benchmark *users*.

        Args:
            tool_config (ToolConfig): Configuration for the tools to be used in the benchmark.
            server_mode (bool): If True, starts a benchmark server.
            available_ports (list[int]): List of ports available for task servers.
            server_host (str): Host for the benchmark & task servers.
            server_port (int): Port for the benchmark server.

        Returns:
            str | None: If server_mode is True, returns the address of the benchmark server, else None.
        """
        self.setup_benchmark_resources(tool_config)  # Prepare benchmark-level resources
        # Then set up server or python mode
        if server_mode:
            # Create session manager
            assert len(available_ports) > 0, "No available ports provided for server mode."
            self._session_manager = SessionManager(benchmark=self, available_ports=available_ports, host=server_host)

            # Create benchmark server app
            app = create_benchmark_server_app(self)

            # Start server in a separate process
            def run_server():
                uvicorn.run(app, host=server_host, port=server_port, log_level="info")
                logger.info(f"Benchmark server for '{self.metadata.name}' started on {server_host}:{server_port}.")

            self._server_process = multiprocessing.Process(target=run_server)
            self._server_process.start()
            return f"http://{server_host}:{server_port}"
        else:
            self._session_manager = None
            self._server_process = None
            return None

    def info(self) -> BenchmarkMetadata:
        """
        Return benchmark metadata.
        cube/info calls this method.
        """
        return self.metadata

    @abstractmethod
    def load_tasks(self, cache=True) -> list[Task]:
        """
        Load and return the list of tasks for this benchmark.
        """
        if len(self._task_list) > 0 and cache:
            return self._task_list
        raise NotImplementedError("load_tasks() must be implemented in subclass.")

    def list_tasks(self, request: TaskRequest) -> TaskListResponse:
        """
        List tasks with optional filtering, offset, and limit.
        cube/tasks calls this method.
        """
        tasks = self.load_tasks()
        total = len(tasks)

        # Apply filtering
        if request.task_id is not None:
            tasks = [task for task in tasks if task.id == request.task_id]

        # Apply offset and limit
        if request.limit == -1:
            limited_tasks = tasks[request.offset :]
        else:
            limited_tasks = tasks[request.offset : request.offset + request.limit]

        task_metadata_list = [task.metadata for task in limited_tasks]
        # TODO: check request.filter for filtering results

        return TaskListResponse(
            tasks=task_metadata_list,
            total=total,
            offset=request.offset,
            limit=request.limit,
        )

    def env_configs(self) -> list[EnvConfig]:
        """Generate environment configurations for all tasks in the benchmark."""
        tasks = self.load_tasks()
        configs = [EnvConfig(task=task) for task in tasks]
        return configs

    def install(self):
        """
        Optional method to download and prepare any resources required by the benchmark.
        """
        pass

    def uninstall(self):
        """
        Optional method to remove any resources used by the benchmark.
        """
        pass

    def spawn(self, request: SpawnRequest) -> SpawnResponse:
        """
        Spawn a new session for a given task.
        cube/spawn calls this method.

        Server mode: Creates subprocess running task server (returns URL)
        Python mode: Creates TaskSession in-process (returns session object)
        """
        logger.debug(f"[ENTRY] Benchmark.spawn - task_id={request.task_id} seed={request.seed}")

        if self._session_manager:
            logger.debug(
                f"[EXIT] Benchmark.spawn - delegating to SessionManager for task {request.task_id}/{request.seed}"
            )
            return self._session_manager.spawn(request)
        else:
            # Python mode: Create TaskSession in-process

            # Find task
            tasks = self.load_tasks()
            task = next((t for t in tasks if t.id == request.task_id), None)
            if not task:
                raise ValueError(f"Task {request.task_id} not found")

            # Create environment
            env_config = EnvConfig(task=task)
            env = env_config.make()
            env.reset()  # TODO: pass the seed

            # Create TaskSession
            session = TaskSession(task_id=request.task_id, env=env)
            # Track sessions locally
            self._local_sessions[session.session_id] = session
            # Update task status
            session.status = TaskStatusEnum.running

            response = SpawnResponse(
                url=None,  # No URL in Python mode
                session_id=session.session_id,
                other={"session": session},
            )
            logger.debug(
                f"[EXIT] Benchmark.spawn - created local session_id={session.session_id} for task {request.task_id}/{request.seed}"
            )
            return response

    def get_task_status(self, request: StatusRequest) -> StatusResponse:
        """
        Get the status of running tasks.
        cube/status calls this method.
        """
        logger.debug("[ENTRY] Benchmark.get_task_status")
        if self._session_manager:
            logger.debug("[EXIT] Benchmark.get_task_status - delegating to SessionManager")
            return self._session_manager.get_status(request)
        else:
            # Python mode: get status from local sessions

            if request.session_id:
                # Single session
                session = self._local_sessions.get(
                    request.session_id
                )  # TODO: check if we can get the task directly attaced to the session
                if session:
                    logger.debug(
                        f"[EXIT] Benchmark.get_task_status - returning status for session_id={request.session_id}"
                    )
                    return StatusResponse(tasks=[session.get_status()])
                logger.debug(
                    f"[EXIT] Benchmark.get_task_status - session_id={request.session_id} not found, returning empty"
                )
                return StatusResponse(tasks=[])
            else:
                # All sessions
                tasks_status = []
                for session_id, session in self._local_sessions.items():
                    tasks_status.append(session.get_status())
                logger.debug(
                    f"[EXIT] Benchmark.get_task_status - returning status for {len(tasks_status)} local sessions"
                )
                return StatusResponse(tasks=tasks_status)

    def _shutdown_one_local_session(self, session_id: str) -> bool:
        session = self._local_sessions.pop(session_id, None)
        if session:
            resp = session.close()
            return resp.success
        return False

    @abstractmethod
    def shutdown(self, request: ShutdownRequest) -> ShutdownResponse:
        """
        Shutdown a running task session.
        cube/shutdown calls this method.
        """
        logger.debug(f"[ENTRY] Benchmark.shutdown for session_id={request.session_id}")
        if self._session_manager:
            logger.debug(
                f"[EXIT] Benchmark.shutdown - delegating to SessionManager for session_id={request.session_id}"
            )
            return self._session_manager.shutdown(request)
        else:
            cleaned = []
            if request.session_id:
                # shutdown single session
                if self._shutdown_one_local_session(request.session_id):
                    cleaned.append(request.session_id)
                else:
                    logger.debug(f"[EXIT] Benchmark.shutdown - failed to close session_id={request.session_id}")
                    return ShutdownResponse(success=False, cleaned=cleaned)
            else:
                # shutdown all sessions
                for session_id in self._local_sessions:
                    if self._shutdown_one_local_session(session_id):
                        cleaned.append(session_id)
                if len(self._local_sessions) != 0:
                    logger.debug("[EXIT] Benchmark.shutdown - failed to close all local sessions")
                    return ShutdownResponse(success=False, cleaned=cleaned)
            logger.debug(f"[EXIT] Benchmark.shutdown - cleaned={cleaned}")
            return ShutdownResponse(success=True, cleaned=cleaned)

    @abstractmethod
    def close(self):
        """
        Clean up resources after all tasks are done.
        """
        running_tasks = self.get_task_status(StatusRequest())
        if len(running_tasks.tasks) > 0:
            raise RuntimeError("Cannot close benchmark while tasks are still running.")
        pass
