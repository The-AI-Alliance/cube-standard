"""
NeMo Gym integration for CUBE.

Provides a CubeResourcesServer that wraps any CUBE BenchmarkConfig as an HTTP
server compatible with NeMo Gym's resource server protocol. NeMo Gym's CubeAgent
calls these endpoints over HTTP -- no NeMo Gym Python dependency required.

Endpoints:
    POST /seed_session  -- pick a task, reset it, return initial observation + tools
    POST /step          -- execute an action, return next observation + reward + done
    POST /verify        -- evaluate the current task state, return reward
    POST /close         -- close a task and free resources
    GET  /tasks         -- list available tasks and their indices

Usage:
    from cube.integrations.nemogym import CubeResourcesServer

    server = CubeResourcesServer(config=my_benchmark_config, infra=my_infra)
    server.run(host="0.0.0.0", port=8080)

Or programmatically:
    app = server.make_app()  # returns a FastAPI instance
"""

import logging
import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from cube.benchmark import Benchmark, BenchmarkConfig
from cube.core import Action
from cube.resource import InfraConfig
from cube.task import Task, TaskConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class SeedSessionRequest(BaseModel):
    task_idx: int


class SeedSessionResponse(BaseModel):
    env_id: str
    observation: list[dict]
    tools: list[dict]


class StepRequest(BaseModel):
    env_id: str
    action: dict


class StepResponse(BaseModel):
    observation: list[dict]
    reward: float
    done: bool
    error: str | None = None


class VerifyRequest(BaseModel):
    env_id: str


class VerifyResponse(BaseModel):
    reward: float
    reward_info: dict = {}


class CloseRequest(BaseModel):
    env_id: str


class TaskListItem(BaseModel):
    idx: int
    task_id: str


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class CubeResourcesServer:
    """Wraps a CUBE BenchmarkConfig as a NeMo Gym-compatible HTTP resource server.

    Manages multiple concurrent sessions (env_id -> Task), each corresponding
    to one episode. Calls ``config.install()`` then ``config.make(infra)`` to
    obtain a live ``Benchmark`` ready to spawn tasks.

    Args:
        config: A CUBE ``BenchmarkConfig`` describing the benchmark to serve.
        infra:  Optional ``InfraConfig`` passed to ``config.make(infra)`` for
                resource provisioning. None for benchmarks that don't need it.
    """

    def __init__(self, config: BenchmarkConfig, infra: InfraConfig | None = None) -> None:
        self.config = config
        config.install()  # populates task execution cache; CompositeBenchmarkConfig overrides as instance method
        self.benchmark: Benchmark = config.make(infra)
        self._task_configs: list[TaskConfig] = list(config.get_task_configs())
        self._sessions: dict[str, Task] = {}
        self._last_obs: dict[str, object] = {}  # env_id -> last Observation for verify

        if not self._task_configs:
            raise ValueError(f"Benchmark '{config.name}' has no task configs")

        logger.info("CubeResourcesServer loaded %d tasks from '%s'", len(self._task_configs), config.name)

    # -- Lifecycle -----------------------------------------------------------

    def _close_all(self) -> None:
        """Best-effort cleanup of all open sessions."""
        for env_id, task in list(self._sessions.items()):
            try:
                task.close()
                logger.debug("Closed session %s", env_id)
            except Exception:
                logger.exception("Error closing session %s", env_id)
        self._sessions.clear()
        self._last_obs.clear()

    # -- Endpoints -----------------------------------------------------------

    def list_tasks(self) -> list[TaskListItem]:
        return [TaskListItem(idx=i, task_id=tc.task_id) for i, tc in enumerate(self._task_configs)]

    def seed_session(self, body: SeedSessionRequest) -> SeedSessionResponse:
        if body.task_idx < 0 or body.task_idx >= len(self._task_configs):
            raise HTTPException(
                status_code=400, detail=f"task_idx {body.task_idx} out of range [0, {len(self._task_configs)})"
            )

        task_config = self._task_configs[body.task_idx]
        task = task_config.make(
            runtime_context=self.benchmark._runtime_context,
            container_backend=self.config.container_backend,
        )
        try:
            obs, _info = task.reset()
        except Exception:
            task.close()
            raise

        env_id = str(uuid.uuid4())
        self._sessions[env_id] = task
        self._last_obs[env_id] = obs

        return SeedSessionResponse(
            env_id=env_id,
            observation=obs.to_llm_messages(),
            tools=[schema.as_dict() for schema in task.action_set],
        )

    def step(self, body: StepRequest) -> StepResponse:
        task = self._sessions.get(body.env_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Unknown env_id: {body.env_id}")

        action = Action.from_openai_tool_call(body.action)
        env_output = task.step(action)

        # Note: env_output.obs is already post-processed by task.obs_postprocess().
        # task.step() calls evaluate() on the raw obs (before postprocessing), so
        # calling task.evaluate(env_output.obs) in /verify is slightly inconsistent
        # for tasks whose obs_postprocess mutates the observation. In practice most
        # evaluate() implementations inspect the environment state rather than the
        # obs argument, so this is rarely observable.
        self._last_obs[body.env_id] = env_output.obs

        return StepResponse(
            observation=env_output.obs.to_llm_messages(),
            reward=env_output.reward,
            done=env_output.done,
            error=str(env_output.error) if env_output.error else None,
        )

    def verify(self, body: VerifyRequest) -> VerifyResponse:
        task = self._sessions.get(body.env_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Unknown env_id: {body.env_id}")

        obs = self._last_obs.get(body.env_id)
        reward, info = task.evaluate(obs)
        return VerifyResponse(reward=reward, reward_info=info)

    def close_session(self, body: CloseRequest) -> dict:
        task = self._sessions.pop(body.env_id, None)
        self._last_obs.pop(body.env_id, None)
        if task is not None:
            task.close()
        return {"status": "ok"}

    # -- App factory ---------------------------------------------------------

    def make_app(self) -> FastAPI:
        """Build a FastAPI application wired to this server's endpoints."""
        server = self

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            yield
            server._close_all()
            server.benchmark.close()

        app = FastAPI(
            title=f"CUBE Resources Server -- {self.config.name}",
            lifespan=lifespan,
        )

        app.get("/tasks", response_model=list[TaskListItem])(self.list_tasks)
        app.post("/seed_session", response_model=SeedSessionResponse)(self.seed_session)
        app.post("/step", response_model=StepResponse)(self.step)
        app.post("/verify", response_model=VerifyResponse)(self.verify)
        app.post("/close")(self.close_session)

        @app.get("/health")
        def health():
            return {"status": "ok", "benchmark": self.config.name, "num_tasks": len(self._task_configs)}

        return app

    def run(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        """Start the server (blocking)."""
        app = self.make_app()
        uvicorn.run(app, host=host, port=port)
