"""
NeMo Gym integration for CUBE.

Provides a CubeResourcesServer that wraps any CUBE Benchmark as an HTTP server
compatible with NeMo Gym's resource server protocol. NeMo Gym's CubeAgent calls
these endpoints over HTTP — no NeMo Gym Python dependency required.

Endpoints:
    POST /seed_session  — pick a task, reset it, return initial observation + tools
    POST /step          — execute an action, return next observation + reward + done
    POST /verify        — evaluate the current task state, return reward
    POST /close         — close a task and free resources

Usage:
    from cube.integrations.nemogym import CubeResourcesServer

    server = CubeResourcesServer(benchmark=my_benchmark)
    server.run(host="0.0.0.0", port=8080)

Or programmatically:
    app = server.make_app()  # returns a FastAPI instance
"""

import atexit
import logging
import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from cube.benchmark import Benchmark
from cube.core import Action
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


class VerifyRequest(BaseModel):
    env_id: str


class VerifyResponse(BaseModel):
    reward: float
    reward_info: dict = {}


class CloseRequest(BaseModel):
    env_id: str


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class CubeResourcesServer:
    """Wraps a CUBE Benchmark as a NeMo Gym-compatible HTTP resource server.

    Manages multiple concurrent sessions (env_id → Task), each corresponding
    to one episode. The benchmark must be set up before constructing this server.

    Args:
        benchmark: A CUBE Benchmark instance (already set up via benchmark.setup()).
    """

    def __init__(self, benchmark: Benchmark) -> None:
        self.benchmark = benchmark
        self._task_configs: list[TaskConfig] = list(benchmark.get_task_configs())
        self._sessions: dict[str, Task] = {}
        self._last_obs: dict[str, object] = {}  # env_id → last Observation for verify

        if not self._task_configs:
            raise ValueError(f"Benchmark '{benchmark.name}' has no task configs")

        logger.info("CubeResourcesServer loaded %d tasks from '%s'", len(self._task_configs), benchmark.name)

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

    def seed_session(self, body: SeedSessionRequest) -> SeedSessionResponse:
        if body.task_idx < 0 or body.task_idx >= len(self._task_configs):
            raise HTTPException(status_code=400, detail=f"task_idx {body.task_idx} out of range [0, {len(self._task_configs)})")

        task_config = self._task_configs[body.task_idx]
        task = task_config.make(
            runtime_context=self.benchmark._runtime_context,
            container_backend=self.benchmark.container_backend,
        )

        obs, _info = task.reset()
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

        self._last_obs[body.env_id] = env_output.obs

        return StepResponse(
            observation=env_output.obs.to_llm_messages(),
            reward=env_output.reward,
            done=env_output.done,
        )

    def verify(self, body: VerifyRequest) -> VerifyResponse:
        task = self._sessions.get(body.env_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Unknown env_id: {body.env_id}")

        obs = self._last_obs.get(body.env_id)
        if obs is None:
            raise HTTPException(status_code=400, detail=f"No observation recorded for env_id: {body.env_id}")

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

        app = FastAPI(
            title=f"CUBE Resources Server — {self.benchmark.name}",
            lifespan=lifespan,
        )
        atexit.register(self._close_all)

        app.post("/seed_session", response_model=SeedSessionResponse)(self.seed_session)
        app.post("/step", response_model=StepResponse)(self.step)
        app.post("/verify", response_model=VerifyResponse)(self.verify)
        app.post("/close")(self.close_session)

        # Health check
        @app.get("/health")
        def health():
            return {"status": "ok", "benchmark": self.benchmark.name, "num_tasks": len(self._task_configs)}

        return app

    def run(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        """Start the server (blocking)."""
        app = self.make_app()
        uvicorn.run(app, host=host, port=port)
