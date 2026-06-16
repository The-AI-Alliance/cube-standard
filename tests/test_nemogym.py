"""Tests for cube.integrations.nemogym — CubeResourcesServer."""

from fastapi.testclient import TestClient

from cube.benchmark import Benchmark, BenchmarkConfig, BenchmarkMetadata, RuntimeContext
from cube.container import Container
from cube.core import Observation
from cube.integrations.nemogym import CubeResourcesServer
from cube.task import Task, TaskConfig, TaskMetadata
from cube.tool import Tool, ToolConfig, tool_action

# ---------------------------------------------------------------------------
# Minimal benchmark fixtures (same pattern as test_benchmark_server.py)
# ---------------------------------------------------------------------------


class _Tool(Tool):
    @tool_action
    def greet(self, name: str) -> str:
        """Say hello."""
        return f"Hello, {name}!"


class _ToolConfig(ToolConfig):
    def make(self, container: Container | None = None) -> Tool:
        return _Tool()


class _Task(Task):
    def reset(self):
        self.tool.reset()
        return Observation.from_text("Welcome! What would you like to do?"), {}

    def evaluate(self, obs: Observation | None = None):
        return 0.5, {"score": 0.5}


class _TaskConfig(TaskConfig):
    def make(self, runtime_context: RuntimeContext | None = None):
        return _Task(
            metadata=self.metadata,
            tool_config=self.tool_config or _ToolConfig(),
            runtime_context=runtime_context,
        )


class _Benchmark(Benchmark):
    def _setup(self):
        pass

    def close(self):
        pass


class _BenchmarkConfig(BenchmarkConfig):
    benchmark_metadata = BenchmarkMetadata(name="test-bench", version="0.1.0", description="test", num_tasks=2)
    task_metadata = {
        "t1": TaskMetadata(id="t1"),
        "t2": TaskMetadata(id="t2"),
    }
    task_config_class = _TaskConfig
    benchmark_class = _Benchmark


def _make_server() -> CubeResourcesServer:
    return CubeResourcesServer(config=_BenchmarkConfig())


def _make_client() -> TestClient:
    server = _make_server()
    return TestClient(server.make_app())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_health():
    with _make_client() as c:
        r = c.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["benchmark"] == "test-bench"
        assert data["num_tasks"] == 2


def test_list_tasks():
    with _make_client() as c:
        r = c.get("/tasks")
        assert r.status_code == 200
        tasks = r.json()
        assert len(tasks) == 2
        assert tasks[0] == {"idx": 0, "task_id": "t1"}
        assert tasks[1] == {"idx": 1, "task_id": "t2"}


def test_seed_session_returns_obs_and_tools():
    with _make_client() as c:
        r = c.post("/seed_session", json={"task_idx": 0})
        assert r.status_code == 200
        data = r.json()
        assert "env_id" in data
        assert isinstance(data["observation"], list)
        assert len(data["observation"]) > 0
        assert isinstance(data["tools"], list)
        assert len(data["tools"]) > 0
        # Tools should be OpenAI function format
        tool = data["tools"][0]
        assert tool["type"] == "function"
        assert "function" in tool


def test_seed_session_invalid_idx():
    with _make_client() as c:
        r = c.post("/seed_session", json={"task_idx": 999})
        assert r.status_code == 400


def test_step():
    with _make_client() as c:
        seed = c.post("/seed_session", json={"task_idx": 0}).json()
        env_id = seed["env_id"]

        r = c.post(
            "/step",
            json={
                "env_id": env_id,
                "action": {"id": "call_1", "name": "greet", "arguments": '{"name": "World"}'},
            },
        )
        assert r.status_code == 200
        data = r.json()
        assert "observation" in data
        assert "reward" in data
        assert "done" in data


def test_step_chat_completions_format():
    """Step accepts Chat Completions tool call format too."""
    with _make_client() as c:
        seed = c.post("/seed_session", json={"task_idx": 0}).json()

        r = c.post(
            "/step",
            json={
                "env_id": seed["env_id"],
                "action": {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "greet", "arguments": '{"name": "World"}'},
                },
            },
        )
        assert r.status_code == 200


def test_verify():
    with _make_client() as c:
        seed = c.post("/seed_session", json={"task_idx": 0}).json()
        env_id = seed["env_id"]

        r = c.post("/verify", json={"env_id": env_id})
        assert r.status_code == 200
        data = r.json()
        assert data["reward"] == 0.5


def test_close():
    with _make_client() as c:
        seed = c.post("/seed_session", json={"task_idx": 0}).json()
        env_id = seed["env_id"]

        r = c.post("/close", json={"env_id": env_id})
        assert r.status_code == 200

        # Subsequent step should return 404
        r = c.post(
            "/step",
            json={
                "env_id": env_id,
                "action": {"id": "call_1", "name": "greet", "arguments": "{}"},
            },
        )
        assert r.status_code == 404


def test_full_episode():
    """Simulate a full episode: seed → step × N → verify → close."""
    with _make_client() as c:
        # Seed
        seed = c.post("/seed_session", json={"task_idx": 1}).json()
        env_id = seed["env_id"]
        assert len(seed["observation"]) > 0
        assert len(seed["tools"]) > 0

        # Step
        step = c.post(
            "/step",
            json={
                "env_id": env_id,
                "action": {"id": "call_1", "name": "greet", "arguments": '{"name": "CUBE"}'},
            },
        ).json()
        assert isinstance(step["reward"], (int, float))

        # Verify
        verify = c.post("/verify", json={"env_id": env_id}).json()
        assert verify["reward"] == 0.5
        assert "score" in verify["reward_info"]

        # Close
        close = c.post("/close", json={"env_id": env_id}).json()
        assert close["status"] == "ok"


def test_multiple_concurrent_sessions():
    with _make_client() as c:
        s1 = c.post("/seed_session", json={"task_idx": 0}).json()
        s2 = c.post("/seed_session", json={"task_idx": 1}).json()
        assert s1["env_id"] != s2["env_id"]

        # Both can step independently
        r1 = c.post(
            "/step",
            json={
                "env_id": s1["env_id"],
                "action": {"id": "c1", "name": "greet", "arguments": '{"name": "A"}'},
            },
        )
        r2 = c.post(
            "/step",
            json={
                "env_id": s2["env_id"],
                "action": {"id": "c2", "name": "greet", "arguments": '{"name": "B"}'},
            },
        )
        assert r1.status_code == 200
        assert r2.status_code == 200
