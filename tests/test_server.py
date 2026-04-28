"""Tests for cube.server — JSON-RPC 2.0 benchmark and task servers.

Uses a minimal inline cube (no counter-cube dependency) and
starlette.testclient.TestClient for in-process testing — no subprocesses, no ports.
"""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from cube.benchmark import Benchmark, BenchmarkConfig, BenchmarkMetadata, RuntimeContext
from cube.container import Container, ContainerBackend
from cube.core import Observation
from cube.server import make_benchmark_jsonrpc_app, make_task_jsonrpc_app
from cube.task import Task, TaskConfig, TaskMetadata
from cube.tool import Tool, ToolConfig, tool_action

# ── Minimal inline cube ────────────────────────────────────────────────────────


class _CounterTool(Tool):
    def __init__(self):
        self._counter = 0

    def reset(self) -> None:
        self._counter = 0

    @tool_action
    def increment(self) -> str:
        """Increment the counter by 1."""
        self._counter += 1
        return f"counter={self._counter}"

    @tool_action
    def get_value(self) -> str:
        """Return the current counter value."""
        return f"counter={self._counter}"


class _CounterToolConfig(ToolConfig):
    def make(self, container: Container | None = None) -> _CounterTool:
        return _CounterTool()


class _CounterTask(Task):
    """Finishes when the counter reaches 2."""

    _TARGET = 2

    def reset(self) -> tuple[Observation, dict[str, Any]]:
        self.tool.reset()
        return Observation.from_text("start"), {}

    def evaluate(self, obs: Observation | None = None) -> tuple[float, dict[str, Any]]:
        assert isinstance(self.tool, _CounterTool)
        done = self.tool._counter >= self._TARGET
        return (1.0 if done else 0.0), {"done": done}

    def finished(self, obs: Observation | None = None) -> bool:
        assert isinstance(self.tool, _CounterTool)
        return self.tool._counter >= self._TARGET


class _CounterTaskConfig(TaskConfig):
    def make(
        self,
        runtime_context: RuntimeContext | None = None,
        container_backend: ContainerBackend | None = None,
    ) -> _CounterTask:
        return _CounterTask(
            metadata=self.metadata,
            tool_config=self.tool_config or _CounterToolConfig(),
            runtime_context=runtime_context,
            container_backend=container_backend,
        )


class _MiniBenchmark(Benchmark):
    def _setup(self) -> None:
        pass

    def close(self) -> None:
        pass


class _MiniBenchmarkConfig(BenchmarkConfig):
    benchmark_metadata = BenchmarkMetadata(name="mini", version="0.1.0", description="test", num_tasks=2)
    task_metadata = {
        "task-1": TaskMetadata(id="task-1"),
        "task-2": TaskMetadata(id="task-2"),
    }
    task_config_class = _CounterTaskConfig
    benchmark_class = _MiniBenchmark


# ── Helpers ───────────────────────────────────────────────────────────────────

_TASK_META_1 = {
    "_type": "cube.task.TaskMetadata",
    "id": "task-1",
    "split": "test",
    "abstract_description": "",
    "recommended_max_steps": None,
    "container_config": None,
}

_TASK_META_2 = {**_TASK_META_1, "id": "task-2"}

_TASK_CONFIG_1 = _CounterTaskConfig(metadata=TaskMetadata(id="task-1")).model_dump(mode="json")
_TASK_CONFIG_2 = _CounterTaskConfig(metadata=TaskMetadata(id="task-2")).model_dump(mode="json")


def _text_content(data: str, tool_call_id: str | None = None) -> dict:
    return {"_type": "cube.core.TextContent", "data": data, "name": None, "tool_call_id": tool_call_id}


def _obs(data: str, tool_call_id: str | None = None) -> dict:
    return {"_type": "cube.core.Observation", "contents": [_text_content(data, tool_call_id)]}


_OBS_START = _obs("start")
_OBS_COUNTER_1 = _obs("counter=1")
_OBS_COUNTER_2 = _obs("counter=2")


def _rpc(client, method, params=None, req_id=1):
    body = {"jsonrpc": "2.0", "method": method, "id": req_id}
    if params:
        body["params"] = params
    return client.post("/", json=body)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def benchmark():
    return _MiniBenchmarkConfig().make()


@pytest.fixture
def bench_client(benchmark):
    app = make_benchmark_jsonrpc_app(benchmark)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def task(benchmark):
    t = benchmark.spawn(_CounterTaskConfig(metadata=TaskMetadata(id="task-1")))
    t.reset()
    return t


@pytest.fixture
def task_client(task):
    app = make_task_jsonrpc_app(task)
    with TestClient(app) as client:
        yield client


# ── Benchmark server ──────────────────────────────────────────────────────────


def test_benchmark_info(bench_client):
    resp = _rpc(bench_client, "cube/info")
    assert resp.status_code == 200
    assert resp.json()["result"] == {
        "_type": "cube.benchmark.BenchmarkMetadata",
        "name": "mini",
        "version": "0.1.0",
        "description": "test",
        "num_tasks": 2,
        "authors": [],
        "license": "",
        "requirements": {},
        "tags": [],
        "reset_isolation": None,
        "named_subsets": {},
    }


def test_benchmark_tasks_all(bench_client):
    resp = _rpc(bench_client, "cube/tasks")
    assert resp.json()["result"] == [_TASK_META_1, _TASK_META_2]


def test_benchmark_tasks_filter_by_id(bench_client):
    resp = _rpc(bench_client, "cube/tasks", {"task_id": "task-1"})
    assert resp.json()["result"] == [_TASK_META_1]


def test_benchmark_tasks_offset_limit(bench_client):
    resp = _rpc(bench_client, "cube/tasks", {"offset": 1, "limit": 1})
    assert resp.json()["result"] == [_TASK_META_2]


def test_benchmark_task_configs_all(bench_client):
    resp = _rpc(bench_client, "cube/task_configs")
    assert resp.json()["result"] == [_TASK_CONFIG_1, _TASK_CONFIG_2]


def test_benchmark_task_configs_filter_by_id(bench_client):
    resp = _rpc(bench_client, "cube/task_configs", {"task_id": "task-1"})
    assert resp.json()["result"] == [_TASK_CONFIG_1]


def test_benchmark_task_configs_offset_limit(bench_client):
    resp = _rpc(bench_client, "cube/task_configs", {"offset": 1, "limit": 1})
    assert resp.json()["result"] == [_TASK_CONFIG_2]


def test_benchmark_shutdown(bench_client):
    resp = _rpc(bench_client, "cube/shutdown")
    assert resp.status_code == 200
    assert resp.json()["result"] is None


# ── Task server ───────────────────────────────────────────────────────────────


def test_tools_list(task_client):
    resp = _rpc(task_client, "tools/list")
    assert resp.status_code == 200
    names = {t["name"] for t in resp.json()["result"]}
    assert names == {"increment", "get_value"}


def test_reset(task_client):
    resp = _rpc(task_client, "cube/reset")
    assert resp.status_code == 200
    assert resp.json()["result"] == {"obs": _OBS_START, "info": {}}


def test_tools_call(task_client):
    resp = _rpc(task_client, "tools/call", {"name": "increment"})
    assert resp.status_code == 200
    assert resp.json()["result"] == _OBS_COUNTER_1


def test_tools_call_with_action_id(task_client):
    resp = _rpc(task_client, "tools/call", {"name": "increment", "action_id": "abc-123"})
    assert resp.status_code == 200
    assert resp.json()["result"] == _obs("counter=1", tool_call_id="abc-123")


def test_step_full_episode(task_client):
    _rpc(task_client, "cube/reset")
    _rpc(task_client, "cube/step", {"name": "increment"})
    resp = _rpc(task_client, "cube/step", {"name": "increment"})
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert result["reward"] == 1.0
    assert result["done"] is True


def test_evaluate_without_obs(task_client):
    _rpc(task_client, "cube/reset")
    _rpc(task_client, "tools/call", {"name": "increment"})
    _rpc(task_client, "tools/call", {"name": "increment"})
    resp = _rpc(task_client, "cube/evaluate")
    assert resp.status_code == 200
    assert resp.json()["result"] == {"reward": 1.0, "info": {"done": True}}


def test_evaluate_with_obs(task_client):
    _rpc(task_client, "cube/reset")
    _rpc(task_client, "tools/call", {"name": "increment"})
    _rpc(task_client, "tools/call", {"name": "increment"})
    resp = _rpc(task_client, "cube/evaluate", {"obs": _OBS_COUNTER_2})
    assert resp.status_code == 200
    assert resp.json()["result"] == {"reward": 1.0, "info": {"done": True}}


def test_status(task_client):
    resp = _rpc(task_client, "cube/status")
    assert resp.status_code == 200
    assert resp.json()["result"] == ""


def test_privileged_info(task_client):
    resp = _rpc(task_client, "cube/privileged_info")
    assert resp.status_code == 200
    assert resp.json()["result"] == {
        "_type": "cube.core.StructuredContent",
        "data": {},
        "name": None,
        "tool_call_id": None,
    }


def test_close(task_client):
    resp = _rpc(task_client, "cube/close")
    assert resp.status_code == 200
    assert resp.json()["result"] is None


# ── JSON-RPC error envelope ───────────────────────────────────────────────────


def test_error_unknown_method(task_client):
    resp = _rpc(task_client, "cube/unknown")
    assert resp.json()["error"]["code"] == -32601


def test_error_missing_name_tools_call(task_client):
    resp = _rpc(task_client, "tools/call", {"arguments": {}})
    assert resp.json()["error"]["code"] == -32602


def test_error_missing_name_cube_step(task_client):
    resp = _rpc(task_client, "cube/step", {"arguments": {}})
    assert resp.json()["error"]["code"] == -32602


def test_error_invalid_json(task_client):
    resp = task_client.post("/", content=b"not json", headers={"content-type": "application/json"})
    assert resp.json()["error"]["code"] == -32700


def test_error_invalid_request(task_client):
    resp = task_client.post("/", json={"method": "tools/list"})  # missing "jsonrpc"
    assert resp.json()["error"]["code"] == -32600


# ── runtime_context JSON helpers ──────────────────────────────────────────────


def test_runtime_context_round_trip_preserves_infra_instance():
    """Regression: every cube that uses infra does ``_runtime_context["infra"] = self.infra``.
    Plain ``json.dumps`` would raise on the Pydantic model. The server's helpers
    must serialize TypedBaseModel values via their own ``model_dump(mode="json")``
    and rehydrate via ``_type`` dispatch.
    """
    from cube.infra_local import LocalInfraConfig
    from cube.server import _dump_runtime_context, _load_runtime_context

    original = {
        "infra": LocalInfraConfig(),
        "server_url": "http://internal:8080",
        "replicas": 3,
    }
    payload = _dump_runtime_context(original)
    assert payload is not None  # non-empty dict

    restored = _load_runtime_context(payload)
    assert restored is not None
    # JSON-native values unchanged
    assert restored["server_url"] == "http://internal:8080"
    assert restored["replicas"] == 3
    # Pydantic model rehydrated to the correct concrete class
    assert isinstance(restored["infra"], LocalInfraConfig)
    assert restored["infra"] == original["infra"]


def test_runtime_context_none_and_empty_return_none():
    from cube.server import _dump_runtime_context, _load_runtime_context

    assert _dump_runtime_context(None) is None
    assert _dump_runtime_context({}) is None
    assert _load_runtime_context(None) is None


def test_runtime_context_rejects_non_json_non_pydantic_values():
    """Non-JSON-native, non-TypedBaseModel values must fail loud rather than silently lose data."""
    from cube.server import _dump_runtime_context

    class _NotSerializable:
        pass

    with pytest.raises(TypeError, match="not JSON-serializable"):
        _dump_runtime_context({"bad": _NotSerializable()})


def test_runtime_context_rejects_non_json_native_builtin_types():
    """Builtin types that aren't JSON-native (set, datetime, function) must also fail loud."""
    import datetime

    from cube.server import _dump_runtime_context

    with pytest.raises(TypeError, match="not JSON-serializable"):
        _dump_runtime_context({"bad": {1, 2, 3}})

    with pytest.raises(TypeError, match="not JSON-serializable"):
        _dump_runtime_context({"bad": datetime.datetime(2026, 1, 1)})

    with pytest.raises(TypeError, match="not JSON-serializable"):
        _dump_runtime_context({"bad": lambda x: x})


def test_runtime_context_rejects_nested_non_json_native_value():
    """A bad value buried in a nested dict/list must still surface as TypeError."""
    from cube.server import _dump_runtime_context

    class _Bad:
        pass

    with pytest.raises(TypeError, match="not JSON-serializable"):
        _dump_runtime_context({"outer": {"inner": [1, _Bad()]}})
