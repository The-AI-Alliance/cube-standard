"""Tests for cube.task - Task, TaskMetadata, STOP_ACTION."""

import asyncio
import json

import pytest

from cube.container import Container
from cube.core import Action, EnvironmentOutput, Observation, TextContent
from cube.task import STOP_ACTION, AgentStop, Task, TaskConfig, TaskExecutionInfo, TaskMetadata, TaskTool
from cube.tool import Tool, ToolConfig, tool_action


class GreetTool(Tool):
    @tool_action
    def greet(self, name: str) -> str:
        """Greet someone."""
        return f"Hello, {name}!"

    @tool_action
    def fail(self) -> str:
        """Always raises."""
        raise ValueError("action failed")


class GreetToolConfig(ToolConfig):
    def make(self, container: Container | None = None) -> GreetTool:
        return GreetTool()


class SimpleTask(Task):
    def reset(self):
        return Observation.from_text("ready"), {}

    def evaluate(self, obs: Observation | None = None):
        return 0.5, {"score": 0.5}


def make_task(**kwargs) -> SimpleTask:
    return SimpleTask(
        metadata=TaskMetadata(id="simple-task"),
        tool_config=GreetToolConfig(),
        **kwargs,
    )


# --- TaskMetadata ---


def test_task_metadata_defaults():
    tm = TaskMetadata(id="my-task")
    assert tm == TaskMetadata(
        id="my-task",
        split="test",
        abstract_description="",
        recommended_max_steps=None,
        container_config=None,
    )


# --- Task.reset ---


def test_task_reset():
    obs, info = make_task().reset()
    assert obs.contents == [TextContent(data="ready")]
    assert info == {}


# --- Task.step ---


def test_task_step_stop_action_marks_done():
    out = make_task().step(Action(name=STOP_ACTION.name, arguments={}))
    assert isinstance(out, EnvironmentOutput)
    assert out.done is True
    assert out.error is None


def test_task_step_regular_action():
    out = make_task().step(Action(name="greet", arguments={"name": "World"}))
    assert isinstance(out, EnvironmentOutput)
    assert out.done is False
    assert out.obs.contents == [TextContent(data="Hello, World!")]


def test_task_step_action_error_becomes_observation():
    # New contract: a tool error is surfaced to the agent as an observation and is
    # NOT terminal — only finished()/evaluate() decide termination.
    out = make_task().step(Action(name="fail", arguments={}))
    assert out.done is False
    assert out.error is None
    assert "ValueError" in out.obs.to_markdown()
    assert "action failed" in out.obs.to_markdown()


def test_task_validate_per_step_triggers_evaluate():
    out = make_task(validate_per_step=True).step(Action(name="greet", arguments={"name": "Alice"}))
    assert out.reward == 0.5
    assert out.info["score"] == 0.5
    assert "profiling" in out.info


def test_task_action_set_comes_from_tool():
    names = {a.name for a in make_task().action_set}
    assert names == {"greet", "fail", STOP_ACTION.name}


# --- TaskTool (the agent-facing view) ---------------------------------------


def test_agent_tools_default_is_single_agent():
    task = make_task()
    tools = task.agent_tools()
    assert len(tools) == 1
    assert isinstance(tools[0], TaskTool)
    assert tools[0]._task is task  # one shared world


def test_task_tool_action_set_mirrors_task():
    task = make_task()
    tool = task.agent_tools()[0]
    assert {a.name for a in tool.action_set} == {a.name for a in task.action_set}


def test_task_tool_execute_action_returns_observation_only():
    tool = make_task().agent_tools()[0]
    obs = tool.execute_action(Action(name="greet", arguments={"name": "World"}))
    assert isinstance(obs, Observation)
    assert obs.contents == [TextContent(data="Hello, World!")]


def test_task_tool_error_becomes_observation():
    obs = make_task().agent_tools()[0].execute_action(Action(name="fail", arguments={}))
    assert isinstance(obs, Observation)
    assert "ValueError" in obs.to_markdown()


def test_task_tool_stop_raises_agent_stop():
    tool = make_task().agent_tools()[0]
    with pytest.raises(AgentStop) as excinfo:
        tool.execute_action(Action(name=STOP_ACTION.name, arguments={}))
    assert "finished" in excinfo.value.observation.to_markdown().lower()


def test_step_and_tool_share_one_core():
    # The two views must not diverge in per-action behavior: same action, same obs.
    action = Action(name="greet", arguments={"name": "Bob"})
    step_obs = make_task().step(action).obs
    tool_obs = make_task().agent_tools()[0].execute_action(action)
    assert step_obs.to_markdown() == tool_obs.to_markdown()


def test_task_tool_async_execute_action_matches_sync():
    tool = make_task().agent_tools()[0]
    obs = asyncio.run(tool.async_execute_action(Action(name="greet", arguments={"name": "Async"})))
    assert isinstance(obs, Observation)
    assert obs.contents == [TextContent(data="Hello, Async!")]


def test_task_tool_async_stop_raises_agent_stop():
    tool = make_task().agent_tools()[0]
    with pytest.raises(AgentStop):
        asyncio.run(tool.async_execute_action(Action(name=STOP_ACTION.name, arguments={})))


def test_task_tracks_last_action_error_for_runtime_telemetry():
    # A tool error is surfaced to the agent as an observation (non-terminal), but the
    # structured error is stashed so the runtime can record it; cleared on success.
    task = make_task()
    tool = task.agent_tools()[0]
    tool.execute_action(Action(name="fail", arguments={}))
    assert task._last_action_error is not None
    assert task._last_action_error.error_type == "ValueError"
    tool.execute_action(Action(name="greet", arguments={"name": "X"}))
    assert task._last_action_error is None


# --- per-step eval callback --------------------------------------------------


def test_eval_callback_fires_on_validate_per_step():
    tool = make_task(validate_per_step=True).agent_tools()[0]
    got: list[tuple[float, dict]] = []
    tool.set_eval_callback(lambda r, i: got.append((r, i)))
    tool.execute_action(Action(name="greet", arguments={"name": "X"}))
    assert len(got) == 1 and got[0][0] == 0.5  # SimpleTask.evaluate -> (0.5, {"score": 0.5})


def test_eval_callback_silent_without_validate_per_step():
    tool = make_task(validate_per_step=False).agent_tools()[0]
    got: list[float] = []
    tool.set_eval_callback(lambda r, i: got.append(r))
    tool.execute_action(Action(name="greet", arguments={"name": "X"}))
    assert got == []  # per-step eval only fires when validate_per_step is set


def test_eval_skipped_when_no_callback_registered():
    # validate_per_step set but no consumer → eval is skipped (no error).
    obs = (
        make_task(validate_per_step=True).agent_tools()[0].execute_action(Action(name="greet", arguments={"name": "X"}))
    )
    assert isinstance(obs, Observation)


# --- multi-agent roles (agent_roles / get_task_tool / make_tool) -------------


def test_agent_roles_default_single_agent():
    assert make_task().agent_roles() == {None: 1}
    assert make_task().get_task_tool().agent_id == "agent"  # single seat keeps "agent"


def test_get_task_tool_carries_role():
    t = make_task().get_task_tool(role="buyer")
    assert (t.role, t.seat, t.agent_id) == ("buyer", 0, "buyer-0")


def test_tool_is_lazy_and_memoized():
    task = make_task()
    assert task._tool is None  # not built at construction (lazy)
    first = task.tool
    assert task._tool is first and task.tool is first  # memoized
    assert task.get_task_tool(None)._tool is first  # no-role seat reuses the task's tool


def test_legacy_build_tool_override_still_runs_eagerly():
    # Back-compat: a cube that still overrides the legacy _build_tool (world-prep + make
    # together) gets it run eagerly at construction, so it isn't broken by lazy creation.
    built: list[str] = []

    class _Legacy(SimpleTask):
        def _build_tool(self) -> None:
            built.append("called")
            self._tool = self.tool_config.make()

    task = _Legacy(metadata=TaskMetadata(id="legacy"), tool_config=GreetToolConfig())
    assert built == ["called"]  # ran eagerly in model_post_init
    assert task._tool is not None  # eager, not lazy


def test_agent_tools_expands_roster_in_order():
    class _RoleTask(SimpleTask):
        def agent_roles(self):
            return {"buyer": 2, "seller": 1}

    task = _RoleTask(metadata=TaskMetadata(id="r"), tool_config=GreetToolConfig())
    assert [t.agent_id for t in task.agent_tools()] == ["buyer-0", "buyer-1", "seller-0"]


def test_per_role_tool_gives_per_role_actions():
    # Per-role action sets fall out of make_tool(role) returning a role-bound tool.
    class _SellerTool(Tool):
        @tool_action
        def sell(self, item: str) -> str:
            """Sell an item."""
            return f"sold {item}"

    class _Market(SimpleTask):
        def agent_roles(self):
            return {"buyer": 1, "seller": 1}

        def make_tool(self, role=None):
            return _SellerTool() if role == "seller" else super().make_tool(role)

    task = _Market(metadata=TaskMetadata(id="m"), tool_config=GreetToolConfig())
    buyer = {a.name for a in task.get_task_tool("buyer").action_set}
    seller = {a.name for a in task.get_task_tool("seller").action_set}
    assert "greet" in buyer and "sell" not in buyer
    assert "sell" in seller and "greet" not in seller


def test_task_action_set_includes_stop_action_by_default() -> None:
    task = make_task()
    stop_schemas = [a for a in task.action_set if a.name == STOP_ACTION.name]
    assert len(stop_schemas) == 1
    assert stop_schemas[0].parameters == {"type": "object", "properties": {}}


def test_task_action_set_excludes_stop_action_when_disabled() -> None:
    task = make_task(accept_agent_stop=False)
    names = {a.name for a in task.action_set}
    assert STOP_ACTION.name not in names


def test_task_action_set_stop_action_not_duplicated_if_filter_adds_it() -> None:
    class TaskWithFilterStop(SimpleTask):
        def filter_actions(self, actions):
            return [*actions, STOP_ACTION]

    task = TaskWithFilterStop(
        metadata=TaskMetadata(id="t"),
        tool_config=GreetToolConfig(),
    )
    stop_count = sum(1 for a in task.action_set if a.name == STOP_ACTION.name)
    assert stop_count == 1


def test_stop_action_parameters_valid_empty_schema() -> None:
    assert STOP_ACTION.parameters == {"type": "object", "properties": {}}


# --- TaskExecutionInfo ---


class _MyExecutionInfo(TaskExecutionInfo):
    instruction: str
    patch: str = ""


def test_task_execution_info_subclass_round_trip():
    """Subclasses round-trip through JSON via the ``_type`` discriminator."""
    info = _MyExecutionInfo(instruction="solve it", patch="diff --git ...")
    payload = info.model_dump_json()
    restored = TaskExecutionInfo.model_validate_json(payload)
    assert isinstance(restored, _MyExecutionInfo)
    assert restored == info


def test_task_execution_info_default_on_task_is_none():
    assert make_task().execution_info is None


def test_task_execution_info_field_round_trips_via_task():
    """The ``execution_info`` slot on Task preserves subclass fields through JSON."""
    info = _MyExecutionInfo(instruction="solve it")
    task = make_task(execution_info=info)
    reloaded = SimpleTask.model_validate_json(task.model_dump_json())
    assert isinstance(reloaded.execution_info, _MyExecutionInfo)
    assert reloaded.execution_info.instruction == "solve it"


# --- TaskConfig cache helpers ---


class _CacheTaskConfig(TaskConfig):
    def make(self, runtime_context=None):
        return SimpleTask(metadata=self.metadata, tool_config=GreetToolConfig())


def _cache_task_config(task_id: str = "task-1") -> _CacheTaskConfig:
    return _CacheTaskConfig(metadata=TaskMetadata(id=task_id), tool_config=GreetToolConfig())


def test_task_execution_cache_dir_falls_back_to_package_when_unowned(monkeypatch, tmp_path):
    """Without a back-stamp, the default keys on the top-level Python package."""
    import cube as cube_pkg

    monkeypatch.setattr(cube_pkg, "_CUBE_CACHE_ROOT", tmp_path)
    # _CacheTaskConfig is defined in this test module: top-level package "tests".
    expected = tmp_path / "tests" / "tasks_execution_info"
    assert _CacheTaskConfig.task_execution_cache_dir() == expected


def test_task_execution_cache_dir_uses_back_stamped_benchmark_dir(monkeypatch, tmp_path):
    """When BenchmarkConfig back-stamps ``_benchmark_cache_dir``, the cache lives under it."""
    monkeypatch.setattr(_CacheTaskConfig, "_benchmark_cache_dir", tmp_path / "my-bench")
    assert _CacheTaskConfig.task_execution_cache_dir() == tmp_path / "my-bench" / "tasks_execution_info"


def test_load_task_execution_info_raises_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(_CacheTaskConfig, "task_execution_cache_dir", classmethod(lambda cls: tmp_path))
    with pytest.raises(RuntimeError, match="No execution data"):
        _cache_task_config("missing-task").load_task_execution_info()


def test_load_task_execution_info_returns_dict_when_present(monkeypatch, tmp_path):
    monkeypatch.setattr(_CacheTaskConfig, "task_execution_cache_dir", classmethod(lambda cls: tmp_path))
    (tmp_path / "task-1.json").write_text(json.dumps({"instruction": "solve it"}))
    assert _cache_task_config("task-1").load_task_execution_info() == {"instruction": "solve it"}


def test_verify_installed_default_is_noop():
    """Base ``verify_installed`` is a no-op so cubes opt in only when they want fail-fast."""
    assert _cache_task_config().verify_installed() is None
