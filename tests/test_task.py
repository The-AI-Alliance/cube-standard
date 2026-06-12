"""Tests for cube.task - Task, TaskMetadata, STOP_ACTION."""

import asyncio
import json

import pytest
from pydantic import PrivateAttr

from cube.container import Container
from cube.core import Action, EnvironmentOutput, Observation, TextContent
from cube.task import STOP_ACTION, AgentStop, AgentView, Task, TaskConfig, TaskExecutionInfo, TaskMetadata
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
    # New contract: a tool error is folded into the observation and is NOT terminal —
    # only finished()/evaluate() decide termination. (The structured obs.error is on the
    # per-action obs; step accumulates into a fresh Observation that merges only contents.)
    out = make_task().step(Action(name="fail", arguments={}))
    assert out.done is False
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


# --- AgentView (the agent-facing view) --------------------------------------


def test_get_agent_view_default_is_single_agent():
    task = make_task()
    view = task.get_agent_view()
    assert isinstance(view, AgentView)
    assert view._task is task  # one shared world
    assert view._tool is task._tool  # no-role seat reuses the task's own tool


def test_agent_view_action_set_mirrors_task():
    task = make_task()
    view = task.get_agent_view()
    assert {a.name for a in view.action_set} == {a.name for a in task.action_set}


def test_agent_view_execute_action_returns_observation_only():
    view = make_task().get_agent_view()
    obs = view.execute_action(Action(name="greet", arguments={"name": "World"}))
    assert isinstance(obs, Observation)
    assert obs.contents == [TextContent(data="Hello, World!")]


def test_agent_view_error_becomes_observation():
    obs = make_task().get_agent_view().execute_action(Action(name="fail", arguments={}))
    assert isinstance(obs, Observation)
    assert obs.error is not None
    assert obs.error.error_type == "ValueError"
    assert "ValueError" in obs.to_markdown()


def test_agent_view_stop_raises_agent_stop():
    view = make_task().get_agent_view()
    with pytest.raises(AgentStop) as excinfo:
        view.execute_action(Action(name=STOP_ACTION.name, arguments={}))
    assert "finished" in excinfo.value.observation.to_markdown().lower()


def test_step_and_view_share_one_core():
    # The two views must not diverge in per-action behavior: same action, same obs.
    action = Action(name="greet", arguments={"name": "Bob"})
    step_obs = make_task().step(action).obs
    view_obs = make_task().get_agent_view().execute_action(action)
    assert step_obs.to_markdown() == view_obs.to_markdown()


def test_agent_view_async_execute_action_matches_sync():
    view = make_task().get_agent_view()
    obs = asyncio.run(view.async_execute_action(Action(name="greet", arguments={"name": "Async"})))
    assert isinstance(obs, Observation)
    assert obs.contents == [TextContent(data="Hello, Async!")]


def test_agent_view_async_stop_raises_agent_stop():
    view = make_task().get_agent_view()
    with pytest.raises(AgentStop):
        asyncio.run(view.async_execute_action(Action(name=STOP_ACTION.name, arguments={})))


# --- per-step eval callback --------------------------------------------------


def test_eval_callback_fires_on_validate_per_step():
    view = make_task(validate_per_step=True).get_agent_view()
    got: list[tuple[float, dict]] = []
    view.set_eval_callback(lambda r, i: got.append((r, i)))
    view.execute_action(Action(name="greet", arguments={"name": "X"}))
    assert len(got) == 1 and got[0][0] == 0.5  # SimpleTask.evaluate -> (0.5, {"score": 0.5})


def test_eval_callback_silent_without_validate_per_step():
    view = make_task(validate_per_step=False).get_agent_view()
    got: list[float] = []
    view.set_eval_callback(lambda r, i: got.append(r))
    view.execute_action(Action(name="greet", arguments={"name": "X"}))
    assert got == []  # per-step eval only fires when validate_per_step is set


def test_eval_raises_when_validate_per_step_but_no_callback():
    # validate_per_step set but no consumer → the per-step reward would be silently dropped,
    # which is a wiring bug, so execute_action raises instead of discarding it.
    view = make_task(validate_per_step=True).get_agent_view()
    with pytest.raises(RuntimeError, match="no eval callback is registered"):
        view.execute_action(Action(name="greet", arguments={"name": "X"}))


# --- multi-agent roles (agent_roles / get_agent_view / _make_tool) -----------


def test_agent_roles_default_single_agent():
    assert make_task().agent_roles() == {None: 1}
    assert make_task().get_agent_view().agent_id == "agent"  # single seat keeps "agent"


def test_base_get_agent_view_raises_for_named_role():
    # The base only handles single-agent; a multi-agent benchmark must override it.
    with pytest.raises(NotImplementedError):
        make_task().get_agent_view(role="buyer")


def test_no_role_seat_reuses_task_tool():
    task = make_task()
    assert task.get_agent_view(None)._tool is task._tool  # no-role seat reuses the task's own tool


def test_multi_agent_override_builds_per_role_seats():
    # A multi-agent benchmark overrides get_agent_view, owning the seat index + per-role
    # tool internally; per-role action sets fall out of the role-bound tool.
    class _SellerTool(Tool):
        @tool_action
        def sell(self, item: str) -> str:
            """Sell an item."""
            return f"sold {item}"

    class _Market(SimpleTask):
        _seats: dict = PrivateAttr(default_factory=dict)  # internal per-role seat counter

        def agent_roles(self):
            return {"buyer": 2, "seller": 1}

        def get_agent_view(self, role=None):
            if role is None:
                return super().get_agent_view(None)
            seat = self._seats.get(role, 0)
            self._seats[role] = seat + 1
            tool = _SellerTool() if role == "seller" else self.tool_config.make()
            return AgentView(self, role=role, tool=tool, seat=seat)

    task = _Market(metadata=TaskMetadata(id="m"), tool_config=GreetToolConfig())
    b0, b1 = task.get_agent_view("buyer"), task.get_agent_view("buyer")
    seller = task.get_agent_view("seller")
    assert (b0.agent_id, b1.agent_id, seller.agent_id) == ("buyer-0", "buyer-1", "seller-0")
    assert "greet" in {a.name for a in b0.action_set} and "sell" not in {a.name for a in b0.action_set}
    assert "sell" in {a.name for a in seller.action_set}


def test_filter_actions_masks_both_gym_and_agent_views():
    # _filter_actions is advisory: it shapes what's advertised, in BOTH views identically.
    class _Restricted(SimpleTask):
        def _filter_actions(self, actions, role=None):
            return [a for a in actions if a.name != "fail"]

    task = _Restricted(metadata=TaskMetadata(id="r"), tool_config=GreetToolConfig())
    gym_names = {a.name for a in task.action_set}
    agent_names = {a.name for a in task.get_agent_view().action_set}
    assert "fail" not in gym_names and "fail" not in agent_names  # masked in both
    assert gym_names == agent_names  # the gym and agent paths never diverge
    assert "greet" in gym_names  # non-masked actions still advertised


def test_obs_postprocess_receives_seat_role():
    # The seat's role threads to obs_postprocess (twin of _filter_actions): None on the gym
    # view, the named role on a multi-agent seat's view.
    seen: list[str | None] = []

    class _Tagged(SimpleTask):
        def obs_postprocess(self, obs, role=None):
            seen.append(role)
            return obs

        def agent_roles(self):
            return {"watcher": 1}

        def get_agent_view(self, role=None):
            if role is None:
                return super().get_agent_view(None)
            return AgentView(self, role=role, tool=self._tool, seat=0)

    task = _Tagged(metadata=TaskMetadata(id="t"), tool_config=GreetToolConfig())
    task.step(Action(name="greet", arguments={"name": "X"}))  # gym view -> role None
    task.get_agent_view("watcher").execute_action(Action(name="greet", arguments={"name": "X"}))
    assert seen == [None, "watcher"]


def test_tool_property_raises_when_no_no_role_tool():
    # A multi-agent task whose tools are strictly per-role opts out of the no-role tool.
    class _NoAdminTool(SimpleTask):
        def agent_roles(self):
            return {"player": 2}

        def _make_tool(self, role=None):
            if role is None:
                raise NotImplementedError("per-role tools only")
            return self.tool_config.make()

    task = _NoAdminTool(metadata=TaskMetadata(id="n"), tool_config=GreetToolConfig())
    assert task._tool is None  # never built
    with pytest.raises(RuntimeError, match="no no-role tool"):
        _ = task.tool


def test_single_agent_make_tool_notimplemented_propagates():
    # Single-agent: the no-role tool is mandatory, so a real NotImplementedError in
    # _make_tool must surface at construction — NOT be masked as a roleless-tool opt-out
    # (which only applies to multi-agent strictly-per-role tasks).
    class _Broken(SimpleTask):
        def _make_tool(self, role=None):
            raise NotImplementedError("forgot to implement the tool")

    with pytest.raises(NotImplementedError, match="forgot to implement"):
        _Broken(metadata=TaskMetadata(id="b"), tool_config=GreetToolConfig())


def test_validate_per_step_with_callback_recuperates_reward():
    task = make_task(validate_per_step=True)
    view = task.get_agent_view()
    seen: list[tuple[float, dict]] = []
    view.set_eval_callback(lambda r, i: seen.append((r, i)))
    view.execute_action(Action(name="greet", arguments={"name": "x"}))
    assert seen == [(0.5, {"score": 0.5})]  # surfaced out-of-band, never in the obs


def test_task_action_set_includes_stop_action_by_default() -> None:
    task = make_task()
    stop_schemas = [a for a in task.action_set if a.name == STOP_ACTION.name]
    assert len(stop_schemas) == 1
    assert stop_schemas[0].parameters == {"type": "object", "properties": {}}


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
