"""Tests for cube.benchmark — BenchmarkConfig, Benchmark, subsetting, make()."""

from __future__ import annotations

import json
from typing import Literal

import pytest

from cube.benchmark import Benchmark, BenchmarkConfig, BenchmarkMetadata
from cube.core import Observation
from cube.seed import AbstractSeedGenerator
from cube.task import Task, TaskConfig, TaskMetadata
from cube.tool import Tool, ToolConfig, tool_action

# ── Minimal fixtures ──────────────────────────────────────────────────────────


class _Tool(Tool):
    @tool_action
    def noop(self) -> str:
        """No-op."""
        return "ok"


class _ToolConfig(ToolConfig):
    def make(self, container=None):
        return _Tool()


class _Task(Task):
    def reset(self):
        return Observation.from_text("ready"), {}

    def evaluate(self, obs=None):
        return 0.0, {}


class _TaskConfig(TaskConfig):
    def make(self, runtime_context=None, container_backend=None):
        return _Task(
            metadata=self.metadata,
            tool_config=self.tool_config or _ToolConfig(),
        )


class MyBenchmark(Benchmark):
    def _setup(self):
        pass

    def close(self):
        pass


class _MyTaskMetadata(TaskMetadata):
    """Subclass that ships a typed ``difficulty`` field — replaces the old
    ``extra_info["difficulty"]`` smuggling channel for filter / glob tests."""

    difficulty: Literal["easy", "hard"] = "easy"


class MyBenchmarkConfig(BenchmarkConfig):
    benchmark_metadata = BenchmarkMetadata(
        name="MyBenchmark",
        version="2.0.0",
        description="Test benchmark",
        num_tasks=4,
    )
    task_metadata = {
        "t1": _MyTaskMetadata(id="t1", split="train", difficulty="easy"),
        "t2": _MyTaskMetadata(id="t2", split="train", difficulty="hard"),
        "t3": _MyTaskMetadata(id="t3", split="val", difficulty="easy"),
        "t4": _MyTaskMetadata(id="t4", split="test", difficulty="hard"),
    }
    task_config_class = _TaskConfig
    benchmark_class = MyBenchmark


# ── BenchmarkMetadata ─────────────────────────────────────────────────────────


def test_benchmark_metadata_defaults():
    bm = BenchmarkMetadata(name="foo", version="1.0", description="bar")
    assert bm == BenchmarkMetadata(
        name="foo",
        version="1.0",
        description="bar",
        authors=[],
        tags=[],
        num_tasks=0,
        license="",
        requirements={},
        named_subsets={},
        reset_isolation=None,
    )


# ── __init_subclass__ validation ──────────────────────────────────────────────


def test_missing_benchmark_class_raises():
    """Concrete BenchmarkConfig subclass without benchmark_class fails at class-def time."""
    with pytest.raises(TypeError, match="benchmark_class"):

        class _Bad(BenchmarkConfig):  # noqa: F841
            benchmark_metadata = BenchmarkMetadata(name="bad", version="1", description="x")
            task_metadata = {"x": TaskMetadata(id="x")}
            task_config_class = _TaskConfig
            # missing: benchmark_class


def test_missing_task_config_class_raises():
    with pytest.raises(TypeError, match="task_config_class"):

        class _Bad(BenchmarkConfig):  # noqa: F841
            benchmark_metadata = BenchmarkMetadata(name="bad", version="1", description="x")
            task_metadata = {"x": TaskMetadata(id="x")}
            benchmark_class = MyBenchmark
            # missing: task_config_class


def test_classvar_override_of_parent_property_is_validated():
    """A subclass that shadows a parent ``@property`` with a ClassVar must be
    *validated* — the parent's property must not bleed through and skip the
    type-check branch in ``__init_subclass__``.

    ``_is_dynamic`` has to look at the *nearest* definition in the MRO, not
    ``any()`` over the whole chain; otherwise an invalid ClassVar on the
    child slips past validation.
    """

    class _DynamicParent(BenchmarkConfig):
        task_config_class = _TaskConfig
        benchmark_class = MyBenchmark

        @property
        def benchmark_metadata(self) -> BenchmarkMetadata:  # type: ignore[override]
            return BenchmarkMetadata(name="dynamic", version="0", description="computed")

        @property
        def task_metadata(self) -> dict[str, TaskMetadata]:  # type: ignore[override]
            return {}

    # An invalid ClassVar override on the child must be rejected — proves the
    # validation branch ran instead of being skipped because of the parent's
    # ``@property``.
    with pytest.raises(TypeError, match="benchmark_metadata"):

        class _BadStaticChild(_DynamicParent):  # noqa: F841
            benchmark_metadata = "not a BenchmarkMetadata"  # type: ignore[assignment]
            task_metadata = {"s1": TaskMetadata(id="s1")}

    # A valid ClassVar override must be accepted, and the ClassVar value must
    # surface on the class.
    class _StaticChildTaskConfig(_TaskConfig):
        pass

    class _StaticChild(_DynamicParent):
        benchmark_metadata = BenchmarkMetadata(name="static", version="1", description="x")
        task_metadata = {"s1": TaskMetadata(id="s1")}
        task_config_class = _StaticChildTaskConfig

    assert _StaticChild.benchmark_metadata.name == "static"
    assert "s1" in _StaticChild.task_metadata

    # Grandchild that adds nothing must inherit the child's ClassVars cleanly,
    # i.e. validation walks the MRO and stops at ``_StaticChild``.
    class _StaticGrandchild(_StaticChild):  # noqa: F841
        pass


def test_init_subclass_back_stamps_benchmark_cache_dir():
    """Concrete BenchmarkConfig subclasses stamp ``cls.cache_dir()`` onto their
    ``task_config_class`` so ``TaskConfig.task_execution_cache_dir()`` lives
    directly under ``BenchmarkConfig.cache_dir()``."""

    class _StampTaskConfig(TaskConfig):
        def make(self, runtime_context=None, container_backend=None):
            return _Task(metadata=self.metadata, tool_config=_ToolConfig())

    class _StampBenchmarkConfig(BenchmarkConfig):
        benchmark_metadata = BenchmarkMetadata(name="stamp-bench", version="1", description="x")
        task_metadata = {"t1": TaskMetadata(id="t1")}
        task_config_class = _StampTaskConfig
        benchmark_class = MyBenchmark

    assert _StampTaskConfig._benchmark_cache_dir == _StampBenchmarkConfig.cache_dir()
    assert _StampTaskConfig.task_execution_cache_dir() == _StampBenchmarkConfig.cache_dir() / "tasks_execution_info"


def test_init_subclass_skips_back_stamp_for_abstract_taskconfig_placeholder():
    """``CompositeBenchmarkConfig`` uses the abstract ``TaskConfig`` as a placeholder
    (it overrides ``get_task_configs``). The back-stamp must skip it so the
    abstract base never carries a benchmark cache dir."""
    assert TaskConfig._benchmark_cache_dir is None


def test_init_subclass_rejects_shared_task_config_class():
    """Two ``BenchmarkConfig`` subclasses pointing at the same ``task_config_class``
    would silently overwrite each other's stamp. Class definition must fail loudly."""

    class _SharedTaskConfig(TaskConfig):
        def make(self, runtime_context=None, container_backend=None):
            return _Task(metadata=self.metadata, tool_config=_ToolConfig())

    class _OwnerOne(BenchmarkConfig):
        benchmark_metadata = BenchmarkMetadata(name="owner-one", version="1", description="x")
        task_metadata = {"t1": TaskMetadata(id="t1")}
        task_config_class = _SharedTaskConfig
        benchmark_class = MyBenchmark

    with pytest.raises(TypeError, match="already owned by benchmark"):

        class _OwnerTwo(BenchmarkConfig):  # noqa: F841
            benchmark_metadata = BenchmarkMetadata(name="owner-two", version="1", description="x")
            task_metadata = {"t1": TaskMetadata(id="t1")}
            task_config_class = _SharedTaskConfig
            benchmark_class = MyBenchmark


def test_task_execution_cache_dir_does_not_inherit_via_mro():
    """A ``TaskConfig`` subclass without its own owning ``BenchmarkConfig`` must
    fall back to the package name — it must NOT silently inherit the parent's
    stamp through the MRO."""

    class _OwnedTaskConfig(TaskConfig):
        def make(self, runtime_context=None, container_backend=None):
            return _Task(metadata=self.metadata, tool_config=_ToolConfig())

    class _OwningBenchmarkConfig(BenchmarkConfig):  # noqa: F841
        benchmark_metadata = BenchmarkMetadata(name="owning-bench", version="1", description="x")
        task_metadata = {"t1": TaskMetadata(id="t1")}
        task_config_class = _OwnedTaskConfig
        benchmark_class = MyBenchmark

    class _DerivedTaskConfig(_OwnedTaskConfig):
        """Test-scaffold subclass with no owning BenchmarkConfig of its own."""

    # Parent is stamped.
    assert _OwnedTaskConfig.task_execution_cache_dir().parent.name == "owning-bench"
    # Derived falls back to top-level package name (here: "tests"), not "owning-bench".
    assert _DerivedTaskConfig.task_execution_cache_dir().parent.name == "tests"


# ── tasks() view (ClassVar filtered by task_ids) ──────────────────────────────


def test_tasks_returns_all_when_task_ids_is_none():
    cfg = MyBenchmarkConfig()
    assert cfg.tasks() == MyBenchmarkConfig.task_metadata
    assert cfg.num_tasks == 4


def test_tasks_filters_by_task_ids():
    cfg = MyBenchmarkConfig(task_ids=["t2", "t4"])
    tasks = cfg.tasks()
    assert list(tasks.keys()) == ["t2", "t4"]
    assert cfg.num_tasks == 2


# ── get_task_configs() ────────────────────────────────────────────────────────


def test_get_task_configs_yields_one_per_task():
    configs = list(MyBenchmarkConfig().get_task_configs())
    assert {c.task_id for c in configs} == {"t1", "t2", "t3", "t4"}


def test_get_task_configs_stamps_metadata_on_each_config():
    """Each emitted TaskConfig carries the full TaskMetadata for its task."""
    configs = {c.task_id: c for c in MyBenchmarkConfig().get_task_configs()}
    assert configs["t1"].metadata == MyBenchmarkConfig.task_metadata["t1"]
    assert configs["t4"].metadata == MyBenchmarkConfig.task_metadata["t4"]


def test_get_task_configs_honours_subset():
    cfg = MyBenchmarkConfig().subset_from_list(["t1", "t3"])
    configs = list(cfg.get_task_configs())
    assert {c.task_id for c in configs} == {"t1", "t3"}


# ── subset_from_list ──────────────────────────────────────────────────────────


def test_subset_from_list_by_ids():
    cfg = MyBenchmarkConfig()
    sub = cfg.subset_from_list(["t2", "t4"])
    assert sub.task_ids == ["t2", "t4"]
    assert list(sub.tasks().keys()) == ["t2", "t4"]
    # original untouched
    assert cfg.task_ids is None


def test_subset_from_list_by_metadata_objects():
    cfg = MyBenchmarkConfig()
    sub = cfg.subset_from_list([cfg.task_metadata["t2"], cfg.task_metadata["t4"]])
    assert sub.task_ids == ["t2", "t4"]


def test_subset_from_list_is_idempotent():
    cfg = MyBenchmarkConfig().subset_from_list(["t1", "t2", "t1"])
    # duplicates removed, order preserved
    assert cfg.task_ids == ["t1", "t2"]


def test_subset_from_list_preserves_subclass_and_instance_fields():
    class _FieldTaskConfig(_TaskConfig):
        pass

    class ConfigWithField(BenchmarkConfig):
        benchmark_metadata = BenchmarkMetadata(
            name="FieldBench",
            version="1.0",
            description="benchmark with extra field",
            num_tasks=3,
            authors=["Alice"],
            tags=["test"],
        )
        task_metadata = {
            "f1": TaskMetadata(id="f1"),
            "f2": TaskMetadata(id="f2"),
            "f3": TaskMetadata(id="f3"),
        }
        task_config_class = _FieldTaskConfig
        benchmark_class = MyBenchmark
        a: str = "hello"

    cfg = ConfigWithField()
    sub = cfg.subset_from_list(["f1", "f2"])

    # Subclass type and instance field survived model_copy
    assert isinstance(sub, ConfigWithField)
    assert sub.a == "hello"

    # task_ids narrowed the view
    assert list(sub.tasks().keys()) == ["f1", "f2"]
    assert sub.num_tasks == 2

    # Class-level metadata is unchanged (ClassVar is authoritative, subsets do NOT mutate it)
    assert sub.benchmark_metadata.name == "FieldBench"
    assert sub.benchmark_metadata.num_tasks == 3
    assert sub.benchmark_metadata.authors == ["Alice"]

    # Original config untouched
    assert cfg.task_ids is None
    assert list(cfg.tasks().keys()) == ["f1", "f2", "f3"]


def test_subset_from_list_invalid_ids_raise():
    with pytest.raises(ValueError, match="do not exist"):
        MyBenchmarkConfig().subset_from_list(["t1", "nonexistent"])


def test_subset_from_list_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        MyBenchmarkConfig().subset_from_list([])


# ── subset_from_glob ──────────────────────────────────────────────────────────


def test_subset_from_glob_by_split():
    sub = MyBenchmarkConfig().subset_from_glob("split", "train")
    assert set(sub.task_ids or ()) == {"t1", "t2"}


def test_subset_from_glob_by_subclass_field():
    """``subset_from_glob`` accepts any top-level field on the (subclassed) TaskMetadata."""
    sub = MyBenchmarkConfig().subset_from_glob("difficulty", "easy")
    assert set(sub.task_ids or ()) == {"t1", "t3"}


def test_subset_from_glob_no_match_raises():
    with pytest.raises(ValueError, match="No tasks found"):
        MyBenchmarkConfig().subset_from_glob("split", "nonexistent_split")


def test_subset_from_glob_composes_with_subset_from_list():
    """A glob applied to an already-subset view only matches within that view."""
    narrowed = MyBenchmarkConfig().subset_from_list(["t1", "t3", "t4"])
    sub = narrowed.subset_from_glob("split", "train")
    # t2 is split=train but was excluded by the first subset
    assert set(sub.task_ids or ()) == {"t1"}


# ── named_subsets ─────────────────────────────────────────────────────────────


def test_named_subsets_and_named_subset():
    class _NamedTaskConfig(_TaskConfig):
        pass

    class ConfigWithNamed(BenchmarkConfig):
        benchmark_metadata = BenchmarkMetadata(
            name="Named",
            version="1",
            description="x",
            num_tasks=4,
            named_subsets={"train": ("split", "train"), "easy": ("difficulty", "easy")},
        )
        task_metadata = MyBenchmarkConfig.task_metadata
        task_config_class = _NamedTaskConfig
        benchmark_class = MyBenchmark

    assert set(ConfigWithNamed.named_subsets()) == {"train", "easy"}

    sub = ConfigWithNamed().named_subset("easy")
    assert set(sub.task_ids or ()) == {"t1", "t3"}


def test_named_subset_unknown_raises():
    with pytest.raises(KeyError, match="Unknown subset"):
        MyBenchmarkConfig().named_subset("nonexistent")


# ── File loading helpers ──────────────────────────────────────────────────────


def test_benchmark_metadata_from_json(tmp_path):
    p = tmp_path / "bm.json"
    p.write_text(json.dumps({"name": "json-bench", "version": "3.0", "description": "from JSON"}))
    bm = BenchmarkConfig.benchmark_metadata_from_json(p)
    assert bm == BenchmarkMetadata(name="json-bench", version="3.0", description="from JSON")


def test_benchmark_metadata_from_csv(tmp_path):
    p = tmp_path / "bm.csv"
    p.write_text(
        'name,version,description,num_tasks,authors,tags\ncsv-bench,4.0,from CSV,5,"[""Alice""]","[""toy""]"\n'
    )
    bm = BenchmarkConfig.benchmark_metadata_from_csv(p)
    assert bm == BenchmarkMetadata(
        name="csv-bench",
        version="4.0",
        description="from CSV",
        num_tasks=5,
        authors=["Alice"],
        tags=["toy"],
    )


def test_task_metadata_from_json(tmp_path):
    p = tmp_path / "tasks.json"
    p.write_text(
        json.dumps(
            [
                {"id": "task-a", "split": "train"},
                {"id": "task-b", "split": "test", "abstract_description": "do other"},
            ]
        )
    )
    tasks = BenchmarkConfig.task_metadata_from_json(p)
    assert tasks == {
        "task-a": TaskMetadata(id="task-a", split="train"),
        "task-b": TaskMetadata(id="task-b", split="test", abstract_description="do other"),
    }


def test_task_metadata_from_csv(tmp_path):
    p = tmp_path / "tasks.csv"
    p.write_text("id,split,abstract_description\ntask-a,train,do something\ntask-b,test,do other\n")
    tasks = BenchmarkConfig.task_metadata_from_csv(p)
    assert tasks == {
        "task-a": TaskMetadata(id="task-a", split="train", abstract_description="do something"),
        "task-b": TaskMetadata(id="task-b", split="test", abstract_description="do other"),
    }


# ── make() + runtime Benchmark ────────────────────────────────────────────────


def test_make_returns_live_benchmark():
    """make() returns a Benchmark whose setup() has already been called."""
    bench = MyBenchmarkConfig().make()
    assert isinstance(bench, MyBenchmark)
    assert bench.config is not None
    # _runtime_context exists (empty for this minimal benchmark)
    assert bench._runtime_context == {}


def test_make_without_infra_skips_provisioning_for_resourceless_benchmark():
    """A benchmark with no resources can be made without an infra."""
    bench = MyBenchmarkConfig().make(infra=None)
    assert isinstance(bench, MyBenchmark)


def test_benchmark_is_context_manager():
    with MyBenchmarkConfig().make() as bench:
        assert isinstance(bench, MyBenchmark)
    # close() was called on exit


def test_make_threads_infra_to_runtime():
    """``make(infra)`` forwards ``infra`` into ``Benchmark.__init__`` so cubes can reach
    it as ``self._infra`` from ``_setup()`` without overriding ``__init__`` or ``make``."""
    from cube.infra_local import LocalInfraConfig

    infra = LocalInfraConfig()
    bench = MyBenchmarkConfig().make(infra=infra)
    assert bench._infra is infra


def test_make_without_infra_leaves_runtime_infra_none():
    """When called without infra, ``self._infra`` stays None — base does not default."""
    bench = MyBenchmarkConfig().make()
    assert bench._infra is None


def test_spawn_returns_ready_task():
    bench = MyBenchmarkConfig().make()
    task = bench.spawn(_TaskConfig(metadata=TaskMetadata(id="t1")))
    assert isinstance(task, Task)
    obs, _ = task.reset()
    assert obs == Observation.from_text("ready")


def test_spawn_unknown_task_raises():
    bench = MyBenchmarkConfig().make()
    with pytest.raises(ValueError, match="not found"):
        bench.spawn(_TaskConfig(metadata=TaskMetadata(id="nonexistent")))


def test_spawn_respects_subset():
    """A benchmark made from a subset config refuses to spawn tasks outside the subset."""
    bench = MyBenchmarkConfig().subset_from_list(["t1", "t2"]).make()
    # t3 is not in the subset — spawn must refuse
    with pytest.raises(ValueError, match="not found"):
        bench.spawn(_TaskConfig(metadata=TaskMetadata(id="t3")))


# ── Round-trip serialization ──────────────────────────────────────────────────


def test_config_round_trip_json():
    """BenchmarkConfig serializes and reloads through JSON preserving state."""
    cfg = MyBenchmarkConfig().subset_from_list(["t2", "t4"])
    payload = cfg.model_dump_json()
    reloaded = MyBenchmarkConfig.model_validate_json(payload)
    assert reloaded == cfg
    # Sanity: reloaded view is still correct
    assert list(reloaded.tasks().keys()) == ["t2", "t4"]


def test_subsetted_config_make_and_spawn():
    """End-to-end: subset → serialize → reload → make() → spawn."""
    cfg = MyBenchmarkConfig().subset_from_list(["t1"])
    reloaded = MyBenchmarkConfig.model_validate_json(cfg.model_dump_json())
    with reloaded.make() as bench:
        # Use a TaskConfig emitted by the reloaded benchmark — metadata is stamped for us.
        tc = next(iter(reloaded.get_task_configs()))
        task = bench.spawn(tc)
        obs, _ = task.reset()
        assert obs == Observation.from_text("ready")


def test_polymorphic_fields_preserve_subclass_state_through_json():
    """Regression: JSON round-trip must preserve subclass-specific fields on every
    polymorphic field of BenchmarkConfig / TaskConfig. Without ``SerializeAsAny``,
    Pydantic silently drops subclass state — the reloaded config looks fine but
    has lost the subclass's extra data.

    Subclasses must be importable by ``TypedBaseModel`` via their ``_type``
    path, so they're defined at module scope just below.
    """
    # BenchmarkConfig polymorphic fields
    original = _BenchWithRichDefaults(
        tool_config=_RichToolConfig(marker="tool-actual"),
        seed_generator=_RichSeedGenerator(marker="seed-actual"),
    )
    reloaded = _BenchWithRichDefaults.model_validate_json(original.model_dump_json())

    assert isinstance(reloaded.tool_config, _RichToolConfig)
    assert reloaded.tool_config.marker == "tool-actual"
    assert isinstance(reloaded.seed_generator, _RichSeedGenerator)
    assert reloaded.seed_generator.marker == "seed-actual"

    # TaskConfig polymorphic fields — every cube subclasses TaskMetadata with
    # per-task data (domain, problem_statement, level, etc.), so this is the
    # most important round-trip path.
    rich_meta = _RichTaskMetadata(id="t1", marker="meta-actual")
    tc = _TaskConfig(metadata=rich_meta, tool_config=_RichToolConfig(marker="tc-tool"))
    reloaded_tc = _TaskConfig.model_validate_json(tc.model_dump_json())
    assert isinstance(reloaded_tc.metadata, _RichTaskMetadata)
    assert reloaded_tc.metadata.marker == "meta-actual"
    assert isinstance(reloaded_tc.tool_config, _RichToolConfig)
    assert reloaded_tc.tool_config.marker == "tc-tool"


# Module-level subclasses for the polymorphic round-trip regression test above.
# Must live at module scope so ``TypedBaseModel._type`` can resolve them.
class _RichToolConfig(_ToolConfig):
    marker: str = "tool-default"

    def make(self, container=None):
        return _Tool()


class _RichSeedGenerator(AbstractSeedGenerator):
    marker: str = "seed-default"

    def __call__(self, task_metadata):
        return []


class _RichTaskMetadata(TaskMetadata):
    marker: str = "meta-default"


class _RichTaskConfig(_TaskConfig):
    pass


class _BenchWithRichDefaults(BenchmarkConfig):
    benchmark_metadata = BenchmarkMetadata(name="rich-defaults-bench", version="1", description="x")
    task_metadata = {"t1": TaskMetadata(id="t1")}
    task_config_class = _RichTaskConfig
    benchmark_class = MyBenchmark
