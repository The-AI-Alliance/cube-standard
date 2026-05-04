# Deltas — task cache dir alignment

**Targets:** `openspec/specs/task/spec.md`, `openspec/specs/benchmark/spec.md`

## ADDED — `TaskConfig._benchmark_cache_dir`

**Spec:** task

`TaskConfig` carries a private `ClassVar[Path | None]` defaulting to `None`:

```python
class TaskConfig[TTMetadata: TaskMetadata](TypedBaseModel, ABC):
    _benchmark_cache_dir: ClassVar[Path | None] = None  # set by BenchmarkConfig.__init_subclass__
```

The owning `BenchmarkConfig.__init_subclass__` stamps `cls.cache_dir()`
onto its `task_config_class._benchmark_cache_dir` at class-definition time.
Storing the resolved `Path` (not just the name) means cube-side overrides
of `cache_dir()` (e.g. cubes that co-locate the cache with other on-disk
state) flow through to the per-task cache automatically. ClassVar is not a
Pydantic field, so the wire payload sent to workers is unchanged. Workers
re-populate it on import of the cube package.

## MODIFIED — `TaskConfig.task_execution_cache_dir()`

**Spec:** task

Default lives directly under `_benchmark_cache_dir`, falling back to the
top-level Python package name when `cls.__dict__.get("_benchmark_cache_dir")`
is unset:

```python
@classmethod
def task_execution_cache_dir(cls) -> Path:
    cache_dir = cls.__dict__.get("_benchmark_cache_dir") or get_cache_dir(cls.__module__.split(".")[0])
    return cache_dir / "tasks_execution_info"
```

`__dict__.get` (not attribute lookup) is deliberate: a `TaskConfig` subclass
without its own owning `BenchmarkConfig` must not silently inherit the
parent's stamp through the MRO. Such subclasses fall back to the package
name as before.

## MODIFIED — `load_task_execution_info` and `verify_installed` are instance methods

**Spec:** task

Signatures change from:

```python
@classmethod
def load_task_execution_info(cls, task_id: str) -> dict[str, Any]: ...

@classmethod
def verify_installed(cls) -> None: ...
```

to:

```python
def load_task_execution_info(self) -> dict[str, Any]: ...

def verify_installed(self) -> None: ...
```

`load_task_execution_info` uses `self.task_id` to locate the file —
no extra parameter needed. Call sites in `TaskConfig.make()` become
`self.load_task_execution_info()` and `self.verify_installed()`.

`task_execution_cache_dir` remains a classmethod because
`BenchmarkConfig.install()` calls it without a task instance.

## ADDED — `BenchmarkConfig.__init_subclass__` back-stamp + shared-class guard

**Spec:** benchmark

After class-level validations pass, `__init_subclass__` stamps the cache
name onto `task_config_class`:

```python
is_abstract = bool(getattr(task_cfg_cls, "__abstractmethods__", None))
if not is_abstract and not _is_dynamic("benchmark_metadata"):
    new_dir = cls.cache_dir()
    existing = task_cfg_cls.__dict__.get("_benchmark_cache_dir")
    if existing is not None and existing != new_dir:
        raise TypeError(
            f"{task_cfg_cls.__qualname__} is already owned by benchmark "
            f"at {existing!s}; cannot reassign to {new_dir!s}. Each "
            f"BenchmarkConfig must declare its own TaskConfig subclass."
        )
    task_cfg_cls._benchmark_cache_dir = new_dir
```

Skipped when:

- `task_config_class` is abstract (e.g. the bare `TaskConfig` placeholder
  used by `CompositeBenchmarkConfig`) — stamping on an abstract class
  would leak the name to every concrete subclass via the MRO.
- `benchmark_metadata` is dynamic (composite `@property` — no fixed name
  at class-creation time).

A `TypeError` raised on conflicting stamps enforces "one `TaskConfig`
subclass per `BenchmarkConfig`" as a hard invariant.
