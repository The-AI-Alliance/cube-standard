# Improve task selection mechanisms in `BenchmarkConfig`

**Author:** Chao Wang

**Scope:** `cube.benchmark`

**Date:** June 2026

**Status:** Draft

---

## Background

In `cube.benchmark.BenchmarkConfig`, two task selection mechanisms are supported:

- Selecting tasks at instantiation time.
  - This is achieved by passing `task_ids` to the constructor of `BenchmarkConfig`.
  - The `BenchmarkConfig` class maintains all tasks by the ClassVar `task_metadata`.
  - Each `BenchmarkConfig` instance maintains the field `task_ids` as its task selector.
  - In a `BenchmarkConfig` instance, only the tasks whose ID is in `task_ids` are available.
  - If `task_ids` is `None` (default), task selection is skipped and all tasks are available.

- Further selecting tasks after instantiation.
  - This is achieved by the functions `subset_from_list()`, `subset_from_glob()`, and `named_subset()`.
  - They further select tasks from the available tasks of a `BenchmarkConfig` instance, and create a pruned copy of the instance, where `task_ids` only covers the further selected tasks.
  - `named_subset()` converts the input subset name to a glob, and pass it to `subset_from_glob()`.
  - `subset_from_glob()` converts the input glob to the entries in the ClassVar `task_metadata`, and pass them to `subset_from_list()`.
  - `subset_from_list()` converts the input entries to a new `task_ids`, makes a copy of the `BenchmarkConfig` instance, and sets its `task_ids` to the new `task_ids`.

## Problem

- When instantiating `BenchmarkConfig`, users may want to select tasks in various ways. For example, by a predefined subset name, a custom glob, or a specific list of task IDs. Currently, only ID-based selection is supported at instantiation time. However, subset-name-based selection and glob-based selection are often more practical, since users rarely keep a list of task IDs in their scripts.

- When not all tasks are available in a `BenchmarkConfig` instance, further selection by `named_subset()` is meaningless, because subset names are defined across all tasks. Similarly, `subset_from_list()` is also meaningless, because the target tasks may not be available. Only `subset_from_glob()` makes sense, because applying a glob does not require all tasks to be available.

- Currently, glob-based selection only supports a single glob. However, users often need to select tasks with a combination of multiple globs. For example, users may want to select tasks whose `split` is `train` and `language` is `en`.

## Solution

1. Define a `AttrGlob` class for matching an object's attribute against a glob pattern. Replace the `(glob_key, glob_pattern)` tuples in the current code with `AttrGlob` instances.

```
    class AttrGlob(TypedBaseModel):
        attr: str
        pattern: str

        def __call__(self, target: Any) -> bool:
            return hasattr(target, self.attr) and fnmatch.fnmatchcase(
                str(getattr(target, self.attr)), self.pattern
            )
```

2. Change the type of `BenchmarkMetadata.named_subsets` from `dict[str, tuple[str, str]]` to `dict[str, list[AttrGlob]]`. In this way, a named subset can be defined by multiple globs.

3. In the `BenchmarkConfig` class, define the following three fields to support task selection at instantiation time:
   - `task_ids: list[str] | None`. The same as in the current code.

   - `attr_globs: list[AttrGlob] | None`. Specifies the custom globs used for task selection. If not `None`, task IDs will be derived and used to cover the value of `task_ids`.

   - `subset_name: str | None`. Specifies the name of the selected subset. If not `None`, globs and task IDs will be derived and used to cover the values of `attr_globs` and `task_ids`.

4. In the `BenchmarkConfig` class, rename the ClassVar `task_metadata` to `_full_task_catalog` to avoid confusion with the `TaskMetadata` class, and define two properties `full_task_catalog` and `task_catalog` for easy access.

```
    _full_task_catalog: ClassVar[dict[str, TaskMetadata]]

    @property
    def full_task_catalog(self) -> dict[str, TaskMetadata]:
        return type(self)._full_task_catalog

    @property
    def task_catalog(self) -> dict[str, TaskMetadata]:
        if self.task_ids is None:
            return self.full_task_catalog
        return {task_id: self.full_task_catalog[task_id] for task_id in self.task_ids}
```

5. In the `BenchmarkConfig` class, define a functiion `_select_tasks_by_glob()` for glob-based task selection.

```
    def _select_tasks_by_glob(
        self, attr_globs: list[AttrGlob], from_all: bool
    ) -> list[str]:
        if not attr_globs:
            raise ValueError("attr_globs is empty.")
        task_catalog = self.full_task_catalog if from_all else self.task_catalog
        return [
            task_id
            for task_id, task_metadata in task_catalog.items()
            if all(glob(task_metadata) for glob in attr_globs)
        ]
```

6. Define a model validator `_resolve_task_selection()` to execute the cascade order for setting the values of `subset_name`, `attr_globs`, and `task_ids`.

```
    @model_validator(mode="after")
    def _resolve_task_selection(self) -> Self:
        if self.subset_name is not None:
            if self.subset_name not in self.metadata.subset_catalog:
                raise ValueError(
                    f"subset_name {self.subset_name!r} is not declared in "
                    "metadata.subset_catalog."
                )
            self.attr_globs = self.metadata.subset_catalog[self.subset_name]
        if self.attr_globs is not None:
            self.task_ids = self._select_tasks_by_glob(
                attr_globs=self.attr_globs, from_all=True
            )
        if self.task_ids is not None:
            unknown_ids = [
                task_id
                for task_id in self.task_ids
                if task_id not in self.full_task_catalog
            ]
            if unknown_ids:
                raise ValueError(
                    "task_ids contains entries not declared in "
                    f"full_task_catalog: {unknown_ids}."
                )
        return self
```

5. Deprecate `subset_from_list()` and `named_subset()`, and change `subset_from_glob()` as below:

```
    def subset_from_glob(self, attr_globs: list[AttrGlob]) -> Self:
        task_ids = self._select_tasks_by_glob(attr_globs=attr_globs, from_all=False)
        return self.model_copy(
            update={"subset_name": None, "attr_globs": None, "task_ids": task_ids}
        )
```
