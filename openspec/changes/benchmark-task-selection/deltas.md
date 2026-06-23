# Improve task selection mechanisms in `BenchmarkConfig`

**Targets:** `openspec/specs/benchmark/spec.md`

---

## ADDED

### `AttrGlob`

```
    class AttrGlob(TypedBaseModel):
        attr: str
        pattern: str

        def __call__(self, target: Any) -> bool:
            return hasattr(target, self.attr) and fnmatch.fnmatchcase(
                str(getattr(target, self.attr)), self.pattern
            )
```

### `BenchmarkConfig`: `attr_globs` and `subset_name`

```
    subset_name: str | None = Field(
        default=None,
        description="...",
    )
    attr_globs: list[AttrGlob] | None = Field(
        default=None,
        description="...",
    )
```

### `BenchmarkConfig`: `_select_tasks_by_glob()`

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

### `BenchmarkConfig`: model validator `_resolve_task_selection()`

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

## MODIFIED

### `BenchmarkMetadata.named_subsets`

Change the type from `dict[str, tuple[str, str]]` to `dict[str, list[AttrGlob]]`.

### `BenchmarkConfig`: ClassVar `task_metadata`

Rename `task_metadata` to `_full_task_catalog` to avoid confusion with the `TaskMetadata` class, and define two properties `full_task_catalog` and `task_catalog` for easy access.

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

### `BenchmarkConfig`: `subset_from_glob()`

```
    def subset_from_glob(self, attr_globs: list[AttrGlob]) -> Self:
        task_ids = self._select_tasks_by_glob(attr_globs=attr_globs, from_all=False)
        return self.model_copy(
            update={"subset_name": None, "attr_globs": None, "task_ids": task_ids}
        )
```

## REMOVED

### BenchmarkConfig: `subset_from_list()` and `named_subset()`
