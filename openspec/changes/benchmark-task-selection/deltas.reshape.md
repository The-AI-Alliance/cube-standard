# Multi-glob task selection in `BenchmarkConfig`

**Targets:** `openspec/specs/benchmark/spec.md`

---

## ADDED

### `AttrGlob`

```python
class AttrGlob(TypedBaseModel):
    attr: str
    pattern: str

    def __call__(self, target: Any) -> bool:
        return hasattr(target, self.attr) and fnmatch.fnmatchcase(
            str(getattr(target, self.attr)), self.pattern
        )
```

A named, serializable callable predicate over an object's attribute. Replaces
the in-flight role of `(glob_key, glob_pattern)` tuples in the multi-glob
form of `subset_from_glob` and in `named_subsets` values. Tuples remain accepted
at every public entry point for back-compat.

### `BenchmarkConfig`: model validator `_resolve_subset_name()`

```python
@model_validator(mode="after")
def _resolve_subset_name(self) -> Self:
    """Honor `subset_name` as a construction-time input.

    If `subset_name` is set and `task_ids` is not, resolve via the existing
    `named_subset()` path so behavior is identical. Has no effect when
    `task_ids` is already set (caller has been explicit) or when
    `subset_name is None`.
    """
    if self.subset_name is not None and self.task_ids is None:
        named = type(self).benchmark_metadata.named_subsets
        if self.subset_name not in named:
            raise ValueError(
                f"subset_name {self.subset_name!r} is not declared in "
                f"named_subsets. Available: {list(named.keys())}"
            )
        resolved = self.named_subset(self.subset_name)
        self.task_ids = resolved.task_ids
    return self
```

## MODIFIED

### `BenchmarkMetadata.named_subsets` — multi-glob, back-compat

Declared type changes from `dict[str, tuple[str, str]]` to
`dict[str, list[AttrGlob]]`. A Pydantic field validator coerces legacy
`(glob_key, glob_pattern)` tuples (from source-declared values and from JSON-
loaded data) to `[AttrGlob(attr=..., pattern=...)]` on load. Spec text on the
existing `# Gotchas` note about JSON-from-file loading tuples as lists is
updated to mention the up-conversion.

### `BenchmarkConfig.subset_from_glob` — multi-glob form, signature back-compat

```python
def subset_from_glob(
    self,
    *globs: AttrGlob | tuple[str, str],
    glob_key: str | None = None,
    glob_pattern: str | None = None,
    subset_name: str | None = None,
) -> Self:
    if glob_key is not None:
        if glob_pattern is None:
            raise ValueError("glob_pattern is required when glob_key is given.")
        globs = (*globs, AttrGlob(attr=glob_key, pattern=glob_pattern))
    norm = [
        g if isinstance(g, AttrGlob) else AttrGlob(attr=g[0], pattern=g[1])
        for g in globs
    ]
    if not norm:
        raise ValueError("subset_from_glob requires at least one glob.")
    matches = [
        tm for tm in self.tasks().values() if all(g(tm) for g in norm)
    ]
    if not matches:
        joined = " AND ".join(f"{g.attr}={g.pattern}" for g in norm)
        raise ValueError(f"No tasks matched globs: {joined}")
    return self.subset_from_list(matches, subset_name=subset_name)
```

Existing two-positional call (`subset_from_glob("split", "train")`) binds to
`glob_key` / `glob_pattern` and routes through the same multi-glob path. Add
`@overload` pairs for clean type-checker UX (single-glob legacy form,
multi-glob form).

### `BenchmarkConfig.named_subset` — multi-glob entry point

```python
def named_subset(self, name: str) -> Self:
    named = type(self).benchmark_metadata.named_subsets
    if name not in named:
        raise KeyError(
            f"Unknown subset {name!r}. Available: {list(named.keys())}"
        )
    return self.subset_from_glob(*named[name], subset_name=name)
```

Same public signature. Internally unpacks the stored `list[AttrGlob]` (or
legacy-coerced single-element list) into `subset_from_glob`'s multi-glob form.

### Spec text — invariant clarification

The existing invariant (`benchmark/spec.md` L127–129) — *"Subsets are
represented entirely by `task_ids` — no ClassVar shadowing or private-attr
hacks. `model_copy(update={"task_ids": [...]})` is all that happens."* — is
**preserved verbatim**. Add a one-line note that `subset_name` is provenance
(records *which* named subset produced the membership) and `task_ids` is
authority (defines the membership); the model validator on construction
resolves `subset_name` *into* `task_ids` without persisting parallel selector
state.

## REMOVED

None.

## DEPRECATED

None in this change. After this lands and multi-glob usage stabilizes, a
follow-up may evaluate whether to deprecate the tuple form of
`named_subsets` values; deferred.
