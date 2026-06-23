# Multi-glob task selection in `BenchmarkConfig`

**Author:** Chao Wang
**Scope:** `cube.benchmark`
**Status:** Draft (reshape of `proposal.md`)
**Date:** June 2026

---

## Problem

A `BenchmarkConfig` can today narrow its task set in three ways:

- `task_ids: list[str] | None` at construction time.
- `subset_from_list(ids_or_metas)` / `subset_from_glob(glob_key, pattern)` /
  `named_subset(name)` as post-instantiation methods that return a
  `model_copy(update={"task_ids": [...]})`.
- `BenchmarkMetadata.named_subsets: dict[str, tuple[str, str]]` maps a name to a
  **single** `(glob_key, glob_pattern)` consumed by `named_subset()`.

Two real limitations fall out of that:

1. **Single-attribute selection only.** `subset_from_glob` and `named_subsets`
   each take exactly one `(glob_key, glob_pattern)`. Authors who need a
   conjunction (e.g. `split == "train"` AND `language == "en"`) cannot express it
   without subclass-side code.
2. **Construction-time selection is id-list-only.** A caller who wants to instantiate
   already scoped to a named subset must construct, then `.named_subset(...)`.
   `subset_name` is already an instance field on `BenchmarkConfig` today, but it
   is set *by* `named_subset()` for provenance round-tripping — it is not honored
   as an *input* at construction.

Both are real; (1) is the central gap (no in-schema workaround), (2) is
ergonomic.

## Solution

**Strictly additive.** No public symbol renamed or removed. The single
source-of-truth invariant (`benchmark/spec.md` L127–129 — *"Subsets are
represented entirely by `task_ids`"*) is preserved: every code path lands in
`task_ids` and clears the selector inputs.

### 1. `AttrGlob` — a named, callable predicate

```python
class AttrGlob(TypedBaseModel):
    attr: str
    pattern: str

    def __call__(self, target: Any) -> bool:
        return hasattr(target, self.attr) and fnmatch.fnmatchcase(
            str(getattr(target, self.attr)), self.pattern
        )
```

Additive. Replaces the in-flight role of `(glob_key, pattern)` tuples without
removing tuple acceptance anywhere.

### 2. Widen `BenchmarkMetadata.named_subsets` to multi-glob, back-compat

Change the declared type:

```python
named_subsets: dict[str, list[AttrGlob]] = {}
```

Existing single-tuple data (in JSON files, in source-declared
`BenchmarkMetadata`s) is **coerced on load** by a Pydantic validator: a bare
`(glob_key, glob_pattern)` tuple becomes `[AttrGlob(attr=..., pattern=...)]`.
No data migration required for downstream cubes.

A named subset with multiple globs is now expressible:

```python
named_subsets = {
    "train_en": [AttrGlob(attr="split", pattern="train"),
                 AttrGlob(attr="language", pattern="en")],
}
```

### 3. Extend `subset_from_glob` to multi-glob without breaking its signature

```python
def subset_from_glob(
    self,
    *globs: AttrGlob | tuple[str, str],
    glob_key: str | None = None,
    glob_pattern: str | None = None,
    subset_name: str | None = None,
) -> Self:
    """Narrow by ANDing one or more attribute globs.

    Back-compat: the existing two-positional form
    ``subset_from_glob("split", "train")`` is still accepted via the
    ``glob_key`` / ``glob_pattern`` keywords being filled positionally — see
    overloads.
    """
    if glob_key is not None:
        if glob_pattern is None:
            raise ValueError("glob_pattern is required when glob_key is given.")
        globs = (*globs, AttrGlob(attr=glob_key, pattern=glob_pattern))
    norm = [g if isinstance(g, AttrGlob) else AttrGlob(attr=g[0], pattern=g[1])
            for g in globs]
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

For the existing single-glob call (`subset_from_glob("split", "train")`), the
two positional args bind to `glob_key` / `glob_pattern` and the multi-glob path
takes a single-element list. A `@overload` pair keeps type-checker UX clean.

### 4. `named_subset` becomes the multi-glob entry point

```python
def named_subset(self, name: str) -> Self:
    named = type(self).benchmark_metadata.named_subsets
    if name not in named:
        raise KeyError(f"Unknown subset {name!r}. Available: {list(named.keys())}")
    return self.subset_from_glob(*named[name], subset_name=name)
```

Same public signature. The stored value is now a `list[AttrGlob]`, unpacked
into the multi-glob form. Single-glob legacy data still works because of the
load-time coercion in (2).

### 5. Construction-time selection via existing fields

`BenchmarkConfig` already accepts `task_ids` and `subset_name` at construction.
Honor `subset_name` as an **input** with a thin model validator that resolves
it into `task_ids` and clears nothing (since `subset_name` is provenance and
`task_ids` is authority — they are not redundant):

```python
@model_validator(mode="after")
def _resolve_subset_name(self) -> Self:
    if self.subset_name is not None and self.task_ids is None:
        named = type(self).benchmark_metadata.named_subsets
        if self.subset_name not in named:
            raise ValueError(
                f"subset_name {self.subset_name!r} is not declared in "
                f"named_subsets. Available: {list(named.keys())}"
            )
        # Resolve via the existing path so behavior is identical to .named_subset().
        resolved = self.named_subset(self.subset_name)
        self.task_ids = resolved.task_ids
    return self
```

This delivers N1 for the `subset_name` case without adding new fields. Callers
who want construction-time multi-glob selection use the existing post-construct
form (`MyConfig().subset_from_glob(AttrGlob(...), AttrGlob(...))`); a one-line
chain is not the problem this change exists to solve.

### What is intentionally **not** in this proposal

- **No `attr_globs` instance field.** Storing it alongside `task_ids` introduces
  parallel authority that the validator must keep in sync; the live spec
  (L127–129) and the recently-archived `2026-04-24-benchmark-config` change
  deliberately moved away from that.
- **No rename of `task_metadata` → `_full_task_catalog`.** That ClassVar is
  declared literally in every cube, hard-coded in the `cube init` template, and
  referenced by `cube.testing`, `cube.server`, examples, tests, and every
  downstream cube package. A rename is cosmetic with ecosystem-wide blast; a
  one-line doc comment on the ClassVar resolves the "confusion with the
  `TaskMetadata` class" concern at zero cost.
- **No removal of `subset_from_list` / `named_subset`.** Both are load-bearing:
  `subset_from_list` is used by harness-side gold-subset construction and the
  `CompositeBenchmarkConfig` usage example in the live spec (`benchmark/spec.md`
  L392); `named_subset` is the public ergonomic API. If after this lands the
  team wants to consolidate, the right move is *deprecate* first and audit call
  sites, not remove.

## Alternatives considered

- **The bundled proposal (`proposal.md` in this directory).** Adds `attr_globs`
  as a persisted field, renames `task_metadata`, and removes
  `subset_from_list` / `named_subset`. Rejected because (a) parallel selectors
  reintroduce drift the framework just removed, (b) the rename costs every
  downstream cube and the scaffold template for a cosmetic gain, (c) the
  removals break load-bearing call sites and the stated motivation ("meaningless
  after narrowing") is a validation question, not a removal one.
- **`AttrGlob` only, no `subset_name` honoring at construction.** Smaller still,
  but leaves N1 unaddressed even though `subset_name` already exists as a
  field. The validator is ~10 lines and reuses `named_subset()`; the asymmetry
  ("settable on the live config but ignored at construction") is more confusing
  than the addition.
- **Replace `tuple[str, str]` outright with `list[AttrGlob]`, no coercion.**
  Cleaner type, but a hard break of every downstream `named_subsets` literal.
  Coercion costs ~5 lines and is invisible to authors.

## Migration notes

- Source-declared `named_subsets = {"x": ("split", "train")}` keeps working
  unchanged (coerced on load).
- JSON-loaded `named_subsets` keeps working unchanged.
- Authors who want multi-glob write the new form directly:
  `"x": [AttrGlob(attr="split", pattern="train"), AttrGlob(attr="language", pattern="en")]`.
- No changes required to `cube init` template, `cube.testing`, `cube.server`,
  examples, or any existing cube package.

## Open questions

1. After this lands, is `named_subset(name)` worth keeping long-term, or
   subsumed by `subset_from_glob(*named[name])` once authors are used to
   `AttrGlob`? Suggest: keep, no deprecation in this change; revisit once
   multi-glob usage data exists.
2. Should `subset_from_glob` accept `dict[str, str]` shorthand (`{"split":
   "train", "language": "en"}`) for ergonomic calls? Not in this change; can be
   added additively later if desired.
