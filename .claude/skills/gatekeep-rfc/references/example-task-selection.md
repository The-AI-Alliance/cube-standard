# Worked example — a task-selection RFC (representative)

A composite, anonymized walkthrough — *not* a transcript of any real submission. It
combines patterns that recur in framework RFCs so the skill has a concrete reference for
how to reason. Treat it as a teaching fixture, not a verdict on anyone's PR.

**The scenario.** An RFC targets `openspec/specs/benchmark/spec.md` and proposes a broad
rework of how `BenchmarkConfig` selects tasks. It is a textbook **RESHAPE**: one genuine
capability gap bundled with a large breaking refactor that fights the existing design.

## Step 1 — ground in the live spec

`origin/dev` `benchmark/spec.md` already provides, today:
- `task_ids: list[str] | None` selection at instantiation.
- `subset_name: str | None` — already exists (from the harness-side official-subset work);
  the RFC re-proposes it.
- `subset_from_list()`, `subset_from_glob(glob_key, pattern)`, `named_subset(name)`,
  `named_subsets()`.
- `named_subsets: dict[str, tuple[str, str]]` — name → single `(glob_key, glob_pattern)`.
- Stated invariant: *"Subsets are represented entirely by `task_ids` — no ClassVar
  shadowing or private-attr hacks. `model_copy(update={"task_ids": [...]})` is all that
  happens."*

## Step 2 — the real need (mechanism stripped)

- **N1.** Select tasks at construction time by a subset name or a glob, not only by an
  explicit id list ("users rarely keep a list of task IDs in their scripts").
- **N2.** Select by a **combination** of attribute globs — e.g. `split == train` AND
  `language == en`. The current `named_subsets` and `subset_from_glob` support only a
  single glob. **This is the one genuine capability gap.**

## Step 3 — the mechanism it proposes (five kernels)

| Kernel | What | Verdict |
|--------|------|---------|
| K1 | `AttrGlob(attr, pattern)` callable type | ACCEPT — additive, clean |
| K2 | `named_subsets` value type → `list[AttrGlob]` (multi-glob subsets) | RESHAPE — keep additive/back-compat; serves N2 |
| K3 | Constructor `attr_globs` + `subset_name` fields + a cascade validator | RESHAPE — serves N1 but multiplies sources of truth |
| K4 | Rename ClassVar `task_metadata` → `_full_task_catalog` (+ `full_task_catalog`/`task_catalog` props) | DECLINE — cosmetic, breaks every cube |
| K5 | Remove `subset_from_list()` and `named_subset()` | DECLINE — load-bearing elsewhere |

## Step 4 — charter assessment

- **K4 is a cosmetic rename of load-bearing public surface.** `task_metadata` is a
  ClassVar **every** cube author declares (`task_metadata: ClassVar[dict[str, TaskMetadata]]`)
  and the framework auto-loads from `task_metadata.{json,csv}`. Renaming it to
  `_full_task_catalog` breaks every existing benchmark and the file-loader convention,
  to avoid "confusion with the `TaskMetadata` class" — a problem a one-line doc comment
  solves. Charter principle 1 (lean / additive over breaking). **Decline.**
- **K5 removes `subset_from_list()`** on the argument that it's "meaningless because the
  target tasks may not be available." Explicit id-list subsetting is exactly how
  harness-side gold/official subsets are built. The "may not be available" case is a
  validation concern, not a reason to delete the method. Charter principle (generality) +
  the "remove the method I don't use" anti-pattern. **Decline** (keep both; if
  `named_subset` is genuinely subsumed by multi-glob, deprecate, don't remove, and audit
  call sites first).
- **K3 stores three overlapping selectors** (`subset_name`, `attr_globs`, `task_ids`) and
  a validator that writes each from the previous. This directly contradicts the spec's
  single-source-of-truth invariant (principle 2): subsets are *represented entirely by
  `task_ids`*. On a serialize→deserialize round trip all three persist and re-derive;
  that's the drift the current design deliberately removed. The *need* (N1, construct
  pre-scoped) is real — but resolve the inputs **into `task_ids` and don't keep the
  intermediate selectors as authority.**
- **Internal consistency (quality note, not a verdict input):** when a proposal names the
  same thing two ways, leaves duplicate step numbering, or references a symbol it never
  defines, that signals it isn't fully baked. Note it as a courtesy fix; never decline
  over packaging.
- **Process:** if there's no Alternatives section, ask for the alternatives the author
  considered (especially the additive multi-glob-only path) — that's where the smaller
  change usually surfaces.

## Step 5 — verdict

**RESHAPE.** Accept K1 + the multi-glob capability (K2). Fold N1 into a thin constructor
path that resolves into `task_ids`. Decline the rename (K4) and the removals (K5).

## Step 6 — the counter-proposal handed to the author

Deliver the whole N2 gap **additively**, no breaking changes:

```python
# Additive: AttrGlob is fine as proposed.
class AttrGlob(TypedBaseModel):
    attr: str
    pattern: str
    def __call__(self, t: Any) -> bool:
        return hasattr(t, self.attr) and fnmatch.fnmatchcase(str(getattr(t, self.attr)), self.pattern)

# Widen named_subsets to accept multiple globs, staying back-compatible:
#   dict[str, tuple[str, str]]  →  dict[str, list[AttrGlob]]
# (one-glob subsets become a single-element list; a validator can up-convert old tuples.)

# subset_from_glob gains an AND-of-globs form WITHOUT losing the single-glob signature:
def subset_from_glob(self, *globs: AttrGlob | tuple[str, str]) -> Self:
    norm = [g if isinstance(g, AttrGlob) else AttrGlob(attr=g[0], pattern=g[1]) for g in globs]
    ids = [tid for tid, md in self.tasks().items() if all(g(md) for g in norm)]
    return self.model_copy(update={"task_ids": ids})   # task_ids stays the only source of truth

# Constructor-time selection (N1): keep it as a classmethod factory that resolves into
# task_ids — no persisted attr_globs/subset_name authority, no cascade validator.
@classmethod
def select(cls, *, subset_name: str | None = None, globs: list[AttrGlob] | None = None,
           task_ids: list[str] | None = None) -> Self:
    cfg = cls()
    if subset_name is not None:
        return cfg.named_subset(subset_name)         # reuses existing path
    if globs is not None:
        return cfg.subset_from_glob(*globs)
    if task_ids is not None:
        return cfg.subset_from_list(task_ids)
    return cfg
```

This gives the author 100% of N1 + N2 with **zero** broken public symbols, one source of
truth preserved, and no validator cascade. `task_metadata` keeps its name;
`subset_from_list` / `named_subset` stay. The author keeps what they actually wanted.

**Apply the leanness bar to your own counter-proposal too.** The genuine new capability
here is multi-glob (`AttrGlob` + the widened `subset_from_glob`) — that earns its place.
The `select()` factory is *convenience* over methods that already exist
(`named_subset` / `subset_from_glob` / `subset_from_list`); under the lean default it's
optional, not core. Offer it as a nice-to-have the author can drop, not as a requirement
— don't trade one over-reach for a smaller one.

**Prior-demand check.** Sweep `openspec/changes/` + archive and `gh issue/pr list
--search` for the same underlying need (multi-glob / attribute-based task selection)
before deciding. Here, suppose it comes back clean — a single instance, no recurring
pattern — so the verdict stays a plain reshape. *If instead* multi-glob selection had been
asked for and declined two or three times before, that recurrence would outweigh the
per-RFC view: escalate the pattern to a human ("this keeps coming back — should the
default selection API just support it?"), rather than reshaping it down a fourth time.

**Escalate to human?** No for the reshape (it's a clean additive change). The *only*
thing worth a maintainer's nod is whether to also deprecate `named_subset` once
multi-glob lands — a small lifecycle call, flag it but don't block on it.
