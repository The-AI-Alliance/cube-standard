# Worked example — a task-selection RFC (representative)

A composite, anonymized walkthrough — *not* a transcript of any real submission. It
combines patterns that recur in framework RFCs so the skill has a concrete reference for
how to reason. Treat it as a teaching fixture, not a verdict on anyone's PR.

**The scenario.** An RFC targets `openspec/specs/benchmark/spec.md` and proposes a broad
rework of how `BenchmarkConfig` selects tasks. It looks like a textbook **RESHAPE** —
a capability gap bundled with a large breaking refactor — but the lesson here is that a
disciplined *workaround hunt* (Step 2.5) collapses the "gap" almost to nothing: most of it
already works today, and what's left is small and demand-gated. The hunt is the move that
separates a real gap from a cosmetic one.

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
  `language == en`. The current `named_subsets` and `subset_from_glob` take a single glob.
  The proposal calls this its central gap — *candidate* gap until Step 2.5 tests it.

## Step 2.5 — hunt for workarounds in today's API (before granting any new symbol)

Don't take the proposal's "there's no workaround" at face value — try it against the live
code.

- **N2 at the call site is already expressible.** `subset_from_glob` filters from
  `self.tasks()` — the *already-narrowed* view, not the full catalog (`benchmark.py`,
  `current = self.tasks()`). So chaining ANDs:
  ```python
  cfg.subset_from_glob("split", "train").subset_from_glob("language", "en")
  #   → split == train  AND  language == en, today, no change
  ```
  That demotes N2 from "hard-block / genuine gap" to **cosmetic for the call-site case** —
  a clean path exists. The proposal (and a first-pass reshape) missed this; the hunt is
  what catches it.
- **The only residue is the *declarative* form.** `named_subsets: dict[str, tuple]` holds
  one glob per name, and a data structure can't "chain" — so a *named subset whose
  definition is an AND* genuinely can't be declared today. Even that is reachable without
  core: a cube can override `named_subset()` in its own subclass to map a name onto chained
  globs (escape hatch — their own code). So the core-only residue is narrow and
  subclass-able, and it only earns core space if **more than this one cube** wants it
  (Step 4, generality).
- **N1** ("construct pre-scoped") is served by one chained call —
  `MyConfig().named_subset("l1")` — so it's ergonomic, not blocking.

Net: the workaround hunt collapses a "broad rework for a central gap" down to *at most* a
small, demand-gated widening of declarative `named_subsets`.

## Step 3 — the mechanism it proposes (five kernels)

| Kernel | What | Verdict |
|--------|------|---------|
| K1 | `AttrGlob(attr, pattern)` callable type | ACCEPT *only if* K2 lands — additive, but pointless on its own |
| K2 | `named_subsets` value type → `list[AttrGlob]` (multi-glob subsets) | RESHAPE, demand-gated — the only residue after Step 2.5; additive/back-compat; keep on the table only if a second cube wants declarative multi-glob |
| K3 | Constructor `attr_globs` + `subset_name` fields + a cascade validator | DECLINE — N1 is already a one-method chain; the fields multiply sources of truth |
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
  that's the drift the current design deliberately removed. And the *need* it serves (N1,
  construct pre-scoped) is already a one-method chain — `MyConfig().named_subset("l1")`
  (Step 2.5) — so this is cost with no capability gain. **Decline;** if a maintainer later
  wants constructor sugar, it must resolve inputs *into* `task_ids` with no persisted
  intermediate selectors.
- **Internal consistency (quality note, not a verdict input):** when a proposal names the
  same thing two ways, leaves duplicate step numbering, or references a symbol it never
  defines, that signals it isn't fully baked. Note it as a courtesy fix; never decline
  over packaging.
- **Process:** if there's no Alternatives section, ask for the alternatives the author
  considered (especially the additive multi-glob-only path) — that's where the smaller
  change usually surfaces.

## Step 5 — verdict

**RESHAPE, mostly down to nothing.** The workaround hunt (Step 2.5) already serves N2 at
the call site (chaining) and N1 (one method) — so most of the proposal is cosmetic. The
*only* core candidate left is declarative multi-glob `named_subsets` (K2 + its K1 type),
and that's **demand-gated**: keep it on the table only if a second cube wants it; otherwise
the subclass override is enough. Decline the rename (K4), the removals (K5), and the
constructor fields/validator (K3).

## Step 6 — the counter-proposal handed to the author

**Lead with the zero-code answer.** Most of what the proposal wants already works:

```python
# N2 — AND of globs at the call site: chain. subset_from_glob narrows the current view.
cfg.subset_from_glob("split", "train").subset_from_glob("language", "en")

# N1 — construct pre-scoped: one method, today.
MyConfig().named_subset("l1")           # or .subset_from_glob(...) / .subset_from_list(...)
```

No new symbols, no break, `task_ids` stays the only source of truth.

**The one thing chaining can't do** is declare an AND *inside* `named_subsets` (data can't
chain). Two paths, smallest first:

```python
# Path A (no core change): the cube overrides named_subset() to chain its own globs.
class MyConfig(BenchmarkConfig):
    def named_subset(self, name: str) -> Self:
        if name == "train_en":
            return self.subset_from_glob("split", "train").subset_from_glob("language", "en")
        return super().named_subset(name)

# Path B (small, additive, ONLY if a second cube wants declarative multi-glob):
#   AttrGlob(attr, pattern) callable type  +  named_subsets: dict[str, tuple] -> dict[str, list[AttrGlob]]
#   with a validator up-converting legacy single-tuple values. named_subset() unpacks the
#   list and chains. No rename, no removals, no constructor fields.
```

**Apply the leanness bar to your own counter-proposal too.** Path A costs core nothing and
should be the default offer. Path B is the *only* part that touches the shared contract,
and even it is gated on generality (Step 4) — if only this cube needs it, Path A is the
answer and B stays out. Don't trade the proposal's over-reach for a smaller core change
that still isn't pulling its weight.

**Prior-demand check.** Sweep `openspec/changes/` + archive and `gh issue/pr list
--search` for the same underlying need (multi-glob / attribute-based task selection)
before deciding. Here, suppose it comes back clean — a single instance, no recurring
pattern — so the verdict stays a plain reshape. *If instead* multi-glob selection had been
asked for and declined two or three times before, that recurrence would outweigh the
per-RFC view: escalate the pattern to a human ("this keeps coming back — should the
default selection API just support it?"), rather than reshaping it down a fourth time.

**Escalate to human?** No. There's no breaking change to weigh and no principled
challenge to an invariant. The single judgment call — whether declarative multi-glob
(Path B) clears the generality bar or stays a subclass override (Path A) — is a small,
demand-gated lean decision; flag it, don't block on it.
