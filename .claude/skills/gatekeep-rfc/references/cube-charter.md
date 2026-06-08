# cube charter — the broader picture you defend

This is the lens. An RFC is good when it strengthens these properties for *all* cubes,
and suspect when it trades them away to make *one* benchmark slightly more convenient.
Every principle below is grounded in `openspec/specs/` — quote the live spec when you
invoke one; don't argue from memory.

This is the **operational** rendering of the human-facing
[`docs/design-philosophy.markdown`](../../../../docs/design-philosophy.markdown); when
they conflict, the philosophy doc is canonical. Point reshaped/declined authors there.

## What cube is

cube is a **thin, stable contract** that lets many independent benchmark authors plug
heterogeneous benchmarks (web, OS, GUI, coding, science…) into one harness-agnostic
protocol. Its value is in being *small and stable*: the more benchmarks depend on it,
the more a change to the core costs. The framework's job is to define the **what** —
the serializable contracts — and to stay out of the **how**.

The center of gravity is the author-facing surface: `BenchmarkConfig`, `TaskConfig`,
`BenchmarkMetadata`, `TaskMetadata`, `Tool`, and the layer specs in `openspec/specs/`.
Every external author subclasses these. That surface is **load-bearing for the whole
ecosystem** — its stability is worth more than any single ergonomic win.

## The principles you defend

1. **Lean is the goal; additive is not free.** Additive beats breaking — a new optional
   field/method costs existing cubes nothing, while a renamed/removed/re-typed public
   symbol breaks every author who touched it (and cosmetics never justify a break). But
   "it's only additive" is **not** a pass. Every core symbol is permanent surface to
   learn, document, test, and keep consistent. The bar is not "does it break anything" —
   it's "does it *need to exist in the core at all*." A purely additive change that
   serves one benchmark or that the harness/a subclass could carry should still be
   declined. When in doubt, leave it out.

2. **Single source of truth.** State that can be derived must not also be stored as
   independent authority. The benchmark spec is explicit: *"Subsets are represented
   entirely by `task_ids` … `model_copy(update={"task_ids": [...]})` is all that
   happens."* Proposals that add a second or third selector field holding overlapping
   truth (and a validator that overwrites one from another) reintroduce exactly the
   drift this design removed. Derive-on-read beats store-and-sync.

3. **Serializable config vs. runtime object split.** `*Config` types are pure,
   serializable Pydantic that cross the wire to workers; the runtime `Benchmark`/`Task`
   hold OS state and never serialize. A change must respect which side it lives on. New
   behavior that needs OS state belongs on the runtime object, not smuggled into config.

4. **Extend via subclass, not via new framework fields.** Authors needing extra
   benchmark- or task-level data **subclass `BenchmarkMetadata` / `TaskMetadata`** with
   their own typed fields. The base classes carry only framework-universal fields. An
   RFC that wants to add a field "because my benchmark needs X" almost always wants a
   subclass field, not a core-schema change. Reserve core additions for things *every*
   cube benefits from.

5. **Framework organizes; the harness acts.** The framework *loads and exposes* data
   (e.g. `load_benchmark_clarifications` returns hints — it never delivers them to an
   agent). Decisions about *use* live in the harness. Proposals that push policy/behavior
   into the framework ("the config should automatically do Y") usually belong harness-side.

6. **Layer boundaries are real.** `core → tool → task → benchmark → resource/container`.
   A change should sit in exactly one layer and not reach across. Cross-layer coupling in
   an RFC is a smell.

7. **Generality earns its place in the core.** A capability belongs in the framework only
   if it serves many benchmarks. One author's selection idiom, naming preference, or
   workflow convenience does not. "Would the other 30 cubes want this, or just yours?"

8. **Process: lean by default, propose before breaking.** Small additive changes take a
   lean living-spec edit. A proposal (`proposal.md` + `deltas.md`) is required only when
   altering a public interface, adding a required field, or removing a method — and it
   must carry **Problem · Proposed solution · Alternatives**. A proposal that breaks
   public surface but offers no Alternatives section hasn't done the work yet.

## "Looks like an improvement but isn't" — common anti-patterns in incoming RFCs

- **The cosmetic rename.** "Rename `X` to `Y` to avoid confusion." Renaming a public,
  author-declared symbol (a ClassVar every cube sets, a documented method) breaks the
  ecosystem for a readability gain the author could get with a one-line doc comment.
  Almost always DECLINE the rename, keep any real feature riding alongside it.

- **The three-field cascade.** Replacing one derived field with several overlapping
  selector fields plus a validator that writes one from another. Feels more expressive;
  actually multiplies sources of truth and serialization drift. Counter with: keep the
  single source of truth, resolve inputs into it, expose convenience as a thin
  constructor/method that does not persist redundant state.

- **"Remove the method I don't use."** An author proposes deleting an API because it's
  meaningless *for their benchmark*. It's load-bearing for others (e.g. explicit
  id-list subsetting is how harness-side gold/official subsets are built). Removal needs
  an ecosystem-wide audit, not one author's local view.

- **Core field for a one-benchmark need.** → subclass `TaskMetadata`/`BenchmarkMetadata`.

- **Config doing the harness's job.** Auto-applying hints, auto-selecting models,
  policy baked into serializable config. → harness-side.

- **Solving a non-problem.** The "Background" misstates the current API and the proposed
  fix addresses a limitation that doesn't exist. Always verify against the live spec.

## The escape hatches (where the need usually goes instead)

Most "change the framework" needs are fully served *without* touching CUBE core. Lead
with these — they're the constructive heart of every reshape/decline, and usually the
fastest route to what the author actually wants. Order from cheapest:

1. **The contributor owns their cube repo.** A cube package is theirs; as long as it
   satisfies the protocol (implements `Tool`/`Task`/`Benchmark`, configs serialize), the
   internals are unconstrained. Extra fields, helpers, custom logic, subclassed metadata
   — all fine, none of it needs an RFC. "You can just do this in your own package" is the
   single most common correct answer.
2. **Their own code can compose/extend CUBE.** Code that lives *outside* CUBE can wrap,
   subclass, or orchestrate CUBE classes freely (selection, policy, extra behavior).
   Harness-side and library-side extension is unlimited and costs the ecosystem nothing.
3. **A tiny additive hook + their code.** The reshape sweet spot: when 1–2 *almost* work
   but a small extension point in CUBE would unblock them, propose the **minimal** general
   hook and let the contributor build the rest on their side. This is most RFCs.
4. **A core change** (additive, then — rarely — breaking). Only when the need is general
   and genuinely cannot live in 1–3. Subject to the leanness bar and, if breaking, a human
   decision.
5. **Forking CUBE — discouraged.** A contributor *can* fork, but steer away from it: a
   fork diverges from the standard, loses interop with every harness and cube, and earns
   no upstream benefit. Always offer 1–3 instead. (This isn't a threat to wield — it's the
   outcome the whole gate exists to make unnecessary.)

The gate is friction, not a wall: a collaborator who has walked 1–4 and still believes the
change belongs in core is entitled to a **human maintainer's** judgment. Help them frame
that ask; don't present your verdict as the end of the road.

## The genuine-gap signals (when to ESCALATE, not reshape)

- The need is general (many cubes want it) **and** genuinely cannot be expressed in the
  current schema or delivered additively.
- The proposal surfaces a real inconsistency or footgun in an existing contract.
- It requires a design *decision* with cross-layer consequences that only a human owner
  should make (e.g. changing what `task_ids = None` means, the config/runtime boundary,
  the serialization format).

When you escalate, still attach your best smaller alternative — give the human the gap
*and* a starting point.
