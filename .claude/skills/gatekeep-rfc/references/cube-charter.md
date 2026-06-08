# cube charter — the broader picture you defend

The lens: a change is good when it strengthens the contract for *all* cubes, suspect when
it trades that away to make *one* benchmark more convenient.

Hold this **humbly**. CUBE is alpha and explicitly invites contributors to *shape* the
standard, not just use it. This charter is current design intent, not scripture — a
proposal that makes a principled, general case against one of these points may be the
evolution the standard needs; escalate it rather than swat it. As new patterns recur,
fold them back into [`docs/design-philosophy.markdown`](../../../../docs/design-philosophy.markdown),
the canonical human version of this document.

And mind the asymmetry: for a young project growing a community, turning away or souring a
good contributor costs more than escalating a marginal case. Bias toward welcome and
toward teaching.

## What cube is

A **thin, stable contract** that lets many independent benchmarks (web, OS, GUI, coding,
science…) plug into one harness-agnostic protocol. Its value is being *small and stable*:
the more benchmarks depend on it, the more a core change costs everyone. The framework
defines the **what** — the serializable contracts — and stays out of the **how**. The
center of gravity is the author-facing surface (`BenchmarkConfig`, `TaskConfig`,
`BenchmarkMetadata`, `TaskMetadata`, `Tool`, the layer specs); it's load-bearing for the
whole ecosystem, so its stability outweighs any single ergonomic win.

## The principles you defend

Ground each in the live spec — quote the actual contract, don't argue from memory.

1. **Lean is the goal; additive is not free.** Additive beats breaking (a new optional
   field/method costs existing cubes nothing; a renamed/removed/re-typed public symbol
   breaks everyone, and cosmetics never justify a break). But "only additive" is not a
   pass — every core symbol is permanent surface to learn, document, test, and keep
   consistent. The bar is "does this *need to exist in the core at all*." When in doubt,
   leave it out.
2. **One source of truth.** State that can be derived must not also be stored as
   independent authority (e.g. subsets are represented entirely by `task_ids`).
   Derive-on-read beats store-and-sync; overlapping fields + a validator that syncs them
   reintroduce drift.
3. **Config vs runtime split.** `*Config` is serializable Pydantic that crosses to
   workers; the runtime object holds OS state and never serializes. A change must respect
   which side it lives on.
4. **Extend by subclass, not by growing the core.** Per-benchmark data goes on a
   `TaskMetadata` / `BenchmarkMetadata` subclass. The base classes carry only what *every*
   cube needs.
5. **Framework organizes; the harness acts.** The framework loads and exposes data;
   deciding what to *do* with it is harness-side. "The config should automatically do X"
   usually belongs in the harness.
6. **One layer per change.** `core → tool → task → benchmark → resource/container`.
   Cross-layer coupling is a smell.
7. **Generality earns core space.** A capability belongs in the framework only if many
   cubes benefit. "Would the other cubes want this, or just yours?"
8. **Process: lean by default.** Small additive changes are a living-spec edit; a formal
   proposal (Problem · Solution · **Alternatives**) is for breaking changes, new required
   fields, or removals. A breaking proposal with no Alternatives hasn't done the work.

## Escape hatches — lead with these

Most "change the framework" needs are served *without* touching core. Cheapest first:

1. **Their cube repo is theirs.** As long as it satisfies the protocol, internals are
   unconstrained — extra fields, helpers, subclassed metadata, custom logic. "Just do it
   in your own package" is the single most common correct answer.
2. **Their own code can extend CUBE.** Code outside CUBE can wrap, subclass, and
   orchestrate its classes freely (selection, policy, behavior). Unlimited, costs the
   ecosystem nothing.
3. **A tiny additive hook + their code.** When 1–2 *almost* work, the minimal general
   extension point in CUBE unblocks the rest on their side. This is most reshapes.
4. **A core change** — only when the need is general and genuinely can't live in 1–3;
   subject to the leanness bar and, if breaking, a human decision.
5. **Forking — discouraged.** They *can*, but it diverges from the standard and loses
   interop. Offer 1–3 instead; forking is the outcome the gate exists to make unnecessary.

## When it's a genuine gap — escalate (don't reshape)

- A general need (many cubes) that truly can't be expressed in the schema or delivered
  additively.
- A real inconsistency or footgun in an existing contract.
- A principled challenge to a current invariant, or a cross-layer design decision a human
  owner should make.
- **Recurring demand.** The same underlying need shows up across multiple prior proposals
  or issues — especially ones repeatedly closed/declined for the same reason — or several
  overlapping requests were never properly resolved. Recurrence is signal: it's aggregate
  demand the per-RFC view misses, and a sign the current answer isn't actually serving
  people. Escalate the *pattern* (with links), not just this instance — and consider that
  a principle may need revisiting rather than the request re-declined.

Attach your best smaller alternative even when you escalate — give the human the gap *and*
a starting point.

## Over-reaches that recur — recognize the pattern, don't pattern-match the list

Illustrative, not a checklist. Each is a surface symptom of a principle above:

- A cosmetic rename of public surface → a doc comment, not a break.
- Replacing one derived field with several overlapping ones + a sync validator → keep one
  source of truth.
- Removing an API because *this* benchmark doesn't use it → it's load-bearing elsewhere;
  audit call sites first.
- A core field for a one-benchmark need → subclass.
- Config doing the harness's job → harness-side.
- A fix for a limitation that doesn't actually exist → verify against the live spec.
