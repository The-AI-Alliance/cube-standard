---
layout: default
title: Design Philosophy
nav_order: 30
has_children: false
---

# Design Philosophy — read before changing the framework

This page is for contributors who want to **change CUBE itself** — add a field, a
method, an abstract hook, or alter a protocol type. (If you only want to *wrap a
benchmark*, you almost never need to touch the framework — see
[Authoring a CUBE]({{site.baseurl}}/authoring-a-cube) instead.)

CUBE is a small, shared contract that many independent benchmarks and harnesses depend
on. The whole value proposition is that the contract is **stable and uniform**: wrap
once, run everywhere. That creates a tension every framework contributor should feel up
front:

> The change that makes *your* benchmark more convenient is paid for by *every other*
> cube and harness that now has to absorb it.

Most proposals to change the framework come from a real, local need — and most of them
are better served by a smaller change that fits the existing schema. This page explains
how we think about that, so you can find the smaller change yourself before opening a PR.

That said — **CUBE is still forming, and we want to evolve it.** The standard is alpha;
shaping it is a contribution we actively invite, not an imposition we tolerate. The bar
below exists to *focus* the conversation on the real question, not to freeze the design.
If you've internalized the trade-off and still believe a principle here is wrong for
where CUBE is heading, that's a genuinely valuable argument — make it. The principles are
current intent, not scripture.

## The mental model

- **The framework defines _what_, not _how_.** It specifies the contracts — `Tool`,
  `Task`, `Benchmark`, `Observation`, `Action`, and the metadata types. It does *not*
  decide how your benchmark scores rewards, how the harness selects models, or how an
  experiment is orchestrated. When a proposal starts encoding *policy*, it's usually in
  the wrong place.
- **Config is serializable; runtime is not.** `*Config` types are pure Pydantic that
  travel to workers; the live `Task`/`Benchmark` hold OS state and never serialize. New
  behavior that needs OS state lives on the runtime object, not in config.
- **You extend by subclassing, not by growing the core.** `TaskMetadata` and
  `BenchmarkMetadata` are *meant* to be subclassed with your own typed fields. The base
  classes carry only fields that **every** cube needs.

## The principles we defend

1. **Lean is the goal; additive is not free.** Additive beats breaking — a new *optional*
   field or method costs existing cubes nothing, whereas renaming or removing a public
   symbol breaks everyone (and cosmetics like a "cleaner name" never justify a break; a
   doc comment does). But "it's only additive" is not a green light. Every symbol added
   to the core is permanent: more surface to learn, document, test, and keep consistent
   forever. The bar is not "does this break anything" — it's "does this *need to exist in
   the core at all*." When in doubt, leave it out: a smaller framework is easier to
   understand and outlives a feature-rich one.
2. **One source of truth.** State that can be *derived* should not also be *stored* as
   independent authority. (Example: benchmark subsets are represented entirely by
   `task_ids` — not by a parallel set of selector fields that can drift on a round trip.)
3. **Respect the config/runtime split** and the **layer boundaries**
   (`core → tool → task → benchmark → resource/container`). A change should sit in one
   layer and not reach across.
4. **The framework organizes; the harness acts.** The framework loads and exposes data
   (e.g. it *returns* benchmark hints); deciding what to *do* with them is the harness's
   job. "The config should automatically do X" usually belongs harness-side.
5. **Generality earns a place in the core.** A capability belongs in the framework only
   if many cubes benefit. One benchmark's idiom, naming preference, or workflow
   convenience does not — that's what subclasses and your own harness code are for.

## Before you propose an API change

Walk this ladder. The cheapest rung that satisfies your need wins — and most needs are
met before rung 3.

1. **Can a subclass do it?** A field on your `TaskMetadata`/`BenchmarkMetadata` subclass,
   or a method override — no framework change at all.
2. **Can the harness do it?** Selection, policy, and orchestration usually live there.
3. **Does it need to live in the core at all?** Even a purely additive change earns its
   place only if it's *general* (many cubes benefit) and *minimal*. If it serves one
   benchmark, it's a subclass field; if it's policy, it's harness code. Prefer the
   smallest addition that works, or none.
4. **Only if 1–3 genuinely can't serve it** is a breaking change on the table — and then
   it needs a proposal with the blast radius named explicitly.

A quick contrast:

| Over-reach | The smaller change that usually fits |
|---|---|
| "Rename `task_metadata` so it reads nicer" | A one-line doc comment. Renaming breaks every cube. |
| "Add a `difficulty` field to `TaskMetadata`" | Subclass `TaskMetadata` with `difficulty: int`. |
| "Make the config auto-apply my hint to the agent" | Return the hint; let the harness apply it. |
| "Replace the one selector field with three that I find expressive" | Keep the single source of truth; add a thin convenience that resolves into it. |
| "Remove the method my benchmark doesn't use" | Leave it — it's load-bearing for others. Audit call sites first. |

## You have escape hatches — you rarely need to change the core

If the framework pushes back on your change, that's almost never a dead end. In order of
preference:

1. **Your cube repo is yours.** As long as your package satisfies the protocol, its
   internals are unconstrained — extra fields, helpers, subclassed metadata, custom logic.
   Most needs are just *"do it in your own package."*
2. **Your own code can extend CUBE.** Code outside CUBE can wrap, subclass, and orchestrate
   its classes freely (selection, policy, extra behavior live great in your harness/library
   code).
3. **A tiny additive hook + your code.** When 1–2 almost work, the right RFC is often a
   small, general extension point in CUBE that lets you build the rest on your side — not a
   large change to the core.
4. **A core change** — only when the need is general and genuinely can't live in 1–3.
5. **Forking CUBE — please don't.** You *can*, but a fork diverges from the standard and
   loses interop with every harness and cube. We'd much rather help you find 1–3.

## This is a friction step, not a wall

The point of the gate — and of the `/gatekeep-rfc` skill — is to **teach the trade-off and
surface the cheaper path**, not to hard-stop you. You can disagree, push back, and insist;
if after the conversation you still believe a change belongs in the core, that's a
legitimate ask for a **human maintainer**. Open a
[Discussion](https://github.com/The-AI-Alliance/cube-standard/discussions) or say so on
your PR. The automated pass just means the human conversation starts already focused on
the real question.

We can't hand-review a flood of framework proposals, so the first pass is automated and
**you can run it on yourself**. The `/gatekeep-rfc` Claude Code skill triages an RFC
against this philosophy: it separates your real *need* from the *mechanism*, checks
whether the schema already covers it, and hands back a concrete in-schema
counter-proposal. Running it before you submit gets you through human review far faster.

→ The skill and its full charter live at
[`.claude/skills/gatekeep-rfc/`](https://github.com/The-AI-Alliance/cube-standard/tree/main/.claude/skills/gatekeep-rfc).

## Where to go next

- **The end-to-end workflow** (branch → RFC → code → smoke → review):
  [CONTRIBUTING.md](https://github.com/The-AI-Alliance/cube-standard/blob/main/CONTRIBUTING.md)
- **Where the project is heading** (so your change rides the roadmap, not against it):
  [ROADMAP.md](https://github.com/The-AI-Alliance/cube-standard/blob/main/ROADMAP.md)
- **The living contracts** your change must respect:
  [`openspec/specs/`](https://github.com/The-AI-Alliance/cube-standard/tree/main/openspec/specs)
- **Wrapping a benchmark instead?**
  [Authoring a CUBE]({{site.baseurl}}/authoring-a-cube)
