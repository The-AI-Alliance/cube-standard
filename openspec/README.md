# OpenSpec — cube-standard

This directory contains the living specifications for cube-standard. It follows
the [OpenSpec](https://github.com/Fission-AI/OpenSpec) methodology — a lightweight,
spec-driven approach designed to keep AI coding agents and human contributors aligned
on the same contracts, without heavyweight processes.

---

## Two tracks, one source of truth

| Track | Audience | Location | Contents |
|-------|----------|----------|----------|
| **Specs** | Coding agents + developers | `openspec/specs/` | Contracts, invariants, signatures — terse and machine-readable |
| **Docs** | Human contributors | `docs/` | Tutorials, architecture, getting-started — narrative and explanatory |

The spec is the contract. Docs cross-reference it. When the two disagree, trust the
spec (and update the doc). When the spec disagrees with the code, trust the code
(and update the spec or file a bug).

---

## The three-habit workflow

### 1. Read the spec before touching a layer

Every layer has a spec in `openspec/specs/<layer>/spec.md`. Before modifying code,
read the **Invariants** and **Gotchas** sections — these capture the non-obvious
constraints that will bite you if you miss them.

The layer map is in `CLAUDE.md` at the root of the repo.

### 2. Sync the spec after merging

After any PR that changes a public contract, run `/update-openspec` in Claude Code.
It reads the source files, compares against the specs, and either:

- **Edits the spec directly** — for minor drift (new method, renamed field, corrected invariant)
- **Creates a change proposal** — for substantive changes (see habit 3)

The PR checklist includes a reminder for this step.

### 3. Propose before breaking

Before coding a change that alters a public interface, adds a required field, or
removes a method — create a delta spec in `openspec/changes/<name>/`:

```
openspec/changes/my-change/
├── proposal.md    # one-page: problem, proposed change, alternatives
└── deltas.md      # structured diff against the current spec
```

No formal approval is required — the proposal makes the change visible to the team
before code lands. Post a link in the team channel when you open the PR.

When the change merges, move the folder to `openspec/changes/archive/YYYY-MM-DD-<name>/`
and apply the deltas to the main spec.

---

## Writing a delta spec

Deltas use three section headers — **ADDED**, **MODIFIED**, **REMOVED** — written in
target-state language (present tense describing what the spec will say after the change
merges). Parallel proposals stay independent; reviewers see exactly what changes.

```markdown
## ADDED

### `Task.truncated` field
`Task.step()` sets `truncated=True` when a step limit is exceeded.
Harnesses must treat `truncated=True` the same as `done=True` for episode termination.

## MODIFIED

### `TaskConfig.make()` signature
**Before:** `make(self) -> Task`
**After:** `make(self, runtime_context: RuntimeContext) -> Task`
`runtime_context` carries infra references from `Benchmark._setup()` to the task.

## REMOVED

### `container_backend` parameter
Removed from `Task.__init__`, `TaskConfig.make()`, and `Benchmark.spawn()`.
Use `RuntimeContext` instead.
```

---

## Spec format

Each spec covers:

- **Purpose** — one sentence
- **Public API** — types, methods, signatures (copy-pasteable, not exhaustive prose)
- **Invariants** — what must always hold; numbered so they can be referenced
- **Contracts** — what implementers must guarantee
- **Gotchas** — non-obvious constraints and known footguns

Specs are terse. They define **what** code must do. They do not explain **why**
(that belongs in a change proposal) or **how to use the library** (that belongs in docs).

---

## Directory layout

```
openspec/
├── specs/                  Living contracts, one dir per layer
│   ├── core/spec.md
│   ├── tool/spec.md
│   ├── task/spec.md
│   ├── benchmark/spec.md
│   ├── resource/spec.md
│   ├── container/spec.md
│   ├── server/spec.md
│   ├── cli/spec.md
│   └── testing/spec.md
└── changes/                Active proposals and archived completed changes
    ├── core-extensions/
    │   ├── proposal.md
    │   └── deltas.md
    └── archive/            # YYYY-MM-DD-<name>/ after merging
```
