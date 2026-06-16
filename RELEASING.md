# Releasing

How a release flows from `dev` to PyPI, across **both** `cube-standard` and
`cube-harness`. This is the maintainer runbook; for day-to-day contribution
rules see [`CONTRIBUTING.md`](CONTRIBUTING.md). Versioning specifics
(`dev` carries the next `rc`, the `>=<rcN>` pin) live in CONTRIBUTING's
[*Releases & dev versioning*](CONTRIBUTING.md#releases--dev-versioning) — read
that first; this doc covers the *mechanics*.

## Branch model

| Branch | Role | Rule |
|--------|------|------|
| `dev`  | Integration. Always carries the **next unreleased `rc`**. | All work branches off `dev` and merges back to `dev`. |
| `main` | **Release ref.** Tags are cut from here; the published commit == `origin/main`. | **Never commit or hotfix directly to `main`.** It only ever moves by promoting `dev`. |

> **Why this matters.** When `main` is treated as "the released branch" people
> are tempted to hotfix straight into it. That orphans the fix from `dev`, and
> the branches drift until they can only be reconciled by hand. A fix belongs in
> `dev`; it reaches `main` at the next promote. Protect `main` (require PRs, no
> direct pushes) so this can't happen by accident.

## The two phases of a release

A release is **(1) promote `dev`→`main`**, then **(2) tag from `main`**. The
tag — not the merge — is what publishes; pushing `<prefix>/v<version>` triggers
each repo's [`release.yml`](.github/workflows/release.yml) to build that one
package and publish it to PyPI via trusted publishing.

### Phase 1 — promote `dev` → `main`

Do this in **both** repos that have changes.

```bash
git checkout main && git pull
git merge --no-ff origin/dev          # NOT a fast-forward — keep the promote as a merge commit
# Conflicts resolve in dev's favour: dev is the source of truth, main is downstream.
git push origin main
```

If `main` carries an orphan commit (someone hotfixed it directly — see above),
the merge will conflict on those files. Resolve to **dev's state** and record
what `main`-only change was intentionally dropped in the merge-commit message.

### Phase 2 — tag from `main`, in dependency order

The packages form a dependency graph spanning both repos, so they must publish
**bottom-up** — a downstream package must not land on PyPI pinning an upstream
version that isn't up yet. The driver enforces this ordering:

| Tier | Packages | Repo | Tag prefix |
|------|----------|------|------------|
| 1 | `cube-standard` | cube-standard | `cube-standard/v*` |
| 2 | `cube-resources/*` | cube-standard | `cube-resources/<name>/v*` |
| 3 | `cube-tools/*` | cube-standard | `cube-tools/<name>/v*` |
| 4 | `cube-harness` | cube-harness | `cube-harness/v*` |
| 5 | `cubes/*` | cube-harness | `cubes/<name>/v*` |

Use the cross-repo driver, [`scripts/release.py`](scripts/release.py) — it reads
every package's version, compares it to the latest published tag, computes the
tier order, pushes tags tier by tier, and **waits for each tier to appear on
PyPI before starting the next**.

```bash
# 1. Dry-run by default — prints the ordered plan, no side effects. Eyeball it.
python scripts/release.py

# 2. First real publish: one package, supervised.
python scripts/release.py --execute --only cube-standard

# 3. Then the rest (or another --only subset). Idempotent: re-running skips
#    already-published packages and resumes at the next tier.
python scripts/release.py --execute
```

The driver is **dry-run by default and refuses to act on ambiguity** — dirty
tree, version not bumped past the last tag, `HEAD != origin/main`, or a PyPI
timeout all stop the run with an actionable message rather than guessing. It
expects to run from `main` (override with `--ref`, but don't unless you know
why).

### Phase 3 — reopen `dev` for the next cycle

After a package publishes `rc<N>`, bump its version on `dev` to `rc<N+1>` so
`dev` once again carries an *unreleased* version. Skipping this lets `uv`
confuse a dev build with the published wheel during cross-repo CI (see
CONTRIBUTING for the full rationale).

## Releasing a subset

You don't have to publish the whole graph. `--only <dist> [--only <dist> ...]`
restricts the run, and the driver still honours tier order among the selected
packages. Common cases:

- **Core only:** `--only cube-standard --only cube-harness` (plus any tools
  cube-harness needs).
- **One new cube:** `--only <cube-dist>` — cubes are tier 5 and independent of
  each other, so a single cube can ship without touching anything else, as long
  as the `cube-standard` version it pins is already on PyPI.

## Pre-flight checklist

- [ ] `dev` is green on both repos at the commit you intend to promote.
- [ ] Each package to publish has its version bumped past its last published tag.
- [ ] `dev` has been promoted to `main` (Phase 1); `main` is clean and pushed.
- [ ] `scripts/release.py` dry-run plan looks right.
- [ ] Publish tier 1 first, confirm on PyPI, then continue.
- [ ] After publishing, bump `dev` to the next `rc` (Phase 3).
