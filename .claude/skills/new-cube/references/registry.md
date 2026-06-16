# Registry submission

Phase 7 — always runs. This doc covers the YAML schema and the `cube registry add` flow.

## Command

From the cube package directory:

```bash
cube registry add              # creates cube-registry-entry.yaml with TODO placeholders
cube registry add --submit     # forks The-AI-Alliance/cube-registry + creates PR via `gh`
```

`cube registry add` reads `pyproject.toml` and pre-fills what it can:
- `id` ← project name with `-cube` suffix stripped (`swebench-verified-cube` → `swebench-verified`). Registry convention is that the id refers to the benchmark, not the wrapper package.
- `version` ← project version
- `description` ← project description
- `package` ← PyPI-normalized name
- `authors[].name` ← project authors
- `dev_install_url` ← `git+https://github.com/...` if detected in pyproject

Everything else is emitted as a placeholder — interview the user and patch the YAML before `--submit`.

## Fields to interview for

### Authors

```yaml
authors:
  - github: <GitHub handle>   # ← interview
    name: <pre-filled>
```

### Legal (3 sub-fields)

```yaml
legal:
  wrapper_license: MIT | Apache-2.0 | ...      # SPDX for the cube wrapper code
  benchmark_license:
    reported: "CC-BY-4.0"                       # self-reported, unverified
    source_url: "https://..."                   # where the license is documented (CI health-checks this)
  notices:
    - type: third_party_data | software_registration | live_website_clone | attribution
      description: "..."
      url: "https://..."
```

Ask:
- What license is your wrapper under? (default MIT if they don't know)
- What license is the underlying benchmark data / software? Where's the license documented?
- Any third-party data caveats? (dataset requires registration; benchmark clones a live site; etc.)

### Links
- `paper` — arXiv or DOI URL
- `getting_started_url` — docs / README / website

### Taxonomy
- `tags` — one or more from: `web`, `coding`, `os`, `gui`, `mobile`, `science`, `math`, `multi-agent`

### Runtime profile
- `supported_infra` — subset of `[aws, azure, gcp, local]`. Default `[aws]` for cloud benchmarks; add `local` if the benchmark runs locally too.
- `max_concurrent_tasks` — how many tasks the user can run in parallel (ask about hardware / API limits).
- `parallelization_mode` — one of:
  - `sequential` — one task at a time per benchmark instance
  - `task-parallel` — multiple tasks concurrently, single benchmark instance
  - `benchmark-parallel` — one task per benchmark instance (heavy infra per task)

## Do NOT fill

These are CI-derived. The bot writes them post-merge:
- `status`, `task_count`, `has_debug_task`, `has_debug_agent`, `resources`, `features.*`, `stress_results_url`

## Invariants

- `version` in YAML must exactly match PyPI `version` (CI quick-check validates).
- `id` must be globally unique in the registry.
- `source_url` in `benchmark_license` must return 200 (CI health-check).

## Flow

1. Run `cube registry add`.
2. Read the generated `cube-registry-entry.yaml`.
3. Interview the user for each placeholder.
4. Patch the YAML.
5. Show the completed file for review.
6. Confirm with the user, then run `cube registry add --submit`.
7. The command forks `The-AI-Alliance/cube-registry`, pushes a branch, opens a PR via `gh`. Show the PR URL to the user.

## What registry CI does on the PR

Four jobs run in sequence; the first three are **hard gates** for auto-merge:

| Gate | What | Hard? |
|---|---|---|
| `ownership-check` | submitter is in `OWNERS.yaml` for the entry (or it's brand new) | yes |
| `quick-compliance` | schema, `pip install`, `Benchmark` import + introspection — hardened Docker sandbox | yes |
| `slow-compliance` | runs the debug task with `provider=local` on the GHA runner | **no** (informational — most real cubes need Docker/VM environments that don't fit a GHA runner) |
| `entry-review` | LLM semantic check (`scripts/entry_review.py` in cube-registry) | yes |

The LLM `entry-review` returns a structured verdict:

```yaml
verdict: PASS | CONCERN
checks:
  description_matches_package: pass | fail | unverified
  authors_consistent_with_git: pass | fail | unverified
  no_id_squat_vs_existing:     pass | fail | unverified
  no_brand_impersonation:      pass | fail | unverified
  wrapper_license_plausible:   pass | fail | unverified
notes: <freeform>
```

**Auto-merge fires when all of:**
- ownership-check ✅ + quick-compliance ✅ + entry-review verdict = `PASS`
- PR diff is strictly additions/modifications under `entries/<id>.yaml`
- PR is from the same repo (fork PRs lack the token scope to merge)

**Otherwise → labeled `ready-for-review` + maintainer merges manually.**

Common causes of `CONCERN`:

- **Package not yet on PyPI** (very common): empty PyPI page can't ground `description_matches_package` or `wrapper_license_plausible`; both stay `unverified` and the verdict tips to `CONCERN`. Publishing to PyPI before submitting flips both to `pass`.
- **README mismatch**: the linked repo's top-level README is for the framework (e.g. `cube-harness`), not the cube subdirectory. Adding a `README.md` inside `cubes/<name>/` covers this.
- **Author handles not traceable**: `authors[].github` not present in the cube subdirectory's commit history.
- **`id` near-duplicate** of an existing entry, or `name`/`description` reads like impersonation of a famous benchmark.

If the user knows the package is `dev_install_url`-only (no PyPI yet), let them know up front that the PR will likely land at `ready-for-review` rather than auto-merge — the verdict is still useful as semantic feedback.
