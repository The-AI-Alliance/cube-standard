# Registry submission

Phase 7 — always runs. This doc covers the YAML schema and the `cube registry add` flow.

## Command

From the cube package directory:

```bash
cube registry add              # creates cube-registry-entry.yaml with TODO placeholders
cube registry add --submit     # forks The-AI-Alliance/cube-registry + creates PR via `gh`
```

`cube registry add` reads `pyproject.toml` and pre-fills what it can:
- `id` ← project name (kebab-case)
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
