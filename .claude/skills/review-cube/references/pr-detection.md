# Identifying a cube-adding PR

Any PR in any repo can introduce a cube package. The reliable **primary signal** is a newly added `pyproject.toml` that declares the `cube.benchmarks` entry-point group.

## Procedure

1. `gh pr diff <pr-url> --name-only --diff-filter=A` — list ADDED files only.
2. Filter to `*/pyproject.toml`.
3. For each candidate, fetch its content (`gh pr view <pr> --json files` + extract, or `gh pr checkout` and read from disk). Parse TOML.
4. Check for:
   ```toml
   [project.entry-points."cube.benchmarks"]
   ```
5. If present, the directory containing that `pyproject.toml` is a new cube package.

If zero candidates match, the PR isn't cube-adding — stop with a clear "no new cube package detected" message.

## Sanity signals (raise confidence; not required)

- `cube-standard` appears in `[project.dependencies]`.
- A sibling `.py` module defines a `class ...(Benchmark)` subclass.
- For cube-harness specifically: the path matches `cubes/<name>/pyproject.toml`.

## Edge cases

- **Multiple cube packages in one PR** — review each; one report section per cube, concatenated into a single report.
- **cube-registry PR** — different shape: the PR adds a YAML entry under `entries/`, NOT a new `pyproject.toml`. Detect this first (path starts with `entries/` and adds a `.yaml`), and follow the YAML's `dev_install_url` instead of running the entry-point detection.
- **PR modifies an existing cube** (no NEW `pyproject.toml`) — not cube-adding per the primary signal. Report "no new cube package detected in this PR" and stop.
- **PR adds a `pyproject.toml` that doesn't declare the entry-point group** — not a cube package; stop.
