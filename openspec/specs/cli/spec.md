# CLI

**Module:** `cube.cli` | **Entry point:** `cube`

## Purpose

`cube` is the command-line entry point for discovering, scaffolding, testing, and
registering benchmarks.

## Commands

### `cube list`
Lists all benchmarks registered under the `cube.benchmarks` entry-point group.
Shows name, version, task count, tags, description.

### `cube init [NAME]`
Scaffolds a new benchmark package from the template at
`src/cube/_template/new_cube_package/`. Copies the template into `<cwd>/<NAME>`.

- Default `NAME`: `my-benchmark`
- Refuses to overwrite existing directories
- Refuses template placeholder names (`cube_package`, `new_cube_package`, `new-cube-package`)
- `NAME` is converted to module-compatible form (hyphens → underscores) for the Python package

### `cube test NAME`
Runs the debug compliance suite. `NAME` is either:
- A benchmark entry-point name (e.g. `counter-cube`) — debug module auto-resolved
- A dotted module path (e.g. `counter_cube.debug`)

The resolved module must expose:
- `get_debug_benchmark() -> Benchmark`
- `make_debug_agent(task_id: str) -> Callable[[Observation, list[ActionSchema]], Action]`

Exits non-zero if any debug task fails to reach `reward == 1.0`.

Options:
- `--max-steps N` — per-episode step budget (default 20)
- `--stress N` — run each task N times to surface flakiness (default 1)
- `--no-reset-check` — skip reset reproducibility check (useful in CI where the check is slow)
- `--output PATH` — save report JSON
- `--ci` — suppress Rich dashboard, plain-text output (also enabled by `CUBE_CI=1`)

### `cube registry add [PATH]`
Generates `cube-registry-entry.yaml` from `pyproject.toml` at `PATH` (default cwd).

- `--submit` — forks `The-AI-Alliance/cube-registry`, uploads the entry, opens a PR
- `--registry=OWNER/REPO` — target a different registry

## Global options
- `--no-color` — disables ANSI colors (also respects `NO_COLOR` env var per [no-color.org](https://no-color.org))

## Invariants

1. Template copy refuses overwrite — users must explicitly remove target to re-scaffold.
2. `cube test` exits non-zero on any debug failure. CI integration depends on this.
3. Debug module discovery auto-resolves entry-point names to their debug module — no
   explicit convention required beyond exposing the two functions.

## Extension points

- Add a new command by editing `cube/cli.py`, registering the subparser, and implementing
  `cmd_<name>()`.
- New benchmark packages register via `pyproject.toml` `[project.entry-points."cube.benchmarks"]`.
