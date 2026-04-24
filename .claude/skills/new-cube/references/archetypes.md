# Registry archetypes

Use this in phase 2 (Reflect). Match the user's benchmark to the closest archetype; if ≥80% fit, fetch that archetype's code as a reference.

## The current archetypes

Registered in the registry today:

| Archetype | Cube | What makes it distinct |
|-----------|------|------------------------|
| **Simple local** | `miniwob` | Local HTTP server serving static pages, 7 browser actions, no cloud infra. Good starting shape for a small static benchmark. |
| **VM / desktop** | `osworld-cube` | One VM per task, qcow2 images, KVM, `benchmark-parallel`. Expensive infra. Use when tasks need a full desktop. |
| **Remote SaaS** | `workarena` | No local infra; connects to a remote ServiceNow PDI per task. Lazy task loading from an upstream browsergym package. |
| **Verified web** | `webarena-verified` | Multi-platform verified web tasks, benchmark-scoped Docker services, task-parallel. |

There are also cubes in `cube-harness/cubes/` that are **not yet in the registry** but are useful as shape references:

- `swebench-verified`, `swebench-live-cube` — per-task Docker images (currently limited by #300; use for the `extra_info` + `install()` pattern)
- `terminalbench` — shell / terminal-based tasks (per-task Docker)

## Fetch strategy

1. If `cube-harness` is cloned as a sibling of `cube-standard` at `../cube-harness/`, prefer reading from there on the `main` branch.
   - Unless the user specifically asks for a non-main branch.
   - Relevant paths:
     - `cube-harness/cubes/miniwob/`
     - `cube-harness/cubes/osworld-cube/`
     - `cube-harness/cubes/webarena-verified/`
     - `cube-harness/cubes/workarena/`
     - `cube-harness/cubes/swebench-live-cube/`
     - `cube-harness/cubes/swebench-verified-cube/`
     - `cube-harness/cubes/terminalbench/`
2. Otherwise fetch from GitHub with WebFetch: `https://github.com/The-AI-Alliance/cube-harness/tree/main/cubes/<name>/`.

Pull at minimum: `benchmark.py`, `task.py`, `tool.py`, `debug.py`, `pyproject.toml`, and any `scripts/create_task_metadata.py`. Scan for structure, not details.

## Matching signals

| Signal from user | Lean toward |
|------------------|-------------|
| "Web pages", "HTML", "DOM", "click / type / submit" | `miniwob` or `webarena-verified` |
| "Desktop", "Ubuntu", "LibreOffice", "screenshot + click" | `osworld-cube` |
| "ServiceNow" / "enterprise SaaS" / "our platform already has tasks" | `workarena` |
| "Terminal", "shell commands", "file system" | `terminalbench` (per-task Docker — flag #300) |
| "Coding", "repo", "fix the bug", "patch" | `swebench-verified` / `swebench-live-cube` (per-task Docker + heavy `extra_info` — flag #300; show `extra_info` pattern) |
| "Audio", "video", "stream" | **Unsupported** — streaming not in protocol yet. Flag. |
| "Multiple agents cooperating" | **Unsupported** — single-agent only for now. Flag; suggest modeling as richer action set. |

## If no archetype is a ≥80% match

- Use the scaffold as-is; don't force a shape.
- Read the closest archetype's `task.py` and `benchmark.py` for structural inspiration only.
- Call this out explicitly in the Requirements Summary so the user knows they're in greenfield territory.
