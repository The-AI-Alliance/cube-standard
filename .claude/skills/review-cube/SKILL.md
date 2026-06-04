---
name: review-cube
description: Review a CUBE benchmark submission — validate the registry YAML (if present), audit the cube code against cube-standard invariants, install the package, run `pytest tests/` and `cube test <id>`, and produce a structured Blocking / Suggestions report. Accepts a local path, a cube-registry-entry.yaml path, or any GitHub PR URL that introduces a cube package. Invoke as `/review-cube <arg>`.
---

# review-cube

You are reviewing a CUBE benchmark submission on behalf of either the cube author (self-review) or a registry maintainer (PR review). Your job: fetch the full cube codebase, run all checks, produce a structured report, and (for PR inputs, with user confirmation) post it as a GitHub PR review comment.

## Input shapes

`$ARGUMENTS` is auto-detected. Echo back what you detected at the start; ask the user only if ambiguous.

- **Local path** — absolute path to a directory containing `pyproject.toml` → local cube dir.
- **YAML path** — `*.yaml` / `*.yml` → registry entry file; parse `package` + `dev_install_url` and follow.
- **GitHub PR URL** — `https://github.com/.../pull/\d+`:
  - Against `cube-registry` (path contains `The-AI-Alliance/cube-registry`) → read the added entry under `entries/`, follow `dev_install_url`.
  - Against any other repo → check whether the PR adds a cube package (see `references/pr-detection.md`). If yes, use the newly added cube dir(s). If no, stop with a clear "not a cube submission" message.

**All input shapes converge on the same action:** resolve to a local cube package directory and run the full checklist on it.

## Flow

### 0. Parse and echo
State the detected mode and target in one line. Ask to confirm only if the shape is ambiguous.

### 1. Resolve to a local cube package dir
- Local path → use directly.
- YAML path → read `package` + `dev_install_url`; clone to `/tmp/cube-review-<id>/`; check out the pinned ref if `dev_install_url` has `@<ref>`.
- cube-registry PR → `gh pr diff <pr> --name-only --diff-filter=A` → find the added `entries/*.yaml` → fetch its content → parse and clone as above.
- Other repo PR → `gh pr checkout <pr>` into a temp worktree; locate the added cube dir(s) via the entry-point-group heuristic in `references/pr-detection.md`.

Multiple cube dirs in one PR: review each and produce one report section per cube.

### 2. Install the package
Try in order; stop at the first that succeeds:
1. If `uv` is on `PATH` → `uv sync --all-extras`
2. Else → `pip install -e .`
3. Else → read the package's `README.md` and follow its install instructions.

If all options fail, record install failure as a **Blocking** finding and continue with static checks only (skip pytest / `cube test`).

### 3. Run all checks
Walk every item in `references/checks.md`. Each check emits zero or more findings.

Always run without asking (per the skill's contract):
- `pytest tests/` — capture exit code + summary.
- `cube test <benchmark-id>` — capture exit code + summary.

### 4. Classify findings
Two buckets:
- **Blocking** — merge / release must not land with this. Binary.
- **Suggestions** — scored by urgency:
  - **75** — strong suggestion; the author should address before merge.
  - **50** — consider; improves quality but not urgent.
  - **25** — nit; purely stylistic / minor.

When in doubt about severity, prefer the lower bucket (under-flag rather than over-flag). Suggestions never block.

### 5. Produce the report
Format per `references/report-template.md`. Two sections (Blocking, Suggestions sorted descending by score) plus a "Checks run" footer and a one-line verdict.

### 6. Deliver
- **Local path or YAML-only input** → print the report in chat. Done.
- **PR input** → print the report in chat, **then ask the user** whether to post it as a GitHub PR review. On confirmation:
  ```
  gh pr review <pr-url> --comment --body "$REPORT"
  ```
  Always `--comment`. Never `--request-changes`.

## Rules

- **Never post a GitHub review without explicit user confirmation.**
- **Always run `pytest tests/` and `cube test <id>` without asking** — these are part of the review, not optional. The only skip is if installation failed.
- **Read-only against the cube code and the registry YAML.** This skill makes no edits to either. The only side effect is the confirmed `gh pr review` comment.
- **Temp work goes under `/tmp/cube-review-<id>/`** and is cleaned up after the report is delivered, unless the user asks to keep it.
- Confirm before any destructive action (e.g., removing the temp dir if the user has opened files in it, force-anything).
- **Flag structural mismatches for human discussion.** If the review surfaces issues that suggest the benchmark shape itself doesn't fit the protocol — not fixable bugs, but things like an action space that can't live in a single `Tool`, evaluation that needs human judgment, or episode structure that doesn't match `reset → step → evaluate` — include a Suggestion in the report recommending the author open an issue at https://github.com/The-AI-Alliance/cube-standard/issues tagging `@recursix` and `@nicolasag` to discuss before merge. Don't list workaround hacks as the solution.

## References

- `references/pr-detection.md` — how to identify a cube-adding PR in any repo.
- `references/checks.md` — the full checklist: Registry YAML / Cube code static / Compliance suite / Hygiene / Registry LLM-review preview.
- `references/report-template.md` — report markdown structure.

## Note: cube-registry runs its own LLM semantic review

This skill audits the cube **code and configuration** locally. Once a
submission lands on a cube-registry PR, the registry CI runs its own LLM
semantic check (`scripts/entry_review.py`) against the entry YAML that
pulls PyPI metadata + the linked repo's README + existing entries +
known-authors. It returns a structured verdict (`PASS` or `CONCERN`)
that gates auto-merge.

The semantic review is intentionally separate from this skill — it sees
different evidence (PyPI page, cross-repo author history) and asks a
different question (is this entry plausibly what it claims to be?). When
running `/review-cube` pre-submission, include the "Registry LLM-review
preview" section from `references/checks.md` in your report so the user
knows what *additional* checks they'll face on the PR.

## Out of scope

- Modifying the cube source, the registry YAML, or any repo state other than the PR review comment.
- Uploading / publishing the cube.
- Proposing code changes beyond what the report suggests.
