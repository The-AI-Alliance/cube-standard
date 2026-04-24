# Report template

Produce the review report in this structure.

## Layout

````
# CUBE review — <cube-id>

**Reviewed:** <local path | YAML file | PR URL>
**Verdict:** 🟢 Approve | 🟡 Suggestions only | 🔴 Blocking issues

## Blocking ({count})

1. **<short title>** — `<file:line>`
   <one-or-two-sentence why>. Fix: <action>.

(If none: "No blocking issues.")

## Suggestions

### Score 75 — strong ({count})
1. **<title>** — `<file:line>`
   <why>. Fix: <action>.

### Score 50 — consider ({count})
1. ...

### Score 25 — nit ({count})
1. ...

## Checks run
- pytest tests/: <pass / fail, N tests>
- cube test <id>: <pass / fail, N tasks>
- YAML validation: <pass | skipped (not in scope)>
- Static code checks: <run>
- Hygiene: <run>

## Next steps
<If Blocking > 0: list the top blockers.>
<Else: suggest the top 3 suggestions to address first, highest score first.>
````

## Rules

- **Verdict** — 🔴 if any Blocking finding, else 🟡 if any Suggestion, else 🟢.
- **Sort order** — Blocking in occurrence order (easier to fix top-down). Suggestions sorted by score descending, then by title.
- **Finding density** — don't over-flag. If you find 10 minor S-25s of the same kind, consolidate into a single line like "Multiple TODO placeholders left in the source (list of file:line)".
- **File:line references** — include them when the finding points at specific code. Optional for configuration / metadata findings.
- **For posting via `gh pr review --comment`** — the body is Markdown. Escape backticks that appear inside code snippets; wrap long multi-line blocks with fenced code blocks. Avoid characters that would corrupt a shell-passed argument; prefer reading the body from a temp file if it's long.
