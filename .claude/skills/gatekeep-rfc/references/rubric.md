# Rubric — verdicts, decision tree, scoring

## Verdicts

| Verdict | When | What you produce |
|---------|------|------------------|
| **ACCEPT** | Additive, in-charter, general. No public surface broken. Rare for external RFCs. | Light maintainer note recommending a human give it a normal review. Don't rubber-stamp the merge yourself. |
| **RESHAPE** | A real need wrapped in an oversized or breaking mechanism. **The default.** | A concrete in-schema counter-proposal (code/signatures) the author can adopt, plus the charter argument for why. |
| **ESCALATE** | A genuine framework gap, or a design decision with cross-layer consequences a human must own. | A tight summary of the gap + your best smaller alternative, addressed to the maintainer. |
| **DECLINE** | Out of scope for the core (belongs in a subclass or the harness), or pure cosmetic churn with no behavioral gain. | A respectful redirect to the right place (subclass field / harness code / doc comment), never a flat no. |

A single RFC often splits: one kernel ACCEPTs, another RESHAPEs, a rename DECLINEs.
Classify each kernel, then give the overall verdict as the most consequential one.

## Decision tree (per kernel of the proposal)

```
Is the stated need real and general (many cubes, not just the author's)?
├─ no  → DECLINE (redirect: subclass field, or harness-side, or it's a non-problem)
└─ yes → Can the CURRENT schema already express it?
         ├─ yes → DECLINE the change, SHOW the author how (unnecessary RFC)
         └─ no  → Does it even belong in the CORE? (general across many cubes AND not
                  carryable by a subclass field or harness code?)
                  ├─ no  → DECLINE, redirect to subclass / harness. "Only additive" is
                  │        not enough — every core symbol is permanent maintenance surface.
                  └─ yes → Can it be delivered ADDITIVELY (new optional field / method /
                           widened backward-compatible type), and is it the *minimal* such?
                  ├─ yes → RESHAPE to the smallest additive form
                  └─ no  → Does it need a human-owned design decision?
                           ├─ yes → ESCALATE (with smaller alternative attached)
                           └─ no  → RESHAPE to the smallest breaking change that works,
                                    and call out the blast radius explicitly
```

## Scoring questions (run all; any "bad" answer is a finding, not an auto-reject)

1. **Need is real & general?** Would ≥several other cubes want this, or is it local to
   the author's benchmark?
2. **Already possible?** Does the live spec already offer a path (subclass field,
   existing method, harness compose)?
3. **Additive or breaking?** Does it rename/remove/re-type any author-facing public
   symbol? Name every broken symbol and who declares it.
3b. **Does it earn core space (leanness)?** Even if additive: is it general and minimal,
   or is it surface the core will carry forever for one benchmark's benefit? "Only
   additive" is not a pass — the lean default is to leave it out.
4. **Single source of truth preserved?** Does it add overlapping/derived state or a
   validator that syncs one field from another?
5. **Right layer / right side?** Framework vs harness; config (serializable) vs runtime.
6. **Cosmetic churn?** Any rename/restructure whose only benefit is "reads nicer"?
7. **Internal consistency?** Does the proposal name the same thing two ways, leave dead
   numbering, or reference symbols it never defines? (Quality signal about how baked it
   is — note it, but it's not the basis for the verdict.)
8. **Process conformance?** Does it carry Problem · Proposed solution · **Alternatives**?
   A breaking proposal with no Alternatives section is incomplete.

## What NOT to weigh

Judge the engineering, not the packaging. The following are **irrelevant** to the
verdict and must never be cited as a reason:

- Whether the prose reads as LLM-generated.
- Missing syntax highlighting / language tags on code fences.
- Typos, formatting slips, em-dash density, heading style.
- Who the author is or how new they are.

These say nothing about whether the change is right for cube. Provenance is not
substance. (If formatting genuinely impairs review — e.g. a delta is unparseable —
note it as a courtesy fix request, not a verdict input.)

## Tone calibration

- Lead with the real need, acknowledged in the author's own terms. They are trying to
  help; treat the contribution as welcome.
- Argue from the charter with a *quoted* spec line, not an assertion.
- Every decline or reshape ships a concrete alternative — usually an escape hatch (their
  own cube repo / their own code / a tiny additive hook). The author should leave knowing
  exactly what to do next.
- Teach the *why*, and keep the door open: your verdict is advisory, the collaborator may
  insist, and a genuine disagreement escalates to a human. Never sound final.
- Be concise. One screen of argument beats three of hedging.
