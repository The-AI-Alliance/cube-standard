# Output templates

Produce **both** blocks every run: the maintainer verdict (for the human routing the
queue) and the author-facing reply (the thing that would be posted to the PR).

---

## 1. Maintainer verdict block (chat only — never posted)

```
## Gatekeeper verdict — <RFC title> (#<pr> / <change-name>)

**Verdict:** RESHAPE | ESCALATE | ACCEPT | DECLINE
**Targets:** <spec layer(s) / files>
**Escalate to human?** yes/no — <one line: why this does or doesn't need a human owner>

**Real need (mechanism stripped out):**
- <user story 1>
- <user story 2>

**What the proposal actually changes:**
- <kernel A> → <ACCEPT/RESHAPE/DECLINE> — <one line>
- <kernel B> → <…>

**Blast radius:** <additive | breaks: list each renamed/removed/re-typed public symbol + who declares it>

**Recommended counter-proposal (one line):** <the smaller in-schema change>

**Charter basis:** <principle # + the one quoted spec line that settles it>
```

---

## 2. Author-facing reply (printed; posted to PR only on confirmation)

Structure, in order:

1. **Acknowledge the real need** — restate, in their terms, the genuine problem the RFC
   solves. Make clear the contribution is welcome.
2. **Separate need from mechanism** — "There are really N things bundled here:" and list
   them, so the conversation can accept/reshape/decline each independently.
3. **The argument, per kernel** — for each part you're not accepting, give the charter
   reason with a quoted spec line, concretely (name the symbols, the blast radius).
4. **The counter-proposal / escape hatch** — the smaller path that gets the author what
   they want, as actual code/signatures. Lead with the cheapest escape hatch that fits
   (charter § Escape hatches): "you can already do this in your own cube package /
   harness code," or "here's the one tiny additive hook that unblocks the rest on your
   side." Never argue against something without handing over what to do instead.
5. **Path forward + the open door** — what you suggest for this PR, and an explicit note
   that this is friction, not a final ruling: *if after this you still think it belongs
   in core, that's a fine conversation to have with a human maintainer — here's how to
   raise it.* Make clear they can push back and escalate; you are not the last word.

Template:

```markdown
Thanks for this — <one-line genuine appreciation of the need it targets>.

Reading it, there are really **N** separate things bundled together, and they don't all
land the same way against cube's design. Let me take them one at a time.

**1. <kernel — the good one>.** <Accept / agree it's a real gap.> <Why it fits.>

**2. <kernel — the over-reach>.** <Charter argument.> The spec says:
> <quoted line from openspec/specs/...>
<Concretely: this would break <symbols>, which every cube declares.> Here's a smaller
way to get the same outcome:

​```python
<the in-schema counter-proposal>
​```

**3. <kernel — decline/redirect>.** <Why it belongs in a subclass / harness / doc.>

**Suggested path:** <narrow the PR to X; split Y and Z into separate issues; or do Z
entirely in your own cube package — here's how>. And to be clear, none of this is the
final word: if you still think <kernel> belongs in core after this, that's a legitimate
call for a human maintainer — ping @<maintainer> / open a Discussion and I'll help frame
it. Happy to shape whichever direction you want.
```

Keep it warm and specific. The author should finish reading it knowing (a) their need was
heard, (b) exactly why the framework pushes back where it does, (c) a concrete next step
(usually in their own code) that gets them most of what they wanted, and (d) that they can
push back and reach a human if they still disagree.
