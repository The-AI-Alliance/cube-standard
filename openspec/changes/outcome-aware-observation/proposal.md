# Outcome-aware Observation: optional `reward` and `done` fields

**Status:** DRAFT
**Date:** 2026-05-29
**Author:** Oleksiy Ostapenko
**Scope:** `cube.core` (`Observation`).
**Targets:** `dev`
**Related:** cube-harness `openspec/changes/multi-episode-rollouts/` (renamed in spirit
to "outcome-aware agents") — the downstream consumer that motivates this change. The
cube-harness side cannot proceed without these fields.

## Problem

Agents in cube-harness receive only `Observation` from `Episode`. They never see the
env's `reward` or `done` flags — both live on `EnvironmentOutput`, which is consumed
framework-side for trajectory recording and loop termination, then discarded before
the next `agent.step()`. This blocks any agent that wants to react to outcomes:

- **Meta-RL** methods that adapt across episodes via reflection (LaMer — arxiv 2512.16848)
  need to know "did the prior step end an episode" and "what was the reward."
- **Online learning / retry-on-failure** — an agent about to submit a final answer could
  re-try a different strategy if the prior step's reward was low.
- **Reward-shaped behaviour** — agents whose policy depends on observed reward signals
  (exploration bonuses, fail-fast heuristics).
- **Outcome-aware logging** — agents that want to record their own per-task metrics
  alongside their state.

Today there's no clean way for an agent to access either signal without breaking the
existing `Agent.step(obs: Observation) -> AgentOutput` contract (which every existing
agent depends on) or leaking env-internal types into agent code.

## Design

Add two optional fields to `Observation`:

```python
class Observation(TypedBaseModel):
    contents: list[Content] = []
    reward: float | None = None     # NEW — additive, optional
    done: bool | None = None        # NEW — additive, optional
```

Both default `None`. Existing Observations and existing agents are byte-identical in
behaviour — the fields are only populated by downstream consumers (cube-harness's
`Episode`) when they want to pass per-step outcome context to the agent.

`Episode` (cube-harness) populates these from the **prior** `env_output` when calling
`agent.step()`:

```
step k:
    obs_for_agent = Observation(
        contents=env_output.obs.contents,
        reward=env_output.reward,    # ← from prior task.step()
        done=env_output.done,        # ← from prior task.step()
    )
    agent.step(obs_for_agent)
```

`EnvironmentOutput.reward` / `.done` remain authoritative (env-side, what the env
emits). `Observation.reward` / `.done` are the agent-side mirror — what the agent
gets to see, set by whatever drives the agent.

## Why this shape

- **Additive optional fields with `None` defaults.** Existing code is unaffected;
  agents that don't read these stay byte-identical. The constitution's Pillar I
  exception for additive backward-compatible changes applies — no breaking change.
- **On `Observation`, not `step()`'s signature.** Putting them on the observation
  keeps `Agent.step(obs) -> AgentOutput` intact (every existing agent's contract
  unchanged). The alternative (`step(obs, reward, done)` or `step(env_output)`) breaks
  every existing agent. Bigger blast radius, no upside.
- **`reward: float | None` and `done: bool | None`** distinguish "no prior step" (None;
  e.g. first call after `reset()`) from "prior step gave reward 0 / done False"
  (0.0 / False). Same shape gym-style envs hint at via "first" observations.
- **`Observation` is the right layer.** It's the agent's view of the world; reward and
  done are part of that view. `EnvironmentOutput` continues to be the env-side
  authoritative record (framework consumes it for trajectory persistence) — the two
  layers play complementary roles, not redundant ones.
- **Doesn't force cubes to populate.** Cubes can keep emitting bare `EnvironmentOutput`
  as today. Only consumers that want to forward reward/done into the agent's view
  (cube-harness's Episode) do the work of construction.

## Why not now: alternatives rejected

- **Pass `EnvironmentOutput` to `step()`.** Bigger change — every existing agent's
  step signature breaks. The fields-on-Observation approach is strictly additive.
- **Pass reward/done as separate kwargs to `step()`.** Same issue: signature change.
  Also puts the "agent's outcome awareness" surface in two places (the kwargs and the
  obs) instead of one.
- **Carry reward/done in `Observation.contents` as special `Content` items.** Couples
  outcome-awareness to content rendering (would these show up in `to_markdown()` /
  `to_llm_messages()`? probably no, which then begs why they're in `contents`).
  Cleaner as typed fields.
- **Add a new `AgentObservation` subclass.** Multiplies types without benefit — agents
  would have to know which subclass to expect; existing code's `Observation` typing
  would lose coverage of the new fields. Plain additive fields on the existing class
  is the minimal-disruption approach.

## Out of scope / follow-ups

- **Consumers populating the fields.** This change adds the fields; cube-harness
  `Episode` populating them is downstream work (in cube-harness, not here).
- **`Task.reset()` semantics.** Reset returns an initial obs; the convention is that
  `reward=None, done=None` there (no prior step). The reset signature itself doesn't
  need changing; consumers that pass the initial obs to an agent just leave the fields
  at their `None` defaults.
- **`Observation.truncated`.** `EnvironmentOutput` has `truncated` (max-steps signal,
  distinct from env-driven `done`). Could be mirrored on `Observation` if a use case
  arises — deferred until then.
- **Cubes that want to expose richer reward shape (vector reward, multi-objective).**
  The `reward: float | None` here matches `EnvironmentOutput.reward`'s shape. Richer
  reward types are a separate design conversation; not blocked by this change.
- **`Agent.finalize()` hook in cube-harness.** Downstream change in cube-harness — uses
  the same primitives this change unlocks. Documented in
  `cube-harness/openspec/changes/multi-episode-rollouts/`.
