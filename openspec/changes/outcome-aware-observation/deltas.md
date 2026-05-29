# Deltas — Outcome-aware Observation

Applies to `openspec/specs/core/spec.md`.

## MODIFIED — `core/spec.md`: `Observation` gains optional `reward` and `done` fields

```python
class Observation(TypedBaseModel):
    contents: list[Content] = []
    reward: float | None = None     # NEW — additive, optional
    done: bool | None = None        # NEW — additive, optional

    @classmethod def from_text(cls, text: str) -> Observation
    def to_llm_messages(self) -> list[dict]
    def to_markdown(self) -> str
    def __add__(self, other: Observation) -> Observation
```

Both fields default `None`. Semantics:

- `reward: float | None` — the reward emitted by the **prior** environment step, if
  the consumer (typically a runner like `Episode`) chose to forward it. `None` when
  no prior step exists (e.g., immediately after `Task.reset()`).
- `done: bool | None` — whether the **prior** step terminated the episode. `None` when
  no prior step exists.

Cubes are not required to populate these — the fields exist for runners that want to
let an agent observe outcomes between steps. `EnvironmentOutput.reward` and `.done`
remain the env-side authoritative source.

## Invariants (additions)

- `reward` and `done` default `None`; existing code paths that construct `Observation`
  without setting them are unaffected.
- `__add__` (concat two Observations) does not merge `reward`/`done` — the operator
  appends `contents` only. Callers that need the fields preserved must construct the
  result explicitly. (Rationale: `__add__` is intentionally for content accumulation;
  outcome fields are per-step, not aggregable.)
- `to_llm_messages()` and `to_markdown()` ignore `reward` and `done` — the rendering
  surface is unchanged. Outcome fields are agent-facing metadata, not LLM-payload.

## Not changed

- `Observation.contents` field and rendering helpers (`to_llm_messages`,
  `to_markdown`, `from_text`, `__add__` content-merging behaviour).
- `EnvironmentOutput.reward` / `.done` / `.truncated` — env-side authoritative source;
  `Observation`'s new fields are a separate, additive surface for agent-facing
  forwarding.
- `Task.reset()` and `Task.step()` signatures.
- `StepError`, `Content` and subclasses, `ActionSchema`, `ActionConfig`,
  `TypedBaseModel`.

## CONSUMED (cube-harness follow-up, not in this change)

- cube-harness `Episode` populates `obs.reward` / `obs.done` from the prior
  `env_output` when calling `agent.step()`. (See cube-harness
  `openspec/changes/multi-episode-rollouts/`.)
- cube-harness `Agent` gets a `finalize(terminal_obs: Observation)` hook that receives
  the terminal observation with these fields populated — the one observation the agent
  never sees via `step()`. (Same change folder.)

## Migration

Fully backward-compatible:

- Existing `Observation(contents=[...])` constructions continue to work; `reward` and
  `done` default to `None`.
- Existing serialized `Observation` JSON / Pydantic dumps deserialize unchanged —
  Pydantic's `model_validate` ignores missing fields when they have defaults.
- Existing agents (any agent that reads `Observation.contents` but not
  `Observation.reward` / `.done`) are byte-identical in behavior.
- No cube changes required. Cubes choosing to populate the fields when constructing
  their own Observations (rare — typically the runner does this) can do so at their
  leisure.

## Why not deprecate `EnvironmentOutput.reward` / `.done`

`EnvironmentOutput` is the env-side return type of `Task.step()` — it's how the env
communicates outcomes to the framework. It stays. The new `Observation` fields are
the agent-side mirror — set by whatever drives the agent (typically the runner)
*from* `EnvironmentOutput` *for* the agent's next `step()` call. The two types play
complementary roles, not redundant ones.
