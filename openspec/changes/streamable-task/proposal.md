# RFC: The Task as a streamable tool

**Status:** DRAFT — high-level (we expand as we go)
**Author:** Alexandre Lacoste (w/ Claude)
**Date:** 2026-06-05
**Related:** `agent-owns-loop`, `stop-action-auto-include`

## The idea

Give `Task` a second *view*: **a Task is a tool an agent interacts with.** The agent
calls `task.execute_action(action)` and gets an `Observation` back — same surface as
any `Tool`. You can **attach a `Streamer`** to the task; every time an action is
executed or an evaluation is produced, the task notifies the streamer.

`Streamer` is an **abstract class in cube-standard** — just the seam. The harness (or
any user) implements a concrete one (write to disk, XRay, OTel, RL-HTTP…). cube-standard
defines *when* it fires and *with what data*, never *where it goes*.

The gym `Task.step()` stays as-is — a second view of the same task, for gym/RL callers.

## Why

Today the harness re-implements the task's execution semantics in a `MonitoredTool`
that wraps the toolbox. Because that wrapper sits *outside* the task, it produced a
cluster of recent bugs — all the same root cause:

- It **replaced the task's own tool**, so the task's `evaluate()`/`setup()` lost access
  to their concrete tool — private attributes, `isinstance`, and `Toolbox.find_tool`
  all broke.
- It had to **register the `final_step` stop sentinel on every tool leaf**, which
  collided as a duplicate action name for any multi-tool task.
- It **bypassed `Task.step()` entirely**, silently disabling per-turn logic a cube had
  put there (workarena invalidates a validation cache per turn; bypassed, its score
  freezes at the first check).

All three exist only because monitoring lives *outside* the task, in a wrapper. If the
task executes its own actions and owns the streamer seam, the wrapper disappears and
that class of bug vanishes by construction.

```mermaid
flowchart LR
  subgraph TODAY["Today — monitoring wraps the toolbox (the tangle)"]
    A1[Agent] -->|execute_action| MT[MonitoredTool wrapper<br/>events + budget + STOP + Task.step semantics]
    MT --> TB1[task.toolbox leaves]
    MT -. clobbers .-> T1[(Task)]
    T1 -. evaluate / finished<br/>see the WRAPPER .-> MT
  end

  subgraph PROPOSED["Proposed — the Task is the tool; streamer is a seam"]
    A2[Agent] -->|execute_action| T2[Task as tool]
    T2 --> TB2[its own concrete tool]
    T2 -->|on_action / on_eval| S[[Streamer abstract]]
    S -. implemented by .-> H[harness / user sink]
    EP[Episode] -->|reset / evaluate / close| T2
  end
```

## What changes in cube-standard (kept minimal)

```python
class Streamer(ABC):                       # NEW — just the seam
    def on_action(self, action: Action, result: Observation | StepError) -> None: ...
    def on_eval(self, reward: float, info: dict) -> None: ...
    # no event types, no storage, no budget — those live downstream

class Task:
    # tool view (NEW): execute one action through the tool + obs_postprocess,
    # notify the streamer. final_step raises AgentStop (today's TaskDone).
    def execute_action(self, action: Action) -> Observation | StepError: ...

    def attach_streamer(self, streamer: Streamer | None) -> None: ...   # NEW

    def on_turn_start(self) -> None: ...    # NEW hook: per-turn cube setup, the
                                            # supported replacement for overriding step()

    def step(self, action) -> EnvironmentOutput: ...   # UNCHANGED gym view
```

Everything else stays downstream: trajectory **event types**, **budget/limits**,
**LLM/agent capture**, and all **persistence**. cube-standard only learns the words
"a task can stream what it executes."

```mermaid
sequenceDiagram
  participant Ag as Agent
  participant Tk as Task (tool view)
  participant St as Streamer
  participant Ep as Episode
  Ag->>Tk: execute_action(a)
  Tk->>Tk: tool.execute_action(a) + obs_postprocess
  Tk-->>St: on_action(a, result)
  Tk-->>Ag: Observation
  Ep->>Tk: evaluate()
  Tk-->>St: on_eval(reward, info)
```

## Is this view complete? (read this first)

You asked whether the view is shallow. It's the right *shape*, but two things in it are
load-bearing and not yet addressed — if we don't, it fails:

1. **A task-streamer only sees the *environment* half of the trajectory.** `on_action` /
   `on_eval` capture what the task did — **not the agent's LLM calls / reasoning.** Those
   originate in the agent, not the task. So "attach a streamer to the task" does **not**
   by itself produce a full trajectory; the harness still has to capture the agent side
   and *merge* the two streams. Decision needed: is a `Streamer` attached to **both** the
   task and the agent, or does the harness own the merge? *(This is the one most likely to
   bite — the mental model "streamer on the task = full capture" is the shallow part.)*

2. **"The Task is the tool the agent holds" reverses a deliberate decision.** The
   agent-owns-loop change (cube-harness `#386`) kept the `task` reference *away* from the
   agent so it couldn't call `reset`/`evaluate`/`close`. If the agent now holds the
   task-as-tool, it can reach the lifecycle. Decision:
   expose a **restricted tool facet** (only `execute_action` / `action_set`) to the agent,
   vs. hand it the whole task and trust it.

Three more that need a decision but won't sink it:

3. **`Streamer`-in-standard reverses an existing invariant.** The `agent-owns-loop`
   companion says *"monitoring is not a cube-standard concern."* Putting even an abstract
   `Streamer` in the standard is a (small, deliberate) reversal — worth stating. Mitigated
   by keeping it a pure seam (no event types, no storage).
4. **`finished` / `done` / granularity.** `on_eval` covers evaluation, but where does
   "task is finished" fire, and is `finished`/`evaluate` per-action or per-turn? (The
   harness loop silently shifted these to per-action; workarena's `validate()` is
   expensive, so per-turn matters.)
5. **Concurrency.** Parallel `execute_action` (the agent-owns-loop fans out via `gather`)
   means concurrent `on_action` emits — the `Streamer` contract must state whether it can
   assume serialized calls.

## Open decisions

- **(1)** streamer on the task only, or task + agent, or harness-owned merge.
- **(2)** restricted tool facet vs whole task to the agent.
- **(3)** accept `Streamer` in cube-standard as a pure seam (vs harness-only).
- **(4)** `finished` / `evaluate` cadence: per-turn (proposed) vs per-action.

`deltas.md` stays intentionally thin until we settle (1)–(4).
