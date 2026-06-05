# RFC: The Task as a streamable tool (+ a path to multi-agent)

**Status:** DRAFT — high-level (we expand as we go)
**Scope note:** single-agent now; multi-agent shape committed now, runtime later (see §Multi-agent).
**Author:** Alexandre Lacoste (w/ Claude)
**Date:** 2026-06-05
**Related:** `agent-owns-loop`, `stop-action-auto-include`

## The idea

A `Task` keeps its lifecycle — `reset` / `evaluate` / `close` — and **the agent must
never reach those.** So the task is *not* the tool. Instead cube-standard adds a thin
**`TaskTool`**: a tool-shaped facet over a task (`execute_action` + `action_set`) that the
agent holds *instead of* the task. Lifecycle stays on the `Task`, driven by the runtime
(Episode); the agent only ever sees the `TaskTool`. (`TaskTool` is the agent-facing
view; distinct from the task's own `tool`/`toolbox`, which it delegates to.)

You **attach a `Streamer`** to the `TaskTool`; every action it executes — and each
evaluation the runtime produces — is streamed. `Streamer` is an **abstract class in
cube-standard** (the seam only): it defines *when* events fire and *with what data*,
never *where they go*. Concrete sinks (disk, XRay, OTel, RL-HTTP…) live downstream.

For multi-agent, a task exposes **one `TaskTool` per agent** (single-agent = N=1). The
gym `Task.step()` stays as a separate view for gym/RL callers.

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

  subgraph PROPOSED["Proposed — thin TaskTool facet over the Task; streamer is a seam"]
    A2[Agent] -->|execute_action| TT[TaskTool facet<br/>execute_action + action_set]
    TT -->|delegates| T2[(Task<br/>lifecycle hidden from agent)]
    T2 --> TB2[its own concrete tool]
    TT -->|on_action| S[[Streamer abstract]]
    EP[Episode] -->|on_eval| S
    EP -->|reset / evaluate / close| T2
    S -. implemented by .-> H[harness / user sink]
  end
```

## What changes in cube-standard (kept minimal)

```python
class Streamer(ABC):                       # NEW — just the seam
    def on_action(self, action: Action, result: Observation | StepError) -> None: ...
    def on_eval(self, reward: float, info: dict) -> None: ...
    # no event types, no storage, no budget — those live downstream

class TaskTool:                            # NEW — the ONLY surface the agent holds
    action_set: list[ActionSchema]                                   # the task's actions
    def execute_action(self, action) -> Observation | StepError: ... # delegates to the
        # task's per-action execution + obs_postprocess; emits on_action; final_step
        # raises AgentStop (today's TaskDone). No reset / evaluate / close.
    def attach_streamer(self, streamer: Streamer | None) -> None: ...

class Task:                                # lifecycle stays here — runtime-driven only
    def agent_tools(self) -> list[TaskTool]: ...  # NEW: one TaskTool per agent (N=1 default)
    def on_turn_start(self) -> None: ...          # NEW hook: per-turn cube setup; the
                                                  # supported replacement for overriding step()
    # reset / evaluate / close / step — UNCHANGED; the agent never sees them
```

The agent is handed a `TaskTool` (via `task.agent_tools()`), never the task. Everything
else stays downstream: trajectory **event types**, **budget/limits**, **LLM/agent
capture**, and all **persistence**. cube-standard only learns "a task can be driven
through a streamable tool facet."

```mermaid
sequenceDiagram
  participant Ag as Agent
  participant TT as TaskTool
  participant Tk as Task
  participant St as Streamer
  participant Ep as Episode
  Ag->>TT: execute_action(a)
  TT->>Tk: per-action exec + obs_postprocess
  TT-->>St: on_action(a, result)
  TT-->>Ag: Observation
  Note over Ep,Tk: lifecycle is runtime-only
  Ep->>Tk: evaluate()
  Ep-->>St: on_eval(reward, info)
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

2. **Lifecycle must stay hidden from the agent — RESOLVED.** The agent holds a `TaskTool`
   facet (`execute_action` / `action_set`) — never the `Task` — so `reset` / `evaluate` /
   `close` are unreachable. This is *why* the task is not itself the tool: a thin facet
   over it keeps the agent-owns-loop guarantee (cube-harness `#386`) that the task
   reference never leaks to the agent.

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

## Multi-agent (shape it now)

The facet generalizes: **`task.agent_tools()` returns one `TaskTool` per agent.** Each
carries its agent's id, action space, observation, and reward attribution; a streamer
records a unified, per-agent-tagged trajectory. **Single-agent is the N=1 case** — a good
sign this is the right shape. We commit to the shape now (it doesn't fork the design); we
don't build the runtime yet.

```mermaid
flowchart LR
  A1[Agent 1 loop] -->|execute_action| Tk1[tool · agent 1]
  A2[Agent 2 loop] -->|execute_action| Tk2[tool · agent 2]
  Tk1 --> W[(shared task state)]
  Tk2 --> W
  W -->|per-agent eval| W
  Tk1 -->|on_action / on_eval · id=1| S[[Streamer]]
  Tk2 -->|on_action / on_eval · id=2| S
  Sch[Scheduler / turn-policy] -. gates .- Tk1
  Sch -. gates .- Tk2
```

**Variants to support:** timing (**async** any agent any time · **turn-based** in order
· **batch** all at once per turn); per-agent **action + observation** spaces; per-agent
or **joint/general-sum reward**; **inter-agent communication** (direct vs through-env);
**shared vs partitioned** world state; **dynamic membership** (join/leave/spawn);
**per-agent vs episode** termination; **partial observability** (an agent's obs reflects
others' intervening actions).

**Load-bearing considerations (what makes it fail):**

1. **Shared state must serialize even in "async".** Tools are views on one world; two
   simultaneous `execute_action`s race. "Async" = interleaved with serialized state
   access; a returned obs is a snapshot others are concurrently changing.
2. **Batch ≠ turns.** No agent acts unilaterally: collect all N actions → resolve jointly
   (incl. conflicts) → return each obs. A batch `execute_action` is *submit + await the
   joint step*, not an immediate execute.
3. **Make the schedule explicit.** "Who acts when" should be a named **scheduler /
   turn-policy**, not ad-hoc blocking inside the env. Blocking N loops to simulate turns
   is uniform for authors but pushes deadlock/ordering risk into the runtime.
4. **Reward attribution needs the joint state.** "Each tool knows its eval" works only if
   that eval can see the **global** state — otherwise general-sum rewards are
   unexpressible.
5. **The streamer is multi-stream.** Every event needs an **agent id**; the trajectory is
   one interleaved timeline *and* per-agent slices. Compounds open decision (1): N agents'
   LLM streams + the env stream all merge.
6. **The harness needs a multi-agent runner.** `Episode` (one loop) → an **arena** running
   N agent loops + the scheduler. The bigger-scope piece; spans cube-standard (task = set
   of agent-tools + per-agent eval + schedule hook) and cube-harness (the runner).

## Open decisions

- **(1)** streamer on the `TaskTool` only, or `TaskTool` + agent, or harness-owned merge.
- ~~**(2)** restricted facet vs whole task~~ — **RESOLVED:** agent holds a `TaskTool`
  facet, never the `Task` (lifecycle hidden).
- **(3)** accept `Streamer` + `TaskTool` in cube-standard as a pure seam (vs harness-only).
- **(4)** `finished` / `evaluate` cadence: per-turn (proposed) vs per-action.
- **(5) multi-agent scope for v1:** which variants land first (async/turn/batch?),
  is inter-agent communication an action or a channel, and is the scheduler a
  cube-standard concept or a harness one?

`deltas.md` stays intentionally thin until we settle (1)–(5).
