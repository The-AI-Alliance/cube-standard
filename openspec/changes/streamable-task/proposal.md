# RFC: The streamable Task — a tool view for agents (+ multi-agent)

**Status:** DRAFT
**Author:** Alexandre Lacoste (w/ Claude)
**Date:** 2026-06
**Companion (harness):** `cube-harness/openspec/changes/multi-agent-episode`.

## The idea

A `Task` is the **world**: it owns shared state + lifecycle (`reset` / `evaluate` /
`close`), and `step()` is its complete **gym view**. Agents never hold the task — they hold
a thin **`TaskTool`**: a tool-shaped view (`execute_action` + a dynamic `action_set`) that
returns just the **observation**. A multi-agent task exposes **one `TaskTool` per agent**
(`agent_tools()`; single-agent = N=1).

The standard stays tiny: it gains the `TaskTool` view, a per-turn hook, and `agent_tools()`.
Trajectory **capture** and **orchestration** (loops, scheduling, budget, sinks) stay in the
harness — so the standard never learns the words *LLM*, *loop*, *turn*, or *budget*.

## What cube-standard adds (minimal)

```python
class Task:                               # the world — runtime-driven; never held by an agent
    def step(self, action) -> EnvironmentOutput: ...   # the complete gym view; do-not-override
    def evaluate(...) -> tuple[float, dict]: ...        # reward — runtime-side only
    def on_turn_start(self) -> None: ...               # NEW: per-turn cube setup
    def agent_tools(self) -> list[TaskTool]: ...        # NEW: one view per agent (N=1 default)

class TaskTool:                           # the ONLY surface an agent holds
    @property
    def action_set(self) -> list[ActionSchema]: ...    # DYNAMIC — recomputed each turn
        # (legal-action masking, phase gating, real-time observe/no-op). Agents re-read it.
    def execute_action(self, action) -> Observation | StepError: ...
        # runs one action through the task + obs_postprocess; returns the OBS ONLY (no
        # reward); final_step raises AgentStop. No reset / evaluate / close. Eval never
        # flows through here — it stays on the Task, which the agent does not hold.
```

## One path — the cube can't be bypassed

`TaskTool.execute_action` is a thin **view** over the same per-action execution `step`
already performs (`step` = `execute_action` looped over a batch → `finished`/`evaluate` →
`EnvironmentOutput`). Both paths run identical cube logic:

- **`step()`** — the complete gym view. **Finalized, do-not-override.** Cubes that need
  per-turn logic use `on_turn_start`, not a `step` override.
- **`TaskTool.execute_action()`** — the agent view: same execution, returns the obs only
  (reward hidden), `final_step` → `AgentStop`.

So whatever a cube puts in `evaluate` / `finished` / `obs_postprocess` / `on_turn_start`
behaves identically under gym `step`, an in-process agent loop, or an external agent —
**cubes customize hooks, never the orchestration.** The one per-caller knob is the
**`StepError` policy** (gym: error ⇒ done; agent loop: error ⇒ returned to the agent to
recover).

## Capture & eval live in the harness — no standard seam

Trajectory capture needs nothing in the standard:

- The agent loop already has each `(action, obs)` (it *called* `execute_action`), so it
  **emits its own tool + LLM events** through the recorder it already uses.
- **Reward** is hidden from the agent; the harness holds the `Task` and **recovers it
  directly** via `task.evaluate()` (per turn / terminal). Keeping eval off the `TaskTool`
  is also what keeps it out of the agent's reach.

```mermaid
sequenceDiagram
  participant Ag as Agent loop · harness
  participant TT as TaskTool
  participant Tk as Task
  participant Rec as Recorder · harness
  Ag->>TT: execute_action(a)
  TT->>Tk: per-action exec + obs_postprocess
  TT-->>Ag: Observation (no reward)
  Ag->>Rec: emit ToolCallEvent(a, obs) + LLM events
  Note over Ag,Tk: eval is runtime-only
  Ag->>Tk: evaluate()  (harness holds the task)
  Ag->>Rec: emit reward (per turn / terminal)
```

## Multi-agent — one task, many tools

`agent_tools()` returns **one `TaskTool` per agent** over a **single shared `Task`**
(single-agent = N=1). Centralizing the world on one task is what makes the hard parts fall
out for free: its `evaluate()` sees the **global** state, so per-agent / general-sum reward
is attributable from the joint outcome; all tools mutate **one coherent world**; lifecycle
runs **once**.

The harness drives it (companion RFC): an **arena** runs N agents under a **scheduler**
(turn-based first), one agent per `TaskTool` from a single `AgentConfig` parameterized by
each tool's identity + action set.

```mermaid
flowchart TB
  subgraph STD["cube-standard — the world"]
    TASK[("Task · shared world<br/>evaluate() sees global state")]
    TASK -->|agent_tools| TT["TaskTool · 1..N<br/>dynamic action_set"]
  end
  subgraph HARN["cube-harness — the runtime"]
    AR["Arena · scheduler (turn-based v1)"]
    AG["Agent · 1..N<br/>(one AgentConfig per tool)"]
    REC[("Recorder · per agent_id<br/>+ eval via task.evaluate()")]
  end
  AR --> AG
  AG -->|execute_action| TT
  AG -. tool + LLM events · agent_id .-> REC
```

**What stays cube-side** (surveyed ~20 multi-agent benchmarks — no standard primitives
needed): **communication** = a `send_message(to, content)` action (observable, routable via
the cube's delivery logic — broadcast / targeted / team-private / neighbor-graph);
**teams / zero-sum / mixed-motive** = how `evaluate()` maps joint state → per-agent reward;
**opponents & partners** = environment-internal agents the cube runs (`agent_tools()`
returns only the seats under test); **multi-metric eval** = `evaluate()` returns a metrics
dict; **chance** = sampled inside `execute_action` / `on_turn_start`.

**Contracts the standard states explicitly:** the world may advance *between/without* an
agent's action (real-time engines tick natively; observe/no-op is a valid action); one
agent's action may mutate another's next observation (one shared task).

**Scheduling vs legality:** the cube expresses *legality* (dynamic `action_set` +
`on_turn_start`); the arena only decides *who acts next*. Timing variants (turn-based first;
`async` / `batch` / real-time later) are harness scheduler policies.

**Non-goal:** JAX-vectorized MARL training libs (JaxMARL, CAMAR). GPU `vmap` over thousands
of functional worlds is a different *resource*, RL-training-shaped, not agent-eval — scoped
out rather than bending the standard.

## Forward extensions (add when needed — not in this change)

- **Parallel tool calls within a turn.** `step` is sequential/batched. For real concurrent
  intra-turn dispatch, factor the per-action execution out of `step` into a shared core both
  views call. Add only when an agent needs sub-turn concurrency.
- **External / black-box agent capture.** An in-process loop self-emits; a CLI / A2A agent
  driven over `cube.server` can't. Add a capture hook (a `Streamer`-style seam) on the
  `TaskTool` when external agents land — server-side, which also gives hard eval isolation.

## Open decisions

- **(1)** `finished` / `evaluate` cadence: **per-turn** (proposed) vs per-action.
- **(2)** `StepError` policy default + whether it's a per-caller knob.
- **(3)** Multi-agent v1 specifics (termination, budget, `AgentConfig.make()` signature)
  live in the harness companion.

`deltas.md` stays thin until (1)–(2) settle.
