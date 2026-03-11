# NeMo Gym → CUBE/AL2: Core Ideas to Adopt

---

## 1. Multi-Dimensional Reward
**Priority: High — trivial to add**

NeMo Gym's `reward_profiles: dict[str, float]` is a better design than a single scalar. RL algorithms need dimensions separately (correctness, format, safety, efficiency) to decide how to combine them.

| | Current CUBE | NeMo Gym |
|---|---|---|
| Reward | `evaluate(obs) → (float, dict)` | `reward_profiles: {"correctness": 1.0, "format": 0.8}` |

**Adopt:** Standardize `reward_breakdown: dict[str, float]` as a first-class key in `EnvironmentOutput.info`. The scalar `reward` stays as the aggregate. One-line convention change, backwards compatible.

---

## 2. Training Data as First-Class Rollout Output
**Priority: High — medium effort**

The most architecturally important insight in NeMo Gym: **every evaluation rollout should simultaneously produce training data**. NeMo Gym captures `prompt_token_ids`, `generation_token_ids`, `generation_log_probs` alongside reward in every episode.

CUBE/AL2 currently treats evaluation and training data collection as separate concerns, forcing an expensive second pass to gather token-level data for RL.

**Adopt in AL2:**
- `LLMCall` already captures `Usage` — extend it to optionally carry token IDs when the model server returns them
- `TrajectoryStep` propagates these fields transparently
- `FileStorage` writes them to `.jsonl` steps automatically — zero cost if not present

---

## 3. Responses API as Native Output Format
**Priority: Medium — medium effort**

NeMo Gym chose OpenAI Responses API as the universal contract between components. Any OpenAI-compatible agent works out of the box, and conversation history is a standard artifact readable by any tool.

CUBE's `Observation` / `Action` are more expressive (multi-modal, typed) but opaque to the wider ecosystem.

**Adopt:**
- `Observation.to_responses_api_message() → dict`
- `Action.from_tool_call(tool_call: dict) → Action`

Zero internal cost, makes CUBE tasks plug directly into OpenAI-compatible harnesses. AL2's `ReactAgent` already parses `tool_calls` from LLM responses — this makes it symmetric.

---

## 4. Async-Native Server for RL Scale
**Priority: Lower for eval — high for RL training**

NeMo Gym's entire stack is async (aiohttp, FastAPI async handlers, global connector pool with configurable per-host limits). CUBE's `server.py` uses sync handlers and spawns per-task uvicorn processes.

For evaluation this is irrelevant. For RL training with thousands of parallel rollouts, the connection pool and rate-limit retry logic in NeMo Gym's `server_utils.py` handle back-pressure correctly.

**Adopt in CUBE `server.py`:**
- Make task endpoint handlers `async def`
- Shared aiohttp client in the benchmark server instead of per-task subprocess spawning
- NeMo Gym's global client singleton pattern (configurable per-host connection limits) is worth copying verbatim

---

## 5. The `verify` Separation Pattern
**Priority: Medium — design-level**

In NeMo Gym, tool execution and reward evaluation are **explicitly separate**: agents call tools via custom routes, then call `/verify` to get the reward. CUBE conflates these in `step()` which calls `evaluate()` internally.

The NeMo Gym pattern makes reward computation pluggable — you can swap the verifier (LLM judge, unit test runner, symbolic checker) without touching the environment. It also allows deferred/batch evaluation after rollout collection.

**Adopt:** CUBE's `task.evaluate()` is already separate from `task.step()` — that's good. But the convention that `step()` auto-calls `evaluate()` hides this. Make `validate_per_step=False` the default and require harnesses to call `evaluate()` explicitly. This matches NeMo Gym's explicit `/verify` call and makes the reward signal auditable.

---

## What NOT to bring over

| Feature | Why not |
|---|---|
| Hydra/OmegaConf config | Pydantic is strictly better for a typed standard library |
| Head server for config distribution | CUBE's registry is a cleaner solution |
| Session cookies for state | CUBE's Python object identity is cleaner; sessions are only needed for HTTP, which `spawn() → URL` already handles |
| Ray built into the protocol | Ray belongs in the harness (AL2 already has it); coupling the protocol to Ray's versioning is a mistake |
