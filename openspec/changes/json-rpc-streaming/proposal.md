# JSON-RPC Implementation Plan for CUBE

## Context

Replace the old multi-endpoint REST server (`make_benchmark_fastapi_app`, `make_task_fastapi_app`)
with a real JSON-RPC 2.0 server (single `POST /` per server, method dispatched from the body),
then add WebSocket and media sideband support in later phases.

---

## Phase 1 — HTTP POST transport ✅ DONE

Single `POST /` endpoint per server.  Method name is in the JSON-RPC 2.0 body.

**Wire format:**

```json
Request:  {"jsonrpc": "2.0", "method": "<name>", "params": {...}, "id": <int|str|null>}
Success:  {"jsonrpc": "2.0", "result": <value>, "id": <same>}
Error:    {"jsonrpc": "2.0", "error": {"code": <int>, "message": "<str>"}, "id": <same>}
```

**Benchmark server methods (`POST /`):**

| Method | Params | Result |
| --- | --- | --- |
| `cube/info` | — | `BenchmarkMetadata` |
| `cube/tasks` | `task_id?`, `offset?`, `limit?` | `list[TaskMetadata]` |
| `cube/task_configs` | `task_id?`, `offset?`, `limit?` | `list[TaskConfig]` |
| `cube/spawn` | `task_config`, `host?`, `port?` | `str` URL |
| `cube/shutdown` | — | `null` |

> `cube/task_configs` returns fully-populated TaskConfig dicts (benchmark fills in all
> benchmark-level fields: tool config, seeds, infra URLs, …).  Clients call this to get
> a ready-to-use config and pass it directly to `cube/spawn` — no manual config construction.

**Task server methods (`POST /`):**

| Method | Params | Result |
| --- | --- | --- |
| `tools/list` | — | `list[ActionSchema]` |
| `tools/call` | `name`, `arguments?`, `action_id?` | `Observation \| StepError` |
| `cube/reset` | — | `{obs, info}` |
| `cube/step` | `name`, `arguments?`, `action_id?` | `EnvironmentOutput` |
| `cube/evaluate` | `obs?` | `{reward, info}` |
| `cube/close` | — | `null` |
| `cube/status` | — | `str` |
| `cube/privileged_info` | — | `Content` |

**`tools/call` and `cube/step` param shape** — flat MCP-compatible:

```json
{"name": "click", "arguments": {"x": 100, "y": 200}, "action_id": "abc-123"}
```

**Serialization:** all Pydantic results via `model_dump(mode="json")`; params validated
with `Model.model_validate(params["key"])`.

**Subprocess design (cross-platform):**

- `TaskConfig` is the sole serialization unit across processes.  `Task` and `Benchmark`
  instances are never pickled (may hold non-serialisable live resources).
- Single module-level entry point `_spawn_task_subprocess(task_config, runtime_ctx, host, port)`
  used by both `cube/spawn` and `make_task_rpc_server` — avoids local-closure pickling
  failure on macOS/Windows (spawn start method).
- `make_benchmark_rpc_server` uses a **daemon thread** (not a subprocess) because
  `Benchmark` is not guaranteed picklable.
  TODO: switch to subprocess once `BenchmarkConfig` lands
  ([RFC: BenchmarkConfig and scaling](https://github.com/The-AI-Alliance/cube-harness/blob/rfc/benchmark-config-and-scaling/docs/rfc-benchmark-config-and-scaling.md)).

**Readiness polling:** `wait_for_server` posts `{}` (valid JSON, invalid JSON-RPC) and
checks for HTTP 200.  Works for both server types with no shared method name.

**Delivered files:**

- `src/cube/server.py` — `make_benchmark_jsonrpc_app`, `make_task_jsonrpc_app`,
  `make_benchmark_rpc_server`, `make_task_rpc_server`
- `src/cube/client.py` — `rpc`, `wait_for_server`, `BenchmarkClient`, `TaskClient`
- `tests/test_server.py` — full coverage via `TestClient` (no subprocesses/ports)
- `examples/counter-cube-remote/` — end-to-end example with `server.py`,
  `client_sdk.py`, `client_raw.py`, and `tests/test_e2e.py`

---

## Phase 2 — WebSocket transport (future)

**Why needed:**

- **Multi-agent:** N agents connect simultaneously; server must push different observations to different agents.
- **Async actions:** agent sends an action, server acknowledges immediately and pushes the result later as a JSON-RPC notification (no `id`).
- **Streaming observations:** continuous server-to-client data (video, stdout, …).

**Proposed design:**

Add `make_task_ws_app(task)` with a WebSocket endpoint at `ws://host/ws`.  The
connection is the session.  JSON-RPC message schema is identical to Phase 1; only
the transport changes.

Server-initiated messages use JSON-RPC *notifications* (no `id`):

```json
{"jsonrpc": "2.0", "method": "cube/action_result",
 "params": {"action_id": "abc-123", "obs": {...}, "done": false}}
```

Async step flow:

1. Client sends `{"method": "cube/step", "params": {"name": "click", ...}, "id": 1}`
2. Server replies `{"result": {"status": "accepted"}, "id": 1}` immediately
3. Server later pushes notification with the result

For multi-agent: each agent opens its own WebSocket.  Optional `agent_id` in params
identifies the sender.

---

## Phase 3 — Media sideband channels (future)

Heavy binary streams (video, audio) must not flow through the JSON-RPC control channel.
The task server negotiates a separate channel:

```json
{"jsonrpc": "2.0", "method": "observation/stream_available",
 "params": {"type": "video", "url": "ws://same-host:PORT/stream/video"}}
```

The client opens a second WebSocket to that URL and receives raw binary frames.
This keeps control-plane latency unaffected by media backpressure.

---

## Possible follow-up examples (not blocking Phase 2/3)

- **`examples/counter-cube-node/`** — Node.js server implementing the CUBE JSON-RPC
  protocol with zero Python dependencies.  Demonstrates language-agnostic protocol.
- **`examples/docker-cube/`** — Docker-backed benchmark where each task's tool runs
  inside a container via `LocalContainerBackend`.
- **Ray parallel test** (in `cube-harness`) — N agents in parallel, each calling
  `cube/spawn` on a shared benchmark server and running independent episodes.
