# JSON-RPC Implementation Plan for CUBE

## Context

`server.py` is currently mislabeled: the functions are named `*_rpc_server` but
the apps they produce are plain REST (multiple GET/POST endpoints).  This plan
replaces that with real JSON-RPC 2.0 and sets up a phased path toward
WebSocket support for multi-agent, async, and streaming use-cases.

---

## Current state (what exists)

| Function | What it actually does |
|---|---|
| `make_benchmark_fastapi_app(benchmark)` | Returns FastAPI with REST endpoints: `GET /cube/info`, `GET /cube/tasks`, `POST /cube/spawn`, `POST /cube/shutdown` |
| `make_task_fastapi_app(task)` | Returns FastAPI with REST endpoints: `GET /tools/list`, `POST /tools/call`, `POST /cube/reset`, `POST /cube/step`, … |
| `make_benchmark_rpc_server(…)` | Wraps above in a subprocess |
| `make_task_rpc_server(…)` | Wraps above in a subprocess |

Problems:
- "RPC" name is misleading; it's REST.
- Multiple endpoints means cross-language clients have to hard-code URL paths per method.
- No uniform error envelope; HTTP status codes carry the error signal, not the body.
- No foundation for server-initiated messages (needed for async steps, multi-agent).

---

## Target: phased JSON-RPC 2.0

### Phase 1 — HTTP POST transport (immediate, this PR)

Replace the multi-endpoint REST apps with a **single `POST /` endpoint** per
server that dispatches on the `method` field in the JSON-RPC 2.0 body.

**Wire format (JSON-RPC 2.0)**

```
Request:
  {"jsonrpc": "2.0", "method": "<name>", "params": {...}, "id": <int|str|null>}

Success response:
  {"jsonrpc": "2.0", "result": <value>, "id": <same id>}

Error response:
  {"jsonrpc": "2.0", "error": {"code": <int>, "message": "<str>", "data": <opt>}, "id": <same id>}
```

Standard error codes used:

| Code | Name | When |
|---|---|---|
| -32700 | Parse error | Body is not valid JSON |
| -32600 | Invalid Request | Missing `jsonrpc` or `method` field |
| -32601 | Method not found | Unknown method name |
| -32602 | Invalid params | Required param missing or wrong type |
| -32603 | Internal error | Unhandled exception in method handler |

**Method registry**

Benchmark server (`POST /`):

| Method | Python equivalent | Params | Result |
|---|---|---|---|
| `cube/info` | `benchmark.benchmark_metadata` | — | `BenchmarkMetadata` |
| `cube/tasks` | `benchmark.task_metadata.values()` | `task_id?`, `offset?`, `limit?` | `list[TaskMetadata]` |
| `cube/spawn` | starts subprocess, returns URL | `task_config`, `host?`, `port?` | `str` URL |
| `cube/shutdown` | `benchmark.close()` | — | `null` |

> Note: `cube/spawn` (network) and `benchmark.spawn()` (Python API) are separate
> concerns.  The network method starts a subprocess and returns a URL.
> `benchmark.spawn()` creates the task in-process and returns `(task, app)` — no
> subprocess.

Task server (`POST /`):

| Method | Python equivalent | Params | Result |
|---|---|---|---|
| `tools/list` | `task.action_set` | — | `list[ActionSchema]` |
| `tools/call` | `task.tool.execute_action(action)` | `action` | `Observation \| StepError` |
| `cube/reset` | `task.reset()` | — | `{obs, info}` |
| `cube/step` | `task.step(action)` | `action` (single or list) | `EnvironmentOutput` |
| `cube/evaluate` | `task.evaluate(obs)` | `obs` | `{reward, info}` |
| `cube/close` | `task.close()` | — | `null` |
| `cube/status` | `task.get_status()` | — | `str` |
| `cube/privileged_info` | `task.get_privileged_info()` | — | `Content` |

**Serialization notes**

All Pydantic models in results are serialized via `model_dump(mode="json")`, which
preserves the `_type` discriminator needed by `TypedBaseModel` for round-trip
deserialization.  Params containing Pydantic models (e.g. `action`, `obs`,
`task_config`) are validated with `Model.model_validate(params["key"])`.

**Public API changes in `server.py`**

| Old name | New name | Notes |
|---|---|---|
| `make_benchmark_fastapi_app` | `make_benchmark_jsonrpc_app` | Single endpoint |
| `make_task_fastapi_app` | `make_task_jsonrpc_app` | Single endpoint |
| `make_benchmark_rpc_server` | unchanged | Wraps new app in a subprocess |
| `make_task_rpc_server` | unchanged | Wraps new app in a subprocess |
| `ServerInfo` type alias | unchanged | `(FastAPI, Process, str)` |

**Changes in `benchmark.py`**

`Benchmark.spawn(task_config)` signature changes:

- Old: starts a subprocess, returns `str` URL
- New: creates the task in-process, returns `tuple[Task, FastAPI]`

Subprocess management is now the caller's responsibility — use
`make_task_rpc_server(task)` if you need a running server.

No new dependencies are required; FastAPI + uvicorn are already in
`pyproject.toml`.

---

### Phase 2 — WebSocket transport (future, not this PR)

**Why WebSocket is needed for advanced use-cases:**

- **Multi-agent**: N agents connect simultaneously; the server must push
  different observations to different agents.  HTTP POST is client-initiated
  only.
- **Async action execution**: agent sends an action, server acknowledges
  immediately, pushes the result later as a JSON-RPC *notification* (no `id`
  field).  HTTP requires polling or long-polling.
- **Streaming observations**: sensor streams, video, stdout — continuous
  server-to-client data flow.  HTTP POST is a single request/response.

**Proposed design (for a follow-up PR):**

Add `make_task_ws_app(task)` that opens a WebSocket endpoint at `ws://host/ws`.
The connection itself is the session.  The JSON-RPC message schema is identical
to Phase 1; only the transport changes.

Server-initiated messages use JSON-RPC *notifications* (requests without `id`):

```json
{"jsonrpc": "2.0", "method": "cube.action_result",
 "params": {"action_id": 42, "obs": {...}, "done": false}}
```

Async step flow over WebSocket:
1. Client sends `{"method": "cube.step", "params": {"action": ...}, "id": 42}`
2. Server replies `{"result": {"status": "accepted"}, "id": 42}` immediately
3. Server later pushes `{"method": "cube.action_result", "params": {...}}` (notification)

For multi-agent: each agent opens its own WebSocket connection.  An optional
`agent_id` in params identifies the sender when the server wants to route
messages.

---

### Phase 3 — Media sideband channels (future, not this PR)

Heavy binary streams (video frames, audio) should not flow through the JSON-RPC
control channel.  Instead, the task server negotiates a separate channel:

```json
{"jsonrpc": "2.0", "method": "observation.stream_available",
 "params": {"type": "video", "url": "ws://same-host:PORT/stream/video"}}
```

The client opens a second WebSocket to that URL and receives raw binary frames.
This keeps the control plane latency unaffected by media backpressure.

---

## Test plan for Phase 1

**File**: `tests/test_server.py`

Uses a minimal inline benchmark + task + tool (no `counter-cube` dependency).
Tests benchmark server and task server independently in-process via
`starlette.testclient.TestClient` — no subprocess, no ports.

```python
# 1. Make the benchmark JSON-RPC app
app = make_benchmark_jsonrpc_app(MinimalBenchmark())
client = TestClient(app)

# 2. cube/info
resp = client.post("/", json={"jsonrpc":"2.0","method":"cube/info","id":1})
assert resp.json()["result"]["name"] == "minimal-benchmark"

# 3. cube/tasks with filtering
resp = client.post("/", json={"jsonrpc":"2.0","method":"cube/tasks","id":2})
assert len(resp.json()["result"]) == 2

# 4. benchmark.spawn() returns (task, app) — use directly
task, task_app = benchmark.spawn(MinimalTaskConfig(task_id="task-1"))
task_client = TestClient(task_app)

# 5. tools/list
resp = task_client.post("/", json={"jsonrpc":"2.0","method":"tools/list","id":1})
# assert action schemas

# 6. cube/reset
resp = task_client.post("/", json={"jsonrpc":"2.0","method":"cube/reset","id":2})
obs = resp.json()["result"]["obs"]

# 7. cube/step until done
action = {"name": "...", "arguments": {}}
resp = task_client.post("/", json={"jsonrpc":"2.0","method":"cube/step",
                                    "params":{"action": action},"id":3})
assert resp.json()["result"]["done"] is True

# 8. Error: unknown method → -32601
resp = task_client.post("/", json={"jsonrpc":"2.0","method":"cube/unknown","id":99})
assert resp.json()["error"]["code"] == -32601

# 9. Error: missing param → -32602
resp = task_client.post("/", json={"jsonrpc":"2.0","method":"cube/step","id":100})
assert resp.json()["error"]["code"] == -32602

# 10. Error: bad JSON → -32700
resp = task_client.post("/", content=b"not json", headers={"content-type":"application/json"})
assert resp.json()["error"]["code"] == -32700
```

Also updates `tests/test_benchmark_server.py` to use `make_benchmark_jsonrpc_app`
and JSON-RPC calls.

**File**: `examples/counter-cube-remote/`

A standalone example showing how to run counter-cube as a remote JSON-RPC server
and interact with it from a plain HTTP client (no cube imports needed on the
client side).  Intended as a learning resource for benchmark authors and harness
developers.  Contains:

- `server.py` — starts the benchmark server and waits
- `client.py` — connects via raw HTTP JSON-RPC, runs a full episode, prints results
- `README.md` — explains the remote protocol and how to run the example

---

## Decisions

1. **Callers of old names**: `tests/test_benchmark_server.py` imports
   `make_benchmark_fastapi_app` and tests REST endpoints — it will be updated
   to use `make_benchmark_jsonrpc_app` and JSON-RPC calls.  No callers in
   `cube-harness`.

2. **`benchmark.spawn()` simplification**: remove subprocess creation from
   `spawn()`.  New signature: `spawn(task_config) → tuple[Task, FastAPI]`.
   The `cube/spawn` network endpoint (in the benchmark server) is separate:
   it still starts a subprocess and returns a URL, because that is the whole
   point of a remote API.  `benchmark.spawn()` and `cube/spawn` serve different
   use-cases and do not need to be consistent with each other.

3. **Method naming**: slashes, matching MCP convention
   (`cube/info`, `cube/tasks`, `tools/list`, `tools/call`, …).

4. **Test client**: `starlette.testclient.TestClient` (already wraps async
   handlers transparently — no `pytest-asyncio` needed).

5. **Test dependencies**: no new deps.  `test_server.py` defines a minimal
   inline cube (benchmark + task + tool) to avoid depending on `counter-cube`.

---

## Session checkpoint — resume from here

### What is done

- [x] `src/cube/server.py` — fully rewritten.
  - Old REST functions (`make_benchmark_fastapi_app`, `make_task_fastapi_app`) replaced
    by `make_benchmark_jsonrpc_app` and `make_task_jsonrpc_app`.
  - Single `POST /` endpoint with JSON-RPC 2.0 dispatch in each.
  - Methods use slash naming: `cube/info`, `cube/tasks`, `cube/spawn`, `cube/shutdown`,
    `tools/list`, `tools/call`, `cube/reset`, `cube/step`, `cube/evaluate`, `cube/close`,
    `cube/status`, `cube/privileged_info`.
  - Serialization via `model_dump(mode="json")` (linter replaced `jsonable_encoder` calls).
  - `_find_free_port()` helper with TOCTOU docstring.
  - `_ok()` / `_err()` helpers with docstrings.
  - `make_benchmark_rpc_server` / `make_task_rpc_server` kept, now wrap the new apps.

### What remains

- [ ] **`src/cube/benchmark.py`** — update `spawn()`:
  - Change return type from `str` to `tuple[Task, FastAPI]`.
  - Remove `from cube.server import make_task_rpc_server` inside the method.
  - New body: `task = task_config.make(...); app = make_task_jsonrpc_app(task); return task, app`
  - Import `make_task_jsonrpc_app` from `cube.server` (lazy import inside the method to
    avoid circular imports, same pattern as the current code).
  - Update docstring accordingly.

- [ ] **`tests/test_benchmark_server.py`** — update existing test:
  - Change import: `make_benchmark_fastapi_app` → `make_benchmark_jsonrpc_app`
  - Replace REST endpoint calls (`GET /cube/info`, etc.) with JSON-RPC POST calls.
  - Keep the same `MinimalBenchmark` / `MinimalTask` / `MinimalTool` fixtures (reuse or
    move to conftest if `test_server.py` also needs them).

- [ ] **`tests/test_server.py`** — write new comprehensive test file:
  - Define a minimal inline cube (no counter-cube dependency).
    The existing `MinimalBenchmark` from `test_benchmark_server.py` is a good starting
    point — move shared fixtures to a conftest or duplicate if test files stay separate.
  - Test benchmark server: `cube/info`, `cube/tasks` (with offset/limit), `cube/shutdown`.
  - Test `benchmark.spawn()` returning `(task, app)` and use `TestClient(app)` directly.
  - Test task server: `tools/list`, `cube/reset`, `cube/step` (full episode to done),
    `cube/evaluate`, `cube/close`, `cube/status`, `cube/privileged_info`.
  - Test JSON-RPC error envelope:
    - Unknown method → `{"error": {"code": -32601}}`
    - Missing required param → `{"error": {"code": -32602}}`
    - Invalid JSON body → `{"error": {"code": -32700}}`
    - Missing `jsonrpc` field → `{"error": {"code": -32600}}`

- [ ] **`examples/counter-cube-remote/`** — new example:
  - `server.py` — imports `CounterBenchmark`, calls `make_benchmark_rpc_server()`,
    prints the URL, then joins the process (blocks).
  - `client.py` — pure HTTP JSON-RPC (uses only `httpx`, no cube imports).
    Calls `cube/info`, `cube/tasks`, `cube/spawn` to get a task URL, then
    `tools/list`, `cube/reset`, `cube/step` in a loop until done, prints final reward.
  - `pyproject.toml` — depends on `cube-standard` and `counter-cube` (via path).
  - Short `README.md` showing how to run (`python server.py` in one terminal,
    `python client.py` in another).

### Key context for next session

- `benchmark.spawn()` currently (old code) does:

  ```python
  from cube.server import make_task_rpc_server
  task = task_config.make(...)
  _app, _process, url = make_task_rpc_server(task)
  return url
  ```

  It needs to become:

  ```python
  from cube.server import make_task_jsonrpc_app
  task = task_config.make(...)
  app = make_task_jsonrpc_app(task)
  return task, app
  ```

- `test_benchmark_server.py` currently uses:

  ```python
  from cube.server import make_benchmark_fastapi_app
  app = make_benchmark_fastapi_app(benchmark)
  client.get("/cube/info")        # → REST
  client.get("/cube/tasks")       # → REST
  client.post("/cube/shutdown")   # → REST
  ```

  Needs to become:

  ```python
  from cube.server import make_benchmark_jsonrpc_app
  app = make_benchmark_jsonrpc_app(benchmark)
  client.post("/", json={"jsonrpc":"2.0","method":"cube/info","id":1})
  client.post("/", json={"jsonrpc":"2.0","method":"cube/tasks","id":2})
  client.post("/", json={"jsonrpc":"2.0","method":"cube/shutdown","id":3})
  ```

- `counter-cube` example lives in `examples/counter-cube/` and is a working reference
  for how a benchmark is structured.  `counter-cube-remote` should `import CounterBenchmark`
  from it.
