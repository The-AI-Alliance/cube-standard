# JSON-RPC Server

**Module:** `cube.server`

## Purpose

Expose Benchmarks and Tasks over JSON-RPC 2.0, MCP-compatible wire format. All requests
dispatch through a single `POST /` endpoint.

## Protocol

### Request / Response
```json
// Request
{"jsonrpc": "2.0", "method": "<name>", "params": {...}, "id": 1}

// Success
{"jsonrpc": "2.0", "result": <value>, "id": 1}

// Error
{"jsonrpc": "2.0", "error": {"code": <int>, "message": "<str>"}, "id": 1}
```

Error codes follow JSON-RPC 2.0:
- `-32700` Parse error
- `-32600` Invalid Request
- `-32601` Method not found
- `-32602` Invalid params
- `-32603` Internal error

### Benchmark methods
| Method | Params | Returns |
|--------|--------|---------|
| `cube/info` | — | `BenchmarkMetadata` |
| `cube/tasks` | `task_id?`, `offset?`, `limit?` | `list[TaskMetadata]` |
| `cube/task_configs` | `task_id?`, `offset?`, `limit?` | `list[TaskConfig]` |
| `cube/spawn` | `task_config`, `host?`, `port?` | URL string |
| `cube/shutdown` | — | `null` |

### Task methods
| Method | Params | Returns |
|--------|--------|---------|
| `tools/list` | — | `list[ActionSchema]` |
| `tools/call` | `name`, `arguments?`, `action_id?` | `Observation \| StepError` |
| `cube/reset` | — | `{obs, info}` |
| `cube/step` | `name`, `arguments?`, `action_id?` | `EnvironmentOutput` |
| `cube/evaluate` | `obs?` | `{reward, info}` |
| `cube/close` | — | `null` |
| `cube/status` | — | string |
| `cube/privileged_info` | — | `Content` |

### Param shape for `tools/call` and `cube/step`
```json
{"name": "click", "arguments": {"x": 100, "y": 200}, "action_id": "abc-123"}
```

Flat MCP-compatible shape — a CUBE task server is drivable by a standard MCP client with
no adapter layer. `Action` is internal; clients never construct it directly.

`action_id` is optional. When provided, forwarded as `Action.id` for logging correlation.
Distinct from the JSON-RPC envelope `id`.

## Public API

### App factories (no subprocess / deployment concerns)
```python
def make_benchmark_jsonrpc_app(benchmark: Benchmark) -> FastAPI
def make_task_jsonrpc_app(task: Task) -> FastAPI
```

Both return plain FastAPI apps usable with `starlette.testclient.TestClient` or any
ASGI host (Modal, fly.io, Cloud Run, …).

### Server launchers (convenience)
```python
def make_benchmark_rpc_server(
    config: BenchmarkConfig, *, infra: InfraConfig | None = None,
    host="127.0.0.1", port=8000,
) -> (Process, str)

def make_task_rpc_server(
    task_config: TaskConfig, host="127.0.0.1", port=8000, runtime_context=None,
) -> (Process, str)
```

- Benchmark → **subprocess** (BenchmarkConfig + InfraConfig are passed directly;
  multiprocessing pickles them across the spawn/fork boundary — pickle handles
  `TypedBaseModel` polymorphism correctly. The worker calls `config.make(infra)`
  locally; the live Benchmark never crosses the process boundary).
- Task → **subprocess** (TaskConfig is picklable; Task is not).

## Serialization boundaries

- **TaskConfig** is the unit of serialization across process boundaries for tasks.
  Always picklable. Self-contained after the Option 1 refactor — carries its own
  `TaskMetadata`, so workers never import the owning `BenchmarkConfig` to resolve
  metadata. Subprocess entry points receive a `TaskConfig` and call
  `task_config.make()` inside the worker.
- **BenchmarkConfig** is the same boundary for benchmarks. Passed directly to
  the subprocess; multiprocessing's pickle handles polymorphism. The worker
  calls `config.make(infra)` locally.
- **Task** and **Benchmark** objects hold live resources — never cross a process
  boundary.

`cube/spawn` (network endpoint) vs `benchmark.spawn(task_config)` (Python API):
- Network endpoint starts a subprocess, returns URL. Subprocess calls `task_config.make()`.
- Python API creates the Task in-process, returns the `Task` object directly.

## Invariants

1. Only one subprocess entry point exists for tasks: `_spawn_task_subprocess`.
   The benchmark equivalent is `_spawn_benchmark_subprocess`. Both are
   module-level functions (macOS/spawn-compat requirement — local closures can't
   be pickled as targets).
2. `container_backend` is NOT forwarded to task subprocesses — legacy parameter being
   replaced by the `infra` / `resource` pattern. Infra state goes via `runtime_context`.
3. All benchmark methods are synchronous; `async def dispatch` is only for
   `await request.json()`. Long-running sync work must use `asyncio.to_thread()`.
4. Results are serialized with `model.model_dump(mode="json")`. Non-JSON-serializable
   fields without a Pydantic serializer are silently excluded.
5. `_spawn_benchmark_subprocess` calls `benchmark.close()` in a `finally` block
   so the worker tears down L2 resources even on uvicorn shutdown or signal.

## Gotchas

- Readiness race in `cube/spawn` and both `make_*_rpc_server` launchers — the URL
  is returned immediately, but uvicorn starts asynchronously. Clients must poll
  until the server responds before sending requests.
- `_find_free_port()` has a TOCTOU window: between port detection and bind, another
  process could claim the port. Callers should retry on `OSError: Address already in use`.
