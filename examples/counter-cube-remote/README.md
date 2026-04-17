# counter-cube-remote

End-to-end example: run `counter-cube` as a remote JSON-RPC 2.0 server and
drive a full episode from a client — two styles available.

## What this demonstrates

- A CUBE benchmark exposed through a single `POST /` JSON-RPC 2.0 endpoint
- The `cube/task_configs` → `cube/spawn` flow: the client fetches fully-populated
  task configs from the benchmark server, then spawns a task subprocess with one
- A readiness-polling pattern (the task server starts asynchronously —
  `BenchmarkClient.spawn()` polls until ready before returning)
- Two client styles side-by-side:
  - `client_sdk.py` — uses `BenchmarkClient` / `TaskClient` from `cube.client`
  - `client_raw.py` — pure `httpx`, zero cube imports; shows the raw wire
    protocol and serves as a template for non-Python clients

## Benchmark wire protocol (reference)

All requests go to `POST /`. Method is in the JSON body.

**Benchmark server methods:**

| Method | Params | Returns |
| --- | --- | --- |
| `cube/info` | — | `BenchmarkMetadata` |
| `cube/tasks` | `task_id?`, `offset?`, `limit?` | `list[TaskMetadata]` |
| `cube/task_configs` | `task_id?`, `offset?`, `limit?` | `list[TaskConfig]` |
| `cube/spawn` | `task_config`, `host?`, `port?` | task server URL |
| `cube/shutdown` | — | `null` |

**Task server methods:**

| Method | Params | Returns |
| --- | --- | --- |
| `tools/list` | — | `list[ActionSchema]` |
| `cube/reset` | — | `{obs, info}` |
| `cube/step` | `name`, `arguments?`, `action_id?` | `EnvironmentOutput` |
| `cube/evaluate` | `obs?` | `{reward, info}` |
| `cube/close` | — | `null` |
| `cube/status` | — | `str` |
| `cube/privileged_info` | — | `Content` |

## Setup

```shell
cd examples/counter-cube-remote
uv sync
```

## Run

**Terminal 1 — start the benchmark server:**

```shell
uv run python server.py [--host HOST] [--port PORT]
# prints: Benchmark server running at http://127.0.0.1:8765
```

**Terminal 2 — run either client, passing the URL printed above:**

```shell
# Using the cube SDK
uv run python client_sdk.py http://127.0.0.1:8765

# Using raw HTTP (no cube imports)
uv run python client_raw.py http://127.0.0.1:8765
```

Both clients print every JSON-RPC response as pretty JSON, then exit cleanly.

## Run the smoke test

```shell
uv run pytest tests/ -v
```

The test starts `server.py` as a subprocess on port 8765, runs both clients
with that URL, and asserts a zero exit code and `"reward": 1.0` in stdout.
