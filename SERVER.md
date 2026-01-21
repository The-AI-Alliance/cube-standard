# CUBE Server Setup Guide

## Quick Start

### 1. Install Dependencies

```bash
# Install in editable mode with dev dependencies
make install

# Or manually with uv
uv sync --all-extras
uv pip install -e ".[dev]"
```

### 2. Run the Server

```bash
# Start the server (default: http://localhost:8000)
uv run python -m cube

# Or with custom configuration
CUBE_HOST=0.0.0.0 CUBE_PORT=8080 uv run python -m cube

# Development mode with auto-reload
CUBE_RELOAD=true uv run python -m cube
```


### 3. Verify Server is Running

```bash
# Check health endpoint
curl http://localhost:8000/health

# View API docs (dev mode only)
# Open in browser: http://localhost:8000/docs
```

## Configuration

The server can be configured via environment variables or a `.env` file.

### Environment Variables

All configuration variables are prefixed with `CUBE_`:

| Variable | Default | Description |
|----------|---------|-------------|
| `CUBE_HOST` | `0.0.0.0` | Server host |
| `CUBE_PORT` | `8000` | Server port |
| `CUBE_RELOAD` | `false` | Auto-reload on code changes (dev only) |
| `CUBE_ENVIRONMENT` | `development` | Runtime environment (development/production) |
| `CUBE_LOG_LEVEL` | `INFO` | Logging level |
| `CUBE_SESSION_TIMEOUT_SECONDS` | `3600` | Session timeout (1 hour) |
| `CUBE_MAX_CONCURRENT_SESSIONS` | `100` | Max concurrent sessions |
| `CUBE_BENCHMARK_MODULE` | `None` | Benchmark module path |
| `CUBE_BENCHMARK_CLASS` | `None` | Benchmark class name |

### Example .env File

```bash
# Server settings
CUBE_HOST=0.0.0.0
CUBE_PORT=8000
CUBE_RELOAD=true
CUBE_ENVIRONMENT=development
CUBE_LOG_LEVEL=DEBUG

# Session settings
CUBE_SESSION_TIMEOUT_SECONDS=3600
CUBE_MAX_CONCURRENT_SESSIONS=50

# Benchmark configuration
CUBE_BENCHMARK_MODULE=examples.simple_math.benchmark
CUBE_BENCHMARK_CLASS=SimpleMathBenchmark
```

## API Endpoints

### Health Check

```bash
GET /health
```

Returns server health status.

**Response:**
```json
{
  "status": "ok",
  "version": "0.1.0",
  "environment": "development"
}
```

### Benchmark-Level APIs (Coming Soon)

- `GET /cube/info` - Get benchmark metadata
- `GET /cube/tasks` - List available tasks
- `POST /cube/spawn` - Spawn a task instance
- `GET /cube/status` - Get status of running tasks
- `POST /cube/shutdown` - Shutdown tasks

### Task-Level APIs (Coming Soon)

- `GET /sessions/{session_id}/tools/list` - List available tools
- `POST /sessions/{session_id}/tools/call` - Execute a tool
- `GET /sessions/{session_id}/resources/list` - List resources
- `GET /sessions/{session_id}/resources/read` - Read a resource
- `GET /sessions/{session_id}/cube/evaluation` - Get evaluation results
- `POST /sessions/{session_id}/cube/reset` - Reset task
- `POST /sessions/{session_id}/cube/close` - Close task

## Development

### Project Structure

```
src/cube/server/
├── __init__.py           # Server package
├── app.py                # FastAPI application ✅
├── config.py             # Configuration management ✅
├── schemas.py            # API request/response models ✅
├── middleware.py         # Error handling and CORS ✅
├── session.py            # Session management (TODO: Dev 2)
├── loader.py             # Benchmark loader (TODO: Dev 2)
├── mcp.py                # MCP protocol (TODO: Dev 3)
├── resources.py          # Resource system (TODO: Dev 3)
├── task_manager.py       # Task lifecycle (TODO: Dev 3)
└── routes/
    ├── __init__.py
    ├── benchmark.py      # Benchmark routes (TODO: Dev 2)
    └── task.py           # Task routes (TODO: Dev 3)
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/cube --cov-report=html

# Run specific test file
pytest tests/unit/test_server.py
```

### Code Quality

```bash
# Format code
make format

# Lint code
make lint
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Server                          │
│  ┌────────────────────────────────────────────────────────┐ │
│  │            Middleware (CORS, Logging, Errors)          │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Benchmark Routes          │   Task Routes             │ │
│  │  /cube/info               │   /sessions/.../tools/*   │ │
│  │  /cube/tasks              │   /sessions/.../cube/*    │ │
│  │  /cube/spawn              │   /sessions/.../resources/*│ │
│  │  /cube/status             │                            │ │
│  │  /cube/shutdown           │                            │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
┌───────▼────────┐  ┌──────▼────────┐  ┌───────▼────────┐
│ Session Manager│  │ Benchmark     │  │ Task Manager   │
│ (TODO: Dev 2) │  │ Loader        │  │ (TODO: Dev 3) │
│                │  │ (TODO: Dev 2) │  │                │
└────────────────┘  └───────────────┘  └────────────────┘
                            │
                    ┌───────▼───────┐
                    │   Benchmark   │
                    │   ┌────────┐  │
                    │   │  Task  │  │
                    │   └───┬────┘  │
                    │       │       │
                    │   ┌───▼────┐  │
                    │   │  Tool  │  │
                    │   └────────┘  │
                    └───────────────┘
```

## Troubleshooting

### Port Already in Use

If port 8000 is already in use:

```bash
# Use a different port
CUBE_PORT=8080 uv run python -m cube
```

### Import Errors

Make sure you've installed the package in editable mode:

```bash
uv pip install -e .
```

### Missing Dependencies

Re-run the install command:

```bash
make install
```

## Next Steps

1. **Dev 2**: Implement session management and benchmark APIs
   - Create `src/cube/server/session.py`
   - Create `src/cube/server/loader.py`
   - Create `src/cube/server/routes/benchmark.py`

2. **Dev 3**: Implement MCP protocol and task APIs
   - Create `src/cube/server/mcp.py`
   - Create `src/cube/server/resources.py`
   - Create `src/cube/server/task_manager.py`
   - Create `src/cube/server/routes/task.py`

3. **Dev 4**: Create example benchmark and tests
   - Create `examples/simple_math/` benchmark
   - Create comprehensive test suite

## Support

See the main [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) for the full project roadmap.
