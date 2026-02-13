# CUBE Standard

Common Unified Benchmark Environments — a standard library for defining and running agent benchmarks with portable container infrastructure.

## Quick Reference

```bash
make install                    # Install deps
make lint                       # Ruff lint
make format                     # Ruff format
uv pip install -e ".[docker,daytona,modal]"  # Install all optional backends
```

## Architecture

Two independent subsystems:

1. **Core** (`base.py` → `core.py` → `tool.py` → `environment.py` → `benchmark.py`): Task/Tool/Environment abstractions for benchmarks.
2. **Container API** (`container.py` → `backends/`): Launch containers across Local Docker, Daytona, and Modal. Separates *what* to run (ContainerSpec) from *how* to run it (ContainerBackend).

## Key Patterns

- `TypedBaseModel` (in `base.py`) adds `_type` field for polymorphic serialization — all config classes inherit from it.
- `ContainerBackend` is serializable (for Ray), `Container` is not (holds live connections).
- `health_check: Callable` uses `Field(exclude=True)` + `ConfigDict(arbitrary_types_allowed=True)`.
- Container exec uses `sh` (not `bash`) for POSIX compatibility with Alpine images.
- Retry decorators from `tenacity` on all I/O and container operations.

## Testing

Integration test scripts in `scripts/` — real backends, no mocks:

```bash
PYTHONPATH=scripts uv run python scripts/test_local.py     # Docker daemon
PYTHONPATH=scripts uv run python scripts/test_daytona.py   # DAYTONA_API_KEY
PYTHONPATH=scripts uv run python scripts/test_modal.py     # Modal token
```

## Dependencies

- Core: `pydantic`, `litellm`, `pillow`
- Optional: `docker + tenacity` (local), `daytona + python-dotenv + tenacity` (daytona), `modal + tenacity` (modal)
- Dev: `ruff`
