# cube-infra-daytona

Daytona `InfraConfig` implementation for the CUBE resource lifecycle.

This package provides `DaytonaInfraConfig`, which launches per-task sandboxes on
[Daytona](https://www.daytona.io/) — each `launch()` returns a live `DaytonaContainer`
backed by a freshly provisioned Daytona sandbox.

---

## Prerequisites

### Credentials

`DaytonaInfraConfig` reads Daytona credentials from environment variables at runtime —
never from fields on the config itself. Set:

```bash
export DAYTONA_API_KEY="<your-api-key>"
# Optional — the SDK defaults to the public API if unset:
export DAYTONA_API_URL="https://api.daytona.io/api"
export DAYTONA_TARGET="us"
```

See the [Daytona SDK docs](https://www.daytona.io/docs) for details.

---

## Usage

```python
from cube.resource import DockerServiceConfig
from cube_infra_daytona import DaytonaInfraConfig

infra = DaytonaInfraConfig(
    ephemeral=True,                # auto-delete when the sandbox stops
    auto_stop_minutes=30,          # idle stop
    launch_timeout_seconds=120,
)
resource = DockerServiceConfig(name="my-task", docker_images=["python:3.12-slim"])

container = infra.launch(resource)
result = container.exec("echo hello")
print(result.stdout)
container.close()
```

---

## Defaults

- **Resources:** 2 CPU / 4 GiB RAM / 10 GiB disk per sandbox.
  `DockerServiceConfig` doesn't carry per-launch CPU/RAM today — override by subclassing
  `DaytonaInfraConfig` and tweaking the `resources_kwargs` branch in `launch()` if needed.
- **Outbound network:** unrestricted. Cubes that run `test.sh` scripts which pull tools
  from the public internet (e.g. terminal-bench, swebench) need this.
- **TTL:** inherited from `resource.default_ttl_seconds`, overridable via
  `DaytonaInfraConfig(default_ttl_seconds=...)`.

---

## Notes

- `cleanup(run_id)` and `list_active()` are no-ops today — Daytona's API doesn't
  surface the labels we'd need for cross-process recovery. When durable cleanup
  matters (e.g. after a crash), rely on `ephemeral=True` + `auto_stop_minutes`
  as the backstop. Follow-up to adopt Daytona labels is tracked in the
  [`resource-convergence` proposal](../../openspec/changes/resource-convergence/proposal.md).
