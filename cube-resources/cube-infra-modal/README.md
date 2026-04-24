# cube-infra-modal

Modal `InfraConfig` implementation for the CUBE resource lifecycle.

This package provides `ModalInfraConfig`, which launches per-task
[Modal Sandboxes](https://modal.com/docs/guide/sandbox) — each `launch()` returns
a live `ModalContainer` backed by a freshly created Modal Sandbox.

---

## Prerequisites

### Credentials

Modal reads credentials from `~/.modal.toml` (written by `modal setup`) or from
`MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` env vars. `ModalInfraConfig` never stores
credentials in its own fields. Run once:

```bash
pip install modal
modal setup
```

See [Modal's auth docs](https://modal.com/docs/guide/auth) for details.

---

## Usage

```python
from cube.resource import DockerServiceConfig
from cube_infra_modal import ModalInfraConfig

infra = ModalInfraConfig(
    app_name="cube-eval",
    launch_timeout_seconds=300,
)
resource = DockerServiceConfig(name="my-task", docker_images=["python:3.12-slim"])

container = infra.launch(resource)
result = container.exec("echo hello")
print(result.stdout)
container.close()
```

---

## Defaults

- **Resources:** inherits the Modal Sandbox defaults; tune via the `ModalInfraConfig`
  fields that map to `modal.Sandbox.create` parameters.
- **TTL:** inherited from `resource.default_ttl_seconds`, overridable via
  `ModalInfraConfig(default_ttl_seconds=...)`.

---

## Notes

- `cleanup(run_id)` and `list_active()` are no-ops today — we don't tag Modal
  Sandboxes with `run_id` yet. Sandboxes self-destruct when the Python process
  exits, so local crashes don't leak; a `launch()` that crashes *after* sandbox
  creation but before the client returns is the narrow leak window.
