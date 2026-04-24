# cube-infra-toolkit

EAI Toolkit `InfraConfig` implementation for the CUBE resource lifecycle.

This package provides `ToolkitInfraConfig`, which launches per-task jobs on the
[EAI Toolkit](https://docs.console.elementai.com/) cluster — each `launch()` returns
a live `ToolkitContainer` backed by a freshly submitted EAI job.

---

## Prerequisites

### EAI CLI

The `eai` CLI must be installed and authenticated on the host that calls `launch()`.
Follow the [EAI Toolkit install guide](https://docs.console.elementai.com/) to install
the CLI and run `eai login` at least once.

### Credentials

Credentials come from `~/.eai/config` (written by `eai login`) or `EAI_PROFILE` env
var — never from fields on `ToolkitInfraConfig` (which is serialized across process
boundaries). Set the profile explicitly if you have more than one:

```bash
export EAI_PROFILE="<profile-name>"
# Optional: pin a specific account (org):
#   ToolkitInfraConfig(account="<account-name>")
```

---

## Usage

```python
from cube.resource import DockerServiceConfig
from cube_infra_toolkit import ToolkitInfraConfig

infra = ToolkitInfraConfig(
    exec_mode="exec_relay",         # HTTP exec relay (default); use "direct" to bypass
    preemptable=True,
    launch_timeout_seconds=600,
)
resource = DockerServiceConfig(name="my-task", docker_images=["python:3.12-slim"])

container = infra.launch(resource)
result = container.exec("echo hello")
print(result.stdout)
container.close()
```

---

## Exec relay

The default `exec_mode="exec_relay"` starts a lightweight HTTP server inside the job
as part of its startup command, then tunnels it via `eai job port-forward`. All
subsequent `.exec()` calls go through the relay instead of `eai job exec`, which
avoids a CLOSE_WAIT hang bug seen on ~6% of `eai job exec` invocations. Images
without `python3` automatically fall back to a bootstrap-via-apt slow path, and
then to direct `eai job exec` if bootstrapping fails.

See [`docs/toolkit-exec-relay-design.md`](../../docs/toolkit-exec-relay-design.md)
for the full design, security model, and failure-mode analysis.

---

## Defaults

- **Resources:** 2 CPU / 4 GiB RAM per job.
- **Launch timeout:** 600 s (tunable via `launch_timeout_seconds`).
- **Retries:** `eai job new` is not idempotent; we use `retries=0` and rely on loud
  logs + manual cleanup for the narrow window where a job may be created but the
  client never learns its ID.

---

## Notes

- `cleanup(run_id)` and `list_active()` are no-ops today — we don't tag EAI jobs
  with `run_id` yet. On crash, orphan jobs can be found via `eai job ls --mine`
  matching the submit timestamps in the log.
