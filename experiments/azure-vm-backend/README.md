# Azure + AWS VM Backend — Experiments

Validates the full CUBE VM backend pipeline (local and bootstrap paths) for Azure and AWS.

## Setup

```bash
cd experiments/azure-vm-backend
uv sync
az login --tenant "8bcff170-9979-491e-8683-d8ced0850bad" --use-device-code
```

## Modules

| File | Purpose |
|------|---------|
| `_common.py` | Shared utilities: `open_tunnel`, `wait_for_ssh`, `convert_image`, `BootstrapMonitor` |
| `azure_backend.py` | `AzureBackend` — full Azure pipeline (upload, gallery, launch, bootstrap) |
| `aws_backend.py` | `AWSBackend` — full AWS pipeline (S3, AMI, EC2, bootstrap) |
| `osworld.py` | OSWorld-specific wrappers + CLI (`python osworld.py create_resources --backend azure\|aws`) |

## Tests

```bash
# Bootstrap path: in-cloud VM downloads from HuggingFace, converts, uploads (~45-60 min, ~$0.06)
python test_bootstrap.py

# Local path: convert local qcow2 → upload → gallery/AMI → launch (~90-120 min)
python test_osworld_parallel.py
```

## Key findings

See `OBJECTIVE.md` for the full design, pipeline diagram, and documented gotchas.
