# Azure VM Backend — Experiments

Exploring the automated VM creation pipeline for CUBE.

## Setup

```bash
cd experiments/azure-vm-backend
uv venv && uv pip install -r pyproject.toml
az login  # select ServiceNow AI Research subscription
```

## Resource Tracking

Every Azure resource we create is:
1. Tagged `project=cube-experiment` in Azure
2. Recorded in `resources.json`

```bash
# List tracked resources (+ Azure query by tag)
python track.py list

# Delete everything we created
python track.py delete
```

## Experiments

### 1. List existing managed images
```bash
python azure_backend.py list-images
```

### 2. Launch a VM from an existing image
```bash
python azure_backend.py launch --image webarena-jeph-image-20250903-2
```

### 3. Stop/delete a VM
```bash
python azure_backend.py stop --vm cube-exp-abc123
```

## Options Under Exploration

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| A | Launch from existing managed image | Fast (image already there) | Manual one-time setup |
| B | qcow2 → VHD → import as managed image | Fully automated | Azure boot issues with raw qcow2 |
| C | Docker container on Azure Container Instances | Simplest | No GPU, container-only |
| D | Azure VM Compute Gallery (Shared Image Gallery) | Multi-region, versioned | More setup |
