# CUBE Resource Management — Core Objective

*Created: 2026-03-25*

---

## The Goal

Build a collection of **resource management tools** in `cube-standard` so that:

- **Benchmark authors** publish one thing: a VM image (qcow2/vmdk on HuggingFace) + a `VMConfig`.
- **Harness users** configure one thing: a `VMBackend` (local, Azure, AWS, GCP, Daytona…).
- **CUBE** handles everything in between — download, convert, upload, provision, tunnel, teardown.

```python
# Benchmark author defines (in their cube):
config = VMConfig(snapshot_name="osworld-ubuntu", cpu_cores=4, ram_gb=8, ...)

# Harness user configures once:
backend = AzureVMBackend(
    resource_group="my-rg",
    hf_qcow2="hf://xlangai/OSWORLD/ubuntu.vmdk",
)

# CUBE does the rest:
vm = backend.launch(config)          # ensure_resource() called automatically
obs, info = task.reset()             # task communicates with vm.endpoint
vm.restore_snapshot("init_state")    # between tasks: ~3-4 min on Azure
vm.stop()
```

---

## The API That Already Exists (do not change)

From `cube/vm.py`:

```python
class VMConfig(TypedBaseModel):
    snapshot_name: str = "init_state"   # maps to gallery image / AMI / GCS image
    cpu_cores: int = 4
    ram_gb: int = 4
    screen_size: tuple[int, int] = (1920, 1080)
    os_type: Literal["Ubuntu", "Windows"] = "Ubuntu"

class VMBackend(TypedBaseModel, ABC):
    def ensure_resource(self, config: VMConfig) -> None: ...   # idempotent, one-time
    def launch(self, config: VMConfig) -> VM: ...              # per eval, returns handle
    def close(self) -> None: ...

class VM(ABC):
    @property
    def endpoint(self) -> str: ...          # "http://localhost:{port}"
    def restore_snapshot(self, name: str) -> None: ...
    def stop(self) -> None: ...
```

`ComputerBase` (in `cube-computer-tool`) takes a `VM` and talks to `vm.endpoint` — it doesn't care which hypervisor is underneath.

---

## What vs How Separation

```
WHAT (benchmark author)          HOW (harness user)
───────────────────────          ──────────────────
VMConfig                         VMBackend
  snapshot_name: "init_state"      AzureVMBackend(hf_qcow2="hf://...", ...)
  cpu_cores: 4                     LocalDockerVMBackend(path_to_vm="...")
  ram_gb: 8                        LocalQEMUVMBackend(path_to_vm="...")
  os_type: "Ubuntu"                AWSVMBackend(...)
                                   GCPVMBackend(...)
```

The benchmark author publishes their image + VMConfig. The harness user picks a backend. Neither knows about the other.

---

## ensure_resource: The Hard Part

`ensure_resource(config)` is **idempotent one-time setup** — expensive but runs once per backend/subscription.

### General pipeline (cloud backends):

```
HuggingFace (or local path)
    ↓  download_image()          — hf_hub_download or direct URL
Local qcow2 / vmdk
    ↓  convert_to_vhd()          — qemu-img, ~5-10 min / 15 GB
Fixed-size VHD
    ↓  upload_image()            — cloud blob storage, ~20-40 min / 15 GB
Cloud blob
    ↓  import_as_disk()          — cloud-specific, ~8 min
Cloud managed disk
    ↓  publish_to_registry()     — gallery / AMI / GCS image, ~8-15 min
Cloud image registry             ← stored permanently, reused for all launches
```

### Per backend:

| Backend | Blob storage | Import | Registry |
|---|---|---|---|
| Azure | Blob Storage (PageBlob) | `createOption: Import` | Compute Gallery |
| AWS | S3 | `ec2 import-snapshot` | AMI |
| GCP | GCS | `gcloud compute images import` (accepts qcow2 directly) | Custom Image Family |

---

## launch: Per-Eval VM Spawning

`launch(config)` runs per eval (~2-4 min for cloud backends). Must:

1. Call `ensure_resource(config)` (no-op if already done)
2. Provision a VM from the cloud image registry
3. Inject SSH key (via `os_profile` on Azure, keypair on AWS, metadata on GCP)
4. Wait for SSH to be available
5. Open SSH tunnel: `localhost:{port} → vm:5000` (bypasses corporate proxies / Zscaler)
6. Wait for HTTP endpoint to be ready
7. Return `VM` handle with `endpoint = "http://localhost:{port}"`

### Guest agent notes

The VM needs an HTTP server on port 5000 that speaks the OSWorld protocol (27 endpoints).

- **OSWorld image**: server already installed; no injection needed (`inject_agent=False`)
- **Generic/test image**: inject a minimal Flask agent via cloud-init (`inject_agent=True`)
- **Production**: `cube-guest-agent` should be a pip-installable systemd service

---

## restore_snapshot: Between Tasks

For cloud backends: `stop() + launch()` — delete VM + re-provision from gallery.
- Azure: ~3-4 min
- AWS: ~3-5 min
- Local QEMU: ~30s (QMP savestate)
- Local Docker: ~30-60s (container restart)

This is the primary throughput bottleneck for multi-task evals. Mitigation: keep N+1 pre-warmed VMs.

---

## Strategy for New / Flaky Backends

Not every backend will work out of the box. The strategy:

1. **Implement the general pipeline** — `ensure_resource` / `launch` / `restore_snapshot` as composable functions (see `cube_azure_pipeline.py` as the template)
2. **Test and document gotchas** — for each backend, record what works and what doesn't in a `FINDINGS_<backend>.md`
3. **Accumulate implementation knowledge in a `.md`** — a structured guide that helps coding agents close the gap for backends not yet implemented or with edge cases

Example gotchas documented so far (Azure/ServiceNow):
- Golden Image Policy → must use Compute Gallery, not Marketplace images
- Zscaler proxy → SSH tunnel is mandatory on corp networks
- `write_files` owner timing → write to `/usr/local/bin/`, not user home
- Cloud-init YAML parse error → use `encoding: b64` for Python code

---

## Bootstrap VM Approach

A key finding from these experiments is that uploading large VM images from a local machine is
the primary bottleneck. Home broadband (~0.1–1 GB/min) makes a 50 GB image take hours.

**Solution**: spin up a cheap VM *inside* the target cloud, download the image from HuggingFace
at datacenter speed (~55–120 MB/s), convert, and upload to cloud storage (~200–500 Mb/s) — all
without any large local transfer.

```
Local machine                    Cloud (bootstrap VM)
─────────────                    ────────────────────
"start bootstrap" ──launch──→   t3.medium / Standard_B2ms
                                    ↓  wget HuggingFace (~3.5 min, 120 MB/s)
                                    ↓  unzip
                                    ↓  qemu-img convert → fixed VHD (~5-10 min)
                                    ↓  upload to S3/Blob (~10-20 min, datacenter speed)
                                    ↓  write sentinel object/blob
                                 VM terminates
poll sentinel ←──────────────── sentinel detected
    ↓  import snapshot / import disk  (~5-10 min)
    ↓  publish AMI / gallery image    (~5-10 min)
 Ready                                                Total: ~30-46 min, ~$0.02-$0.04
```

### Bootstrap gotchas

| Issue | Root cause | Fix |
|-------|-----------|-----|
| Azure Golden Image Policy | Applies to bootstrap VM too — Marketplace images blocked | Bootstrap VM uses our own gallery image (`cube-ubuntu-22-04/1.0.0`) |
| `mkfs.ext4` fails on data disk | udev/udevadm timing on newly-attached block device | Use large OS disk (128 GB) instead of attaching a separate data disk |
| Azure VHD footer corrupt | Local upload timed out, left blob at right size with zeroed footer | Validate `conectix` magic in last 512 bytes; delete + re-upload if invalid |
| AWS rejects sparse VMDK | `ec2 import-snapshot` requires fixed or stream-optimized VMDK; `monolithicSparse` unsupported | Use fixed VHD (`qemu-img convert -O vpc -o subformat=fixed,force_size`) |
| AWS CLI not on Ubuntu 22.04 | `aws` not installed by default | Install `boto3` via pip3; use Python for all S3 operations |
| HuggingFace URL is a zip | `Ubuntu.qcow2.zip`, not bare `.qcow2` | Detect zip with `file(1)`, unzip before conversion |

---

## Current State (2026-03-25)

### Validated (experiments/azure-vm-backend/)

**Azure (cube_azure_pipeline.py)**
- ✅ `ensure_resource`: local convert + upload + gallery (idempotent, skips existing steps)
- ✅ `bootstrap_ensure_resource`: in-cloud bootstrap from HuggingFace URL — tested end-to-end
- ✅ `launch`: gallery → VM + SSH tunnel + cloud-init agent injection
- ✅ `restore_snapshot`: stop + relaunch, ~3.5 min, all endpoints pass
- ✅ SSH tunnel bypasses Zscaler
- ✅ Golden Image Policy bypass via Compute Gallery
- ✅ VHD footer validation prevents corrupt re-upload
- ✅ OSWorld Ubuntu image: bootstrap → gallery → VM → probe passing

**AWS (aws_pipeline.py)**
- ✅ `ensure_resource`: local convert (sparse VMDK) + S3 upload + snapshot import + AMI
- ✅ `bootstrap_ensure_resource`: in-cloud bootstrap from HuggingFace URL — tested end-to-end
- ✅ `launch`: AMI → EC2 + SSH tunnel + user-data agent injection
- ✅ `restore_snapshot`: terminate + relaunch
- ✅ IAM instance profile for credential-free S3 writes from bootstrap EC2
- ✅ OSWorld Ubuntu image: bootstrap → AMI → EC2 → probe passing

**Test scripts**
- ✅ `test_osworld_parallel.py`: full OSWorld pipeline on Azure + AWS (local upload path)
- ✅ `test_bootstrap.py`: bootstrap approach on Azure + AWS from HuggingFace URL

### Not yet done
- ❌ `AzureVMBackend(VMBackend)` / `AWSVMBackend(VMBackend)` classes wired into cube-vm-backend
- ❌ HuggingFace `hf://` scheme in `ensure_resource` (currently takes raw HTTPS URL)
- ❌ Real OSWorld HTTP server tested end-to-end (27 endpoints)
- ❌ `GCPVMBackend`

---

## Next Steps

### Immediate (cube-vm-backend package)
1. `AzureVMBackend(VMBackend)` + `AzureVM(VM)` wrapping `cube_azure_pipeline.py`
2. `AWSVMBackend(VMBackend)` + `AWSVM(VM)` wrapping `aws_pipeline.py`
3. `ensure_resource` accepts `hf://` scheme → resolves to HTTPS bootstrap URL

### Short term
1. Test real OSWorld: `task.reset()` → `vm.endpoint` → full 27-endpoint protocol
2. Document OSWorld-specific `VMConfig` in `osworld-cube`

### Medium term
1. `GCPVMBackend` — same pattern (GCS + `gcloud compute images import`)
2. CI: benchmark pushes qcow2 → CUBE auto-bootstraps to all registries
3. Backend knowledge base (`.md`) for coding agents to implement new backends

---

## The Bigger Picture

The long-term goal is that implementing a new cloud backend is:
1. Copy `AzureVMBackend` as a template
2. Replace Azure SDK calls with target SDK calls (5-6 functions)
3. Document any gotchas in `FINDINGS_<backend>.md`
4. A coding agent can do step 2-3 with the template + findings doc as context

This makes CUBE genuinely backend-agnostic — benchmark authors publish once, researchers run anywhere.
