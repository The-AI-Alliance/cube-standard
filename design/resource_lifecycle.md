# Resource Lifecycle Design

> High-level API proposal for managing benchmark resources across backends.
> Status: **draft v0.8 — for review**

---

## The Three Levels

```
Level 1 — Install-time    one-time image prep per backend/region  (slow, idempotent)
Level 2 — Benchmark-wide  shared server per benchmark run         (e.g. WebArena, WorkArena)
Level 3 — Task-level      per-task ephemeral resources            (e.g. individual VMs)
```

Levels 2 and 3 already exist in the codebase as `VMBackend.launch(VMConfig)`.
This document proposes the glue that connects all three, focused on Level 1
and the downstream user experience.

---

## Core Abstractions

```
ResourceSpec      WHAT the benchmark needs — declared by benchmark author, serializable
BackendConfig     HOW to provision it — declared by harness user, serializable
ResourceHandle    The live runtime object — not serializable, returned by launch()
ProvisionStore    (~/.cube/provisions.json) — maps (spec, backend) → provisioned image info
```

### ResourceSpec

Declared by the benchmark author. Describes a single resource dependency.

```python
class ResourceSpec(TypedBaseModel):
    name: str                           # "osworld-ubuntu-vm", "webarena-server"
    scope: Literal["task", "benchmark"] # task = per-task VM; benchmark = shared server
    max_concurrent_agents: int | None   # capacity hint for Level 2 (e.g. 10 for WebArena)
    source_url: str | None              # canonical image source (HuggingFace URL, etc.)
    source_hash: str | None             # for deduplication across benchmarks
    default_ttl_seconds: int = 3600     # time-to-live: max lifetime before auto-cleanup; 0 = no expiry
```

**Provisioning hints** for semi-manual or agent-assisted setup live as Markdown files
in the benchmark package, one per backend:

```
cube_osworld/
  resources/
    osworld-ubuntu-vm/
      azure.md      ← instructions specific to Azure (Golden Image Policy, etc.)
      aws.md        ← instructions specific to AWS
```

These files can contain code blocks, links, and multi-step procedures. A dedicated
agent skill is responsible for locating and surfacing the right file — this is not
part of the `ResourceSpec` API.

### BackendConfig

Declared by the harness user. Extends `TypedBaseModel` — pure Pydantic, no runtime state.
Credentials are never stored — resolved from env vars at runtime.

```python
class BackendConfig(TypedBaseModel):
    def fingerprint(self) -> str:
        """Stable key for ProvisionStore — e.g. "aws:us-east-2" or "azure:westus2"."""
        ...

# Examples
class AWSEC2BackendConfig(BackendConfig):
    region: str = "us-east-2"
    instance_type: str = "t3.xlarge"
    # credentials via AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY

    def fingerprint(self) -> str:
        return f"aws:{self.region}"

class AzureVMBackendConfig(BackendConfig):
    subscription: str
    location: str = "westus2"
    vm_size: str = "Standard_D4s_v3"
    # credentials via az login / AZURE_CLIENT_ID

    def fingerprint(self) -> str:
        return f"azure:{self.location}"
```

`fingerprint()` is the second half of the ProvisionStore key. It intentionally
excludes instance type / VM size — those are performance knobs, not identity.
Two configs with different sizes but same region resolve to the same image.

### ResourceHandle

Returned by `backend.launch(spec)`. Represents a live cloud resource.
Not serializable — closes on `__exit__`.

```python
class ResourceHandle:
    run_id:     str          # links this resource to a run (for cleanup)
    spec:       ResourceSpec
    backend:    BackendConfig
    endpoint:   str | None   # http://... if a service is exposed
    created_at: datetime
    expires_at: datetime | None

    def close(self) -> None:
        """Tear down this resource (delete VM, stop server, etc.)."""
        ...

    def __enter__(self) -> "ResourceHandle": ...
    def __exit__(self, *_) -> None: self.close()
```

Usage:
```python
with backend.launch(spec, run_id=run_id) as handle:
    result = agent.run(task, endpoint=handle.endpoint)
# VM is deleted on exit
```

### ProvisionStore

Maps `(spec.name, backend.fingerprint())` → provisioned image info.

```python
# Key format: "{spec_name}@{backend_fingerprint}"
# e.g.        "osworld-ubuntu-vm@aws:us-east-2"

class ProvisionStore:
    def get(self, spec: ResourceSpec, backend: BackendConfig) -> dict | None: ...
    def put(self, spec: ResourceSpec, backend: BackendConfig, info: dict) -> None: ...
    def list(self) -> list[tuple[str, dict]]: ...
```

File: `~/.cube/provisions.json` (v1 local). Provisions are written at most once
per `(spec, backend)` pair — read-modify-write over a small JSON file is fine.
Team/CI sharing deferred to v2 (`CUBE_PROVISION_STORE` env var → S3/GCS path).

---

## User-Facing API

### 1. Inspect what a benchmark needs

```python
specs = my_cube.list_resource_specs()
```

Returns a list of `ResourceSpec`, each annotated with:
- `.provision_status(backend_config)` → `"ready" | "needs_provisioning" | "unknown"`
- `.backends` — which `BackendConfig` types can serve this spec

### 2. Register (Level 1 — the core primitive)

```python
spec.register(backend_config, resource_info)
# e.g. resource_info = {"ami_id": "ami-xxx"} or {"gallery_image": "cube-osworld/1.0.0"}
```

`register()` is the **only thing `launch()` depends on**. It records that a resource
is available for a given (spec, backend) pair, whatever the provenance:
manually built, purchased from a Marketplace, provisioned by a teammate, or
produced by the automated `provision()` path below.

Calling `register()` with new info overrides the existing entry and logs a warning.

### 3. Launch (Levels 2 & 3)

```python
# Returns ResourceHandle; raises ResourceNotReadyError if register() wasn't called
resource = backend.launch(spec, run_id=run_id, ttl_seconds=3600)
```

`launch()` reads from the provision store, tags the new resource with `run_id` and
`expires_at`, then delegates to the backend. Fails fast if no entry is found:

```
ResourceNotReadyError: osworld-ubuntu-vm is not registered for aws:us-east-2.
  Run: spec.register(backend_config, {"ami_id": "..."})
  Or:  spec.provision(backend_config)    # automated, ~30 min, if supported
```

### 4. Provision (Level 1 — optional automation)

```python
spec.provision(backend_config)   # idempotent, ~30 min
```

`provision()` automates the full install path for supported (spec, backend) pairs:
download → convert → upload → import as cloud image → calls `register()`.

It is a **convenience wrapper around `register()`**, isolated in `cube.provisioners`.
`launch()` does not know or care whether the entry came from `provision()` or a manual
`register()`. As automation coverage improves, `provision()` reduces manual work —
but it must never be the only path, and its failures must not block the core flow.

---

## Resource Naming & Tagging

Every L2/L3 cloud resource (VM, NIC, IP, disk) is named and tagged consistently
so orphans can be identified and cleaned up by any harness instance.

**Name format:** `cube-{run_id_8}-{type}-{uid_6}`
- `run_id_8` — first 8 chars of the run ID (human-readable in portal)
- `type` — `vm`, `nic`, `ip`, `disk`
- `uid_6` — random hex for uniqueness within a run

```
cube-a3f9b21c-vm-44197d
cube-a3f9b21c-nic-44197d
cube-a3f9b21c-ip-44197d
```

**Required tags on every L2/L3 resource:**

| Tag              | Example                       | Purpose                        |
|------------------|-------------------------------|--------------------------------|
| `cube:run_id`    | `a3f9b21c-...`                | groups resources by run        |
| `cube:spec`      | `osworld-ubuntu-vm`           | which benchmark spec           |
| `cube:created_at`| `2026-03-26T20:04:07Z`        | for age-based cleanup          |
| `cube:expires_at`| `2026-03-26T21:04:07Z`        | TTL; absent = no auto-expiry   |
| `cube:backend`   | `azure:westus2`               | backend fingerprint            |

L1 resources (gallery images, AMIs, S3 buckets) are **not tagged for auto-cleanup** —
they are managed manually as they represent significant invested work.

---

## Orphan Cleanup

L2/L3 resources can outlive their harness process (crash, network loss, OOM kill).
The cleanup API handles this at two granularities:

```python
# List all live cube resources on this backend (L2 + L3 only)
backend.list_active(run_id: str | None = None) -> list[ResourceHandle]

# Delete all resources for a specific run
backend.cleanup(run_id: str) -> None

# Delete all cube resources older than max_age_seconds (regardless of run_id)
# Skips resources without cube:expires_at if max_age_seconds is not set
backend.cleanup_stale(max_age_seconds: int | None = None) -> list[str]
```

**Lifecycle hooks the harness should call:**

```python
# At harness startup — garbage-collect previous crashes
backend.cleanup_stale(max_age_seconds=7200)

# At harness shutdown (normal or signal handler)
backend.cleanup(run_id=run_id)
```

**TTL mechanics:** `cube:expires_at` is set at `launch()` time from
`ttl_seconds` (falls back to `spec.default_ttl_seconds`). `cleanup_stale()`
deletes any resource where `cube:expires_at < now`. Cloud providers do not
enforce TTL natively — deletion is driven by the harness.

**Escape hatch — manual cleanup:**

```python
# CLI: list and delete stale resources interactively
cube resources list --backend azure:westus2
cube resources cleanup --older-than 2h --dry-run
cube resources cleanup --older-than 2h
```

This is the safety net for cases where both the harness and any watchdog have
been dead for a long time (e.g. a dev laptop that was closed mid-run).

---

## Meta-benchmarks

A meta-benchmark's `list_resource_specs()` is the union of all sub-benchmarks'.
Each spec provisions independently. Risk: two benchmarks declare the same
underlying image under different names. Mitigation: `source_hash` lets the provision
store deduplicate — both names point to the same provisioned image.

---

## Resolved

1. **`backend.launch(spec)` vs `spec.launch(backend)`**
   Keeping `backend.launch(spec)` for consistency with the existing codebase.
   `spec.provision(backend)` for Level 1 — install-time feels resource-centric.

2. **Level 2 capacity tracking**: `ResourceHandle` does not expose
   `active_agents` / `available_slots`. The harness manages concurrency itself
   (e.g. via Ray worker count, a semaphore, or a task queue). Level 2 handles
   stay alive across tasks but carry no capacity state.

3. **BackendConfig migration**: `BackendConfig` must extend `TypedBaseModel`
   (already the pattern in `vm.py`). Current experiment code (`azure_backend.py`,
   `aws_backend.py`) uses `@dataclass` and must be migrated before integration.

4. **Public images**: a spec whose backend already has a public image available
   is implicitly pre-registered — no action needed. Calling `register()` with
   new info overrides the entry and logs a warning. No separate
   `spec.use_public_image()` verb needed.
