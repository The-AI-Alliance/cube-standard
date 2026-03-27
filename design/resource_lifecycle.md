# Resource Lifecycle Design

> High-level API proposal for managing benchmark resources across backends.
> Status: **draft v0.9 — for review**

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
      local.md      ← instructions for local QEMU setup
```

These files can contain code blocks, links, and multi-step procedures. A dedicated
agent skill is responsible for locating and surfacing the right file — this is not
part of the `ResourceSpec` API.

### BackendConfig

Declared by the harness user. Extends `TypedBaseModel` — pure Pydantic, no runtime state.
Credentials are never stored — resolved from env vars at runtime.
A `_type` discriminator field enables polymorphic deserialization from JSON
(same pattern as `core.py`), so users can serialize their backend config and pass it
to a harness without code changes. Custom backends follow the same pattern by
subclassing `BackendConfig` and providing a unique `_type`.

```python
class BackendConfig(TypedBaseModel):
    _type: str  # discriminator for polymorphic deserialization

    def fingerprint(self) -> str:
        """Stable key for ProvisionStore.

        Must encode only the properties that determine WHICH image is needed —
        i.e. provider and region/location. Must NOT encode performance knobs
        (instance size, CPU count, memory) because those don't affect the image.

        Two configs with the same fingerprint share the same provisioned image.
        """
        ...

# Examples
class LocalBackendConfig(BackendConfig):
    _type: str = "local"
    # No credentials needed

    def fingerprint(self) -> str:
        return "local"

class DockerBackendConfig(BackendConfig):
    _type: str = "docker"
    registry: str = "docker.io"   # registry where images are pulled from
    # credentials via DOCKER_USERNAME / DOCKER_PASSWORD or docker login

    def fingerprint(self) -> str:
        return f"docker:{self.registry}"

class AWSEC2BackendConfig(BackendConfig):
    _type: str = "aws"
    region: str = "us-east-2"
    instance_type: str = "t3.xlarge"
    # credentials via AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY

    def fingerprint(self) -> str:
        return f"aws:{self.region}"

class AzureVMBackendConfig(BackendConfig):
    _type: str = "azure"
    subscription: str
    location: str = "westus2"
    vm_size: str = "Standard_D4s_v3"
    # credentials via az login / AZURE_CLIENT_ID

    def fingerprint(self) -> str:
        return f"azure:{self.location}"
```

**Custom backends:** a user can bring their own backend (e.g. Kubernetes, private cloud)
by subclassing `BackendConfig` with a unique `_type` and implementing the abstract
backend methods. The config is still serializable to JSON via Pydantic.

### ResourceHandle

Returned by `backend.launch(spec)`. Represents a live cloud resource.
Not serializable — holds live state (tunnel subprocess, cloud client, etc.).

**`close()` is the primary API.** The context manager is a convenience wrapper on top of it
and should be used when the resource lifetime fits within a single process/thread scope.
For multi-process use cases (e.g. Ray workers), the handle is not passed across process
boundaries — instead `run_id` (a plain string) is passed, and any process can call
`backend.cleanup(run_id)` to tear down all resources associated with that run.

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

Usage — single process (context manager):
```python
with backend.launch(spec, run_id=run_id) as handle:
    result = agent.run(task, endpoint=handle.endpoint)
# VM is deleted on exit
```

Usage — multi-process (manual):
```python
# Coordinator process
handle = backend.launch(spec, run_id=run_id)
dispatch_to_workers(handle.endpoint, run_id)  # run_id is serializable

# Any process (including coordinator) at shutdown
backend.cleanup(run_id=run_id)
```

### ResourceSpec vs resource_info

These two concepts are easy to confuse but serve completely different purposes:

| | `ResourceSpec` | `resource_info` |
|---|---|---|
| **Owner** | benchmark author | harness user (or `provision()`) |
| **Where it lives** | benchmark package, source control | `~/.cube/provisions.json` |
| **Cloud-agnostic?** | yes | no — backend-specific |
| **Stable?** | yes, changes only when benchmark changes | changes when image is rebuilt |
| **Content** | what is needed ("OSWorld Ubuntu VM") | where it is ("ami-0abc123 in us-east-2") |

`ResourceSpec` is a static declaration of requirements. `resource_info` is the
runtime answer to "where did you actually put the image for this backend?"

**`register()` is the handoff point.** It does not matter how the image was created —
manually, by `provision()`, by a teammate, or from a Marketplace — once `register()`
is called the harness can launch it. `provision()` is just one path that calls
`register()` internally at the end.

### resource_info

An opaque dict read by `launch()` to locate the provisioned image. Each backend
defines what fields it needs; the ProvisionStore treats it as a blob and only the
backend interprets it. It is created either by `provision()` (automatically) or by
the user calling `register()` manually with backend-specific values.

Examples by backend:

```python
# AWS — AMI in a specific region
{"ami_id": "ami-0abc123def456"}

# Azure — gallery image name + version
{"gallery_image": "cube-osworld", "version": "1.0.0"}

# Docker — image reference
{"image": "happysixd/osworld-docker:latest"}

# Local — path to a local qcow2 or similar
{"image_path": "/data/Ubuntu.qcow2"}
```

Backend implementations may type `resource_info` via a subclass for internal use,
but the store and the `register()` / `get()` interface always use `dict` to keep the
ProvisionStore backend-agnostic.

### ProvisionStore

Maps `(spec.name, backend.fingerprint())` → `resource_info`.

```python
# Key format: "{spec_name}@{backend_fingerprint}"
# e.g.        "osworld-ubuntu-vm@aws:us-east-2"
#             "osworld-ubuntu-vm@docker:docker.io"
#             "osworld-ubuntu-vm@local"

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

If no backend is provided, `LocalBackend` is used by default (see below).

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

### 5. Debug agent (smoke test)

```python
my_cube.run_debug_agent(backend_config)
```

Runs a quick end-to-end smoke test against a given backend before committing to a
full evaluation run. It performs three checks in order:

1. **Resource availability** — queries the ProvisionStore. If a spec is not registered,
   it prints actionable instructions (`register()` or `provision()`) and stops.
2. **Backend compatibility** — attempts to launch a single task-scoped VM. If the
   backend cannot satisfy the resource (wrong region, policy block, quota exceeded),
   it reports the specific challenge and stops.
3. **Functional validation** — runs a minimal debug task (screenshot + execute).
   If this passes, the backend is considered ready for full evaluation.

A clean run guarantees that `my_cube.run(backend_config)` will not fail due to
infrastructure issues. Intended to be run once per (cube, backend) pair before
a batch evaluation.

---

## LocalBackend — Default Backend

`LocalBackend` is a first-class backend that runs resources locally — either via
QEMU (for VM-based benchmarks) or Docker (for container-based benchmarks). It requires
no cloud credentials and is the default when no backend is provided.

```python
# Explicit
backend = LocalBackendConfig()
backend.launch(spec, run_id=run_id)

# Implicit default — no backend argument
my_cube.run(tasks)  # uses LocalBackend
```

`LocalBackend` provisions images to a local path (configured via `CUBE_LOCAL_IMAGE_DIR`,
defaults to `~/.cube/images/`). `provision()` on `LocalBackend` downloads and converts
the image locally — no upload step.

The `resource_info` for `LocalBackend` is `{"image_path": "/path/to/image.qcow2"}`.
For Docker resources the image is pulled on demand; `register()` is a no-op if
`image` is a public Docker Hub reference.

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
Cloud providers do not enforce TTL natively — deletion is driven entirely by the harness.
`cube:expires_at` is set at `launch()` time from `ttl_seconds` (falls back to
`spec.default_ttl_seconds`). `cleanup_stale()` deletes any resource where
`cube:expires_at < now`. This harness-side approach is more reliable than cloud-native
TTL because it works uniformly across all backends (including `LocalBackend`) and is
not subject to provider-specific limitations.

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

---

## CLI

Resource management is exposed as a `cube resource` subgroup of the existing cube CLI —
a thin layer over the Python API with no additional logic.

```
cube resource list   [--backend <fingerprint>]          # show all registered specs
cube resource status [--backend <fingerprint>]          # provision_status per spec
cube resource register <spec> <backend> <info-json>     # manual register()
cube resource provision <spec> <backend>                # automated provision()
cube resource active  [--backend <fingerprint>]         # list live L2/L3 resources
cube resource cleanup --run-id <id>                     # delete a specific run
cube resource cleanup --older-than <duration> [--dry-run]  # age-based cleanup
```

All commands accept `--backend` as either a fingerprint string (`aws:us-east-2`) or a
path to a JSON file containing a serialized `BackendConfig`. The `--dry-run` flag on
cleanup prints what would be deleted without acting.

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

5. **Docker resources**: `DockerBackendConfig` and `LocalBackendConfig` are
   first-class backends. The `resource_info` schema differs per backend but the
   store and the `register()` / `launch()` interface are backend-agnostic.
   Existing Docker resources will be migrated to this schema in a follow-up PR.

---

## Deferred

- **In-cloud provisioning shortcut**: if the harness process is itself running inside
  the target cloud (e.g. an EC2 instance launching an AWS bootstrap), it could skip
  the bootstrap VM entirely and convert the image locally, then upload at datacenter
  speed. Detected via the instance metadata endpoint. No code changes needed now —
  `provision()` on a cloud backend should check for this and optimize accordingly.

- **ProvisionStore v2**: team/CI sharing via `CUBE_PROVISION_STORE` env var pointing
  to an S3/GCS path. v1 is local only (`~/.cube/provisions.json`).
