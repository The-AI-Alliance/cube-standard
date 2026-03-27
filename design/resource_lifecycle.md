# Resource Lifecycle Design

> High-level API proposal for managing benchmark resources across backends.
> Status: **draft v1.0 — for review**

---

## The Three Levels

```
Level 1 — Install-time    one-time image prep per infra/region    (slow, idempotent)
Level 2 — Benchmark-wide  shared server per benchmark run         (e.g. WebArena, WorkArena)
Level 3 — Task-level      per-task ephemeral resources            (e.g. individual VMs)
```

Levels 2 and 3 already exist in the codebase as `VMBackend.launch(VMConfig)`.
This document proposes the glue that connects all three, focused on Level 1
and the downstream user experience.

---

## Core Abstractions

```
ResourceConfig    WHAT the benchmark needs — declared by benchmark author, serializable
InfraConfig       HOW to provision it — declared by harness user, serializable + executable
ResourceHandle    The live runtime object — not serializable, returned by launch()
ProvisionStore    (~/.cube/provisions.json) — maps (resource, infra) → resource_info
```

### ResourceConfig

Declared by the benchmark author. Describes a single resource dependency.

```python
class ResourceConfig(TypedBaseModel):
    name: str                           # "osworld-ubuntu-vm", "webarena-server"
    scope: Literal["task", "benchmark"] # task = per-task VM; benchmark = shared server
    max_concurrent_agents: int | None   # capacity hint for Level 2 (e.g. 10 for WebArena)
    source_url: str | None              # canonical image source (HuggingFace URL, etc.)
    source_hash: str | None             # content hash; informational only
    default_ttl_seconds: int | None = 3600  # max lifetime before auto-cleanup; None = no expiry

    # ── Capability requirements (checked by infra before provisioning) ──────────
    def requirements(self) -> set[str]:
        """Declare what the infra must support to run this resource.

        The infra checks these against its own capabilities() and fails fast
        with a helpful message before any cloud API call.

        Standard capability tokens: "kvm", "docker", "gpu:nvidia", "network:egress"
        """
        return set()

    # ── Script injection (executed on the bootstrap VM, not in harness) ─────────
    bootstrap_script_extra: str | None = None
    """Optional bash fragment appended to the infra's bootstrap script.

    This is the escape hatch for benchmark-specific VM setup that the generic
    infra cannot know about (e.g. installing a custom service, patching the image).
    The fragment runs as root on the ephemeral bootstrap VM.

    Security: this field must be declared in the benchmark's source code — never
    fetched from an external URL at runtime. The trust boundary is the installed
    benchmark package, not this field value. Infra backends should refuse to
    execute bootstrap_script_extra from dynamically loaded or unsigned configs.
    """
```

**ResourceConfig subclasses by infrastructure category.** The infra backend dispatches
on the subtype and raises `UnsupportedResourceType` for categories it does not support:

```python
class VMResourceConfig(ResourceConfig):
    """QEMU-based VM (OSWorld, WindowsAgentArena, macOSWorld, AndroidWorld...)."""
    requires_kvm: bool = True
    def requirements(self) -> set[str]:
        return {"kvm"} if self.requires_kvm else set()

class DockerServiceConfig(ResourceConfig):
    """Multi-container Docker Compose stack (WebArena, WorkArena, TheAgentCompany...)."""
    compose_url: str
    def requirements(self) -> set[str]:
        return {"docker"}

class DockerImageConfig(ResourceConfig):
    """Single Docker image per task (SWE-bench, MLE-bench, CTF...)."""
    image: str
    def requirements(self) -> set[str]:
        return {"docker"}

```

**Provisioning hints** for semi-manual or agent-assisted setup live as Markdown files
in the benchmark package, one per infra provider:

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
part of the `ResourceConfig` API.

### InfraConfig

Declared by the harness user. Extends `TypedBaseModel` — fields are pure Pydantic config,
no runtime state. Credentials are never stored — resolved from env vars at runtime.
Polymorphic deserialization is handled by `TypedBaseModel`: the base class automatically
injects the fully-qualified class name as `_type` on serialization and resolves it back
on deserialization. Subclasses declare no `_type` field — it is provided for free.

`InfraConfig` is both the config **and** the executor: it extends `TypedBaseModel` for
serializability but also carries the `launch()`, `cleanup()`, and `list_active()` methods.
This follows the existing `VMBackend` pattern in `vm.py` where the backend is a Pydantic
model that is instantiated and then called directly:

```python
infra = AWSInfraConfig(region="us-east-2", instance_type="t3.xlarge")
handle = infra.launch(resource_config, run_id=run_id)
```

```python
class InfraConfig(TypedBaseModel):
    def fingerprint(self) -> str:
        """Stable key for ProvisionStore.

        Must encode only the properties that determine WHICH image is needed —
        i.e. provider and region/location. Must NOT encode performance knobs
        (instance size, CPU count, memory) because those don't affect the image.

        Two configs with the same fingerprint share the same provisioned image.
        """
        ...

    def capabilities(self) -> set[str]:
        """Declare what this infra supports.

        Checked against resource.requirements() before provisioning or launch.
        Fails fast with a helpful message if requirements are not met —
        e.g. "This resource requires KVM but LocalInfraConfig does not support it."

        Standard tokens: "kvm", "docker", "gpu:nvidia", "network:egress"
        """
        ...

    def provision(self, resource: ResourceConfig) -> None: ...  # Level 1
    def launch(self, resource: ResourceConfig, run_id: str, ttl_seconds: int | None = None) -> ResourceHandle: ...
    def list_active(self, run_id: str | None = None) -> list[ResourceHandle]: ...
    def cleanup(self, run_id: str) -> None: ...
    def cleanup_stale(self, max_age_seconds: int | None = None) -> list[str]: ...

# Examples
class LocalInfraConfig(InfraConfig):
    # No credentials needed
    def fingerprint(self) -> str:
        return "local"

class DockerInfraConfig(InfraConfig):
    registry: str = "docker.io"
    # credentials via DOCKER_USERNAME / DOCKER_PASSWORD or docker login
    def fingerprint(self) -> str:
        return f"docker:{self.registry}"

class AWSInfraConfig(InfraConfig):
    region: str = "us-east-2"
    instance_type: str = "t3.xlarge"
    # credentials via AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
    def fingerprint(self) -> str:
        return f"aws:{self.region}"

class AzureInfraConfig(InfraConfig):
    subscription: str
    location: str = "westus2"
    vm_size: str = "Standard_D4s_v3"
    # credentials via az login / AZURE_CLIENT_ID
    def fingerprint(self) -> str:
        return f"azure:{self.location}"
```

**Custom infra providers:** subclass `InfraConfig`, implement the abstract methods, and
serialize to JSON via `TypedBaseModel` with no extra boilerplate. This is the extension
point for private clouds, Kubernetes clusters, or any custom provisioning API. No plugin
registration is needed — just import and pass the config directly:

```python
from my_org.infra import KubernetesInfraConfig

infra = KubernetesInfraConfig(cluster="prod-us-east")
run_debug_agent(my_cube, infra)
my_cube.run(infra)
```

### ResourceHandle

Returned by `infra.launch(resource)`. Represents a live cloud resource.
Not serializable — holds live state (tunnel subprocess, cloud client, etc.).

**`close()` is the primary API.** The context manager is a convenience wrapper on top of it
and should be used when the resource lifetime fits within a single process/thread scope.
For multi-process use cases (e.g. Ray workers), the handle is not passed across process
boundaries — instead `run_id` (a plain string) is passed, and any process can call
`infra.cleanup(run_id)` to tear down all resources associated with that run.

```python
class ResourceHandle:
    run_id:     str           # links this resource to a run (for cleanup)
    resource:   ResourceConfig
    infra:      InfraConfig
    endpoint:   str | None    # http://... if a service is exposed
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
with infra.launch(resource, run_id=run_id) as handle:
    result = agent.run(task, endpoint=handle.endpoint)
# VM is deleted on exit
```

Usage — multi-process (manual):
```python
# Coordinator process
handle = infra.launch(resource, run_id=run_id)
dispatch_to_workers(handle.endpoint, run_id)  # run_id is serializable

# Any process (including coordinator) at shutdown
infra.cleanup(run_id=run_id)
```

### ResourceConfig vs resource_info

`ResourceConfig` is owned by the *benchmark author* — static, source-controlled,
infra-agnostic ("I need an OSWorld Ubuntu VM"). `resource_info` is owned by the
*harness user* — dynamic, local, infra-specific ("on aws:us-east-2, that is
`ami-0abc123`"). **`register()` is the handoff point** between them: it does not
matter whether the image was built by `provision()`, a teammate, or a Marketplace
— once registered, the harness can launch it.

### resource_info

An opaque dict read by `launch()` to locate the provisioned image. Each infra provider
defines what fields it needs; the ProvisionStore treats it as a blob and only the
provider interprets it. It is created either by `provision()` (automatically) or by
the user calling `register()` manually with provider-specific values.

Examples by provider:

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

Provider implementations may type `resource_info` via a subclass for internal use,
but the store and the `register()` / `get()` interface always use `dict` to keep the
ProvisionStore provider-agnostic.

### ProvisionStore

Maps `(resource.name, infra.fingerprint())` → `resource_info`.

```python
# Key format: "{resource_name}@{infra_fingerprint}"
# e.g.        "osworld-ubuntu-vm@aws:us-east-2"
#             "osworld-ubuntu-vm@docker:docker.io"
#             "osworld-ubuntu-vm@local"

class ProvisionStore:
    def get(self, resource: ResourceConfig, infra: InfraConfig) -> dict | None: ...
    def put(self, resource: ResourceConfig, infra: InfraConfig, resource_info: dict) -> None: ...
    def list(self) -> list[tuple[str, dict]]: ...
```

File: `~/.cube/provisions.json` (v1 local). Provisions are written at most once
per `(resource, infra)` pair — read-modify-write over a small JSON file is fine.
Team/CI sharing deferred to v2 (`CUBE_PROVISION_STORE` env var → S3/GCS path).

---

## User-Facing API

### 1. Inspect what a benchmark needs

```python
resources = my_cube.list_resources()
```

Returns a list of `ResourceConfig`. To query provision state, call into `infra`:

```python
infra.provision_status(resource)  # → "ready" | "needs_provisioning" | "unknown"
infra.can_serve(resource)         # → bool — checks capabilities() vs requirements()
```

`ResourceConfig` stays a pure data object — store and infra queries live on `InfraConfig`.

### 2. Register (Level 1 — the core primitive)

```python
infra.register(resource, resource_info)
# e.g. resource_info = {"ami_id": "ami-xxx"} or {"gallery_image": "cube-osworld/1.0.0"}
```

`register()` is the **only thing `launch()` depends on**. It records that a resource
is available for a given (resource, infra) pair, whatever the provenance:
manually built, purchased from a Marketplace, provisioned by a teammate, or
produced by the automated `provision()` path below.

`register()` lives on `InfraConfig`, not on `ResourceConfig` — the infra owns the
store interaction; the resource stays a pure data object.

Calling `register()` with new info overrides the existing entry and logs a warning.

### 3. Launch (Levels 2 & 3)

```python
# Returns ResourceHandle; raises ResourceNotReadyError if register() wasn't called
handle = infra.launch(resource, run_id=run_id, ttl_seconds=3600)
```

`launch()` reads from the provision store, tags the new resource with `run_id` and
`expires_at`, then provisions and returns the handle. Fails fast if no entry is found:

```
ResourceNotReadyError: osworld-ubuntu-vm is not registered for aws:us-east-2.
  Run: infra.register(resource, {"ami_id": "..."})
  Or:  infra.provision(resource)    # automated, ~30 min, if supported
```

If no infra is provided, `LocalInfraConfig` is used by default (see below).

### 4. Provision (Level 1 — optional automation)

```python
resource.provision(infra)   # idempotent, ~30 min
```

`provision()` automates the full install path for supported (resource, infra) pairs:
download → convert → upload → import as cloud image → calls `register()`.

It is a **convenience wrapper around `register()`**, isolated in `cube.provisioners`.
`launch()` does not know or care whether the entry came from `provision()` or a manual
`register()`. As automation coverage improves, `provision()` reduces manual work —
but it must never be the only path, and its failures must not block the core flow.

### 5. Debug agent (smoke test)

```python
run_debug_agent(my_cube, infra)
```

Runs a quick end-to-end smoke test against a given infra before committing to a
full evaluation run. It performs three checks in order:

1. **Resource availability** — queries the ProvisionStore. If a resource is not registered,
   it prints actionable instructions (`register()` or `provision()`) and stops.
2. **Infra compatibility** — attempts to launch a single task-scoped VM. If the
   infra cannot satisfy the resource (wrong region, policy block, quota exceeded),
   it reports the specific challenge and stops.
3. **Functional validation** — runs a minimal debug task (screenshot + execute).
   If this passes, the infra is considered ready for full evaluation.

A clean run guarantees that `my_cube.run(infra)` will not fail due to infrastructure
issues. Intended to be run once per (cube, infra) pair before a batch evaluation.
`run_debug_agent` is a standalone function, not a method on the cube, since it requires
both a cube and an infra as equal inputs.

---

## LocalInfraConfig — Default

`LocalInfraConfig` is a first-class provider that runs resources locally — either via
QEMU (for VM-based benchmarks) or Docker (for container-based benchmarks). It requires
no cloud credentials and is the default when no infra is provided.

```python
# Explicit
infra = LocalInfraConfig()
handle = infra.launch(resource, run_id=run_id)

# Implicit default — no infra argument
my_cube.run(tasks)  # uses LocalInfraConfig
```

`LocalInfraConfig` provisions images to a local path (configured via `CUBE_LOCAL_IMAGE_DIR`,
defaults to `~/.cube/images/`). `provision()` on `LocalInfraConfig` downloads and converts
the image locally — no upload step.

The `resource_info` for `LocalInfraConfig` is `{"image_path": "/path/to/image.qcow2"}`.
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
| `cube:resource`  | `osworld-ubuntu-vm`           | which resource config          |
| `cube:created_at`| `2026-03-26T20:04:07Z`        | for age-based cleanup          |
| `cube:expires_at`| `2026-03-26T21:04:07Z`        | TTL; absent = no auto-expiry   |
| `cube:infra`     | `azure:westus2`               | infra fingerprint              |

L1 resources (gallery images, AMIs, S3 buckets) are **not tagged for auto-cleanup** —
they are managed manually as they represent significant invested work.

---

## Orphan Cleanup

L2/L3 resources can outlive their harness process (crash, network loss, OOM kill).
Cloud providers do not enforce TTL natively — deletion is driven entirely by the harness.
`cube:expires_at` is set at `launch()` time from `ttl_seconds` (falls back to
`resource.default_ttl_seconds`). `cleanup_stale()` deletes any resource where
`cube:expires_at < now`. This harness-side approach is more reliable than cloud-native
TTL because it works uniformly across all providers (including `LocalInfraConfig`) and is
not subject to provider-specific limitations.

```python
# List all live cube resources on this infra (L2 + L3 only)
infra.list_active(run_id: str | None = None) -> list[ResourceHandle]

# Delete all resources for a specific run
infra.cleanup(run_id: str) -> None

# Delete all cube resources older than max_age_seconds (regardless of run_id)
# Skips resources without cube:expires_at if max_age_seconds is not set
infra.cleanup_stale(max_age_seconds: int | None = None) -> list[str]
```

**Lifecycle hooks the harness should call:**

```python
# At harness startup — garbage-collect previous crashes
infra.cleanup_stale(max_age_seconds=7200)

# At harness shutdown (normal or signal handler)
infra.cleanup(run_id=run_id)
```

**L2 vs L3 teardown:** the API is identical — `handle.close()` or `infra.cleanup(run_id)`.
The difference is when the harness calls it:
- **L3** (`scope="task"`): close after each task completes.
- **L2** (`scope="benchmark"`): close once at run end, after all tasks are complete.

The cube and infra expose no special L2 protocol — teardown timing is entirely the
harness's responsibility.

---

## CLI

Resource management is exposed as a `cube resource` subgroup of the existing cube CLI —
a thin layer over the Python API with no additional logic.

The ProvisionStore is keyed by `resource.name` — resources are not owned by a single cube,
so most commands are cube-agnostic. `--cube` is an optional filter for discovery and
status commands; `register` and `provision` take a resource name directly.

```
cube resource list     [--cube <name>] [--infra <fingerprint>]     # registered resources
cube resource status   [--cube <name>] [--infra <fingerprint>]     # provision_status per resource
cube resource register <resource-name> <infra> <resource-info-json> # manual register()
cube resource provision <resource-name> <infra>                    # automated provision()
cube resource active   [--infra <fingerprint>]                     # live L2/L3 resources
cube resource cleanup  --run-id <id>                               # delete a specific run
cube resource cleanup  --older-than <duration> [--dry-run]         # age-based cleanup
```

`--infra` accepts either a fingerprint string (`aws:us-east-2`) or a path to a
serialized `InfraConfig` JSON file. The fingerprint form is sufficient for read-only
commands (`list`, `status`, `active`, `cleanup`). The JSON file form is required for
`register` and `provision`, which call `infra.launch()` internally and need full
credentials and config. `--dry-run` on cleanup prints what would be deleted without
acting.

---

## Meta-benchmarks

A meta-benchmark's `list_resources()` is the union of all sub-benchmarks'.
Each resource provisions independently. If two benchmarks declare the same underlying
image under different names, they will provision it twice — `source_hash` is recorded
for informational purposes but store-level deduplication is not implemented in v1.

---

## Resolved

1. **`infra.launch(resource)` vs `resource.launch(infra)`**
   Keeping `infra.launch(resource)` for consistency with the existing `VMBackend` pattern.
   `resource.provision(infra)` for Level 1 — install-time feels resource-centric.

2. **`InfraConfig` is config + executor**: following the existing `VMBackend` pattern in
   `vm.py`, `InfraConfig` extends `TypedBaseModel` (making it serializable) and also
   carries `launch()` / `cleanup()` methods. There is no separate "backend class" distinct
   from the config — instantiating the config IS the backend.

3. **Level 2 capacity tracking**: `ResourceHandle` does not expose
   `active_agents` / `available_slots`. The harness manages concurrency itself
   (e.g. via Ray worker count, a semaphore, or a task queue). Level 2 handles
   stay alive across tasks but carry no capacity state.

4. **InfraConfig migration**: current experiment code (`azure_backend.py`,
   `aws_backend.py`) uses `@dataclass` and must be migrated to `InfraConfig` before
   integration. `VMBackend` in `vm.py` already follows the target pattern.
   `ensure_resource()` on `VMBackend` is superseded by `infra.provision(resource)` +
   `ProvisionStore` and will be removed as part of this migration.

5. **Public images**: a resource whose infra already has a public image available
   is implicitly pre-registered — no action needed. Calling `register()` with
   new info overrides the entry and logs a warning. No separate
   `resource.use_public_image()` verb needed.

6. **Docker resources**: `DockerInfraConfig` and `LocalInfraConfig` are
   first-class providers. The `resource_info` schema differs per provider but the
   store and the `register()` / `launch()` interface are provider-agnostic.
   Existing Docker resources will be migrated to this schema in a follow-up PR.

---

## Deferred

- **Infra providers as optional packages**: cloud-specific providers (`AWSInfraConfig`,
  `AzureInfraConfig`) carry heavy optional dependencies (boto3, azure-sdk). In a future
  phase these should be isolated as separate optional packages (e.g. `cube-infra-aws`,
  `cube-infra-azure`) so users only install what they need. `LocalInfraConfig` and
  `DockerInfraConfig` remain in the core package.

- **In-cloud provisioning shortcut**: if the harness process is itself running inside
  the target cloud (e.g. an EC2 instance launching an AWS bootstrap), it could skip
  the bootstrap VM entirely and convert the image locally, then upload at datacenter
  speed. Detected via the instance metadata endpoint. No code changes needed now —
  `provision()` on a cloud provider should check for this and optimize accordingly.

- **ProvisionStore v2**: team/CI sharing via `CUBE_PROVISION_STORE` env var pointing
  to an S3/GCS path. v1 is local only (`~/.cube/provisions.json`).

- **Provisioner registry**: a `Provisioner` class that encapsulates how to provision
  a specific `(ResourceConfig subtype, InfraConfig subtype)` pair, registered via a
  dispatch table. Benchmarks can register custom provisioners without subclassing either
  InfraConfig or ResourceConfig. The registry walks the MRO to find the best match,
  falling back from specific subtypes to base classes. This cleanly separates
  provisioning logic from both resource declaration and infra config, and is the right
  long-term architecture once there are 3+ provisioners that need to be swapped or
  overridden. The current `infra.provision(resource)` method is the short-term
  equivalent — migrating to a registry is non-breaking.

  ```python
  @provisioner_registry.register(VMResourceConfig, AWSInfraConfig)
  class AWSQcow2Provisioner(Provisioner):
      def provision(self, resource: VMResourceConfig, infra: AWSInfraConfig) -> dict: ...
  ```
