# Resource Lifecycle

**Module:** `cube.resource` | **Design reference:** `design/resource_lifecycle.md`

## Purpose

Model long-lived, shared, and ephemeral infrastructure uniformly. A benchmark declares
WHAT it needs (`ResourceConfig`); an `InfraConfig` decides HOW to provision it. Lifetime
is partitioned into three levels with explicit cleanup semantics so the harness can
reason about safety, cost, and cleanup across crashes.

## Three Lifetime Levels

| Level | Scope | Examples | Created by | Torn down by |
|-------|-------|----------|------------|--------------|
| **L1** | Provisioned images (long-lived) | AWS AMI, Azure Gallery image, local qcow2 | `provision()` or `register()` | `unprovision()` (manual only) |
| **L2** | Benchmark-scoped (per run) | WebArena server, WorkArena ServiceNow | `BenchmarkConfig.make(infra)` → `Benchmark.setup()` | `handle.close()` or `cleanup(run_id)` |
| **L3** | Task-scoped (per task) | Individual OSWorld VM, per-task container | `infra.launch()` | `handle.close()` or `cleanup(run_id)` |

L1 entries live in `ProvisionStore` (`~/.cube/provisions/`).
L2/L3 resources are tracked via cloud tags (`cube:run_id`, `cube:expires_at`).

**Provisioning trigger:** `BenchmarkConfig.make(infra)` iterates
`config.resources` and calls `infra.provision(resource)` on each entry whose
`provision_status(infra) != "ready"` before invoking `benchmark.setup()`. Benchmark
authors do not need to call `provision` manually from `_setup()` — idempotent
provisioning is handled by the factory.

## Public API

### `ResourceConfig` (benchmark-owned, serializable)
```python
class ResourceConfig(TypedBaseModel):
    name: str                                        # stable identifier
    scope: Literal["task", "benchmark"] = "task"     # L3 vs L2
    max_concurrent_agents: int | None = None         # L2 capacity hint
    source_url: str | None = None                    # canonical image source
    source_hash: str | None = None                   # informational; not used for dedup
    default_ttl_seconds: int | None = 3600           # auto-cleanup TTL
    bootstrap_script_extra: str | None = None        # benchmark-specific VM setup
    requires: set[str] = set()                       # explicit extra tokens, e.g. {"container:root"}

    def requirements(self) -> set[str]               # folds `requires`; subclasses super()-union
```

Standard capability tokens: `"kvm"`, `"docker"`, `"gpu:nvidia"`, `"network:egress"`,
`"container:root"` (container processes run as uid 0 — needed by tasks that `apt`-install
or write to `/etc`, `/var`; absent on infras that pin a non-root uid, e.g. EAI Toolkit).

`requires` is the declarative escape hatch: any resource adds tokens here and every
subclass folds them via `super().requirements()`, so `requirements()` is the single
read point used by `can_serve`.

**Subclasses:**
- `VMResourceConfig(requires_kvm: bool = True)` — VM-based (OSWorld, WindowsAgentArena, AndroidWorld…)
- `DockerServiceConfig(docker_images, services, launch_script, endpoint_to_site, volumes)` —
  multi-container stack (WebArena, WorkArena)
- `ContainerConfig(image, ram_gb, cpu_cores, disk_gb, gpu, ports)` — single image per task
  (SWE-bench, terminal-bench, CTF). Defined in `cube.resource`, re-exported from
  `cube.container`; declared on `TaskMetadata.container_config`. `requirements()` →
  `{"docker"}` (+ `"gpu:nvidia"` if `gpu`) ∪ `requires`.

### `VolumeSpec` (used by `DockerServiceConfig`)
```python
class VolumeSpec(TypedBaseModel):
    name: str                        # Docker volume name
    mount_path: str
    source_url: str | None = None    # tarball to pre-populate; baked into snapshot
    tar_subpath: str | None = None
    strip_components: int = 0
```

### `InfraConfig` (harness-owned, serializable + executable)

```python
class InfraConfig(TypedBaseModel, ABC):
    default_ttl_seconds: int | None = 86400          # 1 day; overrides resource TTL
    image_name_suffix: str = ""                       # e.g. "-test" to isolate CI
    on_incompatible: Literal["raise", "force"] = "raise"  # capability-gate policy

    @abstractmethod
    def fingerprint(self) -> str                     # "aws:us-east-2", "azure:westus2", "local"
    @abstractmethod
    def capabilities(self) -> set[str]
    @abstractmethod
    def provision(self, resource) -> None            # L1: download → upload → import → register
    @abstractmethod
    def launch(self, resource) -> ResourceHandle    # L2/L3: spin up from provisioned image
    @abstractmethod
    def list_active(self, run_id=None) -> list[ResourceHandle]
    @abstractmethod
    def cleanup(self, run_id: str) -> None           # L2/L3: delete all live for run_id
    @abstractmethod
    def cleanup_stale(self, max_age_seconds=None) -> list[str]  # L2/L3: TTL-based GC

    def unprovision(self, resource) -> None          # L1: default no-op; override to delete image
```

**Concrete helpers** (provided):
- `register(resource, resource_info: dict)` — record that an image is available
- `provision_status(resource)` → `"ready" | "needs_provisioning"`
- `can_serve(resource)` → bool — `resource.requirements() <= self.capabilities()`. The unit
  of the capability handshake; `BenchmarkConfig.make()` runs it over every task's
  `container_config` and the benchmark's declared `resources` before provisioning, applying
  `on_incompatible`. A meta-infra overrides `can_serve` to delegate per-resource to children.

`fingerprint()` rule: encode provider + region/location only. Two configs with the
same fingerprint share the same provisioned image. Do NOT encode instance size,
CPU count, or other performance knobs.

### `ResourceHandle` (live; NOT serializable)
```python
@dataclass
class ResourceHandle(ABC):
    run_id: str
    resource: ResourceConfig
    infra: InfraConfig
    endpoint: str | None
    endpoints: dict[str, str] = {}          # e.g. {"shopping_admin": "http://..."}
    created_at: datetime
    expires_at: datetime | None = None

    @abstractmethod
    def close(self) -> None                 # tear down this specific resource

    def __enter__(self) / __exit__(self, ...): ...  # context manager
```

`close()` must be idempotent. Callers can rely on `with handle: ...` pattern.

### Exceptions
- `ResourceNotReadyError` — `launch()` called before `provision()` or `register()`
- `UnsupportedResourceType` — infra doesn't support the given `ResourceConfig` subclass
- `IncompatibleInfraError` — raised by `BenchmarkConfig.make()` (pre-provision, pre-episode)
  when `on_incompatible == "raise"` and a resource is not servable

### `on_incompatible` policy
Checked at `make()` by running `can_serve` over each task's `container_config` and the
benchmark's `resources`:
- `"raise"` (default) — abort with `IncompatibleInfraError` if **any** resource is
  incompatible. No provisioning, no episodes, no spend.
- `"force"` — attempt everything anyway (escape hatch to probe a stale requirement).

Future: a per-task mode (`"per-task-raise"`) may let the benchmark proceed while each
incompatible task raises at episode start — recording them as terminal per-task errors
rather than dropping them. A silent `"skip"` is intentionally absent: silently dropping
tasks is the failure mode this gate exists to remove.

## Cleanup Methods Reference

| Method | Levels | When to call | What it does |
|--------|--------|--------------|--------------|
| `handle.close()` | L2/L3 | After each task (L3) or at run end (L2) | Tears down this specific resource |
| `infra.cleanup(run_id)` | L2/L3 | Harness shutdown (catch-all) | Deletes everything tagged with `run_id` |
| `infra.cleanup_stale(max_age)` | L2/L3 | Called automatically by `Benchmark.setup()` (the "harness startup" hook); harnesses may also call it explicitly on lifecycle exit as a defense-in-depth sweep | GCs TTL-expired resources across all runs |
| `infra.unprovision(resource)` | L1 | Manual (retire / switch region) | Removes provisioned image + store entry |

**Experiment-scoped `run_id` (`CUBE_RUN_ID`).** For `cleanup(run_id)` to reap a
whole run's resources in one call, every resource a run launches must share one
`run_id`. A harness signals this by exporting **`CUBE_RUN_ID`** (a stable per-experiment
id) into the environment before launching; tag-based backends adopt it as the
`run_id` tag (falling back to a fresh per-resource id when unset, so non-harness
callers are unaffected — env-resolved like credentials, never an `InfraConfig` field).
This is what lets a harness reap **identity-based, not time-based**: on a clean exit it
calls `cleanup(run_id)`; for a hard-killed/slept client it runs a startup GC that calls
`cleanup(run_id)` for each run its local heartbeat records prove dead (stale/terminal) —
never cancelling a still-heartbeating run, and never touching another session's `run_id`.
The server-side TTL (`cleanup_stale` / `--max-run-time`) remains the last-resort backstop.

## Recommended Harness Lifecycle

```python
benchmark.setup()                           # base setup() auto-calls infra.cleanup_stale()
                                            # to GC orphans from prior crashes, then runs _setup()
for task in tasks:
    handle = infra.launch(resource)         # creates L3 resource
    try:
        run_episode(task, handle)
    finally:
        handle.close()                      # tears down L3 immediately
infra.cleanup(run_id=run_id)                # shutdown: catch-all
benchmark.close()                           # tears down L2 resource
```

## Invariants

1. `provision()` is idempotent and calls `register()` on success.
2. `launch()` raises `ResourceNotReadyError` if the ProvisionStore has no entry for
   `(resource, infra)` — no implicit provisioning.
3. `cleanup(run_id)` is safe to call on already-deleted resources — implementations
   must no-op gracefully.
4. `cleanup_stale()` reads cloud tags directly — works after total local state loss.
5. Effective TTL: `infra.default_ttl_seconds` overrides `resource.default_ttl_seconds`.
6. `image_name_suffix` is appended to both image names AND ProvisionStore keys so CI
   environments can fully isolate from production without renaming resources.
7. `ResourceHandle` is not serializable — never pass across process boundaries. Pass
   `run_id` instead and have the target process call `infra.cleanup(run_id)`.

## Gotchas

- Credentials are NEVER stored in `InfraConfig` fields (would be serialized). They're
  resolved from env vars at runtime.
- `provision()` bakes `VolumeSpec` data into the image — that's why provision is slow
  and `launch()` is fast.
- `bootstrap_script_extra` must be declared in source. Fetching it at runtime breaks
  hermetic reproducibility.
- `list_active(run_id=None)` returns all live resources (no filter). Pass `run_id` to
  narrow.
- Forgetting to `cleanup_stale()` at startup leads to cost leaks over time (orphans
  accumulate across crashes).
