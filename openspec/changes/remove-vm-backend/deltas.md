# Deltas — Remove the legacy `cube.vm` / `VMBackend` provisioning path

**Targets:** `openspec/specs/resource/spec.md`, `openspec/specs/tool/spec.md`

VM analogue of `openspec/changes/remove-container-backend/deltas.md`
(Track 1). The `resource/spec.md` contract already documents
`VMResourceConfig` + `InfraConfig` as the VM path and never carried
`VMBackend` / `VMConfig` text (those lived only as code in `src/cube/vm.py`),
so there is no stale contract prose to delete — only the code-level removals
below. Applied when this change lands.

---

## REMOVED — `cube.vm` module (`VMConfig` / `VM` / `VMBackend`)
**Spec:** resource

The `cube.vm` module (`src/cube/vm.py`) is deleted: the serializable
`VMConfig`, the live `VM` handle (`endpoint`, `restore_snapshot()`,
`stop()`), the `VMBackend(TypedBaseModel, ABC)` factory
(`launch()` / `ensure_resource()`), and the `ResetIsolation` enum are gone.

VM provisioning is owned exclusively by `InfraConfig` — a benchmark declares
a `VMResourceConfig` (see `resource/spec.md` § Public API) and an
`InfraConfig` provisions it via `provision()` / `launch()`, returning a
`ResourceHandle` whose `.endpoint` is the in-VM guest-agent URL. There is no
standalone VM factory/handle abstraction.

## REMOVED — `cube-vm-backend` package
**Spec:** resource

The `cube-vm-backend` package (`cube-resources/cube-vm-backend/`:
`LocalQEMUVMBackend`, `LocalDockerVMBackend`, `QEMUManager`,
`DockerManager`) is deleted. Local QEMU/qcow2 VM provisioning is now
`cube.infra_local.LocalInfraConfig` driving a `VMResourceConfig`; cloud VM
provisioning lives in `cube-infra-aws` / `cube-infra-azure` (unchanged).

## REMOVED — computer tool `vm=` / `attach_vm` API
**Spec:** tool

`cube-computer-tool` no longer imports `cube.vm.VM`.
`ComputerConfig.make()` loses the `vm=` parameter (signature is now
`make(self, container: Container | None = None) -> ComputerBase`);
`ComputerBase.__init__` no longer takes a `vm`; `ComputerBase.attach_vm()`
is removed. The only live attach API is
`ComputerBase.attach_endpoint(endpoint: str)`, called with
`ResourceHandle.endpoint` once the VM is launched — a deferred-launch
pattern that fits the `InfraConfig` / `ResourceHandle` lifecycle.
