# Remove the legacy `cube.vm` / `VMBackend` provisioning path

**Status:** Draft
**Date:** May 2026
**Depends on:** `resource-convergence` (`VMResourceConfig` + InfraConfig), `cube-infra-aws` / `cube-infra-azure` (already on `dev`)
**Tracks:** GitHub issue #94 (Track 2)

---

## Background

This is the VM analogue of the merged container-backend removal
(`remove-container-backend`, Track 1, cube-standard #163). Track 1 deleted the
standalone `ContainerBackend` / `cube.backends` provisioning path once every
in-tree benchmark provisioned containers through the injected `InfraConfig`.
This change does the same for VMs.

`resource-convergence` landed the path the VM migration converges on:
benchmarks declare WHAT VM they need with `cube.resource.VMResourceConfig`
(the `VMResourceConfig.forwarded_ports` delta in
`openspec/changes/resource-convergence/deltas.md` is part of it), and an
`InfraConfig` decides HOW to provision it — `LocalInfraConfig` for local
QEMU/qcow2, `cube-infra-aws` / `cube-infra-azure` for cloud. Both ship a
`provision()` / `launch()` pair returning a `ResourceHandle` whose
`.endpoint` is the in-VM HTTP guest-agent URL.

With `VMResourceConfig` + the cloud/local InfraConfigs already on `dev`, the
old `cube.vm` abstraction (`VMConfig` / `VM` / `VMBackend`) and the
`cube-vm-backend` package that implemented it have no remaining callers and
can be deleted.

---

## What changes

- **Delete the `cube.vm` module** (`src/cube/vm.py`). The serializable
  `VMConfig`, the live `VM` handle (`endpoint`, `restore_snapshot()`,
  `stop()`), the `VMBackend(TypedBaseModel, ABC)` factory
  (`launch()` / `ensure_resource()`) and the `ResetIsolation` enum are gone.
- **Delete the `cube-vm-backend` package** entirely
  (`cube-resources/cube-vm-backend/`): `LocalQEMUVMBackend`,
  `LocalDockerVMBackend`, and the `QEMUManager` / `DockerManager` helpers.
  Local QEMU/qcow2 provisioning is now `cube.infra_local.LocalInfraConfig`
  driving a `VMResourceConfig`; cloud VM provisioning lives in
  `cube-infra-aws` / `cube-infra-azure` (unchanged).
- **Migrate `cube-computer-tool` off `cube.vm.VM`.** No more
  `from cube.vm import VM`. `ComputerConfig.make()` loses the `vm=`
  parameter (signature is now `make(self, container: Container | None =
  None)`); `ComputerBase.__init__(self, config)` no longer takes a `vm`;
  `attach_vm()` is removed. The live API is now
  `ComputerBase.attach_endpoint(endpoint: str)` only — the caller passes
  `ResourceHandle.endpoint` after the VM is launched (deferred-launch
  pattern that fits the `InfraConfig` / `ResourceHandle` lifecycle).
- **Docs/specs/skill references** are updated to the
  `VMResourceConfig` + `InfraConfig` model: `ROADMAP.md`,
  `cube-resources/README.md`, `cube-tools/README.md`, `design/README.md`,
  `.claude/skills/new-cube/references/shared-packages.md`. The obsolete
  `design/vm_backend.md` design doc was already removed; the
  `resource/spec.md` contract already describes `VMResourceConfig` +
  `InfraConfig` and never carried `VMBackend` text, so no spec contract text
  changes — only the code-level removals are recorded in `deltas.md`.

VM provisioning is now exclusively the `InfraConfig` path:
`VMResourceConfig` describes the VM; `LocalInfraConfig` (local QEMU/qcow2) or
`cube-infra-aws` / `cube-infra-azure` (cloud) provision it and return a
`ResourceHandle`; the computer tool connects via
`attach_endpoint(handle.endpoint)`. There is no `VM` / `VMBackend` /
`VMConfig` abstraction anymore.

---

## Impact

- **Breaking** for any caller importing `cube.vm` (`VMConfig`, `VM`,
  `VMBackend`, `ResetIsolation`) or `cube_vm_backend`
  (`LocalQEMUVMBackend`, `LocalDockerVMBackend`). Migration: declare a
  `VMResourceConfig` and provision via `LocalInfraConfig` /
  `cube-infra-aws` / `cube-infra-azure`; connect the computer tool with
  `attach_endpoint(handle.endpoint)`.
- **Breaking** for callers passing `vm=` to `ComputerConfig.make()` or
  calling `ComputerBase.attach_vm()` — use `attach_endpoint()`.
- A paired **cube-harness** PR updates the downstream callers that
  previously constructed `cube.vm` objects or used the computer tool's
  `vm=` / `attach_vm` API, so both land together.

---

## Sequencing

1. `resource-convergence` — `VMResourceConfig` convergence. Done (on `dev`).
2. `cube-infra-aws` / `cube-infra-azure` — cloud VM InfraConfigs. Done (on `dev`).
3. `remove-container-backend` (Track 1, #163) — container analogue. Merged.
4. This change — hard removal of the VM path, paired with the cube-harness
   caller update.
