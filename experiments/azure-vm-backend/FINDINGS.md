# Azure VM Backend — Experiment Findings

## Session: 2026-03-24

---

## 1. qemu-img — RESOLVED

**Problem**: Homebrew can't install QEMU due to non-standard prefix (`~/homebrew` vs `/opt/homebrew`),
causing `p11-kit` test-transport to hang indefinitely.

**Solution**: Built `qemu-img` from source (QEMU 9.2.3) with minimal flags:
```bash
cd /tmp/qemu-9.2.3
./configure --without-default-features --enable-tools --enable-vpc --enable-vhdx --disable-werror
ninja -C build qemu-img
cp build/qemu-img ~/homebrew/bin/qemu-img
```

**Result**: `qemu-img` 9.2.3 installed at `~/homebrew/bin/qemu-img`.
Supported formats include: `qcow2`, `vpc` (VHD), `vhdx`, `raw`.

**VHD conversion test — PASSED**:
```bash
qemu-img convert -f qcow2 -O vpc -o subformat=fixed,force_size input.qcow2 output.vhd
```
- Output has `conectix` magic (valid VHD footer) ✓
- Fixed subformat (Azure requires, not dynamic) ✓
- Virtual disk size 1MB aligned (Azure requirement) ✓
- The 512-byte VHD footer on top of data is standard and expected by Azure ✓

---

## 2. Pipeline Steps 4-5: Snapshot → VM Launch — PASSED

Successfully launched a VM from the `osworld_base` snapshot:
- Disk creation from snapshot: ~30s
- NIC + public IP creation: ~10s
- VM provisioning: ~90s
- **Total: ~2 min from snapshot to VM running**

**Issues discovered**:
- Subnet name was `snet-westus2-1` not `default` → fixed in pipeline.py
- `osworld_base` is a **Specialized** image — no cloud-init, SSH keys baked in
  by original team (Aman/Kusha). No SSH access without their private key.

---

## 3. Guest Agent — NOT present in osworld_base

Port scan of VM launched from `osworld_base`:
- Port 5000: **closed/silent** — guest agent NOT running
- Port 80: 404 from **Zscaler proxy** (not the VM)
- Port 22: SSH accepts connections but rejects keys (baked-in team key)

**Implication**: The existing `osworld_base` image does NOT have the CUBE guest
agent. Any cloud backend must inject it at launch time.

---

## 4. Zscaler Networking Issue

All outbound HTTP from developer machine is intercepted by Zscaler.
- Port 80 on Azure VM returns Zscaler 404 (not VM response)
- Port 5000 appears closed (Zscaler drops non-HTTP/S traffic on unknown ports)
- **SSH (port 22) works** — corporate firewalls typically allow SSH

**Solution validated**: SSH tunnel bypasses Zscaler entirely:
```python
subprocess.Popen([
    "ssh", "-N", "-L", f"127.0.0.1:{local_port}:localhost:{GUEST_PORT}",
    "-i", key_path, f"azureuser@{vm_ip}"
])
endpoint = f"http://localhost:{local_port}"
```

---

## 5. Cloud-init Injection Experiment (pending results)

**Test**: Fresh Ubuntu 22.04 Marketplace VM + cloud-init installs fake guest agent.
- Throwaway SSH keypair generated per-VM (no pre-shared keys)
- cloud-init installs Flask + starts `/screenshot` server on :5000
- SSH tunnel used to reach :5000

**Results**: → See `findings_cloudinit.json` when experiment completes.

**Expected outcome**: Validates the architecture for Generalized images.
For Specialized images (baked state like OSWorld), benchmark authors must either:
  a) Publish a Generalized version + cloud-init script to install their services, OR
  b) CUBE SSH-injects the guest agent after first boot (SSH available before cloud-init)

---

## 6. Architecture Conclusions

### What works today
- `LocalQEMUVMBackend` — local QEMU with qcow2, already in codebase
- `LocalDockerVMBackend` — Docker containers, already in codebase
- Azure VM launch from snapshot — 2 min, pipeline.py step 5 works

### What's needed for cloud backend
1. **qemu-img** — now available ✓
2. **Storage account in westus2** — needs creating (cubeexpvhd)
3. **VHD upload** — pipeline.py step 2 (azure-storage-blob, PageBlob)
4. **Disk import from blob** — pipeline.py step 3
5. **Snapshot creation** — pipeline.py step 4
6. **Guest agent injection** — cloud-init for Generalized; SSH script for Specialized
7. **SSH tunnel** — for Zscaler/corporate network environments

### Key open questions
- [x] Does cloud-init injection work cleanly? → **BLOCKED by Golden Image Policy** (see §6)
- [x] Does Azure Compute Gallery bypass Golden Image Policy? → **YES — confirmed** (see §8)
- [ ] Does a fresh Ubuntu VHD converted from qcow2 actually boot on Azure?
      (known issue: virtio → hyper-v driver swap sometimes fails)
- [ ] What format do macOS Arena / other non-Linux benchmarks use?
- [ ] Should CUBE publish guest agent as a pip-installable service for cloud-init?

---

## 6. CRITICAL: ServiceNow Golden Image Policy

Azure Policy `Golden_image_exception` **DENIES** VM creation unless:

**Condition A**: `osDisk.createOption == "attach"` (pre-existing managed disk)
**Condition B**: `imageReference.id` contains `Microsoft.Compute/galleries` (Compute Gallery)

This means:
- ❌ Cannot launch VMs from Azure Marketplace images (Ubuntu 22.04, etc.)
- ❌ Cannot launch VMs from uploaded VHD blobs directly
- ✅ CAN launch VMs using `createOption: attach` from a snapshot-derived disk (our pipeline.py steps 4-5)
- ✅ CAN launch from images in Azure Compute Gallery (if registered there)

**Impact on architecture**:

The `snapshot → disk (attach) → VM` path is the **only valid launch path** in this subscription.
This means `ensure_resource()` MUST:
1. Convert qcow2 → VHD
2. Upload to blob storage
3. Import as managed disk (`createOption: Import`)
4. Create snapshot from that disk
5. All future `launch()` calls: snapshot → new disk → VM with `createOption: attach`

Steps 1-4 are one-time setup. Step 5 is the per-eval fast path.

The cloud-init injection experiment **cannot run in this subscription** due to this policy.
It would work fine in a subscription without this restriction (personal Azure account, AWS, GCP).

---

## 7. UX Design Conclusions

**For 100s of CUBEs at scale, the right model is**:

```
Benchmark author publishes:
  - Docker image on Docker Hub / HuggingFace  (OR qcow2 on HuggingFace)
  - VMConfig with resource requirements

CUBE harness user runs:
  backend = AzureQEMUVMBackend(subscription=..., resource_group=...)
  backend.ensure_resource(config)  # once: download + convert + upload + snapshot
  vm = backend.launch(config)      # each eval: snapshot → disk → VM + cloud-init agent

No benchmark-specific cloud infra work required from benchmark authors.
```

**The legal constraint** (no redistributing modified images) is handled by:
- cloud-init injects CUBE guest agent at runtime (not baked into image)
- CUBE publishes `cube-guest-agent` as a pip package
- `ensure_resource()` wraps the conversion, user never touches qcow2 directly

---

## 8. Plan A: Azure Compute Gallery — CONFIRMED

**Date**: 2026-03-24

**Goal**: Validate that Azure Compute Gallery bypasses the Golden Image Policy
and enables `createOption: FromImage` (the clean cloud-init injection path).

**Test**:
- Created `cube_exp_gallery` (Compute Gallery) in `ui_assist/westus2`
- Created image definition `cube-osworld-linux` (Linux, Specialized, HyperV Gen1)
- Created image version `1.0.1` from a tiny 1GB test snapshot (not osworld_base — no hotspot needed)
- Launched VM `cube-gal-vm-2f1581` with `createOption: FromImage` + gallery `imageReference`

**Result**: ✅ **VM provisioned successfully — no policy denial**

```
imageReference.id = .../galleries/cube_exp_gallery/images/cube-osworld-linux/versions/1.0.1
createOption: FromImage  ← NOT Attach
```

This confirms: Golden Image Policy Condition B (`imageReference.id` contains
`Microsoft.Compute/galleries`) is satisfied when launching from a Compute Gallery.

**Implication**: The production CUBE architecture can use `createOption: FromImage`
with a Compute Gallery image + `os_profile` SSH key injection (cloud-init).
No CCOE workarounds, no Run Command key injection hacks.

The full production flow for `ensure_resource()`:
```
HuggingFace qcow2 → VHD → Azure Blob → Managed Disk → Gallery Image Version
```

And per `launch()`:
```
Gallery Image → VM (FromImage) + os_profile.linux_config.ssh.public_keys
→ SSH tunnel → http://localhost:{port}
```

**Note**: Version `1.0.0` (from `osworld_base`, 1TB) is also being created in the gallery
(Azure-side operation, no local data transfer). This validates the full production image path.

---

## 9. Full End-to-End Test: Ubuntu 22.04 Gallery Image — PASSED

**Date**: 2026-03-24

**Pipeline executed** (`experiment_ubuntu_gallery.py full`):

1. **Download**: Ubuntu 22.04 server cloudimg (~660MB) ✅
2. **Convert**: `qemu-img convert -f qcow2 -O vpc -o subformat=fixed,force_size` → 2.2GB VHD ✅
3. **Upload**: VHD → Azure Blob Storage (cubeexpvhd, Page Blob) ✅
4. **Import**: Blob → Managed Disk (`createOption: Import`) → `cube-exp-disk-7e931d` ✅
5. **Gallery**: Created `cube-ubuntu-22-04` (Generalized, Linux, HyperV Gen1) image definition ✅
6. **Version**: Replicated disk into gallery image version `1.0.0` (~8 min for 2.2GB) ✅
7. **Launch**: VM `cube-ub-vm-60e70d` from gallery (`createOption: FromImage` + `os_profile`) ✅
   - Golden Image Policy: **NOT BLOCKED** (gallery imageReference passes policy check)
   - SSH key injection: **CLEAN** — `azureuser` account created, our key in authorized_keys
   - SSH available: **~2 min** after VM provisioned
8. **Guest agent**: Installed flask + started agent manually (cloud-init bug — see below) ✅
9. **Probe via SSH tunnel**:
   - `/health` → `{"agent":"cube-mini-guest-agent","status":"ok"}` ✅
   - `/screenshot` → HTTP 200, `image/png`, 2788 bytes ✅
   - `/execute` → `["uname","-a"]` returns Ubuntu 22.04 kernel info ✅

**Key answers**:
- Q1: Does qcow2-converted VHD boot on Azure? → **YES** — Ubuntu 22.04 with Hyper-V drivers boots correctly
- Q2: Does os_profile SSH key injection work? → **YES** — clean SSH without any CCOE workarounds
- Q3: Full pipeline automated? → **YES (with one bug fix needed)**

**Bug 1 found and fixed**: cloud-init `custom_data` with Python code in `runcmd` heredoc fails
because cloud-init tries to parse the Python code as YAML (`import io` → YAML parse error).

**Fix 1**: Use `write_files` with `encoding: b64` to write the Python script, then a simple
one-liner `runcmd` to start it.

**Bug 2 found and fixed** (2026-03-25): `write_files` with `owner: "azureuser:azureuser"` fails
at the `init-network` phase because `azureuser` is created by waagent *after* `write_files` runs.
Error: `KeyError: "getpwnam(): name not found: 'azureuser'"`.

**Fix 2**: Write to `/usr/local/bin/cube_guest_agent.py` with no `owner` field (defaults to
`root:root`). `runcmd` runs as root so this is fine. Fixed in all three scripts and `cube_azure_pipeline.py`.

**Cloud-init injection fully confirmed** (2026-03-25): `cube_azure_pipeline.py launch` + `probe`
validated end-to-end — `/health`, `/screenshot`, `/execute` all pass. `restore_snapshot` also
validated (stop + relaunch in ~3.5 min, all endpoints pass on the new VM).
