# Findings — Spot + VM-side TTL spike

**Run by:** Claude (working with Alex Lacoste)
**Date:** 2026-05-21
**Subscription:** `ServiceNow AI Research` (aeb958d3-a614-450e-94bc-88f284dc0664)
**Resource group:** `ui_assist`
**Region:** `westus2`
**VM SKU:** `Standard_D4s_v3`
**Smoke runs:** 10 iterations (incremental fixes to plumbing), last 3 reached end-to-end

## Headline result

**SMOKE FAIL** on Option A. Empirical data shows OS-level `shutdown -h now` from inside a Linux VM **does NOT trigger Azure-side cleanup** (deallocation or delete) — the VM transitions to `PowerState/stopped` and stays there indefinitely, **continuing to bill for compute**.

The spike has done its job: it conclusively rules out Option A. The orphan-elimination mechanism we need is different.

## Run log (smoke #10, the conclusive one)

```
11:00:48 Auto-discovered: subscription=..., location=westus2, gallery=cube_exp_gallery
11:00:48 Registered 'smoke-spot-ttl-426f5f' @ 'azure:westus2'
11:00:51 Spot patch applied: priority=Spot eviction=Delete max_price=-1.0
11:02:00 launch: VM ready in 68s
11:02:07 Azure-side Spot facts: {'priority': 'Spot', 'eviction_policy': 'Delete',
         'billing_profile.max_price': -1.0, 'vm_size': 'Standard_D4s_v3'}
11:02:09 TTL agent: cube:expires_at=2026-05-21T15:03:48 (was set via
         AzureInfraConfig(default_ttl_seconds=180), not the resource — required
         fix because infra-level default 86400 overrides resource-level)
11:02:09 TTL agent: scheduled units (cube-ttl-shutdown.timer: 1min 38s left)
11:03:48 (TTL hits — systemd-run timer fires `shutdown -h now`)
11:04:10 [+22s after TTL] vm state=PowerState/stopped
11:13:14 [+9.5 min after TTL] still PowerState/stopped (smoke gave up waiting)
SMOKE FAIL: VM did not self-delete in time; final_state=PowerState/stopped
```

The smoke's `finally` block then ran `handle.close()` which successfully deleted the VM and cascaded its disk, NIC, and public IP. So the cube-infra-azure delete path works correctly on a stopped VM — the issue isn't deletion, it's *triggering* deletion.

## Measurements

| Measurement | Observed | Notes |
|---|---|---|
| Provision time (launch → SSH ready) | 68-80s | Consistent across 3 successful launches |
| Spot attribute round-trip | ✓ confirmed | `priority`, `eviction_policy`, `billing_profile` all set correctly on Azure side |
| TTL agent install duration | ~2s | Single SSH round-trip; very fast |
| Time from TTL deadline to VM stopped | ~22s | systemd-run fires immediately; `shutdown` adds a brief grace period |
| Time from VM stopped to Deallocated | **never** | VM stayed in `PowerState/stopped` for 9+ minutes |
| Time from VM stopped to Spot-evicted | **never** | Despite `eviction_policy=Delete`, Azure did not evict the stopped Spot VM in 9 minutes |
| Cascade delete on `_delete_vm()` (cleanup path) | ✓ works | Full disk/NIC/IP cascade on explicit delete |

## Bugs found and fixed during the spike

1. **Multiple resources in ui_assist required explicit overrides.** Added `--storage-account`, `--vnet-name`, `--nsg-name`, `--gallery-name` CLI flags.
2. **ProvisionStore registration.** `launch()` requires resource-to-image registration; spike calls `infra.register()` with the pre-existing `cube-ubuntu-22-04/1.0.0`.
3. **Spot patch must hook `_compute()`, not `launch()`.** `AzureInfraConfig._compute()` returns a fresh client each call, so patching the local var inside an overridden `launch()` didn't survive into `super().launch()`. Fixed by patching `_compute()` itself.
4. **SSH user is `cube`, not `infra.user`.** cube-infra-azure pins Linux VM admin to "cube"; there is no `user` attribute on AzureInfraConfig.
5. **IMDS tag parsing.** `awk -F: '/^cube:expires_at:/ {print $2}'` broke on the colons in ISO-8601 timestamps. Switched to `sed -n 's/^cube:expires_at://p'` which strips only the prefix.
6. **`at` not installable on cube-ubuntu-22-04.** Switched to `systemd-run --on-calendar` which is built into systemd on Ubuntu 22.04+ (no apt install needed).
7. **TTL bug: AzureInfraConfig.default_ttl_seconds=86400 overrides resource-level setting.** The spike was passing a 5-minute TTL on the resource, but the infra's 24h default won. Set `default_ttl_seconds=ttl_minutes*60` on the infra config explicitly.

## The architectural finding

The single most important finding is this confirmation of Azure VM behavior:

> **Linux `shutdown -h now` from inside an Azure VM transitions the VM to `PowerState/stopped`, NOT `PowerState/deallocated`.** Azure continues to bill for compute in the `Stopped` state. To stop compute billing, the VM must be explicitly *deallocated* (or *deleted*) via the Azure REST API.

And further:

> **`eviction_policy=Delete` on a Spot VM does NOT auto-fire on user-initiated `shutdown -h`.** The eviction policy only triggers when Azure itself reclaims the Spot capacity. A user-stopped Spot VM stays in `Stopped` state until either Azure decides to reclaim it (non-deterministic timing) or someone calls the Azure API to delete/deallocate it.

These two facts mean **Option A (SSH-installed OS-level TTL shutdown) cannot save compute cost on its own**. The VM stops processing work at TTL — useful for safety — but the bill keeps accruing until a separate mechanism cleans it up.

## Decision

**NO-GO on Option A as a standalone mechanism.** It does not deliver the cost-recovery property that was its motivation.

The viable paths forward are:

### Path 1 — Skip the VM-side agent entirely, rely on existing cleanup layers (RECOMMENDED)

The harness PR #422 already adds:
- **L2** `cleanup_stale()` at `_experiment_lifecycle` exit (with bounded grace + Ctrl+C escalation)
- **L3** `cleanup_stale()` at OSWorld/WAA `_setup()`

`cleanup_stale()` deletes any cube-tagged VM whose `cube:expires_at` has passed — and the existing `_delete_vm()` path successfully deletes VMs in any power state (verified by the smoke's own teardown). This means the existing infrastructure catches the orphan problem at L2/L3 cadence.

Worst-case window between TTL expiry and L3 sweep: ~6 hours (between benchmark runs) = $1.14 of waste per VM @ D4s_v3. Compare to current state with no L3 (orphans live 14-45 days = $70-$200 per VM). **L2+L3 alone delivers the bulk of the savings without needing any VM-side mechanism.**

### Path 2 — Spot priority for task VMs (orthogonal cost win, RECOMMENDED to combine with Path 1)

Adding `priority=Spot` + `eviction_policy=Delete` to `VMResourceConfig` is a small, additive change (3 fields in `vm_spec`). Benefits:
- **50-70% cheaper compute** on task VMs
- **Azure-driven eviction → automatic cascade delete** when capacity events fire
- Existing retry logic (`RETRIABLE_STATUSES`) handles mid-task preemption
- No new infrastructure, no IAM changes

This is the "free" cost win independent of the orphan question. Worth an openspec RFC in cube-standard.

### Path 3 — Option B (full VM-side self-delete with Managed Identity)

If we want VM-side cleanup that works without an external sweeper, we need:
- `vm_spec["identity"] = {"type": "SystemAssigned"}` at create time
- A role assignment giving the identity `Microsoft.Compute/virtualMachines/delete` on its own resource ID
- VM-side script that fetches an IMDS token and calls the Azure REST API to delete itself

This is non-trivial: role assignment requires `Owner` or `User Access Administrator` on the RG, which the spike runner may not have. The complexity probably isn't worth it relative to Path 1, given Path 1 already covers most of the gap.

## Recommendations

1. **Land cube-harness PR #422 as-is.** L2+L3 deliver the bulk of the orphan-cleanup value.
2. **Open an openspec RFC in cube-standard for Spot task VMs** (Path 2). Add `use_spot: bool = False` and `max_spot_price: float | None = None` to `VMResourceConfig`. Wire them into the `vm_spec` builder in `AzureInfraConfig.launch()`. Default to off; turn on for OSWorld + WAA task VMs after a one-week validation period to measure eviction-rate impact.
3. **Drop Option A from active work.** The spike confirmed it is not viable as a standalone mechanism. The smoke remains in `scripts/smoke/spot_ttl_lifecycle.py` as a reproducible record of the finding.
4. **Defer Option B** until Path 1 + Path 2 are measured in production for a month. If the residual orphan cost is still material, revisit with the IAM design work.

## Next steps

1. Update the spike `README.md` to point at this findings document for the conclusion.
2. Land PR #422 (L2+L3).
3. Open the openspec RFC for Path 2 (Spot priority on VMResourceConfig).
4. Re-run the smoke after the openspec RFC lands, with the production API (no monkey-patch) to validate the integration.
