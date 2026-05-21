# Spike: Spot VMs with VM-side TTL shutdown

**Status:** active spike (not production code)
**Owner:** alexandre.lacoste@servicenow.com
**Question:** Can we provision Spot VMs with a VM-side TTL self-shutdown to eliminate the orphan-cost problem at its source, without baking new images and without per-team scheduler infrastructure?

## Background

May 2026 Azure spend in the `ui_assist` resource group hit ~$20K MTD (forecast $29K), with ~$15.9K attributable to VM compute. Investigation revealed that ~5% of VMs survived as orphans for 14-45 days after the per-task `finally` block failed to fire (Ray-worker crashes, SIGKILL, OOM). Each long-tail orphan accrued $70-$206 in compute before manual cleanup.

The harness-side mitigation (PR [cube-harness#422](https://github.com/The-AI-Alliance/cube-harness/pull/422)) adds two layers of defense:
- **L2** — `cleanup_stale()` on `_experiment_lifecycle` exit (with SIGTERM handler so graceful shutdowns run finally blocks)
- **L3** — `cleanup_stale()` at OSWorld/WAA `_setup()` (sweeps prior-run orphans on next launch)

Those two cover the normal-exit and next-launch cases. They do **not** cover:
- SIGKILL / OOM / hard kills that bypass Python entirely
- Long gaps between benchmark runs where prior orphans bill indefinitely
- Network-partitioned runs where the harness can't reach Azure to clean up

A cloud-native solution that runs *independently of the harness* is needed for those.

## Hypothesis

A combination of two cloud-native mechanisms catches the residual orphans without requiring any harness-side scheduler:

1. **Azure Spot priority with `eviction_policy="Delete"`.** Azure eventually reclaims Spot capacity, deleting the VM and triggering the existing `delete_option=Delete` cascade on disk/NIC/IP. Compute is 50-70% cheaper as a bonus. Eviction is non-deterministic, so this alone is insufficient.

2. **VM-side TTL shutdown via `at` job, SSH-installed at launch.** At launch time, the harness already opens an SSH connection. One extra command schedules `sudo shutdown -h now` at `cube:expires_at` (read from the VM's tag via Azure Instance Metadata Service). The VM stops billing compute at TTL with no external dependency.

Combined, these two mechanisms catch:
- **Spot eviction**: any Azure capacity event → automatic cleanup, no external trigger
- **TTL shutdown**: every VM has a guaranteed end-of-life regardless of any external state

Neither mechanism requires:
- Image rebuilds (the SSH install runs on existing images)
- IAM / Managed Identity changes (the VM only does an OS-level `shutdown`, not a cloud API call)
- Cloud-side scheduler infrastructure (Azure Function, Logic App, GitHub Action)
- Per-team OIDC / federated identity setup

This is the "Option A" path from the architecture discussion — minimum-IAM, ships fastest. Option B (full self-delete via cloud API + Managed Identity) is a follow-up that requires provision-time IAM changes.

## What this spike measures

Run the smoke at [`../../scripts/smoke/spot_ttl_lifecycle.py`](../../scripts/smoke/spot_ttl_lifecycle.py) and capture:

| Measurement | How to read it | Decision impact |
|---|---|---|
| Does the Spot VM provision successfully? | Script exits 0 at launch | Validates `priority=Spot` works in our subscription |
| Provisioning time (Spot vs regular) | Wall-clock in launch log | Spot should be similar; if significantly slower, capacity is constrained |
| Spot $/hr at launch time | `billing_profile.max_price` echoed by Azure | Quantifies savings vs $0.19/hr regular |
| Does the SSH `at` install land? | `atq` output shows the scheduled job | Validates the install path |
| Does the VM actually shut down at TTL? | VM power state goes to "stopped" by `T+TTL+1m` | Confirms the kill mechanism works |
| Does the cascade-delete fire? | VM record + disk + NIC + IP all gone by `T+TTL+5m` | Validates `delete_option=Delete` + Spot interaction |
| What's the Azure eviction signal handling? | Check `ScheduledEvents` metadata endpoint during the run | Informs how the harness should observe preemption |

Capture results in `findings.md` (template in this directory).

## What this spike does NOT cover

- **AWS / GCP equivalents.** This is the cube-infra-azure spike. AWS spot semantics are different (preemption ≠ delete unless `InstanceInitiatedShutdownBehavior=terminate` + `DeleteOnTermination=true`). Each cloud needs its own validation.
- **Production integration.** This is exploratory code, not a refactor of `AzureInfraConfig.launch()`. The production change requires an openspec RFC modifying `VMResourceConfig` (adding `use_spot`, `max_spot_price`) and the `vm_spec` builder.
- **Eviction-rate measurement.** A real eviction-rate study would need to run hundreds of VMs over multiple days across regions and times. Out of scope for the spike — defer to a "phase 2" study if Phase 1 shows the mechanism works.
- **IAM-based self-delete.** Option B (VM calls cloud API to delete itself) requires Managed Identity setup. Deferred.
- **The harness's retry behavior on preemption.** Already validated to exist (`RETRIABLE_STATUSES = {"FAILED", "CANCELLED", "STALE"}`). Empirical testing of how often it kicks in under Spot belongs in the phase 2 study.

## How to run

**Prerequisites:**
- `az login` against the target subscription (uses `AzureCliCredential`)
- A pre-existing L1 image in the resource group's compute gallery (defaults to `cube-ubuntu-22-04/1.0.0`)
- The Azure CLI for inspecting results (`az vm list`, `az network public-ip list`, etc.)

**Cost estimate:** ~$0.01 per run (a single Spot D4s_v3 for 5-10 minutes, evicted or self-shutdown).

```bash
# From cube-standard repo root:
uv run cube-resources/cube-infra-azure/scripts/smoke/spot_ttl_lifecycle.py \
    --resource-group ui_assist --ttl-minutes 5
```

Exit codes follow the cube-standard smoke convention: 0 = SMOKE OK, 1 = SMOKE FAIL,
2 = SMOKE SKIP. The banner line is the only signal a CI runner or another
agent needs; the verbose log is for humans diagnosing failures.

Then in a second terminal, watch the VM lifecycle:

```bash
# VM should appear within ~3 minutes of script start:
watch -n 30 'az vm list -g ui_assist --query "[?contains(name, '"'"'spike-spot'"'"')].{name:name, state:powerState, size:hardwareProfile.vmSize}" -o table'

# After TTL: VM should be Stopped (deallocated), then deleted within a few minutes.
```

## Decision criteria

Based on the spike outcome, recommend one of:

- **GO** — Spot + SSH TTL works as designed. Open openspec RFC for the production change in cube-standard. Estimated cost reduction: ~$10K/month for ServiceNow's cube workloads, generalizable to all cube users.
- **GO with caveats** — Mechanism works but eviction rate is too high for some benchmarks (e.g. WAA's 30-min episodes). Make `use_spot` opt-in per-VMResourceConfig, default off, document for short-task benchmarks.
- **NO-GO** — Mechanism doesn't work (Spot unavailable for our SKUs, SSH install too flaky, cascade delete doesn't fire on shutdown). Fall back to Option B (full self-delete via cloud API + IAM).
