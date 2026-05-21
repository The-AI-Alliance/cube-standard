# Findings — Spot + VM-side TTL spike

**Run by:**
**Date:**
**Subscription:** `ServiceNow AI Research` (aeb958d3-a614-450e-94bc-88f284dc0664)
**Resource group:**
**Region:**
**VM SKU:**

## Run log

Paste relevant log excerpts from `spike_spot_vm_ttl.py` here.

## Measurements

| Measurement | Observed | Notes |
|---|---|---|
| Provision time (launch → SSH ready) | ___ s | Compare to regular-priority baseline (typically 3-5 min) |
| Spot vs regular hourly rate | $___ / $0.19 | Read `billing_profile.max_price` and Azure pricing page |
| TTL agent install duration | ___ s | The added SSH round-trip overhead |
| Time from TTL deadline to VM stopped | ___ s | `at` job firing latency |
| Time from VM stopped to VM record deleted | ___ s | Cascade-delete behaviour |
| Total resources cleaned (VM + disk + NIC + IP) | ___ / 4 | Should be 4/4 if `delete_option=Delete` cascade works |

## Behaviour observed

### Did the Spot VM provision successfully?
- [ ] Yes — `priority=Spot, eviction_policy=Delete, max_price=...`
- [ ] No — failure mode:

### Did the SSH install land?
- [ ] Yes — `atq` showed the scheduled job
- [ ] No — failure mode:

### Did the VM self-shutdown at TTL?
- [ ] Yes — power state went to Stopped within X seconds of deadline
- [ ] No — power state stayed Running past deadline + Ys

### Did the cascade delete fire?
- [ ] Yes — VM + OS disk + NIC + public IP all deleted
- [ ] Partial — VM gone but ___ remained
- [ ] No — VM record still present, manually checked with `az vm show`

## Decision

- [ ] **GO** — open openspec RFC for production change
- [ ] **GO with caveats:**
- [ ] **NO-GO** — fall back to Option B (Managed Identity + cloud API self-delete)

## Open questions raised by the spike

- Anything surprising about Spot pricing, capacity, or eviction signalling
- Any benchmark-specific compatibility issues (e.g. WAA Windows-on-Spot pricing)
- Anything that breaks if we make this opt-in vs default-on

## Next steps

If GO:
1. Open openspec change folder in cube-standard for `VMResourceConfig.use_spot` + `max_spot_price`
2. Promote the SSH-install logic from monkey-patch into a clean `_install_ttl_agent()` method on `AzureInfraConfig`
3. Add a smoke test (`scripts/smoke/spot_ttl_lifecycle.py`) that runs the same flow but exits in <5 min
4. Add `use_spot=True` default on OSWorld and WAA `VMResourceConfig`s (with explicit opt-out for the rare long-running tasks)
5. Plan eviction-rate observability via `RETRIABLE_STATUSES` counts in the harness summary
