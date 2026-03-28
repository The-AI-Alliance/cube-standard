"""
Integration test: full resource lifecycle + run_debug_agent(OSWorldBenchmark, AzureInfraConfig)

Uses image_name_suffix="-test" so the test creates its own gallery image definition
("osworld-ubuntu-vm-test") and ProvisionStore key, leaving the team's manually-created
"osworld-ubuntu-vm/1.0.0" untouched.

Steps:
  1. Clean up any stale "-test" VMs from previous runs (tests list_active + close)
  2. Unprovision the test image if it exists (delete gallery image + ProvisionStore entry)
  3. Provision from scratch: full ~40-min bootstrap pipeline
  4. run_debug_agent: provision check → capability check → launch + HTTP probe
  5. Unprovision: clean up the test gallery image after the run

Run:
    cd experiments/azure-vm-backend
    .venv/bin/python test_run_debug_agent.py
"""
from __future__ import annotations

import json
import logging
import sys

from _common import configure_logging
from cube_infra_azure import AzureInfraConfig
from osworld_cube.benchmark import OSWorldBenchmark

from cube.testing import run_debug_agent

configure_logging(debug=False)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

INFRA = AzureInfraConfig(
    resource_group="ui_assist",
    # needed because this resource group has multiple of each:
    storage_account="cubeexpvhd",
    vnet_name="vnet-westus2",
    nsg_name="osworld-nsg",
    # isolates test from the team's manually-created "osworld-ubuntu-vm/1.0.0"
    image_name_suffix="-test",
)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("=== integration test: resource lifecycle + run_debug_agent ===")
    log.info("infra: %s", INFRA.fingerprint())

    benchmark = OSWorldBenchmark()
    resources = benchmark.list_resources()
    log.info("benchmark resources: %s", [r.name for r in resources])

    # ── Step 1: clean up any stale test VMs from previous runs ────────────────
    active = INFRA.list_active()
    if active:
        log.info("Step 1: found %d active VM(s) — cleaning up", len(active))
        for handle in active:
            log.info("  closing run_id=%s", handle.run_id[:8])
            handle.close()
        log.info("Step 1: cleanup done")
    else:
        log.info("Step 1: no active VMs")

    # ── Step 2: unprovision test image (clean slate for full reprovision) ──────
    for resource in resources:
        status = INFRA.provision_status(resource)
        log.info("Step 2: provision_status(%s-test) = %s", resource.name, status)
        if status == "ready":
            log.info("Step 2: unprovisioning stale test image for %s …", resource.name)
            INFRA.unprovision(resource)
            log.info("Step 2: unprovision done")

    # ── Step 3: provision from scratch (~40 min bootstrap pipeline) ───────────
    for resource in resources:
        log.info("Step 3: provisioning %s-test (this takes ~40 min) …", resource.name)
        INFRA.provision(resource)
        log.info("Step 3: provision_status(%s-test) = %s", resource.name, INFRA.provision_status(resource))

    # ── Step 4: run_debug_agent (provision + capability + launch checks) ───────
    log.info("Step 4: running run_debug_agent …")
    report = run_debug_agent(benchmark, INFRA)

    log.info("=== Report ===")
    print(json.dumps(report, indent=2, default=str))

    # ── Step 5: unprovision test image (cleanup) ───────────────────────────────
    for resource in resources:
        log.info("Step 5: cleaning up test image for %s …", resource.name)
        INFRA.unprovision(resource)
    log.info("Step 5: cleanup done")

    if report.get("launch_ok"):
        log.info("SUCCESS — infra is ready for a full OSWorld evaluation run.")
    else:
        log.error("FAILED — see 'error' field in report above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
