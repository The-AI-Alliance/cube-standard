"""
Integration test: provision + launch a DockerServiceConfig (WebArena shopping_admin) on Azure.

Uses a single Docker container (shopping_admin) to keep iteration fast:
  - Docker image: am1n3e/webarena-verified-shopping_admin (~4 GB)
  - Provision time: ~20-30 min (first run only — idempotent on subsequent runs)
  - Launch time: ~3-5 min

Steps:
  1. Unprovision stale test image (clean slate)
  2. provision() — launches marketplace Ubuntu VM, installs Docker, pulls image, snapshots
  3. launch() — starts VM from snapshot, runs docker container, opens SSH tunnels
  4. Health-check both tunneled endpoints (web + env-ctrl)
  5. handle.close() — stops VM
  6. unprovision() — deletes gallery image + ProvisionStore entry

Run:
    cd cube-resources/cube-infra-azure
    uv run python test_docker_provision_azure.py

To skip reprovisioning (reuse an already-provisioned image):
    SKIP_PROVISION=1 uv run python test_docker_provision_azure.py
"""

from __future__ import annotations

import logging
import os
import sys
import urllib.request

from cube.resource import DockerServiceConfig
from cube_infra_azure import AzureInfraConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
for _noisy in ("azure.core.pipeline.policies.http_logging_policy", "azure.identity", "urllib3.connectionpool"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

log = logging.getLogger(__name__)

# ── Resource declaration ───────────────────────────────────────────────────────
# shopping_admin only — single container, tasks 0+1 in WAV debug suite.
# Ports match webarena-verified default ContainerConfig for SHOPPING_ADMIN.

WEBARENA_SHOPPING_ADMIN = DockerServiceConfig(
    name="webarena-shopping-admin-test",
    scope="benchmark",
    docker_images=["am1n3e/webarena-verified-shopping_admin"],
    services={
        "shopping_admin": 7780,  # web UI (container_port=80 mapped to 7780)
        "shopping_admin_ctrl": 7781,  # env-ctrl API (env_ctrl_port=8877 mapped to 7781)
    },
    launch_script="""\
docker run -d \\
    --name webarena_shopping_admin \\
    -p 7780:80 \\
    -p 7781:8877 \\
    am1n3e/webarena-verified-shopping_admin
# Wait for the container to be healthy (up to 60s)
for i in $(seq 1 30); do
    curl -sf http://localhost:7780/ > /dev/null 2>&1 && echo "healthy" && break
    sleep 2
done
""",
)

INFRA = AzureInfraConfig(
    resource_group="ui_assist",
    storage_account="cubeexpvhd",
    vnet_name="vnet-westus2",
    nsg_name="osworld-nsg",
    # image_name_suffix isolates this test from production images
    image_name_suffix="-test",
)

SKIP_PROVISION = os.environ.get("SKIP_PROVISION", "").strip() in ("1", "true", "yes")


def _healthcheck(name: str, url: str, timeout: int = 10) -> bool:
    """Try a GET request; return True if we get any HTTP response."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            log.info("  %s → HTTP %d ✓", name, resp.status)
            return True
    except Exception as exc:
        log.warning("  %s → %s", name, exc)
        return False


def main() -> None:
    log.info("=== integration test: DockerServiceConfig provision + launch (Azure) ===")
    log.info("infra: %s", INFRA.fingerprint())
    log.info("resource: %s  images=%s", WEBARENA_SHOPPING_ADMIN.name, WEBARENA_SHOPPING_ADMIN.docker_images)

    # ── Step 1: clean up any active test VMs ──────────────────────────────────
    active = INFRA.list_active()
    if active:
        log.info("Step 1: found %d active VM(s) — cleaning up", len(active))
        for h in active:
            h.close()
    else:
        log.info("Step 1: no active VMs")
    INFRA.cleanup_orphaned_resources()

    # ── Step 2: optionally unprovision stale test image ───────────────────────
    if not SKIP_PROVISION:
        if INFRA.provision_status(WEBARENA_SHOPPING_ADMIN) == "ready":
            log.info("Step 2: unprovisioning stale test image …")
            INFRA.unprovision(WEBARENA_SHOPPING_ADMIN)
        else:
            log.info("Step 2: no stale image — skipping")

        # ── Step 3: provision from scratch ────────────────────────────────────
        log.info("Step 3: provisioning (install Docker + docker pull) — ~20-30 min …")
        INFRA.provision(WEBARENA_SHOPPING_ADMIN)
        log.info("Step 3: provisioned ✓  status=%s", INFRA.provision_status(WEBARENA_SHOPPING_ADMIN))
    else:
        log.info("Step 2+3: SKIP_PROVISION=1 — reusing existing image")
        if INFRA.provision_status(WEBARENA_SHOPPING_ADMIN) != "ready":
            log.error("No provisioned image found — run without SKIP_PROVISION=1 first")
            sys.exit(1)

    # ── Step 4: launch VM + start container + open tunnels ────────────────────
    log.info("Step 4: launching VM and starting shopping_admin container (~3-5 min) …")
    handle = INFRA.launch(WEBARENA_SHOPPING_ADMIN)
    log.info("Step 4: launched ✓  endpoints=%s", handle.endpoints)

    # ── Step 5: health-check tunneled endpoints ───────────────────────────────
    log.info("Step 5: health-checking endpoints …")
    results = {}
    for name, url in handle.endpoints.items():
        results[name] = _healthcheck(name, url)

    if not all(results.values()):
        log.error("Health-check failures: %s", {k: v for k, v in results.items() if not v})
        handle.close()
        sys.exit(1)
    log.info("Step 5: all endpoints reachable ✓")

    # ── Step 6: close (stop VM) ───────────────────────────────────────────────
    log.info("Step 6: closing handle (stopping VM) …")
    handle.close()
    log.info("Step 6: closed ✓")

    # ── Step 7: unprovision (cleanup gallery image) ───────────────────────────
    log.info("Step 7: unprovisioning test image …")
    INFRA.unprovision(WEBARENA_SHOPPING_ADMIN)
    log.info("Step 7: unprovisioned ✓")

    log.info("=== SUCCESS — DockerServiceConfig provision + launch on Azure works ===")


if __name__ == "__main__":
    main()
