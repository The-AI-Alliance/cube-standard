#!/usr/bin/env python3
"""Spike: launch a Spot VM with VM-side TTL shutdown.

End-to-end demonstration of the proposed orphan-elimination mechanism:

1. Launch a Spot D4s_v3 VM with ``eviction_policy="Delete"`` so an Azure
   capacity event triggers the existing ``delete_option=Delete`` cascade
   on the OS disk, NIC, and public IP.
2. Over the existing SSH tunnel opened by ``launch()``, install an ``at``
   job that runs ``sudo shutdown -h now`` at ``cube:expires_at``. The VM
   is responsible for its own demise — no harness-side scheduler,
   no Managed Identity, no image rebuild.

This script is NOT a refactor of ``AzureInfraConfig.launch()``. It monkey-
patches the Spot fields onto ``vm_spec`` via a subclass so the spike stays
self-contained. The production change would go through openspec and add
``use_spot`` / ``max_spot_price`` fields to ``VMResourceConfig``.

Cost per run: ~$0.01 (single Spot D4s_v3 for 5-10 min before self-shutdown).

Usage:

    cd cube-resources/cube-infra-azure
    uv run python spikes/spot_with_ttl/spike_spot_vm_ttl.py \\
        --resource-group ui_assist \\
        --ttl-minutes 5

Then watch the VM lifecycle in another terminal:

    watch -n 30 'az vm list -g ui_assist --query "[?contains(name, \\'spike-spot\\')].{name:name, state:powerState}" -o table'
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any

import typer
from cube_infra_azure import AzureInfraConfig
from cube_infra_azure.azure import AzureResourceHandle

from cube.resource import VMResourceConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
for _noisy in ("azure.core.pipeline.policies.http_logging_policy", "azure.identity", "urllib3.connectionpool"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

log = logging.getLogger("spike_spot_vm_ttl")


class SpotAzureInfraConfig(AzureInfraConfig):
    """Spike-only: inject Spot priority + eviction policy into vm_spec.

    A clean production refactor would route these through ``VMResourceConfig``
    fields (``use_spot``, ``max_spot_price``) and read them at the same point
    where ``vm_spec`` is built. For the spike, monkey-patching the dict via
    a subclass keeps the diff to the prod code minimal.
    """

    use_spot: bool = True
    max_spot_price: float = -1.0  # -1 = pay up to standard rate (best-effort capacity)

    def _patch_vm_spec_for_spot(self, vm_spec: dict[str, Any]) -> dict[str, Any]:
        if not self.use_spot:
            return vm_spec
        vm_spec["priority"] = "Spot"
        vm_spec["eviction_policy"] = "Delete"  # Delete → triggers delete_option=Delete cascade
        vm_spec["billing_profile"] = {"max_price": self.max_spot_price}
        return vm_spec

    def launch(self, resource):  # type: ignore[no-untyped-def]
        # Wrap parent launch() with a monkey-patch on the SDK's create_or_update
        # — we cannot easily override only the vm_spec construction without
        # duplicating ~100 lines of parent code. Spike-friendly hack.
        compute = self._compute()
        original = compute.virtual_machines.begin_create_or_update

        def patched(rg, name, parameters, *args, **kwargs):  # type: ignore[no-untyped-def]
            patched_spec = self._patch_vm_spec_for_spot(parameters)
            log.info(
                "Spot patch applied: priority=%s eviction=%s max_price=%s",
                patched_spec.get("priority"),
                patched_spec.get("eviction_policy"),
                patched_spec.get("billing_profile", {}).get("max_price"),
            )
            return original(rg, name, patched_spec, *args, **kwargs)

        compute.virtual_machines.begin_create_or_update = patched
        try:
            return super().launch(resource)
        finally:
            compute.virtual_machines.begin_create_or_update = original


# ── VM-side TTL install ──────────────────────────────────────────────────────


_TTL_INSTALL_SCRIPT = """\
set -euo pipefail

# 1. Read cube:expires_at from instance metadata
EXPIRES_AT=$(curl -sf -H Metadata:true \\
    'http://169.254.169.254/metadata/instance/compute/tags?api-version=2021-02-01&format=text' \\
    | tr ';' '\\n' | awk -F: '/^cube:expires_at:/ {print $2}')

if [ -z "$EXPIRES_AT" ]; then
    echo "ERROR: cube:expires_at tag not found on VM" >&2
    exit 1
fi

echo "TTL agent: cube:expires_at=$EXPIRES_AT"

# 2. Convert ISO-8601 to `at` format and schedule
AT_TIME=$(date -d "$EXPIRES_AT" '+%H:%M %Y-%m-%d')
echo "TTL agent: scheduling sudo shutdown -h at $AT_TIME"

# Install `at` if missing (Ubuntu base images usually have it; safety check)
which at >/dev/null 2>&1 || sudo apt-get install -y -qq at

# Ensure atd is running
sudo systemctl is-active atd >/dev/null 2>&1 || sudo systemctl start atd

echo "sudo shutdown -h now 'cube TTL expired'" | at $AT_TIME

# 3. Echo the scheduled queue so the caller can verify
echo "TTL agent: scheduled jobs:"
atq
"""


def install_vm_side_ttl(public_ip: str, ssh_user: str, ssh_key_path: str) -> None:
    """SSH to the VM and install the at-based TTL self-shutdown.

    Best-effort: if this fails, the VM still has the cube:expires_at tag
    (for cleanup_stale to find later) and will be evicted whenever Azure
    needs the Spot capacity. The TTL agent is an additional belt to the
    Spot-eviction suspenders.
    """
    log.info("Installing VM-side TTL agent over SSH to %s@%s", ssh_user, public_ip)
    cmd = [
        "ssh",
        "-i",
        ssh_key_path,
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "ConnectTimeout=30",
        f"{ssh_user}@{public_ip}",
        "bash -s",
    ]
    result = subprocess.run(cmd, input=_TTL_INSTALL_SCRIPT, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        log.error("TTL install failed (exit %d). stderr:\n%s", result.returncode, result.stderr)
        raise RuntimeError("VM-side TTL install failed — see stderr above")
    log.info("TTL agent install stdout:\n%s", result.stdout.strip())


# ── Verification helpers ─────────────────────────────────────────────────────


def verify_spot_attributes(infra: AzureInfraConfig, vm_name: str) -> dict[str, Any]:
    """Read back the VM's Azure record and confirm Spot fields are set."""
    compute = infra._compute()
    vm = compute.virtual_machines.get(infra.resource_group, vm_name)
    facts = {
        "priority": getattr(vm, "priority", None),
        "eviction_policy": getattr(vm, "eviction_policy", None),
        "billing_profile.max_price": getattr(getattr(vm, "billing_profile", None), "max_price", None),
        "vm_size": vm.hardware_profile.vm_size if vm.hardware_profile else None,
    }
    log.info("Azure-side Spot facts: %s", facts)
    return facts


def wait_for_vm_deleted(infra: AzureInfraConfig, vm_name: str, deadline_s: float) -> str:
    """Poll until the VM is either gone or deadline expires. Returns final state."""
    compute = infra._compute()
    t0 = time.time()
    last_state = "unknown"
    while time.time() - t0 < deadline_s:
        try:
            vm = compute.virtual_machines.get(infra.resource_group, vm_name)
            iv = compute.virtual_machines.instance_view(infra.resource_group, vm_name)
            power = [s.code for s in (iv.statuses or []) if s.code and s.code.startswith("PowerState/")]
            last_state = ",".join(power) if power else (vm.provisioning_state or "unknown")
            log.info("  [%.0fs] vm=%s state=%s", time.time() - t0, vm_name, last_state)
        except Exception as exc:
            # When the VM is deleted, .get() raises ResourceNotFoundError
            log.info("  [%.0fs] vm=%s state=DELETED (%s)", time.time() - t0, vm_name, type(exc).__name__)
            return "DELETED"
        time.sleep(30)
    return last_state


# ── Spike entry ──────────────────────────────────────────────────────────────


def main(
    resource_group: Annotated[str, typer.Option(help="Azure resource group with the provisioned image")],
    ttl_minutes: Annotated[int, typer.Option(help="VM self-shutdown time from launch")] = 5,
    image_def: Annotated[str, typer.Option(help="Compute Gallery image definition name")] = "cube-ubuntu-22-04",
    vm_size: Annotated[str, typer.Option(help="Azure VM SKU")] = "Standard_D4s_v3",
    wait_for_delete_minutes: Annotated[int, typer.Option(help="How long to wait after TTL for delete to complete")] = 8,
) -> None:
    """Run the Spot + VM-side TTL spike end-to-end."""
    log.info("=" * 70)
    log.info("Spot + VM-side TTL spike")
    log.info("  resource_group=%s ttl=%dm image=%s size=%s", resource_group, ttl_minutes, image_def, vm_size)
    log.info("=" * 70)

    infra = SpotAzureInfraConfig(resource_group=resource_group)
    log.info(
        "Auto-discovered: subscription=%s location=%s gallery=%s",
        infra.subscription,
        infra.location,
        infra.gallery_name,
    )

    expires_at = datetime.now(timezone.utc) + timedelta(minutes=ttl_minutes)
    resource = VMResourceConfig(
        name=f"spike-spot-ttl-{uuid.uuid4().hex[:6]}",
        os_type="linux",
        requires_kvm=False,
        default_ttl_seconds=ttl_minutes * 60,
    )

    log.info("Launching Spot VM (expected TTL kick: %s UTC)", expires_at.strftime("%H:%M:%S"))
    t_launch = time.time()
    handle: AzureResourceHandle = infra.launch(resource)
    launch_elapsed = time.time() - t_launch
    log.info("Launched in %.1fs: vm=%s endpoint=%s", launch_elapsed, handle._vm_name, handle.endpoint)

    try:
        facts = verify_spot_attributes(infra, handle._vm_name)
        assert facts["priority"] == "Spot", f"Expected priority=Spot, got {facts['priority']!r}"
        assert facts["eviction_policy"] == "Delete", (
            f"Expected eviction_policy=Delete, got {facts['eviction_policy']!r}"
        )
        log.info("✓ Spot attributes confirmed on Azure side")

        public_ip = handle.endpoint.split("@")[-1].split(":")[0] if "@" in handle.endpoint else None
        if not public_ip:
            # Fall back: read the public IP from the handle's PIP resource name
            pip = infra._network().public_ip_addresses.get(infra.resource_group, handle._pip_name)
            public_ip = pip.ip_address

        ssh_user = infra.user
        ssh_key = infra.ssh_privkey_path
        install_vm_side_ttl(public_ip, ssh_user, ssh_key)
        log.info("✓ VM-side TTL agent installed")

        log.info("Waiting %dm for TTL to fire + cascade delete...", ttl_minutes + wait_for_delete_minutes)
        final_state = wait_for_vm_deleted(
            infra,
            handle._vm_name,
            deadline_s=(ttl_minutes + wait_for_delete_minutes) * 60,
        )

        if final_state == "DELETED":
            log.info("✓ VM auto-deleted after TTL — cascade fired")
            log.info("SPIKE OK: end-to-end Spot + VM-side TTL works in this subscription")
            sys.exit(0)
        else:
            log.warning("⚠ VM did not self-delete in time. Final state: %s", final_state)
            log.warning("Possible causes: at job didn't fire, shutdown didn't trigger cascade,")
            log.warning("or the cascade is in progress but slower than wait window.")
            log.warning("Inspect manually:")
            log.warning("  az vm show -g %s -n %s --query 'powerState' -o tsv", resource_group, handle._vm_name)
            sys.exit(2)
    finally:
        # Belt-and-suspenders: if the VM is still alive, force-delete it so the
        # spike doesn't leak its own test resources. The whole point of the
        # spike is to prove the mechanism works without this fallback.
        log.info("Cleanup: handle.close() (no-op if VM already gone)")
        try:
            handle.close()
        except Exception as exc:
            log.info("  close() noted: %s", exc)


if __name__ == "__main__":
    typer.run(main)
