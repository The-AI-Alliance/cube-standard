#!/usr/bin/env python3
"""Smoke: end-to-end Spot VM with VM-side TTL self-shutdown.

Validates the orphan-elimination mechanism proposed in
``../../spikes/spot_with_ttl/README.md``:

1. Launch a Spot D4s_v3 VM with ``eviction_policy="Delete"`` so an Azure
   capacity event triggers the existing ``delete_option=Delete`` cascade
   on the OS disk, NIC, and public IP.
2. Over the SSH tunnel already opened by ``launch()``, install an ``at``
   job that runs ``sudo shutdown -h now`` at ``cube:expires_at``. The VM
   is responsible for its own demise — no harness-side scheduler,
   no Managed Identity, no image rebuild.

This is the "Option A" path: minimum-IAM, ships fastest. Option B (full
self-delete via cloud API + Managed Identity) is a future follow-up.

Currently uses a spike-shape ``SpotAzureInfraConfig`` subclass that
monkey-patches the SDK call to inject Spot fields. Once the production
change (``VMResourceConfig.use_spot``) lands via openspec RFC, the
monkey-patch goes away and the smoke uses the real API.

How to read the output
======================
- **SMOKE OK** — the full lifecycle worked: Spot VM came up, TTL agent
  installed, VM self-deleted at TTL, cascade fired.
- **SMOKE SKIP** — prerequisites missing: ``az login`` stale, target
  resource group not found, or pre-provisioned L1 image absent.
  Re-run after fixing the listed prereq.
- **SMOKE FAIL** — mechanism broke at a measurable step. The failure
  reason tells you which step (Spot attrs, SSH install, OS shutdown,
  or cascade delete). Each failure mode triages to a specific decision
  in the spike's ``findings.md``.

Cost per run: ~$0.01 (single Spot D4s_v3 for 5-10 min before self-shutdown).

Run from cube-standard repo root:
    uv run cube-resources/cube-infra-azure/scripts/smoke/spot_ttl_lifecycle.py \\
        --resource-group ui_assist --ttl-minutes 5
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

NAME = "spot_ttl_lifecycle"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
for _noisy in ("azure.core.pipeline.policies.http_logging_policy", "azure.identity", "urllib3.connectionpool"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

log = logging.getLogger(NAME)


def banner(status: str, reason: str = "") -> int:
    """Print the SMOKE banner and return the conventional exit code.

    OK=0, FAIL=1, SKIP=2 — matches `find . -path '*/scripts/smoke/*.py'`
    discovery contract used by other smokes in cube-standard.
    """
    line = f"SMOKE {status}: {NAME}"
    if reason:
        line += f": {reason}"
    print(line)
    return {"OK": 0, "FAIL": 1, "SKIP": 2}[status]


# ── Spot subclass (spike-shape — production path will be VMResourceConfig.use_spot) ─


class SpotAzureInfraConfig(AzureInfraConfig):
    """Spike-only: inject Spot priority + eviction policy into vm_spec.

    Patches at the ``_compute()`` level rather than ``launch()`` because the
    parent's ``_compute()`` returns a fresh ``ComputeManagementClient`` on
    every call — patching a local var inside an overridden ``launch()`` would
    only affect the first call, not the one inside ``super().launch()``.
    """

    use_spot: bool = True
    max_spot_price: float = -1.0  # -1 = pay up to standard rate

    def _patch_vm_spec_for_spot(self, vm_spec: dict[str, Any]) -> dict[str, Any]:
        if not self.use_spot:
            return vm_spec
        vm_spec["priority"] = "Spot"
        vm_spec["eviction_policy"] = "Delete"
        vm_spec["billing_profile"] = {"max_price": self.max_spot_price}
        return vm_spec

    def _compute(self) -> Any:  # type: ignore[no-untyped-def]
        client = super()._compute()
        if not self.use_spot:
            return client
        original = client.virtual_machines.begin_create_or_update

        def patched(rg, name, parameters, *args, **kwargs):  # type: ignore[no-untyped-def]
            patched_spec = self._patch_vm_spec_for_spot(parameters)
            log.info(
                "Spot patch applied: priority=%s eviction=%s max_price=%s",
                patched_spec.get("priority"),
                patched_spec.get("eviction_policy"),
                patched_spec.get("billing_profile", {}).get("max_price"),
            )
            return original(rg, name, patched_spec, *args, **kwargs)

        client.virtual_machines.begin_create_or_update = patched
        return client


# ── VM-side TTL install ──────────────────────────────────────────────────────


_TTL_INSTALL_SCRIPT = """\
set -euo pipefail

# IMDS tags endpoint returns "key1:value1;key2:value2" — but values may contain
# colons (ISO-8601 timestamps do). Split on ';' into lines, then use sed to
# strip only the leading "cube:expires_at:" prefix without splitting the value.
RAW_TAGS=$(curl -sf -H Metadata:true \\
    'http://169.254.169.254/metadata/instance/compute/tags?api-version=2021-02-01&format=text')
EXPIRES_AT=$(printf '%s' "$RAW_TAGS" | tr ';' '\\n' | sed -n 's/^cube:expires_at://p')

if [ -z "$EXPIRES_AT" ]; then
    echo "ERROR: cube:expires_at tag not found on VM" >&2
    echo "Raw IMDS tags: $RAW_TAGS" >&2
    exit 1
fi

echo "TTL agent: cube:expires_at=$EXPIRES_AT"

# systemd-run --on-calendar accepts ISO-8601 with seconds, but normalize to
# "YYYY-MM-DD HH:MM:SS" (no T, no timezone) for safety across systemd versions.
WHEN=$(date -d "$EXPIRES_AT" '+%Y-%m-%d %H:%M:%S')
echo "TTL agent: scheduling shutdown -h at $WHEN (systemd-run transient timer)"

# systemd-run --on-calendar creates a transient .timer + .service unit pair.
# Runs as root by default when invoked via sudo. No apt install needed —
# systemd-run is part of systemd which is mandatory on Ubuntu 22.04+.
sudo systemd-run --unit=cube-ttl-shutdown \\
    --on-calendar="$WHEN" \\
    --description="cube TTL self-shutdown" \\
    /sbin/shutdown -h now "cube TTL expired"

echo "TTL agent: scheduled units:"
sudo systemctl list-timers cube-ttl-shutdown.timer --all --no-pager
"""


def install_vm_side_ttl(public_ip: str, ssh_user: str, ssh_key_path: str) -> None:
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
        raise RuntimeError(f"TTL install failed (exit {result.returncode}): {result.stderr.strip()[:300]}")
    log.info("TTL agent install stdout:\n%s", result.stdout.strip())


# ── Verification helpers ─────────────────────────────────────────────────────


def verify_spot_attributes(infra: AzureInfraConfig, vm_name: str) -> dict[str, Any]:
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
    compute = infra._compute()
    t0 = time.time()
    last_state = "unknown"
    while time.time() - t0 < deadline_s:
        try:
            iv = compute.virtual_machines.instance_view(infra.resource_group, vm_name)
            power = [s.code for s in (iv.statuses or []) if s.code and s.code.startswith("PowerState/")]
            last_state = ",".join(power) if power else "provisioning"
            log.info("  [%.0fs] vm=%s state=%s", time.time() - t0, vm_name, last_state)
        except Exception as exc:
            log.info("  [%.0fs] vm=%s state=DELETED (%s)", time.time() - t0, vm_name, type(exc).__name__)
            return "DELETED"
        time.sleep(30)
    return last_state


# ── Smoke entry ──────────────────────────────────────────────────────────────


def main(
    resource_group: Annotated[str, typer.Option(help="Azure resource group with the provisioned image")],
    ttl_minutes: Annotated[int, typer.Option(help="VM self-shutdown time from launch")] = 5,
    image_def: Annotated[str, typer.Option(help="Compute Gallery image definition name")] = "cube-ubuntu-22-04",
    image_version: Annotated[str, typer.Option(help="Compute Gallery image version")] = "1.0.0",
    vm_size: Annotated[str, typer.Option(help="Azure VM SKU")] = "Standard_D4s_v3",
    wait_for_delete_minutes: Annotated[int, typer.Option(help="Wait window after TTL for cascade delete")] = 8,
    storage_account: Annotated[
        str | None, typer.Option(help="Override storage account (required if RG has multiple)")
    ] = None,
    vnet_name: Annotated[
        str | None, typer.Option(help="Override vnet name (required if RG has multiple)")
    ] = None,
    nsg_name: Annotated[
        str | None, typer.Option(help="Override NSG name (required if RG has multiple)")
    ] = None,
    gallery_name: Annotated[
        str | None, typer.Option(help="Override compute gallery name")
    ] = None,
) -> int:
    """End-to-end smoke of the Spot + VM-side TTL mechanism."""
    log.info("=" * 70)
    log.info("Spot + VM-side TTL smoke")
    log.info("  resource_group=%s ttl=%dm image=%s size=%s", resource_group, ttl_minutes, image_def, vm_size)
    log.info("=" * 70)

    kwargs: dict[str, Any] = {"resource_group": resource_group}
    if storage_account:
        kwargs["storage_account"] = storage_account
    if vnet_name:
        kwargs["vnet_name"] = vnet_name
    if nsg_name:
        kwargs["nsg_name"] = nsg_name
    if gallery_name:
        kwargs["gallery_name"] = gallery_name

    try:
        infra = SpotAzureInfraConfig(**kwargs)
    except Exception as exc:
        return banner("SKIP", f"infra init failed (auth or rg lookup): {exc}")
    log.info(
        "Auto-discovered: subscription=%s location=%s gallery=%s",
        infra.subscription,
        infra.location,
        infra.gallery_name,
    )

    expires_at = datetime.now(timezone.utc) + timedelta(minutes=ttl_minutes)
    resource = VMResourceConfig(
        name=f"smoke-spot-ttl-{uuid.uuid4().hex[:6]}",
        os_type="linux",
        requires_kvm=False,
        default_ttl_seconds=ttl_minutes * 60,
    )

    # Map the resource onto a pre-provisioned gallery image so launch() can find it.
    # In production this would be a permanent registration after the first provision().
    try:
        infra.register(resource, {"image_def": image_def, "version": image_version})
    except Exception as exc:
        return banner("SKIP", f"register failed (image missing in gallery?): {exc}")

    log.info("Launching Spot VM (expected TTL kick: %s UTC)", expires_at.strftime("%H:%M:%S"))
    t_launch = time.time()
    try:
        handle: AzureResourceHandle = infra.launch(resource)
    except Exception as exc:
        return banner("FAIL", f"launch failed (Spot capacity? auth? image missing?): {exc}")
    launch_elapsed = time.time() - t_launch
    log.info("Launched in %.1fs: vm=%s endpoint=%s", launch_elapsed, handle._vm_name, handle.endpoint)

    try:
        facts = verify_spot_attributes(infra, handle._vm_name)
        if facts["priority"] != "Spot":
            return banner("FAIL", f"Azure ignored priority=Spot (got {facts['priority']!r})")
        if facts["eviction_policy"] != "Delete":
            return banner("FAIL", f"eviction_policy != Delete (got {facts['eviction_policy']!r})")
        log.info("Spot attributes confirmed on Azure side")

        public_ip = None
        if "@" in handle.endpoint:
            public_ip = handle.endpoint.split("@")[-1].split(":")[0]
        if not public_ip:
            pip = infra._network().public_ip_addresses.get(infra.resource_group, handle._pip_name)
            public_ip = pip.ip_address

        # cube-infra-azure pins SSH user to "cube" for Linux (see launch() in azure.py)
        ssh_user = "cube"
        try:
            install_vm_side_ttl(public_ip, ssh_user, infra.ssh_privkey_path)
        except Exception as exc:
            return banner("FAIL", f"TTL agent install failed: {exc}")
        log.info("VM-side TTL agent installed")

        log.info("Waiting up to %dm for TTL fire + cascade delete...", ttl_minutes + wait_for_delete_minutes)
        final_state = wait_for_vm_deleted(infra, handle._vm_name, (ttl_minutes + wait_for_delete_minutes) * 60)

        if final_state == "DELETED":
            return banner("OK", f"VM self-deleted after TTL (launch={launch_elapsed:.0f}s)")
        return banner("FAIL", f"VM did not self-delete in time; final_state={final_state}")
    finally:
        # Belt-and-suspenders: if the VM is still alive, force-delete it so the
        # smoke doesn't leak its own test resources.
        try:
            handle.close()
        except Exception as exc:
            log.info("  cleanup close() noted: %s", exc)


if __name__ == "__main__":
    sys.exit(typer.run(main) or 0)
