#!/usr/bin/env python3
"""Smoke: launch an Azure VM with or without Spot pricing, end-to-end.

Exercises the production ``VMResourceConfig.use_spot`` field added in
this PR. Launches a real Azure VM via cube-infra-azure, then reads back
the VM record from the Azure ARM API to assert the ``priority`` field
matches the requested mode:

  --use-spot      → expect ``priority == "Spot"`` + ``eviction_policy == "Delete"``
                    + ``billing_profile.max_price`` set
  --no-use-spot   → expect ``priority`` unset (Regular pricing)

This is the live-cloud counterpart to the unit tests in
``cube-resources/cube-infra-azure/tests/test_azure_infra.py`` — those
mock the SDK and only verify what cube-infra-azure *intends* to send;
this smoke verifies that Azure *accepts* the request and the VM
actually comes up with the right pricing tier.

Why a flag rather than two tests
================================

The two modes share 95% of the lifecycle (provision check, register,
launch, verify, cleanup). Splitting into two smokes would duplicate the
plumbing and double the wall-clock for a no-regression check. One
parameterised smoke lets you re-run either mode on demand, and pick the
mode that matches what you just changed.

How to read the output
======================

- **SMOKE OK** — VM launched, Azure-side ``priority`` matched the
  requested mode, teardown completed.
- **SMOKE FAIL** — VM launched but the priority round-trip didn't match,
  OR launch failed mid-flight. The failure reason names the step.
- **SMOKE SKIP** — ``cube_infra_azure`` not installed OR ``az login``
  credentials missing OR the configured resource group can't be reached.

Cost: ~$0.01-0.03 per run (one D4s_v3 VM for the 90s of provisioning +
verification + teardown). Spot mode is ~70% cheaper than Regular mode
but both are dominated by the provisioning overhead, so the price
difference is negligible at this duration.

Run from cube-standard repo root:

    uv run cube-resources/cube-infra-azure/scripts/smoke/spot_vm_launch.py \\
        --resource-group ui_assist --use-spot

    uv run cube-resources/cube-infra-azure/scripts/smoke/spot_vm_launch.py \\
        --resource-group ui_assist --no-use-spot
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import time
import uuid
from typing import Annotated, Any

import typer

NAME = "spot_vm_launch"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
for _noisy in ("azure.core.pipeline.policies.http_logging_policy", "azure.identity", "urllib3.connectionpool"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

log = logging.getLogger(NAME)


def banner(status: str, reason: str = "") -> int:
    """Print the SMOKE banner and return the conventional exit code."""
    line = f"SMOKE {status}: {NAME}"
    if reason:
        line += f": {reason}"
    print(line)
    return {"OK": 0, "FAIL": 1, "SKIP": 2}[status]


def main(
    resource_group: Annotated[str, typer.Option(help="Azure resource group with the provisioned image.")],
    use_spot: Annotated[
        bool,
        typer.Option(
            "--use-spot/--no-use-spot",
            help="Whether to request Spot pricing for the launched VM.",
        ),
    ] = True,
    max_spot_price: Annotated[
        float | None,
        typer.Option(
            help="Max hourly USD price for Spot (None = pay up to standard rate). "
            "Only meaningful when --use-spot is set."
        ),
    ] = None,
    image_def: Annotated[str, typer.Option(help="Compute Gallery image definition name.")] = "cube-ubuntu-22-04",
    image_version: Annotated[str, typer.Option(help="Compute Gallery image version.")] = "1.0.0",
    storage_account: Annotated[str | None, typer.Option(help="Override storage account.")] = None,
    vnet_name: Annotated[str | None, typer.Option(help="Override vnet name.")] = None,
    nsg_name: Annotated[str | None, typer.Option(help="Override NSG name.")] = None,
    gallery_name: Annotated[str | None, typer.Option(help="Override compute gallery name.")] = None,
) -> int:
    """Launch a VM with the requested pricing tier and verify Azure honoured it."""
    if importlib.util.find_spec("cube_infra_azure") is None:
        return banner("SKIP", "cube_infra_azure not installed")

    # Lazy import so SKIP cases above don't pay the cost.
    from cube_infra_azure import AzureInfraConfig
    from cube_infra_azure.azure import AzureResourceHandle

    from cube.resource import VMResourceConfig

    log.info("=" * 70)
    log.info("Spot VM launch smoke")
    log.info(
        "  rg=%s, use_spot=%s, max_spot_price=%s, image=%s/%s",
        resource_group,
        use_spot,
        max_spot_price,
        image_def,
        image_version,
    )
    log.info("=" * 70)

    kwargs: dict[str, Any] = {
        "resource_group": resource_group,
        "default_ttl_seconds": 600,  # 10-min TTL — generous for the smoke
    }
    if storage_account:
        kwargs["storage_account"] = storage_account
    if vnet_name:
        kwargs["vnet_name"] = vnet_name
    if nsg_name:
        kwargs["nsg_name"] = nsg_name
    if gallery_name:
        kwargs["gallery_name"] = gallery_name

    try:
        infra = AzureInfraConfig(**kwargs)
    except Exception as exc:
        return banner("SKIP", f"infra init failed (auth or rg lookup): {exc}")
    log.info(
        "Auto-discovered: subscription=%s location=%s gallery=%s",
        infra.subscription,
        infra.location,
        infra.gallery_name,
    )

    resource = VMResourceConfig(
        name=f"smoke-spot-vm-{uuid.uuid4().hex[:6]}",
        os_type="linux",
        requires_kvm=False,
        use_spot=use_spot,
        max_spot_price=max_spot_price,
    )

    # Register the resource against the pre-existing L1 image (so launch() can find it
    # without a 30-90 min provision()).
    try:
        infra.register(resource, {"image_def": image_def, "version": image_version})
    except Exception as exc:
        return banner("SKIP", f"register failed (image missing in gallery?): {exc}")

    log.info("Launching VM (this typically takes ~60-90s)...")
    t_launch = time.time()
    try:
        handle: AzureResourceHandle = infra.launch(resource)
    except Exception as exc:
        return banner("FAIL", f"launch failed: {exc}")
    launch_elapsed = time.time() - t_launch
    log.info("Launched in %.0fs: vm=%s", launch_elapsed, handle._vm_name)

    try:
        # Read the VM back from Azure to verify the priority round-trip.
        compute = infra._compute()
        vm = compute.virtual_machines.get(infra.resource_group, handle._vm_name)
        observed_priority = getattr(vm, "priority", None)
        observed_eviction = getattr(vm, "eviction_policy", None)
        observed_max_price = getattr(getattr(vm, "billing_profile", None), "max_price", None)
        log.info(
            "Azure-side facts: priority=%r, eviction_policy=%r, max_price=%r",
            observed_priority,
            observed_eviction,
            observed_max_price,
        )

        expected_priority = "Spot" if use_spot else None
        if observed_priority != expected_priority:
            return banner(
                "FAIL",
                f"priority mismatch — requested use_spot={use_spot} but Azure reports "
                f"priority={observed_priority!r} (expected {expected_priority!r})",
            )

        if use_spot:
            if observed_eviction != "Delete":
                return banner(
                    "FAIL",
                    f"eviction_policy mismatch — Spot VMs must carry 'Delete' "
                    f"(for cascade-cleanup); got {observed_eviction!r}",
                )
            expected_max_price = max_spot_price if max_spot_price is not None else -1.0
            if observed_max_price != expected_max_price:
                return banner(
                    "FAIL",
                    f"max_price mismatch — expected {expected_max_price}, got {observed_max_price}",
                )

        return banner(
            "OK",
            f"VM launched with priority={observed_priority!r} as requested (launch={launch_elapsed:.0f}s)",
        )
    finally:
        log.info("Cleanup: handle.close() — deleting %s + cascade (disk/NIC/IP)", handle._vm_name)
        try:
            handle.close()
        except Exception as exc:
            log.warning("  close() noted: %s", exc)


if __name__ == "__main__":
    sys.exit(typer.run(main) or 0)
