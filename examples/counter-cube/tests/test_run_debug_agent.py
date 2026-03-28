"""
Tests for run_debug_agent with counter-cube and LocalInfraConfig.

Covers three paths:
1. Benchmark with no resources — trivially passes (counter-cube)
2. Benchmark with a VMResourceConfig not yet provisioned — provision check fails
3. Benchmark with registered resource but infra lacks capability — capability check fails
4. Both checks pass, launch raises (StubInfra) — launch_ok=False, error set
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from counter_cube.benchmark import CounterBenchmark
from cube.resource import InfraConfig, ResourceConfig, ResourceHandle, VMResourceConfig
from cube.provision_store import ProvisionStore
from cube.testing import run_debug_agent
from cube import LocalInfraConfig


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def tmp_store(tmp_path):
    """Return a ProvisionStore backed by a temp file, and patch the module default."""
    store_path = tmp_path / "provisions.json"
    store = ProvisionStore(path=store_path)
    # Patch the default so any ProvisionStore() call inside InfraConfig methods
    # uses our temp file rather than ~/.cube/provisions.json.
    with patch("cube.provision_store._DEFAULT_STORE_PATH", store_path):
        yield store


# ── Minimal benchmark that declares one VMResourceConfig ─────────────────────


class VMBenchmark(CounterBenchmark):
    """CounterBenchmark subclass that pretends to need a VM."""

    # Re-declare so Benchmark.__init_subclass__ finds them in cls.__dict__
    benchmark_metadata = CounterBenchmark.benchmark_metadata
    task_metadata = CounterBenchmark.task_metadata
    task_config_class = CounterBenchmark.task_config_class

    def list_resources(self) -> list[ResourceConfig]:
        return [VMResourceConfig(name="test-vm", scope="task", source_url=None)]


# ── InfraConfig stub (no actual QEMU) ────────────────────────────────────────


class StubInfra(InfraConfig):
    """InfraConfig that reports kvm capability but never actually launches."""

    def fingerprint(self) -> str:
        return "stub:local"

    def capabilities(self) -> set[str]:
        return {"kvm"}

    def provision(self, resource: ResourceConfig) -> None:  # pragma: no cover
        pass

    def launch(self, resource, run_id, ttl_seconds=None) -> ResourceHandle:
        raise NotImplementedError("StubInfra cannot actually launch VMs")

    def list_active(self, run_id=None):
        return []

    def cleanup(self, run_id: str) -> None:
        pass

    def cleanup_stale(self, max_age_seconds=None):
        return []


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_no_resources_passes(tmp_store):
    """Counter-cube declares no resources → run_debug_agent returns all-OK."""
    result = run_debug_agent(CounterBenchmark(), LocalInfraConfig())
    assert result["resources_checked"] == 0
    assert result["provision_ok"] is True
    assert result["capabilities_ok"] is True
    assert result["launch_ok"] is True
    assert result["error"] is None


def test_provision_check_fails_when_unregistered(tmp_store):
    """Benchmark with VMResourceConfig but nothing in the store → provision check fails."""
    result = run_debug_agent(VMBenchmark(), StubInfra())

    assert result["resources_checked"] == 1
    assert result["provision_ok"] is False
    assert result["capabilities_ok"] is False  # never reached
    assert result["launch_ok"] is False         # never reached
    assert "not registered" in result["error"]


def test_capability_check_fails_when_infra_lacks_kvm(tmp_store):
    """Resource requires kvm but infra doesn't have it → capability check fails."""

    class NoKvmInfra(StubInfra):
        def capabilities(self) -> set[str]:
            return set()

    infra = NoKvmInfra()
    resource = VMResourceConfig(name="test-vm", scope="task")
    tmp_store.put(resource, infra, {"image_path": "/tmp/fake.qcow2"})

    result = run_debug_agent(VMBenchmark(), infra)

    assert result["provision_ok"] is True
    assert result["capabilities_ok"] is False
    assert result["launch_ok"] is False
    assert "capability mismatch" in result["error"]


def test_launch_check_runs_after_both_checks_pass(tmp_store):
    """Provision + capability OK, launch raises NotImplementedError → error captured."""
    infra = StubInfra()
    resource = VMResourceConfig(name="test-vm", scope="task")
    tmp_store.put(resource, infra, {"image_path": "/tmp/fake.qcow2"})

    result = run_debug_agent(VMBenchmark(), infra)

    assert result["provision_ok"] is True
    assert result["capabilities_ok"] is True
    assert result["launch_ok"] is False
    assert "NotImplementedError" in result["error"]
