"""Unit tests for resource lifecycle abstractions.

Tests ProvisionStore, ResourceConfig subclasses, InfraConfig base behaviour
(register, provision_status, can_serve), and ResourceHandle context manager.

All tests run without cloud credentials — no Azure/AWS SDK calls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass as _dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from cube.provision_store import ProvisionStore
from cube.resource import (
    DockerImageConfig,
    DockerServiceConfig,
    InfraConfig,
    ResourceConfig,
    ResourceHandle,
    ResourceNotReadyError,
    UnsupportedResourceType,
    VMResourceConfig,
)

# ── Minimal concrete InfraConfig for testing ──────────────────────────────────


@_dataclass
class _StubHandle(ResourceHandle):
    closed: bool = False

    def close(self) -> None:
        self.closed = True


class _StubInfra(InfraConfig):
    """Minimal concrete InfraConfig — no cloud calls, for unit testing."""

    name: str = "stub"
    caps: list[str] = []

    def fingerprint(self) -> str:
        return f"stub:{self.name}"

    def capabilities(self) -> set[str]:
        return set(self.caps)

    def provision(self, resource: ResourceConfig) -> None:
        raise NotImplementedError

    def launch(self, resource: ResourceConfig, run_id: str, ttl_seconds: int | None = None) -> ResourceHandle:
        raise NotImplementedError

    def list_active(self, run_id: str | None = None) -> list[ResourceHandle]:
        return []

    def cleanup(self, run_id: str) -> None:
        pass

    def cleanup_stale(self, max_age_seconds: int | None = None) -> list[str]:
        return []


# ── ProvisionStore ─────────────────────────────────────────────────────────────


class TestProvisionStore:
    def test_get_returns_none_when_empty(self, tmp_path: Path) -> None:
        store = ProvisionStore(tmp_path / "provisions.json")
        resource = VMResourceConfig(name="my-vm")
        infra = _StubInfra(name="us-east")
        assert store.get(resource, infra) is None

    def test_put_then_get_roundtrip(self, tmp_path: Path) -> None:
        store = ProvisionStore(tmp_path / "provisions.json")
        resource = VMResourceConfig(name="my-vm")
        infra = _StubInfra(name="us-east")
        info = {"ami_id": "ami-abc123"}
        store.put(resource, infra, info)
        assert store.get(resource, infra) == info

    def test_key_format(self, tmp_path: Path) -> None:
        store = ProvisionStore(tmp_path / "provisions.json")
        resource = VMResourceConfig(name="osworld-ubuntu-vm")
        infra = _StubInfra(name="westus2")
        store.put(resource, infra, {"x": 1})
        raw = json.loads((tmp_path / "provisions.json").read_text())
        assert "osworld-ubuntu-vm@stub:westus2" in raw

    def test_put_overwrites_existing(self, tmp_path: Path) -> None:
        store = ProvisionStore(tmp_path / "provisions.json")
        resource = VMResourceConfig(name="my-vm")
        infra = _StubInfra(name="a")
        store.put(resource, infra, {"v": 1})
        store.put(resource, infra, {"v": 2})
        assert store.get(resource, infra) == {"v": 2}

    def test_delete_removes_entry(self, tmp_path: Path) -> None:
        store = ProvisionStore(tmp_path / "provisions.json")
        resource = VMResourceConfig(name="my-vm")
        infra = _StubInfra(name="a")
        store.put(resource, infra, {"x": 1})
        deleted = store.delete(resource, infra)
        assert deleted is True
        assert store.get(resource, infra) is None

    def test_delete_returns_false_when_not_present(self, tmp_path: Path) -> None:
        store = ProvisionStore(tmp_path / "provisions.json")
        resource = VMResourceConfig(name="my-vm")
        infra = _StubInfra(name="a")
        assert store.delete(resource, infra) is False

    def test_list_returns_all_entries(self, tmp_path: Path) -> None:
        store = ProvisionStore(tmp_path / "provisions.json")
        r1 = VMResourceConfig(name="vm-a")
        r2 = VMResourceConfig(name="vm-b")
        infra = _StubInfra(name="r1")
        store.put(r1, infra, {"a": 1})
        store.put(r2, infra, {"b": 2})
        entries = dict(store.list())
        assert "vm-a@stub:r1" in entries
        assert "vm-b@stub:r1" in entries

    def test_different_infra_fingerprints_are_isolated(self, tmp_path: Path) -> None:
        store = ProvisionStore(tmp_path / "provisions.json")
        resource = VMResourceConfig(name="my-vm")
        infra_a = _StubInfra(name="us-east")
        infra_b = _StubInfra(name="eu-west")
        store.put(resource, infra_a, {"region": "us"})
        assert store.get(resource, infra_b) is None

    def test_persists_to_disk(self, tmp_path: Path) -> None:
        path = tmp_path / "p.json"
        resource = VMResourceConfig(name="my-vm")
        infra = _StubInfra(name="x")
        ProvisionStore(path).put(resource, infra, {"k": "v"})
        # New instance reads same file
        assert ProvisionStore(path).get(resource, infra) == {"k": "v"}

    def test_missing_file_is_treated_as_empty(self, tmp_path: Path) -> None:
        store = ProvisionStore(tmp_path / "nonexistent.json")
        assert store.list() == []


# ── ResourceConfig subclasses ─────────────────────────────────────────────────


class TestResourceConfig:
    def test_base_requirements_empty(self) -> None:
        r = ResourceConfig(name="generic")
        assert r.requirements() == set()

    def test_vm_resource_requires_kvm_by_default(self) -> None:
        r = VMResourceConfig(name="vm")
        assert r.requirements() == {"kvm"}

    def test_vm_resource_no_kvm_when_disabled(self) -> None:
        r = VMResourceConfig(name="vm", requires_kvm=False)
        assert r.requirements() == set()

    def test_docker_service_requires_docker(self) -> None:
        r = DockerServiceConfig(name="webarena", compose_url="http://example.com/compose.yml")
        assert r.requirements() == {"docker"}

    def test_docker_image_requires_docker(self) -> None:
        r = DockerImageConfig(name="swebench", image="swebench/swe-bench:latest")
        assert r.requirements() == {"docker"}

    def test_default_ttl_is_one_hour(self) -> None:
        r = VMResourceConfig(name="vm")
        assert r.default_ttl_seconds == 3600

    def test_scope_defaults_to_task(self) -> None:
        r = VMResourceConfig(name="vm")
        assert r.scope == "task"


# ── InfraConfig base: register / provision_status / can_serve ─────────────────


class TestInfraConfigBase:
    def test_provision_status_needs_provisioning_when_empty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_store_path(monkeypatch, tmp_path)
        infra = _StubInfra(name="x")
        resource = VMResourceConfig(name="my-vm")
        assert infra.provision_status(resource) == "needs_provisioning"

    def test_provision_status_ready_after_register(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_store_path(monkeypatch, tmp_path)
        infra = _StubInfra(name="x")
        resource = VMResourceConfig(name="my-vm")
        infra.register(resource, {"image_path": "/tmp/img.qcow2"})
        assert infra.provision_status(resource) == "ready"

    def test_provision_status_values_are_exactly_spec_strings(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_store_path(monkeypatch, tmp_path)
        infra = _StubInfra(name="x")
        resource = VMResourceConfig(name="my-vm")
        status_before = infra.provision_status(resource)
        infra.register(resource, {})
        status_after = infra.provision_status(resource)
        assert status_before == "needs_provisioning"
        assert status_after == "ready"

    def test_can_serve_true_when_caps_superset(self) -> None:
        infra = _StubInfra(name="x", caps=["kvm", "docker"])
        resource = VMResourceConfig(name="vm")  # requires "kvm"
        assert infra.can_serve(resource) is True

    def test_can_serve_false_when_caps_missing(self) -> None:
        infra = _StubInfra(name="x", caps=["docker"])
        resource = VMResourceConfig(name="vm")  # requires "kvm"
        assert infra.can_serve(resource) is False

    def test_can_serve_true_for_no_requirements(self) -> None:
        infra = _StubInfra(name="x", caps=[])
        resource = ResourceConfig(name="nothing")
        assert infra.can_serve(resource) is True

    def test_register_isolates_by_infra_fingerprint(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_store_path(monkeypatch, tmp_path)
        infra_a = _StubInfra(name="a")
        infra_b = _StubInfra(name="b")
        resource = VMResourceConfig(name="my-vm")
        infra_a.register(resource, {"x": 1})
        assert infra_b.provision_status(resource) == "needs_provisioning"


# ── ResourceHandle context manager ────────────────────────────────────────────


class TestResourceHandle:
    def _make_handle(self) -> _StubHandle:
        infra = _StubInfra()
        resource = VMResourceConfig(name="vm")
        return _StubHandle(
            run_id="run-1",
            resource=resource,
            infra=infra,
            endpoint="http://localhost:5000",
            created_at=datetime.now(UTC),
        )

    def test_context_manager_calls_close(self) -> None:
        handle = self._make_handle()
        with handle:
            assert not handle.closed
        assert handle.closed

    def test_close_is_idempotent(self) -> None:
        handle = self._make_handle()
        handle.close()
        handle.close()  # should not raise
        assert handle.closed


# ── Exceptions ────────────────────────────────────────────────────────────────


class TestExceptions:
    def test_resource_not_ready_error_message(self) -> None:
        infra = _StubInfra(name="us")
        resource = VMResourceConfig(name="my-vm")
        err = ResourceNotReadyError(resource, infra)
        assert "my-vm" in str(err)
        assert "stub:us" in str(err)
        assert "register" in str(err)

    def test_unsupported_resource_type_message(self) -> None:
        infra = _StubInfra()
        resource = VMResourceConfig(name="my-vm")
        err = UnsupportedResourceType(resource, infra)
        assert "_StubInfra" in str(err)
        assert "VMResourceConfig" in str(err)
        assert "my-vm" in str(err)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _patch_store_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Redirect the default ProvisionStore path to a temp dir for isolation."""
    monkeypatch.setattr(
        "cube.provision_store._DEFAULT_STORE_PATH",
        tmp_path / "provisions.json",
    )
