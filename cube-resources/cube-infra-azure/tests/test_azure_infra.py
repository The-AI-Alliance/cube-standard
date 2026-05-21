"""Unit tests for AzureInfraConfig.

All tests run without Azure credentials — no SDK calls.
AzureInfraConfig instances are constructed with model_construct() to bypass
the _autodiscover validator, which requires live Azure credentials.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from cube_infra_azure.azure import (
    AzureInfraConfig,
    AzureResourceHandle,
)

from cube.provision_store import ProvisionStore
from cube.resource import (
    ResourceConfig,
    ResourceNotReadyError,
    UnsupportedResourceType,
    VMResourceConfig,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_infra(image_name_suffix: str = "", location: str = "eastus") -> AzureInfraConfig:
    """Construct AzureInfraConfig without triggering the autodiscovery validator."""
    return AzureInfraConfig.model_construct(
        subscription="sub-12345",
        resource_group="cube-rg",
        location=location,
        gallery_name="cube_gallery",
        vm_size="Standard_D4s_v3",
        storage_account="cubestorage",
        vnet_name="cube-vnet",
        subnet_name="cube-subnet",
        nsg_name="cube-nsg",
        guest_port=5000,
        tags={"project": "cube"},
        image_name_suffix=image_name_suffix,
    )


def _patch_store_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("cube.provision_store._DEFAULT_STORE_DIR", tmp_path / "provisions")


# ── AzureInfraConfig: basic interface ────────────────────────────────────────


class TestAzureInfraConfigBasic:
    def test_fingerprint(self) -> None:
        infra = _make_infra(location="eastus")
        assert infra.fingerprint() == "azure:eastus"

    def test_fingerprint_different_locations(self) -> None:
        assert _make_infra(location="westeurope").fingerprint() == "azure:westeurope"
        assert _make_infra(location="southeastasia").fingerprint() == "azure:southeastasia"

    def test_capabilities_contains_kvm(self) -> None:
        assert "kvm" in _make_infra().capabilities()

    def test_can_serve_vm_resource(self) -> None:
        infra = _make_infra()
        assert infra.can_serve(VMResourceConfig(name="vm")) is True

    def test_image_name_no_suffix(self) -> None:
        infra = _make_infra()
        resource = VMResourceConfig(name="osworld-ubuntu-vm")
        assert infra._image_name(resource) == "osworld-ubuntu-vm"

    def test_image_name_with_suffix(self) -> None:
        infra = _make_infra(image_name_suffix="-test")
        resource = VMResourceConfig(name="osworld-ubuntu-vm")
        assert infra._image_name(resource) == "osworld-ubuntu-vm-test"

    def test_resource_shim_name(self) -> None:
        infra = _make_infra(image_name_suffix="-test")
        resource = VMResourceConfig(name="osworld-ubuntu-vm")
        assert infra._resource_shim(resource).name == "osworld-ubuntu-vm-test"


# ── AzureInfraConfig: provision_status / register ────────────────────────────


class TestAzureProvisionStatus:
    def test_needs_provisioning_when_empty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_store_path(monkeypatch, tmp_path)
        infra = _make_infra()
        assert infra.provision_status(VMResourceConfig(name="vm")) == "needs_provisioning"

    def test_ready_after_register(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_store_path(monkeypatch, tmp_path)
        infra = _make_infra()
        resource = VMResourceConfig(name="vm")
        infra.register(resource, {"image_def": "vm", "version": "1.0.0", "image_id": "/galleries/..."})
        assert infra.provision_status(resource) == "ready"

    def test_image_name_suffix_isolates_provision_status(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_store_path(monkeypatch, tmp_path)
        infra_prod = _make_infra()
        infra_test = _make_infra(image_name_suffix="-test")
        resource = VMResourceConfig(name="vm")
        infra_prod.register(resource, {"image_def": "vm", "version": "1.0.0", "image_id": "..."})
        assert infra_prod.provision_status(resource) == "ready"
        assert infra_test.provision_status(resource) == "needs_provisioning"

    def test_different_locations_are_isolated(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_store_path(monkeypatch, tmp_path)
        infra_east = _make_infra(location="eastus")
        infra_west = _make_infra(location="westeurope")
        resource = VMResourceConfig(name="vm")
        infra_east.register(resource, {"image_def": "vm", "version": "1.0.0", "image_id": "..."})
        assert infra_west.provision_status(resource) == "needs_provisioning"

    def test_provision_store_key_format(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_store_path(monkeypatch, tmp_path)
        infra = _make_infra(image_name_suffix="-test")
        resource = VMResourceConfig(name="vm")
        ProvisionStore().put(
            infra._resource_shim(resource), infra, {"image_def": "vm-test", "version": "1.0.0", "image_id": "..."}
        )
        store_dir = tmp_path / "provisions"
        stored_keys = [p.stem for p in store_dir.glob("*.json")]
        assert any("vm-test@azure:eastus" in k for k in stored_keys)


# ── AzureInfraConfig: provision / unprovision ────────────────────────────────


class TestAzureProvision:
    def test_provision_raises_for_non_vm_resource(self) -> None:
        infra = _make_infra()
        with pytest.raises(UnsupportedResourceType):
            infra.provision(ResourceConfig(name="generic"))

    def test_provision_raises_without_source_url(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_store_path(monkeypatch, tmp_path)
        infra = _make_infra()
        with pytest.raises(ValueError, match="source_url"):
            infra.provision(VMResourceConfig(name="vm"))

    def test_provision_skips_if_already_registered(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_store_path(monkeypatch, tmp_path)
        infra = _make_infra()
        resource = VMResourceConfig(name="vm", source_url="http://example.com/img.qcow2.zip")
        infra.register(resource, {"image_def": "vm", "version": "1.0.0", "image_id": "..."})
        with patch.object(infra, "_bootstrap") as mock_bootstrap:
            infra.provision(resource)
            mock_bootstrap.assert_not_called()

    def test_provision_calls_bootstrap_and_stores_result(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_store_path(monkeypatch, tmp_path)
        infra = _make_infra()
        resource = VMResourceConfig(name="vm", source_url="http://example.com/img.qcow2.zip")
        with patch.object(
            infra, "_bootstrap", return_value="/galleries/.../images/vm/versions/1.0.0"
        ) as mock_bootstrap:
            infra.provision(resource)
            mock_bootstrap.assert_called_once_with(
                url="http://example.com/img.qcow2.zip",
                image_name="vm",
                version="1.0.0",
                uefi=False,
                trusted_launch=False,
                specialized=False,
            )
        assert infra.provision_status(resource) == "ready"

    def test_unprovision_raises_for_non_vm_resource(self) -> None:
        infra = _make_infra()
        with pytest.raises(UnsupportedResourceType):
            infra.unprovision(ResourceConfig(name="generic"))

    def test_unprovision_noop_when_not_registered(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_store_path(monkeypatch, tmp_path)
        infra = _make_infra()
        infra.unprovision(VMResourceConfig(name="vm"))  # should not raise


# ── AzureInfraConfig: launch errors ──────────────────────────────────────────


class TestAzureLaunch:
    def test_launch_raises_for_non_vm_resource(self) -> None:
        infra = _make_infra()
        with pytest.raises(UnsupportedResourceType):
            infra.launch(ResourceConfig(name="generic"))

    def test_launch_raises_resource_not_ready(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_store_path(monkeypatch, tmp_path)
        infra = _make_infra()
        with pytest.raises(ResourceNotReadyError):
            infra.launch(VMResourceConfig(name="vm"))

    def test_launch_resource_not_ready_message(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_store_path(monkeypatch, tmp_path)
        infra = _make_infra()
        try:
            infra.launch(VMResourceConfig(name="my-vm"))
        except ResourceNotReadyError as err:
            assert "my-vm" in str(err)
            assert "azure:eastus" in str(err)

    def test_launch_reads_pubkey_from_ssh_pubkey_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """launch() must read the caller's pubkey from ssh_pubkey_path at launch time."""
        _patch_store_path(monkeypatch, tmp_path)
        infra = _make_infra()
        resource = VMResourceConfig(name="vm")
        infra.register(
            resource, {"image_def": "vm", "version": "1.0.0", "image_id": "/galleries/.../vm/versions/1.0.0"}
        )

        pubkey_content = "ssh-ed25519 AAAAC3Nza... user@host"
        pubkey_file = tmp_path / "id_ed25519.pub"
        pubkey_file.write_text(pubkey_content)
        object.__setattr__(infra, "ssh_pubkey_path", str(pubkey_file))

        vm_spec_captured: dict = {}

        def fake_begin_create(*args, **kwargs):
            vm_spec_captured.update(args[2] if len(args) > 2 else {})
            mock_poller = MagicMock()
            mock_poller.result.return_value = None
            return mock_poller

        mock_compute = MagicMock()
        mock_compute.virtual_machines.begin_create_or_update.side_effect = fake_begin_create
        mock_network = MagicMock()
        mock_pip = MagicMock()
        mock_pip.ip_address = "1.2.3.4"
        mock_network.public_ip_addresses.get.return_value = mock_pip

        with (
            patch.object(infra, "_compute", return_value=mock_compute),
            patch.object(infra, "_network", return_value=mock_network),
            patch.object(infra, "_create_network_resources", return_value=(MagicMock(), MagicMock(), "pip-1", "nic-1")),
            patch("cube_infra_azure.azure.wait_for_ssh", return_value="cube"),
            patch("cube_infra_azure.azure.open_tunnel", return_value=(MagicMock(), 15000)),
        ):
            infra.launch(resource)

        os_profile = vm_spec_captured.get("os_profile", {})
        assert os_profile.get("admin_username") == "cube"
        linux_cfg = os_profile.get("linux_configuration", {})
        pub_keys = linux_cfg.get("ssh", {}).get("public_keys", [])
        assert len(pub_keys) == 1
        assert pub_keys[0]["key_data"] == pubkey_content

    def test_launch_vm_spec_is_generalized_no_baked_key(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """os_profile must be present; there must be no custom_data (key via os_profile only)."""
        _patch_store_path(monkeypatch, tmp_path)
        infra = _make_infra()
        resource = VMResourceConfig(name="vm")
        infra.register(
            resource, {"image_def": "vm", "version": "1.0.0", "image_id": "/galleries/.../vm/versions/1.0.0"}
        )

        pubkey_file = tmp_path / "id_ed25519.pub"
        pubkey_file.write_text("ssh-ed25519 AAAA... user@host")
        object.__setattr__(infra, "ssh_pubkey_path", str(pubkey_file))

        vm_spec_captured: dict = {}

        def fake_begin_create(*args, **kwargs):
            vm_spec_captured.update(args[2] if len(args) > 2 else {})
            mock_poller = MagicMock()
            mock_poller.result.return_value = None
            return mock_poller

        mock_compute = MagicMock()
        mock_compute.virtual_machines.begin_create_or_update.side_effect = fake_begin_create
        mock_network = MagicMock()
        mock_pip = MagicMock()
        mock_pip.ip_address = "1.2.3.4"
        mock_network.public_ip_addresses.get.return_value = mock_pip

        with (
            patch.object(infra, "_compute", return_value=mock_compute),
            patch.object(infra, "_network", return_value=mock_network),
            patch.object(infra, "_create_network_resources", return_value=(MagicMock(), MagicMock(), "pip-1", "nic-1")),
            patch("cube_infra_azure.azure.wait_for_ssh", return_value="cube"),
            patch("cube_infra_azure.azure.open_tunnel", return_value=(MagicMock(), 15000)),
        ):
            infra.launch(resource)

        assert "os_profile" in vm_spec_captured
        assert "custom_data" not in vm_spec_captured.get("os_profile", {})

    def test_launch_writes_expireon_tag_with_z_format(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Launched resources must carry the org-standard ``expireOn`` tag
        (in addition to ``cube:expires_at`` for internal cleanup_stale).
        Format must match Christian's budget-automation convention:
        ISO-8601 ``YYYY-MM-DDTHH:MM:SSZ`` — second precision, ``Z`` suffix."""
        _patch_store_path(monkeypatch, tmp_path)
        infra = _make_infra()
        resource = VMResourceConfig(name="vm")
        infra.register(
            resource, {"image_def": "vm", "version": "1.0.0", "image_id": "/galleries/.../vm/versions/1.0.0"}
        )

        pubkey_file = tmp_path / "id_ed25519.pub"
        pubkey_file.write_text("ssh-ed25519 AAAA... user@host")
        object.__setattr__(infra, "ssh_pubkey_path", str(pubkey_file))

        vm_spec_captured: dict = {}

        def fake_begin_create(*args, **kwargs):
            vm_spec_captured.update(args[2] if len(args) > 2 else {})
            mock_poller = MagicMock()
            mock_poller.result.return_value = None
            return mock_poller

        mock_compute = MagicMock()
        mock_compute.virtual_machines.begin_create_or_update.side_effect = fake_begin_create
        mock_network = MagicMock()
        mock_pip = MagicMock()
        mock_pip.ip_address = "1.2.3.4"
        mock_network.public_ip_addresses.get.return_value = mock_pip

        with (
            patch.object(infra, "_compute", return_value=mock_compute),
            patch.object(infra, "_network", return_value=mock_network),
            patch.object(infra, "_create_network_resources", return_value=(MagicMock(), MagicMock(), "pip-1", "nic-1")),
            patch("cube_infra_azure.azure.wait_for_ssh", return_value="cube"),
            patch("cube_infra_azure.azure.open_tunnel", return_value=(MagicMock(), 15000)),
        ):
            infra.launch(resource)

        tags = vm_spec_captured.get("tags", {})
        # Both tag names must be present and carry the same value (org-standard
        # ``expireOn`` for budget automation + internal ``cube:expires_at`` for
        # cleanup_stale).
        assert "expireOn" in tags, f"expireOn tag missing; got {sorted(tags)}"
        assert "cube:expires_at" in tags, f"cube:expires_at tag missing; got {sorted(tags)}"
        assert tags["expireOn"] == tags["cube:expires_at"]
        # Format: Z suffix, no microseconds.
        assert tags["expireOn"].endswith("Z"), f"expected Z suffix, got {tags['expireOn']!r}"
        assert "+" not in tags["expireOn"], f"expected Z (not +00:00), got {tags['expireOn']!r}"
        assert "." not in tags["expireOn"], f"unexpected microseconds: {tags['expireOn']!r}"
        # cube:created_at follows the same convention.
        assert tags["cube:created_at"].endswith("Z")
        assert "." not in tags["cube:created_at"]


# ── Module-level helpers ──────────────────────────────────────────────────────


class TestIsoUtc:
    """Unit tests for the ``_iso_utc`` formatter used by tag writes."""

    def test_strips_microseconds(self) -> None:
        from cube_infra_azure.azure import _iso_utc

        dt = datetime(2026, 5, 21, 17, 17, 29, 131757, tzinfo=UTC)
        assert _iso_utc(dt) == "2026-05-21T17:17:29Z"

    def test_uses_z_suffix_not_plus_offset(self) -> None:
        from cube_infra_azure.azure import _iso_utc

        dt = datetime(2026, 6, 15, 18, 0, 0, tzinfo=UTC)
        assert _iso_utc(dt) == "2026-06-15T18:00:00Z"
        assert "+00:00" not in _iso_utc(dt)

    def test_roundtrip_through_fromisoformat(self) -> None:
        """The format we write must round-trip through datetime.fromisoformat —
        this is what cleanup_stale uses to read ``cube:expires_at`` back."""
        from cube_infra_azure.azure import _iso_utc

        dt = datetime(2026, 5, 21, 17, 17, 29, tzinfo=UTC)
        parsed = datetime.fromisoformat(_iso_utc(dt))
        assert parsed == dt


# ── Bootstrap script ──────────────────────────────────────────────────────────


class TestBootstrapScript:
    def test_no_ssh_pubkey_placeholder(self) -> None:
        """Bootstrap script must not contain {ssh_pubkey} — key injection removed."""
        from cube_infra_azure.azure import _AZURE_BOOTSTRAP_SCRIPT

        assert "{ssh_pubkey}" not in _AZURE_BOOTSTRAP_SCRIPT

    def test_walinuxagent_installed(self) -> None:
        """Bootstrap script must install walinuxagent for Generalized image support."""
        from cube_infra_azure.azure import _AZURE_BOOTSTRAP_SCRIPT

        assert "walinuxagent" in _AZURE_BOOTSTRAP_SCRIPT

    def test_waagent_deprovision_called(self) -> None:
        """Bootstrap script must deprovision the image so waagent re-provisions at launch."""
        from cube_infra_azure.azure import _AZURE_BOOTSTRAP_SCRIPT

        assert "waagent" in _AZURE_BOOTSTRAP_SCRIPT
        assert "deprovision" in _AZURE_BOOTSTRAP_SCRIPT

    def test_no_authorized_keys_injection(self) -> None:
        """Bootstrap script must not write to authorized_keys — keys injected at launch."""
        from cube_infra_azure.azure import _AZURE_BOOTSTRAP_SCRIPT

        assert "authorized_keys" not in _AZURE_BOOTSTRAP_SCRIPT


class TestDockerBootstrapScript:
    def test_no_ssh_pubkey_placeholder(self) -> None:
        """Docker bootstrap script must not contain {ssh_pubkey} — key injected at launch."""
        from cube_infra_azure.azure import _DOCKER_BOOTSTRAP_SCRIPT

        assert "{ssh_pubkey}" not in _DOCKER_BOOTSTRAP_SCRIPT

    def test_no_authorized_keys_injection(self) -> None:
        """Docker bootstrap script must not append to authorized_keys — key injected via os_profile."""
        from cube_infra_azure.azure import _DOCKER_BOOTSTRAP_SCRIPT

        # Check for the actual write operation, not just the word in a comment.
        assert ">> " not in _DOCKER_BOOTSTRAP_SCRIPT or not any(
            "authorized_keys" in part for part in _DOCKER_BOOTSTRAP_SCRIPT.split(">> ")[1:]
        )

    def test_walinuxagent_installed(self) -> None:
        """Docker bootstrap script must install walinuxagent for Generalized image support."""
        from cube_infra_azure.azure import _DOCKER_BOOTSTRAP_SCRIPT

        assert "walinuxagent" in _DOCKER_BOOTSTRAP_SCRIPT

    def test_waagent_deprovision_called(self) -> None:
        """Docker bootstrap script must deprovision so Azure can inject caller's key at launch."""
        from cube_infra_azure.azure import _DOCKER_BOOTSTRAP_SCRIPT

        assert "waagent" in _DOCKER_BOOTSTRAP_SCRIPT
        assert "deprovision" in _DOCKER_BOOTSTRAP_SCRIPT


# ── AzureResourceHandle ───────────────────────────────────────────────────────


def test_windows_admin_password_excluded_from_repr() -> None:
    """windows_admin_password must not appear in repr or serialization."""
    from cube_infra_azure import AzureInfraConfig

    fields = AzureInfraConfig.model_fields
    assert "windows_admin_password" in fields
    field_info = fields["windows_admin_password"]
    assert field_info.exclude is True


# ── AzureResourceHandle ───────────────────────────────────────────────────────


class TestAzureResourceHandle:
    def _make_handle(self) -> AzureResourceHandle:
        infra = _make_infra()
        resource = VMResourceConfig(name="vm")
        return AzureResourceHandle(
            run_id="run-test",
            resource=resource,
            infra=infra,
            endpoint="http://localhost:15000",
            created_at=datetime.now(UTC),
            _vm_name="cube-abc-vm-def123",
            _pip_name="pip-1",
            _nic_name="nic-1",
        )

    def test_context_manager_calls_close(self) -> None:
        handle = self._make_handle()
        with patch.object(handle.infra, "_delete_vm") as mock_delete:
            with handle:
                pass
            mock_delete.assert_called_once_with("cube-abc-vm-def123", "pip-1", "nic-1")

    def test_close_terminates_tunnel(self) -> None:
        handle = self._make_handle()
        tunnel = MagicMock()
        handle._tunnels = [tunnel]
        with patch.object(handle.infra, "_delete_vm"):
            handle.close()
        tunnel.terminate.assert_called_once()

    def test_close_clears_tunnel_reference(self) -> None:
        handle = self._make_handle()
        handle._tunnels = [MagicMock()]
        with patch.object(handle.infra, "_delete_vm"):
            handle.close()
        assert handle._tunnels == []
