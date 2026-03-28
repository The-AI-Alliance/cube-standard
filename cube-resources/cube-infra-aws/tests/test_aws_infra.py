"""Unit tests for AWSInfraConfig.

All tests run without AWS credentials — no boto3/EC2 API calls.
AWSInfraConfig instances are constructed with model_construct() to bypass
the _autodiscover validator, which requires live AWS credentials.
"""
from __future__ import annotations

from dataclasses import dataclass as _dataclass
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from cube_infra_aws.aws import (
    AWSInfraConfig,
    AWSResourceHandle,
    _dict_to_ec2_tags,
    _ec2_tags_to_dict,
)

from cube.provision_store import ProvisionStore
from cube.resource import (
    ResourceConfig,
    ResourceNotReadyError,
    UnsupportedResourceType,
    VMResourceConfig,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_infra(image_name_suffix: str = "", region: str = "us-east-2") -> AWSInfraConfig:
    """Construct AWSInfraConfig without triggering the autodiscovery validator."""
    return AWSInfraConfig.model_construct(
        region=region,
        account_id="123456789012",
        s3_bucket="cube-vmimages-123456789012",
        vpc_id="vpc-12345",
        subnet_id="subnet-12345",
        security_group_name="cube-sg",
        key_pair_name="cube-key",
        instance_type="t3.xlarge",
        guest_port=5000,
        ssh_privkey_path="/home/user/.ssh/id_ed25519",
        ssh_pubkey_path="/home/user/.ssh/id_ed25519.pub",
        tags={"project": "cube"},
        bootstrap_instance_type="t3.medium",
        bootstrap_root_volume_gb=128,
        bootstrap_role_name="cube-bootstrap-role",
        bootstrap_profile_name="cube-bootstrap",
        image_name_suffix=image_name_suffix,
    )


def _patch_store_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Redirect the default ProvisionStore path to a temp dir for isolation."""
    monkeypatch.setattr(
        "cube.provision_store._DEFAULT_STORE_PATH",
        tmp_path / "provisions.json",
    )


# ── Tag helpers ───────────────────────────────────────────────────────────────


class TestTagHelpers:
    def test_dict_to_ec2_tags_basic(self) -> None:
        result = _dict_to_ec2_tags({"project": "cube", "env": "test"})
        assert {"Key": "project", "Value": "cube"} in result
        assert {"Key": "env", "Value": "test"} in result

    def test_dict_to_ec2_tags_empty(self) -> None:
        assert _dict_to_ec2_tags({}) == []

    def test_ec2_tags_to_dict_basic(self) -> None:
        tags = [{"Key": "cube:infra", "Value": "aws:us-east-2"},
                {"Key": "cube:run_id", "Value": "run-abc"}]
        result = _ec2_tags_to_dict(tags)
        assert result == {"cube:infra": "aws:us-east-2", "cube:run_id": "run-abc"}

    def test_ec2_tags_to_dict_empty(self) -> None:
        assert _ec2_tags_to_dict([]) == {}

    def test_roundtrip(self) -> None:
        original = {"a": "1", "b": "2", "c": "3"}
        assert _ec2_tags_to_dict(_dict_to_ec2_tags(original)) == original


# ── AWSInfraConfig: basic interface ───────────────────────────────────────────


class TestAWSInfraConfigBasic:
    def test_fingerprint(self) -> None:
        infra = _make_infra(region="us-east-2")
        assert infra.fingerprint() == "aws:us-east-2"

    def test_fingerprint_different_regions(self) -> None:
        assert _make_infra(region="eu-west-1").fingerprint() == "aws:eu-west-1"
        assert _make_infra(region="ap-southeast-1").fingerprint() == "aws:ap-southeast-1"

    def test_capabilities_contains_kvm(self) -> None:
        infra = _make_infra()
        assert "kvm" in infra.capabilities()

    def test_can_serve_vm_resource(self) -> None:
        infra = _make_infra()
        resource = VMResourceConfig(name="vm")
        assert infra.can_serve(resource) is True

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
        shim = infra._resource_shim(resource)
        assert shim.name == "osworld-ubuntu-vm-test"


# ── AWSInfraConfig: provision_status / register ───────────────────────────────


class TestAWSProvisionStatus:
    def test_needs_provisioning_when_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_store_path(monkeypatch, tmp_path)
        infra = _make_infra()
        resource = VMResourceConfig(name="osworld-ubuntu-vm")
        assert infra.provision_status(resource) == "needs_provisioning"

    def test_ready_after_register(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_store_path(monkeypatch, tmp_path)
        infra = _make_infra()
        resource = VMResourceConfig(name="osworld-ubuntu-vm")
        infra.register(resource, {"ami_id": "ami-abc123"})
        assert infra.provision_status(resource) == "ready"

    def test_image_name_suffix_isolates_provision_status(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_store_path(monkeypatch, tmp_path)
        infra_prod = _make_infra()
        infra_test = _make_infra(image_name_suffix="-test")
        resource = VMResourceConfig(name="osworld-ubuntu-vm")

        infra_prod.register(resource, {"ami_id": "ami-prod"})

        assert infra_prod.provision_status(resource) == "ready"
        assert infra_test.provision_status(resource) == "needs_provisioning"

    def test_provision_status_for_non_vm_resource(self) -> None:
        infra = _make_infra()
        resource = ResourceConfig(name="generic")
        assert infra.provision_status(resource) == "needs_provisioning"

    def test_provision_store_key_format(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ProvisionStore key must be {image_name}@{fingerprint}."""
        import json
        _patch_store_path(monkeypatch, tmp_path)
        infra = _make_infra(image_name_suffix="-test")
        resource = VMResourceConfig(name="osworld-ubuntu-vm")
        infra.register(resource, {"ami_id": "ami-test"})

        raw = json.loads((tmp_path / "provisions.json").read_text())
        assert "osworld-ubuntu-vm-test@aws:us-east-2" in raw

    def test_different_regions_are_isolated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_store_path(monkeypatch, tmp_path)
        infra_east = _make_infra(region="us-east-2")
        infra_west = _make_infra(region="us-west-2")
        resource = VMResourceConfig(name="vm")
        infra_east.register(resource, {"ami_id": "ami-east"})
        assert infra_west.provision_status(resource) == "needs_provisioning"


# ── AWSInfraConfig: provision / unprovision (with mocked EC2) ─────────────────


class TestAWSProvision:
    def test_provision_raises_for_non_vm_resource(self) -> None:
        infra = _make_infra()
        resource = ResourceConfig(name="generic")
        with pytest.raises(UnsupportedResourceType):
            infra.provision(resource)

    def test_provision_raises_without_source_url(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_store_path(monkeypatch, tmp_path)
        infra = _make_infra()
        resource = VMResourceConfig(name="vm")  # no source_url
        with pytest.raises(ValueError, match="source_url"):
            infra.provision(resource)

    def test_provision_skips_if_already_registered(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_store_path(monkeypatch, tmp_path)
        infra = _make_infra()
        resource = VMResourceConfig(name="vm", source_url="http://example.com/img.qcow2.zip")
        infra.register(resource, {"ami_id": "ami-existing"})

        # _bootstrap should NOT be called
        with patch.object(infra, "_bootstrap") as mock_bootstrap:
            infra.provision(resource)
            mock_bootstrap.assert_not_called()

    def test_provision_calls_bootstrap_and_stores_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_store_path(monkeypatch, tmp_path)
        infra = _make_infra()
        resource = VMResourceConfig(name="vm", source_url="http://example.com/img.qcow2.zip")

        with patch.object(infra, "_bootstrap", return_value="ami-new123") as mock_bootstrap:
            infra.provision(resource)
            mock_bootstrap.assert_called_once_with(
                url="http://example.com/img.qcow2.zip", image_name="vm"
            )

        assert infra.provision_status(resource) == "ready"
        store = ProvisionStore()
        info = store.get(infra._resource_shim(resource), infra)
        assert info == {"ami_id": "ami-new123"}

    def test_unprovision_raises_for_non_vm_resource(self) -> None:
        infra = _make_infra()
        resource = ResourceConfig(name="generic")
        with pytest.raises(UnsupportedResourceType):
            infra.unprovision(resource)

    def test_unprovision_noop_when_not_registered(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_store_path(monkeypatch, tmp_path)
        infra = _make_infra()
        resource = VMResourceConfig(name="vm")
        # Should not raise
        infra.unprovision(resource)

    def test_unprovision_deregisters_ami_and_deletes_snapshot(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_store_path(monkeypatch, tmp_path)
        infra = _make_infra()
        resource = VMResourceConfig(name="vm")
        infra.register(resource, {"ami_id": "ami-deadbeef"})

        mock_ec2 = MagicMock()
        mock_ec2.describe_images.return_value = {
            "Images": [{
                "BlockDeviceMappings": [
                    {"DeviceName": "/dev/sda1", "Ebs": {"SnapshotId": "snap-12345"}},
                ]
            }]
        }

        with patch.object(infra, "_ec2", return_value=mock_ec2):
            infra.unprovision(resource)

        mock_ec2.describe_images.assert_called_once_with(ImageIds=["ami-deadbeef"])
        mock_ec2.deregister_image.assert_called_once_with(ImageId="ami-deadbeef")
        mock_ec2.delete_snapshot.assert_called_once_with(SnapshotId="snap-12345")

        assert infra.provision_status(resource) == "needs_provisioning"

    def test_unprovision_retrieves_snapshot_before_deregistering(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Critical: describe_images must be called BEFORE deregister_image."""
        _patch_store_path(monkeypatch, tmp_path)
        infra = _make_infra()
        resource = VMResourceConfig(name="vm")
        infra.register(resource, {"ami_id": "ami-order-test"})

        call_order: list[str] = []
        mock_ec2 = MagicMock()
        mock_ec2.describe_images.side_effect = lambda **_: (
            call_order.append("describe") or {
                "Images": [{"BlockDeviceMappings": [
                    {"Ebs": {"SnapshotId": "snap-abc"}}
                ]}]
            }
        )
        mock_ec2.deregister_image.side_effect = lambda **_: call_order.append("deregister")
        mock_ec2.delete_snapshot.side_effect = lambda **_: call_order.append("delete_snap")

        with patch.object(infra, "_ec2", return_value=mock_ec2):
            infra.unprovision(resource)

        assert call_order == ["describe", "deregister", "delete_snap"]


# ── AWSInfraConfig: launch errors ─────────────────────────────────────────────


class TestAWSLaunch:
    def test_launch_raises_for_non_vm_resource(self) -> None:
        infra = _make_infra()
        resource = ResourceConfig(name="generic")
        with pytest.raises(UnsupportedResourceType):
            infra.launch(resource, run_id="run-1")

    def test_launch_raises_resource_not_ready(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_store_path(monkeypatch, tmp_path)
        infra = _make_infra()
        resource = VMResourceConfig(name="vm")
        with pytest.raises(ResourceNotReadyError):
            infra.launch(resource, run_id="run-1")

    def test_launch_resource_not_ready_message(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_store_path(monkeypatch, tmp_path)
        infra = _make_infra()
        resource = VMResourceConfig(name="my-vm")
        try:
            infra.launch(resource, run_id="run-1")
        except ResourceNotReadyError as err:
            assert "my-vm" in str(err)
            assert "aws:us-east-2" in str(err)


# ── AWSInfraConfig: list_active ───────────────────────────────────────────────


class TestAWSListActive:
    def _make_ec2_instance(
        self,
        instance_id: str,
        run_id: str = "run-abc",
        resource_name: str = "vm",
        infra_fp: str = "aws:us-east-2",
        state: str = "running",
    ) -> dict:
        return {
            "InstanceId": instance_id,
            "State": {"Name": state},
            "Tags": _dict_to_ec2_tags({
                "cube:infra": infra_fp,
                "cube:run_id": run_id,
                "cube:resource": resource_name,
            }),
        }

    def test_list_active_empty_when_no_instances(self) -> None:
        infra = _make_infra()
        mock_ec2 = MagicMock()
        mock_ec2.describe_instances.return_value = {"Reservations": []}
        with patch.object(infra, "_ec2", return_value=mock_ec2):
            result = infra.list_active()
        assert result == []

    def test_list_active_returns_handles(self) -> None:
        infra = _make_infra()
        mock_ec2 = MagicMock()
        mock_ec2.describe_instances.return_value = {
            "Reservations": [{
                "Instances": [
                    self._make_ec2_instance("i-111", run_id="run-abc"),
                    self._make_ec2_instance("i-222", run_id="run-def"),
                ]
            }]
        }
        with patch.object(infra, "_ec2", return_value=mock_ec2):
            handles = infra.list_active()

        assert len(handles) == 2
        ids = {h._instance_id for h in handles}
        assert ids == {"i-111", "i-222"}

    def test_list_active_endpoint_is_none(self) -> None:
        infra = _make_infra()
        mock_ec2 = MagicMock()
        mock_ec2.describe_instances.return_value = {
            "Reservations": [{"Instances": [self._make_ec2_instance("i-111")]}]
        }
        with patch.object(infra, "_ec2", return_value=mock_ec2):
            handles = infra.list_active()
        assert handles[0].endpoint is None

    def test_list_active_passes_run_id_filter(self) -> None:
        infra = _make_infra()
        mock_ec2 = MagicMock()
        mock_ec2.describe_instances.return_value = {"Reservations": []}

        with patch.object(infra, "_ec2", return_value=mock_ec2):
            infra.list_active(run_id="run-xyz")

        filters = mock_ec2.describe_instances.call_args[1]["Filters"]
        run_id_filter = next((f for f in filters if f["Name"] == "tag:cube:run_id"), None)
        assert run_id_filter is not None
        assert run_id_filter["Values"] == ["run-xyz"]

    def test_list_active_no_run_id_filter_when_none(self) -> None:
        infra = _make_infra()
        mock_ec2 = MagicMock()
        mock_ec2.describe_instances.return_value = {"Reservations": []}

        with patch.object(infra, "_ec2", return_value=mock_ec2):
            infra.list_active()

        filters = mock_ec2.describe_instances.call_args[1]["Filters"]
        run_id_filters = [f for f in filters if f["Name"] == "tag:cube:run_id"]
        assert run_id_filters == []

    def test_list_active_filters_by_infra_fingerprint(self) -> None:
        infra = _make_infra(region="us-east-2")
        mock_ec2 = MagicMock()
        mock_ec2.describe_instances.return_value = {"Reservations": []}

        with patch.object(infra, "_ec2", return_value=mock_ec2):
            infra.list_active()

        filters = mock_ec2.describe_instances.call_args[1]["Filters"]
        infra_filter = next((f for f in filters if f["Name"] == "tag:cube:infra"), None)
        assert infra_filter is not None
        assert infra_filter["Values"] == ["aws:us-east-2"]


# ── AWSInfraConfig: cleanup_stale ─────────────────────────────────────────────


class TestAWSCleanupStale:
    def _make_instance_with_tags(
        self, instance_id: str, tags: dict[str, str]
    ) -> dict:
        return {
            "InstanceId": instance_id,
            "Tags": _dict_to_ec2_tags({"cube:infra": "aws:us-east-2", **tags}),
        }

    def test_cleanup_stale_terminates_expired_instances(self) -> None:
        infra = _make_infra()
        now = datetime.now(timezone.utc)
        expired = (now - timedelta(hours=2)).isoformat()

        mock_ec2 = MagicMock()
        mock_ec2.describe_instances.return_value = {
            "Reservations": [{
                "Instances": [
                    self._make_instance_with_tags("i-expired", {"cube:expires_at": expired}),
                ]
            }]
        }

        with patch.object(infra, "_ec2", return_value=mock_ec2):
            deleted = infra.cleanup_stale()

        assert "i-expired" in deleted
        mock_ec2.terminate_instances.assert_called_once()

    def test_cleanup_stale_keeps_unexpired_instances(self) -> None:
        infra = _make_infra()
        now = datetime.now(timezone.utc)
        future = (now + timedelta(hours=2)).isoformat()

        mock_ec2 = MagicMock()
        mock_ec2.describe_instances.return_value = {
            "Reservations": [{
                "Instances": [
                    self._make_instance_with_tags("i-live", {"cube:expires_at": future}),
                ]
            }]
        }

        with patch.object(infra, "_ec2", return_value=mock_ec2):
            deleted = infra.cleanup_stale()

        assert deleted == []
        mock_ec2.terminate_instances.assert_not_called()

    def test_cleanup_stale_age_fallback(self) -> None:
        infra = _make_infra()
        now = datetime.now(timezone.utc)
        old_created = (now - timedelta(hours=3)).isoformat()

        mock_ec2 = MagicMock()
        mock_ec2.describe_instances.return_value = {
            "Reservations": [{
                "Instances": [
                    self._make_instance_with_tags("i-old", {"cube:created_at": old_created}),
                ]
            }]
        }

        with patch.object(infra, "_ec2", return_value=mock_ec2):
            deleted = infra.cleanup_stale(max_age_seconds=3600)

        assert "i-old" in deleted

    def test_cleanup_stale_expires_at_takes_priority(self) -> None:
        """cube:expires_at is checked first; max_age_seconds is a fallback."""
        infra = _make_infra()
        now = datetime.now(timezone.utc)
        future_expiry = (now + timedelta(hours=2)).isoformat()
        old_created = (now - timedelta(hours=3)).isoformat()

        mock_ec2 = MagicMock()
        mock_ec2.describe_instances.return_value = {
            "Reservations": [{
                "Instances": [
                    self._make_instance_with_tags(
                        "i-not-expired",
                        {
                            "cube:expires_at": future_expiry,
                            "cube:created_at": old_created,
                        },
                    ),
                ]
            }]
        }

        with patch.object(infra, "_ec2", return_value=mock_ec2):
            # max_age_seconds would match old_created, but expires_at says keep it
            deleted = infra.cleanup_stale(max_age_seconds=3600)

        assert deleted == []

    def test_cleanup_stale_returns_empty_on_api_error(self) -> None:
        infra = _make_infra()
        mock_ec2 = MagicMock()
        mock_ec2.describe_instances.side_effect = Exception("network error")

        with patch.object(infra, "_ec2", return_value=mock_ec2):
            deleted = infra.cleanup_stale()

        assert deleted == []


# ── AWSResourceHandle ─────────────────────────────────────────────────────────


@_dataclass
class _MockTunnel:
    terminated: bool = False

    def terminate(self) -> None:
        self.terminated = True


class TestAWSResourceHandle:
    def _make_handle(self, instance_id: str = "i-12345") -> AWSResourceHandle:
        infra = _make_infra()
        resource = VMResourceConfig(name="vm")
        return AWSResourceHandle(
            run_id="run-test",
            resource=resource,
            infra=infra,
            endpoint="http://localhost:15000",
            created_at=datetime.now(UTC),
        )

    def test_context_manager_calls_close(self) -> None:
        handle = self._make_handle()
        handle._instance_id = "i-abc"

        with patch.object(handle.infra, "_terminate_instance") as mock_terminate:
            with handle:
                pass
            mock_terminate.assert_called_once_with("i-abc")

    def test_close_terminates_tunnel(self) -> None:
        handle = self._make_handle()
        tunnel = MagicMock()
        handle._tunnel = tunnel

        with patch.object(handle.infra, "_terminate_instance"):
            handle.close()

        tunnel.terminate.assert_called_once()

    def test_close_clears_tunnel_reference(self) -> None:
        handle = self._make_handle()
        handle._tunnel = MagicMock()

        with patch.object(handle.infra, "_terminate_instance"):
            handle.close()

        assert handle._tunnel is None

    def test_close_no_instance_id_skips_terminate(self) -> None:
        handle = self._make_handle()
        handle._instance_id = ""  # no instance

        with patch.object(handle.infra, "_terminate_instance") as mock_terminate:
            handle.close()

        mock_terminate.assert_not_called()
