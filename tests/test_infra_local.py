"""Unit tests for LocalInfraConfig Docker support.

Covers build_volume_setup_script, _kill_entry, list_active, and
LocalDockerServiceHandle.close(). All tests run without a real Docker
daemon — subprocess.run is mocked wherever Docker calls would be made.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import cube.infra_local as _mod
from cube.infra_local import (
    LocalDockerServiceHandle,
    LocalInfraConfig,
    LocalResourceHandle,
    _kill_entry,
)
from cube.infra_utils import build_volume_setup_script
from cube.resource import DockerServiceConfig, VolumeSpec

# ── build_volume_setup_script ─────────────────────────────────────────────────


class TestBuildVolumeSetupScript:
    def test_empty_list_returns_empty_string(self) -> None:
        assert build_volume_setup_script([]) == ""

    def test_volume_without_source_url_creates_empty_volume(self) -> None:
        vol = VolumeSpec(name="tiles", mount_path="/data/tiles")
        script = build_volume_setup_script([vol])
        assert 'docker volume create "tiles"' in script
        assert "curl" not in script
        assert "tar" not in script

    def test_volume_with_source_url_downloads_and_extracts(self) -> None:
        vol = VolumeSpec(name="osm", mount_path="/data", source_url="https://example.com/data.tar")
        script = build_volume_setup_script([vol])
        assert "curl" in script
        assert "https://example.com/data.tar" in script
        assert "tar -xf" in script

    def test_vol_name_is_quoted_in_all_docker_commands(self) -> None:
        vol = VolumeSpec(name="my-vol", mount_path="/data", source_url="https://example.com/a.tar")
        script = build_volume_setup_script([vol])
        assert '"my-vol"' in script
        # Bare unquoted name must not appear as a standalone shell token
        assert "create my-vol" not in script
        assert "-v my-vol:" not in script

    def test_download_filename_prefixed_with_vol_name(self) -> None:
        vol = VolumeSpec(name="myvol", mount_path="/data", source_url="https://example.com/archive.tar")
        script = build_volume_setup_script([vol])
        assert "myvol_archive.tar" in script

    def test_two_volumes_same_url_basename_use_separate_files(self) -> None:
        vol_a = VolumeSpec(name="vol-a", mount_path="/a", source_url="https://host-a.com/data.tar")
        vol_b = VolumeSpec(name="vol-b", mount_path="/b", source_url="https://host-b.com/data.tar")
        script = build_volume_setup_script([vol_a, vol_b])
        assert "vol-a_data.tar" in script
        assert "vol-b_data.tar" in script
        assert "https://host-a.com/data.tar" in script
        assert "https://host-b.com/data.tar" in script

    def test_same_url_shared_by_two_volumes_downloaded_once(self) -> None:
        url = "https://example.com/shared.tar"
        vol_a = VolumeSpec(name="va", mount_path="/a", source_url=url)
        vol_b = VolumeSpec(name="vb", mount_path="/b", source_url=url)
        script = build_volume_setup_script([vol_a, vol_b])
        assert script.count(url) == 1

    def test_strip_components_added_to_tar_command(self) -> None:
        vol = VolumeSpec(name="v", mount_path="/d", source_url="https://example.com/a.tar", strip_components=2)
        script = build_volume_setup_script([vol])
        assert "--strip-components=2" in script

    def test_no_strip_components_by_default(self) -> None:
        vol = VolumeSpec(name="v", mount_path="/d", source_url="https://example.com/a.tar")
        script = build_volume_setup_script([vol])
        assert "--strip-components" not in script

    def test_tar_subpath_appended_to_tar_command(self) -> None:
        vol = VolumeSpec(name="v", mount_path="/d", source_url="https://example.com/a.tar", tar_subpath="sub/dir")
        script = build_volume_setup_script([vol])
        assert "sub/dir" in script

    def test_script_ends_with_completion_message(self) -> None:
        vol = VolumeSpec(name="v", mount_path="/d")
        script = build_volume_setup_script([vol])
        assert "Volume setup complete" in script


# ── _kill_entry ───────────────────────────────────────────────────────────────


class TestKillEntry:
    def test_docker_service_entry_stops_and_removes_each_container(self) -> None:
        entry = {"type": "docker_service", "container_ids": ["abc123", "def456"]}
        with patch("cube.infra_local.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            _kill_entry(entry)

        issued = [c.args[0] for c in mock_run.call_args_list]
        assert ["docker", "stop", "abc123"] in issued
        assert ["docker", "rm", "abc123"] in issued
        assert ["docker", "stop", "def456"] in issued
        assert ["docker", "rm", "def456"] in issued

    def test_docker_service_entry_tolerates_no_such_container(self) -> None:
        entry = {"type": "docker_service", "container_ids": ["gone"]}
        with patch("cube.infra_local.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="Error: No such container: gone")
            _kill_entry(entry)  # must not raise

    def test_empty_container_ids_issues_no_docker_calls(self) -> None:
        entry = {"type": "docker_service", "container_ids": []}
        with patch("cube.infra_local.subprocess.run") as mock_run:
            _kill_entry(entry)
        mock_run.assert_not_called()

    def test_vm_entry_without_containers_issues_no_docker_calls(self) -> None:
        # No pid alive → nothing to kill; no container_ids → no docker calls.
        entry = {"pid": None}
        with patch("cube.infra_local.subprocess.run") as mock_run:
            _kill_entry(entry)
        mock_run.assert_not_called()


# ── list_active ───────────────────────────────────────────────────────────────


def _write_active(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def _docker_entry(**overrides) -> dict:
    base = {
        "type": "docker_service",
        "run_id": "run-abc",
        "resource_name": "my-svc",
        "infra_fingerprint": "local",
        "created_at": datetime.utcnow().isoformat(),
        "expires_at": None,
        "endpoint": "http://127.0.0.1:7780",
        "endpoints": {"svc": "http://127.0.0.1:7780"},
        "container_ids": ["cid1", "cid2"],
    }
    base.update(overrides)
    return base


def _vm_entry(**overrides) -> dict:
    base = {
        "run_id": "run-vm",
        "resource_name": "my-vm",
        "infra_fingerprint": "local",
        "created_at": datetime.utcnow().isoformat(),
        "expires_at": None,
        "endpoint": "http://127.0.0.1:5000",
        # No pid field — treated as alive by list_active.
    }
    base.update(overrides)
    return base


class TestListActive:
    def test_docker_service_entry_returns_local_docker_service_handle(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        active_path = tmp_path / "active.json"
        monkeypatch.setattr(_mod, "_ACTIVE_JSON", active_path)
        _write_active(active_path, {"e1": _docker_entry()})

        handles = LocalInfraConfig().list_active()

        assert len(handles) == 1
        h = handles[0]
        assert isinstance(h, LocalDockerServiceHandle)
        assert h.run_id == "run-abc"
        assert h._container_ids == ["cid1", "cid2"]
        assert h.endpoints == {"svc": "http://127.0.0.1:7780"}

    def test_vm_entry_returns_local_resource_handle(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        active_path = tmp_path / "active.json"
        monkeypatch.setattr(_mod, "_ACTIVE_JSON", active_path)
        _write_active(active_path, {"e2": _vm_entry()})

        handles = LocalInfraConfig().list_active()

        assert len(handles) == 1
        assert isinstance(handles[0], LocalResourceHandle)
        assert handles[0].run_id == "run-vm"

    def test_mixed_entries_return_correct_handle_types(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        active_path = tmp_path / "active.json"
        monkeypatch.setattr(_mod, "_ACTIVE_JSON", active_path)
        _write_active(active_path, {"e1": _docker_entry(), "e2": _vm_entry()})

        handles = LocalInfraConfig().list_active()

        assert len(handles) == 2
        types = {type(h) for h in handles}
        assert LocalDockerServiceHandle in types
        assert LocalResourceHandle in types

    def test_list_active_filtered_by_run_id(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        active_path = tmp_path / "active.json"
        monkeypatch.setattr(_mod, "_ACTIVE_JSON", active_path)
        _write_active(
            active_path,
            {
                "e1": _docker_entry(run_id="run-aaa"),
                "e2": _docker_entry(run_id="run-bbb"),
            },
        )

        handles = LocalInfraConfig().list_active(run_id="run-aaa")

        assert len(handles) == 1
        assert handles[0].run_id == "run-aaa"

    def test_entries_for_other_infra_fingerprints_are_ignored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        active_path = tmp_path / "active.json"
        monkeypatch.setattr(_mod, "_ACTIVE_JSON", active_path)
        _write_active(active_path, {"e1": _docker_entry(infra_fingerprint="aws:us-east-1")})

        assert LocalInfraConfig().list_active() == []


# ── LocalDockerServiceHandle.close() ─────────────────────────────────────────


class TestLocalDockerServiceHandleClose:
    def _make_handle(
        self,
        container_ids: list[str],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> LocalDockerServiceHandle:
        active_path = tmp_path / "active.json"
        monkeypatch.setattr(_mod, "_ACTIVE_JSON", active_path)
        _write_active(active_path, {"entry-1": {}})

        return LocalDockerServiceHandle(
            run_id="run-1",
            resource=DockerServiceConfig(name="svc", scope="task"),
            infra=LocalInfraConfig(),
            endpoint="http://127.0.0.1:7780",
            created_at=datetime.utcnow(),
            _entry_id="entry-1",
            _container_ids=container_ids,
        )

    def test_close_stops_and_removes_each_container(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        handle = self._make_handle(["cid1", "cid2"], tmp_path, monkeypatch)
        with patch("cube.infra_local.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            handle.close()

        issued = [c.args[0] for c in mock_run.call_args_list]
        assert ["docker", "stop", "cid1"] in issued
        assert ["docker", "rm", "cid1"] in issued
        assert ["docker", "stop", "cid2"] in issued
        assert ["docker", "rm", "cid2"] in issued

    def test_close_tolerates_no_such_container(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        handle = self._make_handle(["gone"], tmp_path, monkeypatch)
        with patch("cube.infra_local.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="Error: No such container: gone")
            handle.close()  # must not raise

    def test_close_deregisters_entry_from_active_json(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        handle = self._make_handle([], tmp_path, monkeypatch)
        active_path = tmp_path / "active.json"

        with patch("cube.infra_local.subprocess.run"):
            handle.close()

        assert "entry-1" not in json.loads(active_path.read_text())
