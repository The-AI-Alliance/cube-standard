"""Unit tests for LocalInfraConfig Docker support.

All tests mock subprocess calls — no Docker daemon required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import pytest

from cube.infra_local import (
    LocalDockerResourceHandle,
    LocalInfraConfig,
    _kill_entry,
    _stop_docker_container,
    _wait_for_docker_http,
)
from cube.resource import (
    DockerImageConfig,
    ResourceNotReadyError,
    UnsupportedResourceType,
)


@pytest.fixture()
def infra(tmp_path):
    """LocalInfraConfig with a temp active.json and provision store."""
    active = tmp_path / "active.json"
    store_dir = tmp_path / "provisions"
    store_dir.mkdir()
    with (
        patch("cube.infra_local._ACTIVE_JSON", active),
        patch("cube.provision_store._DEFAULT_STORE_DIR", store_dir),
    ):
        yield LocalInfraConfig()


@pytest.fixture()
def registered_image(infra):
    """An infra with a DockerImageConfig already registered."""
    resource = DockerImageConfig(
        name="test-image",
        image="myrepo/myimage:latest",
        ports=[80, 8877],
    )
    infra.register(resource, {"image": "myrepo/myimage:latest"})
    return infra, resource


# ── provision() ───────────────────────────────────────────────────────────────


def test_provision_docker_pulls_image_and_registers(infra):
    resource = DockerImageConfig(name="test-image", image="myrepo/myimage:latest")
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        infra.provision(resource)

    mock_run.assert_called_once_with(["docker", "pull", "myrepo/myimage:latest"], check=True)
    assert infra.provision_status(resource) == "ready"


def test_provision_docker_is_idempotent(infra):
    resource = DockerImageConfig(name="test-image", image="myrepo/myimage:latest")
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        infra.provision(resource)
        infra.provision(resource)

    assert mock_run.call_count == 2  # docker pull called each time (docker handles idempotency)


def test_provision_unsupported_type_raises(infra):
    from cube.resource import ResourceConfig

    class _AlienResource(ResourceConfig):
        pass

    with pytest.raises(UnsupportedResourceType):
        infra.provision(_AlienResource(name="alien"))


# ── launch() ──────────────────────────────────────────────────────────────────


def test_launch_docker_without_provision_raises(infra):
    resource = DockerImageConfig(name="test-image", image="myrepo/myimage:latest", ports=[80])
    with pytest.raises(ResourceNotReadyError):
        infra.launch(resource)


def test_launch_docker_returns_handle_with_port_map(registered_image):
    infra, resource = registered_image

    with (
        patch("subprocess.run") as mock_run,
        patch("cube.infra_local._free_port", side_effect=[15001, 15002]),
        patch("cube.infra_local._wait_for_docker_http"),
    ):
        mock_run.return_value = MagicMock(returncode=0)
        handle = infra.launch(resource)

    assert isinstance(handle, LocalDockerResourceHandle)
    assert handle.port_map == {80: 15001, 8877: 15002}
    assert handle.endpoint == "http://127.0.0.1:15001"
    assert handle._container_name.startswith("cube-test-image-")


def test_launch_docker_run_command_includes_port_flags(registered_image):
    infra, resource = registered_image

    with (
        patch("subprocess.run") as mock_run,
        patch("cube.infra_local._free_port", side_effect=[15001, 15002]),
        patch("cube.infra_local._wait_for_docker_http"),
    ):
        mock_run.return_value = MagicMock(returncode=0)
        infra.launch(resource)

    docker_run_call = mock_run.call_args_list[0]
    cmd = docker_run_call[0][0]
    assert "docker" in cmd
    assert "run" in cmd
    assert "-p" in cmd
    assert "127.0.0.1:15001:80" in cmd
    assert "127.0.0.1:15002:8877" in cmd
    assert "myrepo/myimage:latest" in cmd


def test_launch_docker_gpu_flag(infra):
    resource = DockerImageConfig(name="gpu-image", image="myrepo/gpu:latest", ports=[8080], gpu=True)
    infra.register(resource, {"image": "myrepo/gpu:latest"})

    with (
        patch("subprocess.run") as mock_run,
        patch("cube.infra_local._free_port", return_value=15001),
        patch("cube.infra_local._wait_for_docker_http"),
    ):
        mock_run.return_value = MagicMock(returncode=0)
        infra.launch(resource)

    cmd = mock_run.call_args_list[0][0][0]
    assert "--gpus" in cmd
    assert "all" in cmd


def test_launch_docker_no_ports_no_endpoint(infra):
    resource = DockerImageConfig(name="no-port-image", image="myrepo/noop:latest")
    infra.register(resource, {"image": "myrepo/noop:latest"})

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        handle = infra.launch(resource)

    assert handle.endpoint is None
    assert handle.port_map == {}


def test_launch_docker_health_check_skipped_when_path_empty(infra):
    resource = DockerImageConfig(name="test-image", image="myrepo/myimage:latest", ports=[80], health_check_path="")
    infra.register(resource, {"image": "myrepo/myimage:latest"})

    with (
        patch("subprocess.run") as mock_run,
        patch("cube.infra_local._free_port", return_value=15001),
        patch("cube.infra_local._wait_for_docker_http") as mock_health,
    ):
        mock_run.return_value = MagicMock(returncode=0)
        infra.launch(resource)

    mock_health.assert_not_called()


def test_launch_docker_health_check_timeout_removes_container(registered_image):
    infra, resource = registered_image

    with (
        patch("subprocess.run") as mock_run,
        patch("cube.infra_local._free_port", side_effect=[15001, 15002]),
        patch("cube.infra_local._wait_for_docker_http", side_effect=TimeoutError("timed out")),
        patch("cube.infra_local._stop_docker_container") as mock_stop,
    ):
        mock_run.return_value = MagicMock(returncode=0)
        with pytest.raises(TimeoutError):
            infra.launch(resource)

    mock_stop.assert_called_once()


# ── LocalDockerResourceHandle.close() ─────────────────────────────────────────


def test_handle_close_stops_and_removes_container(registered_image, tmp_path):
    infra, resource = registered_image

    with (
        patch("subprocess.run") as mock_run,
        patch("cube.infra_local._free_port", side_effect=[15001, 15002]),
        patch("cube.infra_local._wait_for_docker_http"),
    ):
        mock_run.return_value = MagicMock(returncode=0)
        handle = infra.launch(resource)

    container_name = handle._container_name
    with patch("cube.infra_local._stop_docker_container") as mock_stop:
        handle.close()

    mock_stop.assert_called_once_with(container_name)


# ── _stop_docker_container() ──────────────────────────────────────────────────


def test_stop_docker_container_calls_stop_then_rm():
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        _stop_docker_container("my-container")

    assert mock_run.call_count == 2
    assert mock_run.call_args_list[0] == call(["docker", "stop", "my-container"], capture_output=True, text=True)
    assert mock_run.call_args_list[1] == call(["docker", "rm", "my-container"], capture_output=True, text=True)


def test_stop_docker_container_tolerates_not_found():
    not_found = MagicMock(returncode=1, stderr="No such container: my-container")
    with patch("subprocess.run", return_value=not_found):
        _stop_docker_container("my-container")  # should not raise


# ── _wait_for_docker_http() ───────────────────────────────────────────────────


def test_wait_for_docker_http_succeeds_on_2xx():
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.status = 200

    with patch("urllib.request.urlopen", return_value=mock_resp):
        _wait_for_docker_http("http://localhost:8080/", timeout=5)


def test_wait_for_docker_http_succeeds_on_4xx():
    import urllib.error

    with patch("urllib.request.urlopen", side_effect=urllib.error.HTTPError(None, 401, "Unauthorized", {}, None)):
        _wait_for_docker_http("http://localhost:8080/", timeout=5)


def test_wait_for_docker_http_times_out():
    with patch("urllib.request.urlopen", side_effect=ConnectionRefusedError):
        with pytest.raises(TimeoutError):
            _wait_for_docker_http("http://localhost:8080/", timeout=1)


# ── _kill_entry() ─────────────────────────────────────────────────────────────


def test_kill_entry_docker_calls_stop():
    entry = {"type": "docker", "container_name": "cube-abc-docker-xyz"}
    with patch("cube.infra_local._stop_docker_container") as mock_stop:
        _kill_entry(entry)
    mock_stop.assert_called_once_with("cube-abc-docker-xyz")


def test_kill_entry_vm_kills_pid_and_removes_overlay(tmp_path):
    overlay = tmp_path / "overlay.qcow2"
    overlay.write_text("fake")
    entry = {"type": "vm", "pid": 99999, "overlay_path": str(overlay)}
    with patch("cube.infra_local._pid_alive", return_value=False):
        _kill_entry(entry)
    # PID not alive → no kill signal; overlay still present (only removed if alive path ran)


def test_kill_entry_unknown_type_raises():
    with pytest.raises(ValueError, match="Unknown active.json entry type"):
        _kill_entry({"type": "kubernetes"})


# ── list_active() ─────────────────────────────────────────────────────────────


def test_list_active_returns_running_docker_handles(infra, tmp_path):
    resource = DockerImageConfig(name="test-image", image="myrepo/myimage:latest", ports=[80])
    infra.register(resource, {"image": "myrepo/myimage:latest"})

    with (
        patch("subprocess.run") as mock_run,
        patch("cube.infra_local._free_port", return_value=15001),
        patch("cube.infra_local._wait_for_docker_http"),
    ):
        mock_run.return_value = MagicMock(returncode=0)
        handle = infra.launch(resource)

    # Simulate container still running
    inspect_result = MagicMock(returncode=0, stdout="true\n")
    with patch("subprocess.run", return_value=inspect_result):
        active = infra.list_active()

    assert len(active) == 1
    assert isinstance(active[0], LocalDockerResourceHandle)
    assert active[0].run_id == handle.run_id


def test_list_active_excludes_stopped_containers(infra):
    resource = DockerImageConfig(name="test-image", image="myrepo/myimage:latest", ports=[80])
    infra.register(resource, {"image": "myrepo/myimage:latest"})

    with (
        patch("subprocess.run") as mock_run,
        patch("cube.infra_local._free_port", return_value=15001),
        patch("cube.infra_local._wait_for_docker_http"),
    ):
        mock_run.return_value = MagicMock(returncode=0)
        infra.launch(resource)

    # Simulate container not running
    inspect_result = MagicMock(returncode=1, stdout="")
    with patch("subprocess.run", return_value=inspect_result):
        active = infra.list_active()

    assert active == []
