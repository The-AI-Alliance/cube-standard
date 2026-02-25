"""Shared test harness for all container backend integration tests.

Each backend test script creates a backend + spec, then calls ``run_all()``.
"""

from __future__ import annotations

import sys
import traceback

from cube.container import (
    ContainerBackend,
    ContainerConfig,
    HealthCheckError,
)


def log(msg: str) -> None:
    print(f"  {msg}")


def _run_one(name: str, fn) -> bool:
    print(f"\n[TEST] {name}")
    try:
        fn()
        print("  \u2713 PASSED")
        return True
    except Exception as exc:
        print(f"  \u2717 FAILED: {exc}")
        traceback.print_exc()
        return False


# ---------------------------------------------------------------------------
# Generic tests — work for any ContainerBackend
# ---------------------------------------------------------------------------


def make_container_common_tests(backend: ContainerBackend, spec: ContainerConfig):
    """Return list of (name, callable) test pairs for the given backend."""

    tests: list[tuple[str, callable]] = []

    def test_launch_and_echo():
        container = backend.launch(spec)
        try:
            result = container.exec("echo hello")
            assert result.stdout.strip() == "hello", f"got: {result.stdout!r}"
            assert result.exit_code == 0
            assert result.duration_seconds > 0
            log(f"stdout={result.stdout.strip()!r} exit={result.exit_code} dur={result.duration_seconds}s")
        finally:
            container.stop()

    tests.append(("launch + echo", test_launch_and_echo))

    def test_exec_env_and_workdir():
        container = backend.launch(spec)
        try:
            r = container.exec("echo $MY_VAR", env={"MY_VAR": "cube_test"})
            assert "cube_test" in r.stdout, f"env not set: {r.stdout!r}"
            log(f"env: {r.stdout.strip()!r}")

            r = container.exec("pwd", workdir="/tmp")
            assert r.stdout.strip() == "/tmp", f"workdir: {r.stdout!r}"
            log(f"workdir: {r.stdout.strip()!r}")
        finally:
            container.stop()

    tests.append(("exec env & workdir", test_exec_env_and_workdir))

    def test_nonzero_exit_and_stderr():
        container = backend.launch(spec)
        try:
            r = container.exec("exit 42")
            assert r.exit_code == 42, f"exit_code={r.exit_code}"
            log(f"exit_code={r.exit_code}")

            r = container.exec("echo out; echo err >&2")
            assert "out" in r.stdout, f"stdout: {r.stdout!r}"
            assert "err" in r.stderr, f"stderr: {r.stderr!r}"
            log(f"stdout={r.stdout.strip()!r} stderr={r.stderr.strip()!r}")
        finally:
            container.stop()

    tests.append(("nonzero exit & stderr", test_nonzero_exit_and_stderr))

    def test_exec_timeout():
        container = backend.launch(spec)
        try:
            result = container.exec("sleep 60", timeout=3)
            # 124 = GNU timeout, 143 = BusyBox SIGTERM (128+15)
            assert result.exit_code in (124, 143), f"expected 124/143, got {result.exit_code}"
            log(f"timed out correctly, exit_code={result.exit_code}")
        finally:
            container.stop()

    tests.append(("exec timeout", test_exec_timeout))

    def test_container_id_and_status():
        container = backend.launch(spec)
        try:
            assert isinstance(container.id, str) and len(container.id) > 0
            log(f"id={container.id[:16]}...")

            status = container.get_status()
            assert status.running is True
            log(f"running={status.running} healthy={status.healthy}")
        finally:
            container.stop()

    tests.append(("container id & status", test_container_id_and_status))

    def test_stop_idempotent():
        container = backend.launch(spec)
        container.stop()
        container.stop()  # must not raise
        log("double stop: ok")

    tests.append(("stop idempotent", test_stop_idempotent))

    def test_status_after_stop():
        container = backend.launch(spec)
        container.stop()
        status = container.get_status()
        assert status.running is False, f"expected running=False, got {status.running}"
        log(f"after stop: running={status.running} healthy={status.healthy}")

    tests.append(("status after stop", test_status_after_stop))

    def test_serialization_roundtrip():
        data = backend.model_dump()
        assert "_type" in data
        log(f"serialized keys: {sorted(data.keys())}")

        restored = ContainerBackend.model_validate(data)
        assert type(restored).__name__ == type(backend).__name__
        assert restored.timeout_seconds == backend.timeout_seconds
        log(f"deserialized: {type(restored).__name__}")

    tests.append(("serialization roundtrip", test_serialization_roundtrip))

    return tests


def make_container_health_check_tests(
    backend: ContainerBackend,
    spec: ContainerConfig,
):
    """Return health-check tests (create backend subclasses that override health_check)."""

    backend_cls = type(backend)
    backend_kwargs = {k: v for k, v in backend.model_dump().items() if k != "_type"}

    tests: list[tuple[str, callable]] = []

    def test_health_check_pass():
        class PassingBackend(backend_cls):
            def health_check(self, container):
                return True

        b = PassingBackend(**backend_kwargs)
        container = b.launch(spec)
        try:
            assert container.get_status().running is True
            log("health check passed, container running")
        finally:
            container.stop()

    tests.append(("health check pass", test_health_check_pass))

    def test_health_check_fail():
        class FailingBackend(backend_cls):
            def health_check(self, container):
                return False

        b = FailingBackend(**backend_kwargs)
        try:
            b.launch(spec)
            assert False, "should have raised HealthCheckError"
        except HealthCheckError as exc:
            log(f"correctly raised: {exc}")

    tests.append(("health check fail", test_health_check_fail))

    def test_health_check_exception():
        class ErrorBackend(backend_cls):
            def health_check(self, container):
                raise RuntimeError("boom")

        b = ErrorBackend(**backend_kwargs)
        try:
            b.launch(spec)
            assert False, "should have raised HealthCheckError"
        except HealthCheckError as exc:
            assert "boom" in str(exc)
            log(f"correctly raised: {exc}")

    tests.append(("health check exception", test_health_check_exception))

    return tests


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_all(
    label: str,
    tests: list[tuple[str, callable]],
) -> None:
    print("=" * 60)
    print(label)
    print("=" * 60)

    passed = failed = 0
    for name, fn in tests:
        if _run_one(name, fn):
            passed += 1
        else:
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    print("=" * 60)
    sys.exit(1 if failed else 0)
