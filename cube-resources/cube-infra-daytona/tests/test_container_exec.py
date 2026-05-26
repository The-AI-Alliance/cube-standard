"""DaytonaContainer.exec uses async-submit + idempotent poll, so a transient
proxy read-timeout retries on a poll instead of crashing the whole exec."""

from __future__ import annotations

from typing import Any

import pytest
from cube_infra_daytona import container as dc
from cube_infra_daytona.container import DaytonaContainer

from cube.container import ContainerExecError


class _Resp:
    def __init__(self, cmd_id: str | None, exit_code: int | None = None, output: str = "") -> None:
        self.cmd_id = cmd_id
        self.exit_code = exit_code
        self.output = output


class _Cmd:
    def __init__(self, exit_code: int | None) -> None:
        self.exit_code = exit_code


class _Logs:
    def __init__(self, stdout: str = "", stderr: str = "") -> None:
        self.stdout = stdout
        self.stderr = stderr


class _FakeProcess:
    """Scriptable stand-in for daytona's sync `process`."""

    def __init__(
        self,
        *,
        poll_exit_codes: list[int | None],
        logs: _Logs,
        submit: _Resp | None = None,
        submit_exc: Exception | None = None,
        poll_errors: int = 0,
    ) -> None:
        self._poll_exit_codes = list(poll_exit_codes)
        self._logs = logs
        self._submit = submit if submit is not None else _Resp(cmd_id="cid")
        self._submit_exc = submit_exc
        self._poll_errors = poll_errors
        self.last_request: Any = None
        self.poll_calls = 0

    def create_session(self, session_id: str) -> None:  # called in __init__
        pass

    def execute_session_command(self, session_id: str, request: Any, timeout: int) -> _Resp:
        self.last_request = request
        if self._submit_exc is not None:
            raise self._submit_exc
        return self._submit

    def get_session_command(self, session_id: str, cmd_id: str) -> _Cmd:
        self.poll_calls += 1
        if self._poll_errors > 0:
            self._poll_errors -= 1
            raise TimeoutError("transient proxy read timeout")
        return _Cmd(self._poll_exit_codes.pop(0) if self._poll_exit_codes else 0)

    def get_session_command_logs(self, session_id: str, cmd_id: str) -> _Logs:
        return self._logs


class _FakeSandbox:
    def __init__(self, process: _FakeProcess) -> None:
        self.process = process
        self.id = "sandbox-x"


def _container(proc: _FakeProcess) -> DaytonaContainer:
    return DaytonaContainer(sandbox=_FakeSandbox(proc), client=object())  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _fast_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dc, "_POLL_INTERVAL_S", 0.0)


def test_exec_submits_async_and_returns_split_streams() -> None:
    proc = _FakeProcess(poll_exit_codes=[None, 0], logs=_Logs(stdout="hello\n", stderr="warn\n"))
    res = _container(proc).exec("echo hello", timeout=30)
    assert proc.last_request.run_async is True  # async submit, not the old blocking read
    assert res.stdout == "hello" and res.stderr == "warn"
    assert res.exit_code == 0


def test_exec_propagates_nonzero_exit() -> None:
    proc = _FakeProcess(poll_exit_codes=[3], logs=_Logs())
    assert _container(proc).exec("exit 3", timeout=30).exit_code == 3


def test_exec_retries_transient_poll_error() -> None:
    # First poll raises a transient transport error; @_retry_io re-polls (idempotent) → 0.
    proc = _FakeProcess(poll_exit_codes=[0], logs=_Logs(stdout="ok"), poll_errors=1)
    res = _container(proc).exec("echo ok", timeout=30)
    assert res.exit_code == 0 and res.stdout == "ok"
    assert proc.poll_calls == 2  # one transient failure + one success


def test_exec_deadline_reports_124(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dc, "_POLL_GRACE_S", 0.0)
    proc = _FakeProcess(poll_exit_codes=[], logs=_Logs())  # never resolves
    # timeout=0 ⇒ deadline == start ⇒ poll loop body never runs ⇒ unresolved ⇒ 124.
    assert _container(proc).exec("sleep 999", timeout=0).exit_code == 124


def test_exec_missing_cmd_id_raises() -> None:
    proc = _FakeProcess(poll_exit_codes=[0], logs=_Logs(), submit=_Resp(cmd_id=None))
    with pytest.raises(ContainerExecError):
        _container(proc).exec("echo x", timeout=30)


def test_exec_wraps_submit_failure() -> None:
    proc = _FakeProcess(poll_exit_codes=[0], logs=_Logs(), submit_exc=RuntimeError("boom"))
    with pytest.raises(ContainerExecError, match="Daytona exec failed"):
        _container(proc).exec("echo x", timeout=30)
