"""
Shared SSH + bootstrap monitoring utilities for cloud InfraConfig implementations.

Used by cube-infra-aws, cube-infra-azure, and any future cloud backends.
"""

from __future__ import annotations

import logging
import re
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

# ── SSH utilities ─────────────────────────────────────────────────────────────


def free_port(start: int = 15000, count: int = 200) -> int:
    """Find a free local TCP port. Raises RuntimeError if none found."""
    for port in range(start, start + count):
        try:
            with socket.socket() as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"No free port in {start}–{start + count - 1}")


def open_tunnel(
    vm_ip: str,
    ssh_user: str,
    ssh_privkey: str,
    local_port: int,
    remote_port: int = 5000,
) -> subprocess.Popen:
    """Open SSH tunnel localhost:{local_port} → vm_ip:{remote_port}.

    Returns the subprocess — caller must .terminate() it.
    Waits 2 seconds for the tunnel to establish.
    """
    proc = subprocess.Popen(
        [
            "ssh",
            "-N",
            "-L",
            f"127.0.0.1:{local_port}:localhost:{remote_port}",
            "-i",
            ssh_privkey,
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "IdentitiesOnly=yes",
            f"{ssh_user}@{vm_ip}",
        ],
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)
    return proc


def open_tunnels(
    vm_ip: str,
    ssh_user: str,
    ssh_privkey: str,
    service_ports: dict[str, int],
) -> tuple[dict[str, str], list[subprocess.Popen]]:
    """Open one SSH tunnel per service and return (endpoints dict, tunnel procs).

    Each entry in ``service_ports`` maps a service name to a guest port.
    The returned ``endpoints`` maps the same service names to
    ``http://localhost:{local_port}`` URLs.
    """
    endpoints: dict[str, str] = {}
    procs: list[subprocess.Popen] = []
    for name, remote_port in service_ports.items():
        local_port = free_port()
        proc = open_tunnel(vm_ip, ssh_user, ssh_privkey, local_port, remote_port)
        procs.append(proc)
        endpoints[name] = f"http://localhost:{local_port}"
    return endpoints, procs


def wait_for_ssh(
    public_ip: str,
    ssh_user: str,
    ssh_privkey: str,
    fallback_users: list[str] | None = None,
    timeout: int = 300,
    retry_interval: int = 10,
) -> str:
    """Block until SSH is accepting connections. Returns the user that succeeded.

    Tries ssh_user first, then fallback_users in order on each retry.
    """
    log = logging.getLogger(__name__)
    users = [ssh_user] + (fallback_users or [])
    deadline = time.time() + timeout
    while time.time() < deadline:
        for user in users:
            r = subprocess.run(
                [
                    "ssh",
                    "-i",
                    ssh_privkey,
                    "-o",
                    "IdentitiesOnly=yes",
                    "-o",
                    "StrictHostKeyChecking=no",
                    "-o",
                    "UserKnownHostsFile=/dev/null",
                    "-o",
                    "ConnectTimeout=5",
                    "-o",
                    "BatchMode=yes",
                    f"{user}@{public_ip}",
                    "echo OK",
                ],
                capture_output=True,
                text=True,
            )
            if "OK" in r.stdout:
                log.info("SSH available as %s@%s", user, public_ip)
                return user
        time.sleep(retry_interval)
    raise TimeoutError(f"SSH not available after {timeout}s")


def ssh_run(
    public_ip: str,
    ssh_user: str,
    ssh_privkey: str,
    script: str,
) -> None:
    """Run a bash script on the remote host via SSH. Raises on non-zero exit.

    Stdout is forwarded at DEBUG level; stderr at WARNING on failure.
    """
    log = logging.getLogger(__name__)
    result = subprocess.run(
        [
            "ssh",
            "-i",
            ssh_privkey,
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=30",
            f"{ssh_user}@{public_ip}",
            "bash -s",
        ],
        input=script,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        log.debug("[ssh] %s", line)
    if result.returncode != 0:
        for line in result.stderr.splitlines():
            log.warning("[ssh stderr] %s", line)
        raise subprocess.CalledProcessError(result.returncode, "ssh", result.stdout, result.stderr)


# ── BootstrapMonitor ──────────────────────────────────────────────────────────


@dataclass
class BootstrapMonitor:
    """SSH-tails /var/log/cube-bootstrap.log and parses it into structured log events.

    Runs two background threads:
    - _tail_thread: SSHs into the bootstrap VM, tails the log, parses and emits lines.
    - _poll_thread: calls sentinel_fn() every poll_interval seconds; sets _done on success.

    Usage::

        monitor = BootstrapMonitor(
            public_ip="20.x.x.x",
            ssh_privkey="/home/user/.ssh/id_ed25519",
            ssh_user="azureuser",
            sentinel_fn=lambda: backend.blob_exists("Ubuntu.vhd.bootstrap_done"),
        )
        with monitor:
            monitor.wait(timeout=7200)
    """

    public_ip: str
    ssh_privkey: str
    ssh_user: str = "azureuser"
    log_path: str = "/var/log/cube-bootstrap.log"
    sentinel_fn: Callable[[], bool] | None = None
    poll_interval: int = 30
    timeout: int = 7200

    _log: logging.Logger = field(init=False)
    _tail_thread: threading.Thread | None = field(init=False, default=None)
    _poll_thread: threading.Thread | None = field(init=False, default=None)
    _done: threading.Event = field(init=False, default_factory=threading.Event)
    _failed: threading.Event = field(init=False, default_factory=threading.Event)
    _failure_msg: str | None = field(init=False, default=None)
    _tail_proc: subprocess.Popen | None = field(init=False, default=None)

    _boto3_uploaded: int = field(init=False, default=0)
    _last_pct: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self._log = logging.getLogger("cube.bootstrap.vm")

    # ── Line patterns ─────────────────────────────────────────────────────────

    _AZCOPY_RE = re.compile(r"([\d.]+) %.*?2-sec Throughput \(Mb/s\): ([\d.]+)")
    _WGET_PCT = re.compile(r"(\d+)%\s+([\d.]+[KMG])=")
    _BOTO3_RE = re.compile(r"uploaded (\d+) bytes")
    _STAGE_RE = re.compile(r"\[bootstrap\] ")

    def _parse_line(self, line: str) -> tuple[int, str] | None:
        """Return (log_level, message) or None to suppress."""
        if self._STAGE_RE.search(line):
            return logging.INFO, line

        m = self._AZCOPY_RE.search(line)
        if m:
            pct, mbps = float(m.group(1)), float(m.group(2))
            if pct - self._last_pct >= 5.0 or pct >= 99.9:
                self._last_pct = pct
                return logging.INFO, f"  upload: {pct:.0f}%  {mbps:.0f} Mb/s"
            return None

        m = self._WGET_PCT.search(line)
        if m and m.group(1) == "100":
            return logging.INFO, f"  download: 100% ({m.group(2)})"

        m = self._BOTO3_RE.search(line)
        if m:
            self._boto3_uploaded += int(m.group(1))
            gb = self._boto3_uploaded / 1024**3
            if self._boto3_uploaded % (5 * 1024 * 1024 * 1024) < int(m.group(1)):
                return logging.INFO, f"  upload: {gb:.0f} GB uploaded"
            return None

        return logging.DEBUG, line

    # ── Thread bodies ─────────────────────────────────────────────────────────

    def _tail_body(self) -> None:
        attempt = 0
        while not self._done.is_set():
            attempt += 1
            try:
                proc = subprocess.Popen(
                    [
                        "ssh",
                        "-i",
                        self.ssh_privkey,
                        "-o",
                        "IdentitiesOnly=yes",
                        "-o",
                        "StrictHostKeyChecking=no",
                        "-o",
                        "BatchMode=yes",
                        "-o",
                        "ConnectTimeout=30",
                        f"{self.ssh_user}@{self.public_ip}",
                        f"sudo tail -f {self.log_path}",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                )
                self._tail_proc = proc
                for line in proc.stdout:  # type: ignore[union-attr]
                    if self._done.is_set():
                        break
                    line = line.rstrip()
                    result = self._parse_line(line)
                    if result is not None:
                        level, msg = result
                        self._log.log(level, "%s", msg)
                proc.wait()
                if self._done.is_set():
                    return
            except Exception as e:
                self._log.debug("SSH tail attempt %d failed: %s", attempt, e)
            if not self._done.is_set():
                time.sleep(30)

    def _poll_body(self) -> None:
        if self.sentinel_fn is None:
            return
        deadline = time.time() + self.timeout
        t0 = time.time()
        while time.time() < deadline:
            try:
                if self.sentinel_fn():
                    elapsed = int(time.time() - t0)
                    self._log.info("Bootstrap complete after %ds", elapsed)
                    self._done.set()
                    return
            except RuntimeError as e:
                self._failure_msg = str(e)
                self._failed.set()
                self._done.set()
                return
            except Exception:
                pass
            time.sleep(self.poll_interval)
        self._failure_msg = f"Bootstrap timed out after {self.timeout}s"
        self._failed.set()
        self._done.set()

    # ── Public interface ──────────────────────────────────────────────────────

    def start(self) -> None:
        self._tail_thread = threading.Thread(target=self._tail_body, daemon=True, name="bootstrap-tail")
        self._tail_thread.start()
        if self.sentinel_fn is not None:
            self._poll_thread = threading.Thread(target=self._poll_body, daemon=True, name="bootstrap-poll")
            self._poll_thread.start()

    def stop(self) -> None:
        self._done.set()
        if self._tail_proc:
            try:
                self._tail_proc.terminate()
            except Exception:
                pass
        if self._tail_thread:
            self._tail_thread.join(timeout=5)
        if self._poll_thread:
            self._poll_thread.join(timeout=5)

    def wait(self, timeout: int | None = None) -> None:
        self._done.wait(timeout=timeout or self.timeout)
        if self._failed.is_set():
            raise RuntimeError(self._failure_msg or "Bootstrap failed")
        if not self._done.is_set():
            raise TimeoutError(f"Bootstrap did not complete within {timeout or self.timeout}s")

    def __enter__(self) -> "BootstrapMonitor":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()
