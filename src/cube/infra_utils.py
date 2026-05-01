"""
Shared SSH + bootstrap monitoring utilities for cloud InfraConfig implementations.

Used by cube-infra-aws, cube-infra-azure, and any future cloud backends.
"""

from __future__ import annotations

import logging
import os
import re
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

_TUNNEL_LOG_DIR = Path(os.environ.get("CUBE_SSH_TUNNEL_LOG_DIR", "/tmp/cube-tunnels"))

# Tracks ports already handed out by free_port() in this process. Closing the
# probe socket releases the OS-level reservation, so without this set, two
# concurrent callers can both probe-bind, both close, and both return the
# same number — and the slower SSH tunnel binds last and wins, sending the
# first caller's traffic to the wrong VM.
#
# EX-002 documented exception: _RESERVED_PORTS is an intentional module-level
# singleton. Ray forks a fresh worker process per episode, so each worker has
# its own isolated copy — no cross-episode sharing occurs. This is the same
# pattern as the OTel TracerProvider exception in the constitution (Pillar II,
# "No Global State"). The lock ensures thread-safety within a single worker.
_RESERVED_PORTS: set[int] = set()
_RESERVED_PORTS_LOCK = threading.Lock()

# ── SSH utilities ─────────────────────────────────────────────────────────────


def free_port(start: int = 15000, count: int = 1000) -> int:
    """Find a free local TCP port. Raises RuntimeError if none found.

    Thread-safe: callers in the same process never receive duplicates, even if
    the port hasn't been bound yet by the eventual consumer (e.g. ssh -L).
    Released ports are not recycled — assumes the process lifetime is bounded
    (an eval run, a probe), which keeps the set small in practice.

    Iteration starts at a process-PID-derived offset within ``[start, start+count)``
    to spread parallel workers across the range — without this, 50 Ray workers
    all start at ``start`` and stomp on each other 90%+ of the time when the
    SSH tunnel races to actually bind the port.
    """
    pid_offset = (os.getpid() * 7919) % count  # cheap deterministic spread
    with _RESERVED_PORTS_LOCK:
        for i in range(count):
            port = start + ((pid_offset + i) % count)
            if port in _RESERVED_PORTS:
                continue
            try:
                with socket.socket() as s:
                    s.bind(("127.0.0.1", port))
            except OSError:
                continue
            _RESERVED_PORTS.add(port)
            return port
        raise RuntimeError(f"No free port in {start}–{start + count - 1}")


_TUNNEL_BOUND_RE = re.compile(r"Local forwarding listening on 127\.0\.0\.1 port \d+\.")


def open_tunnel(
    vm_ip: str,
    ssh_user: str,
    ssh_privkey: str,
    remote_port: int = 5000,
    max_attempts: int = 30,
    establish_timeout: float = 30.0,
) -> tuple[subprocess.Popen, int]:
    """Open SSH tunnel localhost:<picked-port> → vm_ip:{remote_port}.

    Returns ``(subprocess, local_port)`` — caller must .terminate() the subprocess
    when done.

    Picks a local port via ``free_port()``, spawns SSH with ``-L``, then waits
    for SSH to log "Local forwarding listening on 127.0.0.1 port N." (which
    SSH only writes after auth completes AND the local bind succeeds). If SSH
    exits before that line (almost always a port collision: free_port's
    test-bind window let a parallel worker grab the same port first), retries
    with a different port.

    Why we have to wait for the log line, not just check ``proc.poll()`` after
    a fixed sleep: SSH binds the local listener *after* SSH-level auth, which
    on Azure routinely takes 2–5 s. A naive ``sleep(2) + poll()`` returns
    "alive" mid-auth, before the bind has even been attempted, and we'd
    wrongly hand back an endpoint pointing at a tunnel that crashes seconds
    later — which then routes to whatever else binds that port, surfacing as
    a mysterious "dead Flask" 502 from a Caddy on a *different* VM. See
    `recipes/waa/DEAD_FLASK_FINDINGS.md`.

    Raises ``RuntimeError`` after ``max_attempts`` consecutive failures.
    """
    _TUNNEL_LOG_DIR.mkdir(parents=True, exist_ok=True)
    last_log_path: Path | None = None
    last_rc: int | None = None
    for attempt in range(1, max_attempts + 1):
        local_port = free_port()
        log_path = _TUNNEL_LOG_DIR / f"{vm_ip}_{local_port}_{remote_port}.log"
        log_fh = open(log_path, "w")
        proc = subprocess.Popen(
            [
                "ssh",
                "-N",
                "-vv",  # log connection-level events to stderr; captured in log_fh
                "-L",
                f"127.0.0.1:{local_port}:127.0.0.1:{remote_port}",
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
                # Default ServerAliveCountMax=3 → ssh exits after 90s of network silence.
                # Bumping to 30 means we tolerate up to 15 min of host-side network blips
                # before the tunnel gives up. Mac networks have intermittent multi-second
                # hiccups (DNS, WiFi roaming, VPN reconnect) that would otherwise drop
                # 8+ parallel tunnels simultaneously.
                "-o",
                "ServerAliveCountMax=30",
                "-o",
                "TCPKeepAlive=yes",
                "-o",
                "IdentitiesOnly=yes",
                f"{ssh_user}@{vm_ip}",
            ],
            stderr=log_fh,
        )
        # Wait for either: SSH writes the bind-success line, SSH exits, or
        # we exceed the establish_timeout. Polling cadence ~100ms.
        deadline = time.time() + establish_timeout
        established = False
        while time.time() < deadline:
            if proc.poll() is not None:
                break  # SSH exited — collision or auth failure
            try:
                if _TUNNEL_BOUND_RE.search(log_path.read_text()):
                    established = True
                    break
            except OSError:
                pass
            time.sleep(0.1)
        if established:
            if attempt > 1:
                logger.info(
                    "open_tunnel: established localhost:%d → %s:%d on attempt %d",
                    local_port,
                    vm_ip,
                    remote_port,
                    attempt,
                )
            return proc, local_port
        # Either SSH exited (collision) or timeout. Tear down and retry.
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        last_rc = proc.returncode
        last_log_path = log_path
        log_fh.flush()
        log_fh.close()
    try:
        tail = last_log_path.read_text()[-1500:] if last_log_path else "<no log>"
    except Exception:
        tail = "<could not read tunnel log>"
    raise RuntimeError(
        f"SSH tunnel to {vm_ip}:{remote_port} failed to establish after "
        f"{max_attempts} attempts (last rc={last_rc}). Most recent tunnel "
        f"log tail:\n{tail}"
    )


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
        proc, local_port = open_tunnel(vm_ip, ssh_user, ssh_privkey, remote_port)
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


# ── Volume setup ─────────────────────────────────────────────────────────────


def build_volume_setup_script(volumes: list) -> str:
    """Generate a bash script that downloads archives and extracts to Docker volumes.

    Each VolumeSpec with a ``source_url`` is downloaded (idempotent — skips if the
    file already exists) and extracted into the named Docker volume.  Archives
    referenced by multiple VolumeSpec entries are downloaded once.

    Volumes without ``source_url`` are created empty.

    Args:
        volumes: list of VolumeSpec objects from DockerServiceConfig.volumes.

    Returns:
        Bash script fragment (can be empty if no volumes).
    """
    if not volumes:
        return ""

    lines: list[str] = [
        "# ── Volume setup (generated by build_volume_setup_script) ────────────────────",
        "VOLUME_DATA_DIR=/opt/cube/volume-data",
        "mkdir -p $VOLUME_DATA_DIR",
    ]

    # Collect unique URLs to download, keyed by vol.name to avoid basename collisions.
    urls_seen: set[str] = set()
    for vol in volumes:
        if vol.source_url and vol.source_url not in urls_seen:
            urls_seen.add(vol.source_url)
            basename = vol.source_url.rsplit("/", 1)[-1]
            filename = f"{vol.name}_{basename}"
            lines.append(f'echo "[bootstrap] Downloading {basename} for {vol.name} ..."')
            lines.append(
                f'[ -f "$VOLUME_DATA_DIR/{filename}" ] || '
                f'curl -L --retry 3 --retry-delay 10 -o "$VOLUME_DATA_DIR/{filename}" "{vol.source_url}"'
            )

    # Create and populate volumes
    for vol in volumes:
        lines.append(f'docker volume create "{vol.name}" 2>/dev/null || true')
        if vol.source_url:
            basename = vol.source_url.rsplit("/", 1)[-1]
            filename = f"{vol.name}_{basename}"
            # Skip extraction if volume already has data
            lines.append(
                f'if ! docker run --rm -v "{vol.name}:/vol:ro" alpine sh -c "ls -A /vol | head -1" | grep -q .; then'
            )
            tar_cmd = f"tar -xf /tar/{filename}"
            if vol.strip_components > 0:
                tar_cmd += f" --strip-components={vol.strip_components}"
            tar_cmd += " -C /vol"
            if vol.tar_subpath:
                tar_cmd += f" {vol.tar_subpath}"
            lines.append(f'  echo "[bootstrap] Extracting {basename} into {vol.name} ..."')
            lines.append(
                f'  docker run --rm -v "$VOLUME_DATA_DIR:/tar:ro" -v "{vol.name}:/vol" alpine sh -c "{tar_cmd}"'
            )
            lines.append("fi")

    lines.append('echo "[bootstrap] Volume setup complete"')
    return "\n".join(lines)
