"""Shared utilities for Azure and AWS backends."""
from __future__ import annotations

import logging
import re
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


# ── Logging ───────────────────────────────────────────────────────────────────

def configure_logging(debug: bool = False) -> None:
    """Configure root logger. Called once by CLI entry points, never by library code."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


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
            "ssh", "-N",
            "-L", f"127.0.0.1:{local_port}:localhost:{remote_port}",
            "-i", ssh_privkey,
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ExitOnForwardFailure=yes",
            "-o", "ServerAliveInterval=30",
            "-o", "IdentitiesOnly=yes",
            f"{ssh_user}@{vm_ip}",
        ],
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)
    return proc


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
                    "ssh", "-i", ssh_privkey,
                    "-o", "IdentitiesOnly=yes",
                    "-o", "StrictHostKeyChecking=no",
                    "-o", "UserKnownHostsFile=/dev/null",
                    "-o", "ConnectTimeout=5",
                    "-o", "BatchMode=yes",
                    f"{user}@{public_ip}", "echo OK",
                ],
                capture_output=True, text=True,
            )
            if "OK" in r.stdout:
                log.info("SSH available as %s@%s", user, public_ip)
                return user
        time.sleep(retry_interval)
    raise TimeoutError(f"SSH not available after {timeout}s")


# ── Image conversion ──────────────────────────────────────────────────────────

def qemu_img_info(image_path: str) -> dict:
    """Return parsed output of `qemu-img info --output=json`."""
    import json
    result = subprocess.run(
        ["qemu-img", "info", "--output=json", str(image_path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def convert_image(
    src: Path,
    dst: Path,
    output_format: str,
    options: str,
    log: logging.Logger,
) -> None:
    """Run qemu-img convert. Logs format/size at INFO, skips if dst exists."""
    if dst.exists():
        log.info("convert: %s already exists (%.1f GB) — skipping", dst.name, dst.stat().st_size / 1024**3)
        return
    info = qemu_img_info(str(src))
    fmt = info["format"]
    vsize_gb = info["virtual-size"] / 1024**3
    dsize_gb = info.get("disk-size", info["virtual-size"]) / 1024**3
    log.info("convert: %s  format=%s  virtual=%.1f GB  on-disk=%.1f GB", src.name, fmt, vsize_gb, dsize_gb)
    log.info("convert: → %s (%s)", dst.name, output_format)
    t0 = time.time()
    subprocess.run(
        ["qemu-img", "convert", "-f", fmt, "-O", output_format, "-o", options, str(src), str(dst)],
        check=True,
    )
    log.info("convert: done in %.0fs (%.1f GB on disk)", time.time() - t0, dst.stat().st_size / 1024**3)


# ── BootstrapMonitor ─────────────────────────────────────────────────────────

@dataclass
class BootstrapMonitor:
    """SSH-tails /var/log/cube-bootstrap.log and parses it into structured log events.

    Runs two background threads:
    - _tail_thread: SSHs into the bootstrap VM, tails the log, parses and emits lines.
    - _poll_thread: calls sentinel_fn() every poll_interval seconds; sets _done on success.

    Usage:
        monitor = BootstrapMonitor(
            public_ip="20.x.x.x",
            ssh_privkey="/home/user/.ssh/id_ed25519",
            ssh_user="azureuser",
            sentinel_fn=lambda: backend.blob_exists("Ubuntu.vhd.bootstrap_done"),
        )
        with monitor:
            monitor.wait(timeout=7200)
    """
    public_ip:     str
    ssh_privkey:   str
    ssh_user:      str = "azureuser"
    log_path:      str = "/var/log/cube-bootstrap.log"
    sentinel_fn:   Callable[[], bool] | None = None
    poll_interval: int = 30
    timeout:       int = 7200

    _log:         logging.Logger = field(init=False)
    _tail_thread: threading.Thread | None = field(init=False, default=None)
    _poll_thread: threading.Thread | None = field(init=False, default=None)
    _done:        threading.Event = field(init=False, default_factory=threading.Event)
    _failed:      threading.Event = field(init=False, default_factory=threading.Event)
    _failure_msg: str | None = field(init=False, default=None)
    _tail_proc:   subprocess.Popen | None = field(init=False, default=None)

    # Running total of boto3 bytes (to suppress per-MB spam)
    _boto3_uploaded: int = field(init=False, default=0)

    # azcopy / wget progress: only emit every N% change or N MB/s change
    _last_pct: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self._log = logging.getLogger("cube.bootstrap.vm")

    # ── Line patterns ─────────────────────────────────────────────────────────

    _AZCOPY_RE  = re.compile(r"([\d.]+) %.*?2-sec Throughput \(Mb/s\): ([\d.]+)")
    _WGET_PCT   = re.compile(r"(\d+)%\s+([\d.]+[KMG])=")
    _BOTO3_RE   = re.compile(r"uploaded (\d+) bytes")
    _STAGE_RE   = re.compile(r"\[bootstrap\] ")

    def _parse_line(self, line: str) -> tuple[int, str] | None:
        """Return (log_level, message) or None to suppress."""
        # Stage markers → INFO
        if self._STAGE_RE.search(line):
            return logging.INFO, line

        # azcopy: emit every 5% change
        m = self._AZCOPY_RE.search(line)
        if m:
            pct, mbps = float(m.group(1)), float(m.group(2))
            if pct - self._last_pct >= 5.0 or pct >= 99.9:
                self._last_pct = pct
                return logging.DEBUG, f"  upload: {pct:.0f}%  {mbps:.0f} Mb/s"
            return None  # suppress

        # wget final line: "100%  97.0M=3m28s"
        m = self._WGET_PCT.search(line)
        if m and m.group(1) == "100":
            return logging.DEBUG, f"  download: 100% ({m.group(2)})"

        # boto3: accumulate and emit every 512 MB
        m = self._BOTO3_RE.search(line)
        if m:
            self._boto3_uploaded += int(m.group(1))
            gb = self._boto3_uploaded / 1024**3
            # emit every ~512 MB
            if self._boto3_uploaded % (512 * 1024 * 1024) < int(m.group(1)):
                return logging.DEBUG, f"  upload: {gb:.1f} GB uploaded"
            return None  # suppress

        return logging.DEBUG, line

    # ── Thread bodies ─────────────────────────────────────────────────────────

    def _tail_body(self) -> None:
        """Background thread: SSH into VM, tail log, parse and emit each line."""
        for attempt in range(3):
            if self._done.is_set():
                return
            try:
                proc = subprocess.Popen(
                    [
                        "ssh", "-i", self.ssh_privkey,
                        "-o", "IdentitiesOnly=yes",
                        "-o", "StrictHostKeyChecking=no",
                        "-o", "ConnectTimeout=30",
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
                return
            except Exception as e:
                self._log.debug("SSH tail attempt %d failed: %s", attempt + 1, e)
                if not self._done.is_set():
                    time.sleep(10)

    def _poll_body(self) -> None:
        """Background thread: poll sentinel_fn. Sets _done on success, _failed on error."""
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
        """Start both background threads."""
        self._tail_thread = threading.Thread(target=self._tail_body, daemon=True, name="bootstrap-tail")
        self._tail_thread.start()
        if self.sentinel_fn is not None:
            self._poll_thread = threading.Thread(target=self._poll_body, daemon=True, name="bootstrap-poll")
            self._poll_thread.start()

    def stop(self) -> None:
        """Kill the tail subprocess and join threads."""
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
        """Block until done or failed. Raises RuntimeError / TimeoutError."""
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
