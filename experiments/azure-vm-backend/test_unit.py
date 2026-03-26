"""
Unit tests for azure_backend, aws_backend, and _common utilities.

These tests run locally with no cloud credentials required.
They cover pure logic, regex parsing, and script generation.

Run:
    .venv/bin/python -m pytest test_unit.py -v
"""
from __future__ import annotations

import base64
import re
import socket
import unittest
from unittest.mock import MagicMock, patch


# ── _common ───────────────────────────────────────────────────────────────────

class TestFreePort(unittest.TestCase):
    def test_returns_bindable_port(self):
        from _common import free_port
        port = free_port()
        self.assertGreaterEqual(port, 15000)
        # Confirm we can actually bind to it
        with socket.socket() as s:
            s.bind(("127.0.0.1", port))

    def test_raises_when_range_exhausted(self):
        from _common import free_port
        # Bind all ports in a tiny range, then ask for one
        sockets = []
        start = 19900
        try:
            for p in range(start, start + 5):
                s = socket.socket()
                try:
                    s.bind(("127.0.0.1", p))
                    sockets.append(s)
                except OSError:
                    pass
            with self.assertRaises(RuntimeError):
                free_port(start=start, count=5)
        finally:
            for s in sockets:
                s.close()


class TestBootstrapMonitorParseLine(unittest.TestCase):
    """Test _parse_line — the line classifier for live bootstrap log streaming."""

    def setUp(self):
        from _common import BootstrapMonitor
        import logging
        self.monitor = BootstrapMonitor(
            public_ip="1.2.3.4",
            ssh_privkey="/tmp/key",
            ssh_user="ubuntu",
        )

    def test_stage_marker_is_info(self):
        import logging
        result = self.monitor._parse_line("[bootstrap] Downloading: https://huggingface.co/...")
        self.assertIsNotNone(result)
        level, msg = result
        self.assertEqual(level, logging.INFO)
        self.assertIn("[bootstrap]", msg)

    def test_stage_marker_starting(self):
        import logging
        result = self.monitor._parse_line("[bootstrap] Starting at Thu Mar 26")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], logging.INFO)

    def test_azcopy_5pct_threshold(self):
        import logging
        # First 4% — suppressed
        result = self.monitor._parse_line("  4.0 %, Average Speed:  500 KB/s, 2-sec Throughput (Mb/s): 4.0")
        self.assertIsNone(result)
        # 5% exactly — emitted
        result = self.monitor._parse_line("  5.0 %, Average Speed:  500 KB/s, 2-sec Throughput (Mb/s): 10.5")
        self.assertIsNotNone(result)
        level, msg = result
        self.assertEqual(level, logging.INFO)
        self.assertIn("5%", msg)
        self.assertIn("10 Mb/s", msg)

    def test_azcopy_cumulative_threshold(self):
        import logging
        # Jump to 50% — emitted
        self.monitor._last_pct = 0.0
        result = self.monitor._parse_line(" 50.0 %, 2-sec Throughput (Mb/s): 300.0")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], logging.INFO)
        # 51% — only 1% change, suppressed
        result = self.monitor._parse_line(" 51.0 %, 2-sec Throughput (Mb/s): 300.0")
        self.assertIsNone(result)
        # 55% — 5% change, emitted
        result = self.monitor._parse_line(" 55.0 %, 2-sec Throughput (Mb/s): 300.0")
        self.assertIsNotNone(result)

    def test_azcopy_100pct_always_emitted(self):
        import logging
        self.monitor._last_pct = 99.0
        result = self.monitor._parse_line("100.0 %, 2-sec Throughput (Mb/s): 150.0")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], logging.INFO)

    def test_wget_100pct_emitted(self):
        import logging
        result = self.monitor._parse_line("    100%  25.0G=18m32s")
        self.assertIsNotNone(result)
        level, msg = result
        self.assertEqual(level, logging.INFO)
        self.assertIn("100%", msg)

    def test_wget_partial_suppressed(self):
        # wget intermediate lines (not 100%) fall through to debug
        import logging
        result = self.monitor._parse_line("     50%  12.5G=9m16s")
        # Not a wget 100% match — falls through to debug
        if result is not None:
            self.assertEqual(result[0], logging.DEBUG)

    def test_boto3_512mb_boundary(self):
        import logging
        half_gb = 512 * 1024 * 1024
        # Just under 512 MB — suppressed
        result = self.monitor._parse_line(f"  uploaded {half_gb - 1} bytes")
        self.assertIsNone(result)
        # Cross the 512 MB boundary — emitted
        result = self.monitor._parse_line("  uploaded 2 bytes")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], logging.INFO)
        self.assertIn("0.5 GB", result[1])

    def test_unknown_line_is_debug(self):
        import logging
        result = self.monitor._parse_line("some random log line")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], logging.DEBUG)

    def test_empty_line_is_debug(self):
        import logging
        result = self.monitor._parse_line("")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], logging.DEBUG)


# ── AWSBackend ────────────────────────────────────────────────────────────────

class TestMakeUserData(unittest.TestCase):
    """Test _make_user_data — the EC2 user-data script that injects SSH keys."""

    def setUp(self):
        # AWSBackend needs boto3 at import time but not at instantiation
        import sys
        # Use the venv's boto3 via the venv python — or just mock it
        try:
            from aws_backend import AWSBackend
            self.AWSBackend = AWSBackend
        except ImportError:
            self.skipTest("aws_backend not importable (missing dependencies)")

    def _decode_user_data(self, encoded: str) -> str:
        return base64.b64decode(encoded).decode()

    def test_contains_sshd_install(self):
        b = self.AWSBackend()
        script = self._decode_user_data(b._make_user_data("ssh-ed25519 AAAA testkey"))
        self.assertIn("openssh-server", script)
        self.assertIn("systemctl start ssh", script)

    def test_injects_key_into_user_account(self):
        b = self.AWSBackend()
        pubkey = "ssh-ed25519 AAAA testkey comment"
        script = self._decode_user_data(b._make_user_data(pubkey))
        self.assertIn(pubkey, script)
        self.assertIn("/home/user/.ssh/authorized_keys", script)

    def test_injects_key_into_ubuntu_account(self):
        b = self.AWSBackend()
        pubkey = "ssh-ed25519 AAAA testkey comment"
        script = self._decode_user_data(b._make_user_data(pubkey))
        self.assertIn("/home/ubuntu/.ssh/authorized_keys", script)

    def test_is_valid_base64(self):
        b = self.AWSBackend()
        encoded = b._make_user_data("ssh-ed25519 AAAA testkey")
        # Should not raise
        decoded = base64.b64decode(encoded)
        self.assertTrue(decoded.startswith(b"#!/bin/bash"))

    def test_shebang_first_line(self):
        b = self.AWSBackend()
        script = self._decode_user_data(b._make_user_data("ssh-ed25519 AAAA key"))
        self.assertTrue(script.startswith("#!/bin/bash"))


class TestAWSBackendDefaults(unittest.TestCase):
    def setUp(self):
        try:
            from aws_backend import AWSBackend
            self.AWSBackend = AWSBackend
        except ImportError:
            self.skipTest("aws_backend not importable")

    def test_default_region(self):
        b = self.AWSBackend()
        self.assertEqual(b.region, "us-east-2")

    def test_default_instance_type(self):
        b = self.AWSBackend()
        self.assertEqual(b.instance_type, "t3.xlarge")

    def test_open_tunnel_parameter_name(self):
        """Regression: parameter was renamed do_open_tunnel → open_tunnel."""
        import inspect
        b = self.AWSBackend()
        sig = inspect.signature(b.launch)
        self.assertIn("open_tunnel", sig.parameters)
        self.assertNotIn("do_open_tunnel", sig.parameters)


# ── AzureBackend ──────────────────────────────────────────────────────────────

class TestAzureBackendDefaults(unittest.TestCase):
    def setUp(self):
        try:
            from azure_backend import AzureBackend
            self.AzureBackend = AzureBackend
        except ImportError:
            self.skipTest("azure_backend not importable")

    def test_default_location(self):
        b = self.AzureBackend()
        self.assertEqual(b.location, "westus2")

    def test_default_vm_size(self):
        b = self.AzureBackend()
        self.assertEqual(b.vm_size, "Standard_D4s_v3")

    def test_open_tunnel_parameter_name(self):
        """Regression: parameter was renamed do_open_tunnel → open_tunnel."""
        import inspect
        b = self.AzureBackend()
        sig = inspect.signature(b.launch)
        self.assertIn("open_tunnel", sig.parameters)
        self.assertNotIn("do_open_tunnel", sig.parameters)

    def test_cloud_init_contains_provision_vm_agent_false(self):
        """Regression: OSProvisioningTimedOut fix — provision_vm_agent must be False."""
        import inspect
        src = inspect.getsource(self.AzureBackend.launch)
        self.assertIn("provision_vm_agent", src)
        self.assertIn("False", src)

    def test_cloud_init_uses_custom_data_not_ssh_keys(self):
        """Regression: SSH key must be injected via custom_data (cloud-init), not os_profile.ssh."""
        import inspect
        src = inspect.getsource(self.AzureBackend.launch)
        self.assertIn("custom_data", src)
        # The old waagent-dependent approach should not be present
        self.assertNotIn('"ssh": {"public_keys"', src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
