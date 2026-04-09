"""
ProvisionStore — maps (ResourceConfig, InfraConfig) → resource_info.

Each entry is stored as a separate JSON file under ~/.cube/provisions/<key>.json.
This makes reads/writes atomic at the file level and safe for parallel provision()
calls — two processes provisioning different resources never contend, and two
processes provisioning the same resource will both write the same data on success
(idempotent), so a last-writer-wins race is harmless.

Key format: "{resource.name}@{infra.fingerprint()}"
e.g.        "osworld-ubuntu-vm@aws:us-east-2"
            "osworld-ubuntu-vm@local"
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cube.resource import InfraConfig, ResourceConfig

logger = logging.getLogger(__name__)

_DEFAULT_STORE_DIR = Path(os.environ.get("CUBE_CACHE_DIR", str(Path.home() / ".cube"))) / "provisions"

# Characters safe in filenames on all platforms (keeps @, :, -, . which appear in keys).
_SAFE_KEY_RE = re.compile(r"[^A-Za-z0-9._:@-]")


def _key_to_filename(key: str) -> str:
    return _SAFE_KEY_RE.sub("_", key) + ".json"


class ProvisionStore:
    """File-per-entry store mapping (resource, infra) pairs to resource_info dicts.

    Each entry lives in its own JSON file under the store directory.  This makes
    parallel provision() calls safe: different resources never contend, and
    concurrent calls for the same resource are idempotent (same data written).

    Usage:
        store = ProvisionStore()
        store.put(resource, infra, {"ami_id": "ami-0abc123"})
        info = store.get(resource, infra)   # → {"ami_id": "ami-0abc123"} or None
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self._dir = Path(path) if path else _DEFAULT_STORE_DIR

    # ── Key ───────────────────────────────────────────────────────────────────

    @staticmethod
    def key(resource: "ResourceConfig", infra: "InfraConfig") -> str:
        """Build the store key: "{resource.name}@{infra.fingerprint()}"."""
        return f"{resource.name}@{infra.fingerprint()}"

    def _path_for(self, key: str) -> Path:
        return self._dir / _key_to_filename(key)

    # ── Public API ────────────────────────────────────────────────────────────

    def get(self, resource: "ResourceConfig", infra: "InfraConfig") -> dict | None:
        """Return resource_info for (resource, infra), or None if not registered."""
        p = self._path_for(self.key(resource, infra))
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except Exception:
            return None

    def put(
        self,
        resource: "ResourceConfig",
        infra: "InfraConfig",
        resource_info: dict,
    ) -> None:
        """Write or overwrite resource_info for (resource, infra)."""
        key = self.key(resource, infra)
        p = self._path_for(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file then rename for atomic replace.
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(resource_info, indent=2))
        tmp.replace(p)
        logger.debug("ProvisionStore: wrote %r", key)

    def delete(self, resource: "ResourceConfig", infra: "InfraConfig") -> bool:
        """Remove the entry for (resource, infra). Returns True if it existed."""
        key = self.key(resource, infra)
        p = self._path_for(key)
        if not p.exists():
            return False
        p.unlink()
        logger.debug("ProvisionStore: deleted %r", key)
        return True

    def list(self) -> list[tuple[str, dict]]:
        """Return all (key, resource_info) pairs in the store."""
        if not self._dir.exists():
            return []
        result = []
        for p in sorted(self._dir.glob("*.json")):
            try:
                data = json.loads(p.read_text())
                # Filename is the key with .json appended (safe chars are preserved as-is).
                result.append((p.stem, data))
            except Exception:
                pass
        return result
