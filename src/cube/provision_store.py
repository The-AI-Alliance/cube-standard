"""
ProvisionStore — maps (ResourceConfig, InfraConfig) → resource_info.

Backed by ~/.cube/provisions.json (v1 local).
Key format: "{resource.name}@{infra.fingerprint()}"
e.g.        "osworld-ubuntu-vm@aws:us-east-2"
            "osworld-ubuntu-vm@local"

The store treats resource_info as an opaque dict — only the InfraConfig
provider that wrote it knows how to interpret it. launch() reads it; the
store never inspects its contents.

v2 (deferred): team/CI sharing via CUBE_PROVISION_STORE env var → S3/GCS path.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cube.resource import InfraConfig, ResourceConfig

logger = logging.getLogger(__name__)

_DEFAULT_STORE_PATH = Path(os.environ.get("CUBE_CACHE_DIR", str(Path.home() / ".cube"))) / "provisions.json"


class ProvisionStore:
    """Local JSON-backed store mapping (resource, infra) pairs to resource_info dicts.

    Thread-safe for single-process use (read-modify-write over a small JSON file).
    Not safe for concurrent multi-process writes — use v2 (S3/GCS) for that.

    Usage:
        store = ProvisionStore()
        store.put(resource, infra, {"ami_id": "ami-0abc123"})
        info = store.get(resource, infra)   # → {"ami_id": "ami-0abc123"} or None
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path else _DEFAULT_STORE_PATH

    # ── Key ───────────────────────────────────────────────────────────────────

    @staticmethod
    def key(resource: "ResourceConfig", infra: "InfraConfig") -> str:
        """Build the store key: "{resource.name}@{infra.fingerprint()}"."""
        return f"{resource.name}@{infra.fingerprint()}"

    # ── Read / write ──────────────────────────────────────────────────────────

    def _load(self) -> dict:
        if not self._path.exists():
            return {}
        with open(self._path) as f:
            return json.load(f)

    def _save(self, data: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(data, f, indent=2)

    # ── Public API ────────────────────────────────────────────────────────────

    def get(self, resource: "ResourceConfig", infra: "InfraConfig") -> dict | None:
        """Return resource_info for (resource, infra), or None if not registered."""
        return self._load().get(self.key(resource, infra))

    def put(
        self,
        resource: "ResourceConfig",
        infra: "InfraConfig",
        resource_info: dict,
    ) -> None:
        """Write or overwrite resource_info for (resource, infra)."""
        data = self._load()
        data[self.key(resource, infra)] = resource_info
        self._save(data)
        logger.debug("ProvisionStore: wrote %r", self.key(resource, infra))

    def delete(self, resource: "ResourceConfig", infra: "InfraConfig") -> bool:
        """Remove the entry for (resource, infra). Returns True if it existed."""
        data = self._load()
        k = self.key(resource, infra)
        if k not in data:
            return False
        del data[k]
        self._save(data)
        logger.debug("ProvisionStore: deleted %r", k)
        return True

    def list(self) -> list[tuple[str, dict]]:
        """Return all (key, resource_info) pairs in the store."""
        return list(self._load().items())
