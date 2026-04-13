"""CUBE Standard - Common Unified Benchmark Environments."""

import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    __version__ = version("cube-standard")
except PackageNotFoundError:
    __version__ = "unknown"

_CUBE_CACHE_ROOT = Path(os.environ.get("CUBE_CACHE_DIR", str(Path.home() / ".cube")))


def get_cache_dir(benchmark_name: str) -> Path:
    """Get the cache directory for a specific benchmark."""
    return _CUBE_CACHE_ROOT / benchmark_name


__all__ = ["__version__", "get_cache_dir"]

# Resource lifecycle API (design/resource_lifecycle.md)
from cube.infra_local import LocalInfraConfig  # noqa: E402
from cube.provision_store import ProvisionStore  # noqa: E402
from cube.resource import (  # noqa: E402
    DockerImageConfig,
    DockerServiceConfig,
    InfraConfig,
    ResourceConfig,
    ResourceHandle,
    ResourceNotReadyError,
    UnsupportedResourceType,
    VMResourceConfig,
)

__all__ += [
    "ResourceConfig",
    "VMResourceConfig",
    "DockerServiceConfig",
    "DockerImageConfig",
    "InfraConfig",
    "ResourceHandle",
    "ResourceNotReadyError",
    "UnsupportedResourceType",
    "ProvisionStore",
    "LocalInfraConfig",
]
