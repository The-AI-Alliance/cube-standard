"""CUBE Standard - Common Unified Benchmark Environments."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cube-standard")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["__version__"]
