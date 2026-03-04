"""CUBE container backends — re-exports for convenience.

Backends depend on optional third-party packages (docker, daytona, modal, etc.).
This module exposes whichever backends are importable in the current environment.
"""

__all__: list[str] = []

try:
    from cube.backends.local import LocalContainer, LocalContainerBackend

    __all__ += ["LocalContainer", "LocalContainerBackend"]
except ImportError:
    pass

try:
    from cube.backends.daytona import DaytonaContainer, DaytonaContainerBackend

    __all__ += ["DaytonaContainer", "DaytonaContainerBackend"]
except ImportError:
    pass

try:
    from cube.backends.modal import ModalContainer, ModalContainerBackend

    __all__ += ["ModalContainer", "ModalContainerBackend"]
except ImportError:
    pass

try:
    from cube.backends.toolkit import ToolkitContainer, ToolkitContainerBackend

    __all__ += ["ToolkitContainer", "ToolkitContainerBackend"]
except ImportError:
    pass
