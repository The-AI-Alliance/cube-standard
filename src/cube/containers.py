from abc import ABC, abstractmethod
from typing import List

from cube.core import TypedBaseModel


class Container(ABC, TypedBaseModel):
    pass


# Separate WHAT to run from HOW to run it.


class ContainerConfig(ABC, TypedBaseModel):
    """
    WHAT to run. Owned by CUBE benchmark/task. Serializable (json)
    Part of TaskConfig
    """

    image: str
    ram_gb: float
    cpu_cores: float
    gpu: bool = False
    ports: List[int] | None = None


class ContainerBackend(ABC, TypedBaseModel):
    """
    HOW to run. Owned by harness users. Contain execution strategy (local,Modal,...).
    Serializable (can pass to Ray workers)
    User chooses which backend to use
    """

    @abstractmethod
    def launch(self, conf: ContainerConfig) -> Container:
        """Launch container from config. Blocks until ready."""
        pass


# Concrete implementations
class LocalContainerBackend(ContainerBackend):
    # TODO
    pass


class ModalContainerBackend(ContainerBackend):
    # TODO
    pass
