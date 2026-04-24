"""Modal InfraConfig for CUBE — per-task sandboxes via Modal Sandboxes."""

from cube.backends.modal import ModalContainer
from cube_infra_modal.modal_infra import ModalInfraConfig

__all__ = ["ModalInfraConfig", "ModalContainer"]
