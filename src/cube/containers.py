from cube.types import TypedBaseModel

# TODO: flesh out container management interfaces and implementations as we add support for containerized environments and tasks


class Container(TypedBaseModel):
    pass


class ContainerConfig(TypedBaseModel):
    def make(self) -> Container:
        return Container()
