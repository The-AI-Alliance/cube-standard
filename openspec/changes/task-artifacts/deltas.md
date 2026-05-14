# Deltas — Task Artifacts

## ADDED — `cube.core`: Artifact types

```python
class BinaryArtifactBlob(TypedBaseModel):
    type: Literal["binary"] = "binary"
    blob: bytes

class FileArtifactBlob(TypedBaseModel):
    type: Literal["file"] = "file"
    path: Path

class RemoteFileArtifactBlob(TypedBaseModel):
    type: Literal["remote_file"] = "remote_file"
    url: str

class TextArtifactBlob(TypedBaseModel):
    type: Literal["text"] = "text"
    text: str

ArtifactBlob = Annotated[
    BinaryArtifactBlob | TextArtifactBlob | FileArtifactBlob | RemoteFileArtifactBlob,
    Field(discriminator="type"),
]

class Artifact(TypedBaseModel):
    mime: str
    id: str
    blob: ArtifactBlob
```

Typed envelope for side-channel outputs (traces, logs, screenshots)
produced by tools and tasks. The discriminated `ArtifactBlob` union
allows the harness to handle each storage form differently (read from
disk, upload bytes, reference a remote URL).

## ADDED — `cube.tool`: `AbstractTool.artifacts()` / `AbstractAsyncTool.artifacts()`

```python
class AbstractTool(ABC):
    @abstractmethod
    def artifacts(self) -> list[Artifact]: ...

class AbstractAsyncTool(ABC):
    @abstractmethod
    def artifacts(self) -> list[Artifact]: ...
```

Called after `close()` to collect side-channel outputs. Must not raise.
Abstract — direct `AbstractTool`/`AbstractAsyncTool` subclasses must
implement it. The concrete `Tool` and `AsyncTool` bases provide a
default returning `[]`.

`Toolbox.artifacts()` and `AsyncToolbox.artifacts()` concatenate
results from all contained tools.

## ADDED — `cube.task`: `Task.task_artifacts()` and `Task.artifacts()`

```python
class Task:
    def task_artifacts(self) -> list[Artifact]:
        return []

    def artifacts(self) -> list[Artifact]:
        return self.task_artifacts() + self.tool.artifacts()
```

`task_artifacts()` is the override point for task-specific artifacts.
`artifacts()` combines task and tool artifacts — not intended to be
overridden.

## Migration impact

**`Tool` / `AsyncTool` subclasses** — no change required (default
returns `[]`).

**Direct `AbstractTool` / `AbstractAsyncTool` subclasses** — must add
`artifacts() -> list[Artifact]`. No known cases in cube-harness today
(`ToolWithTelemetry` subclasses `Tool`, not `AbstractTool`).

**`Task` subclasses** — no change required (defaults return `[]`).
