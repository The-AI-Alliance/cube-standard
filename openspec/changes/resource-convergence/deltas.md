# Deltas — Resource / Infra convergence

**Targets:** `openspec/specs/resource/spec.md`, `openspec/specs/container/spec.md`, `openspec/specs/task/spec.md`

Applied as each phase lands. The deltas below cover contract changes in `feat/daytona-infra-config` (PR #116); the four items in `proposal.md` are scoped for a separate follow-up once this proposal reaches consensus.

---

## MODIFIED — `ResourceHandle` fields become optional-with-defaults
**Spec:** resource

`ResourceHandle` fields (`run_id`, `resource`, `infra`, `endpoint`) were required dataclass fields. They now carry defaults (`""`, `None`, `None`, `None`) so that `Container` — which subclasses `ResourceHandle` — can call `super().__init__()` with no bookkeeping and let the launcher populate fields post-construction.

Callers that construct a handle directly MUST still populate `run_id`, `resource`, and `infra` for cross-process cleanup tooling (`infra.cleanup(run_id)`, `infra.list_active()`) to find the resource. The InfraConfig launchers in `cube-infra-*` do this immediately after construction.

## ADDED — `Container` subclasses `ResourceHandle`
**Spec:** container, resource

`Container` is now a `ResourceHandle` directly (`class Container(ResourceHandle, ABC)`), eliminating the wrapper-handle dataclasses (`LocalDockerServiceHandle`, `DaytonaResourceHandle`, `ToolkitResourceHandle`, `ModalResourceHandle`) previously used to carry `run_id`/`resource`/`infra` bookkeeping alongside a live container.

An `InfraConfig.launch()` that returns a single `Container` therefore satisfies the `ResourceHandle` protocol directly — no wrapping required. Harness code that reads `handle.container` continues to work via a default `ResourceHandle.container` property that returns `self` for `Container` subclasses.

## ADDED — `Container.exec_long_running`
**Spec:** container

New abstract-with-default method on `Container`:

```python
def exec_long_running(self, command: str, *, poll_interval: float = 10.0, timeout: int = 3600, ...) -> ExecResult:
    """Execute a command that may run longer than the backend's per-exec timeout.
    Default implementation: background the command on the container, poll for completion."""
```

Backends whose per-exec timeout ceiling is well above task-level operations (local docker, Modal) fall back to `exec()`. Backends with lower ceilings (ToolkitContainer — see `docs/toolkit-exec-relay-design.md`) override with a background-and-poll implementation to sidestep CLI-level hangs.

Backwards-compatible: callers that only use `exec()` are unaffected.

## ADDED — `Task._build_tool()` and `Task._resource_handle`
**Spec:** task

`Task` gains two new members so cube authors don't have to override `model_post_init` for the common case:

- `_resource_handle: ResourceHandle | None` (`PrivateAttr`) — stores the handle returned by `InfraConfig.launch()` on the infra path; `Task.close()` tears it down.
- `_build_tool(self) -> Tool` — hook called from `model_post_init` after the container is launched. Default implementation: `return self.tool_config.make(container=self._container)`. Cubes override to do cube-specific tool setup (e.g. directory relocation) without touching `model_post_init`.

## MODIFIED — `Task.model_post_init` handles infra-path launch inline
**Spec:** task

`Task.model_post_init` now accepts two launch modes:

1. **Infra path.** If `runtime_context["infra"]` is present and `metadata.container_config` is set, call `launch_task_container(...)` (from `cube.task_infra`) to provision the container via the injected `InfraConfig`, and store the returned handle in `self._resource_handle`.
2. **Container-backend path.** If `container_backend` is set, call `container_backend.launch(metadata.container_config)` directly.

Then call `self._build_tool()`. Cubes no longer need to override `model_post_init` purely to route between infra and backend paths — the base class does it. The container-backend path is preserved for back-compat with the legacy `cube.backends.*` stubs, but is scheduled for removal in a follow-up (see PR #116 review discussion).

---

See `proposal.md` for items deferred to a future follow-up (launch-label convention, `ResourceHandle.container` typed property, network capability tokens).
