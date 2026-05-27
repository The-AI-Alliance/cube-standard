# Deltas — Structured evaluation results

## ADDED — `cube.core`: `EvaluationCheck` and `TaskResult`

```python
class EvaluationCheck(TypedBaseModel):
    name: str
    passed: bool
    expected: str | None
    actual: str | None
    comment: str | None

class TaskResult(TypedBaseModel):
    reward: float
    checks: list[EvaluationCheck]
    info: dict[str, Any]
```

No defaults — all fields mandatory.

## MODIFIED — `cube.task`: `Task.evaluate()` return type

**Before:**
```python
def evaluate(self, obs: Observation | None = None) -> Tuple[float, dict]:
```

**After:**
```python
def evaluate(self, obs: Observation | None = None) -> TaskResult:
```

## MODIFIED — `cube.task`: `Task.step()` evaluate call site

**Before:**
```python
reward, info = self.evaluate(obs)
```

**After:**
```python
result = self.evaluate(obs)
reward = result.reward
info = result.info
```

`EnvironmentOutput` continues to receive `reward` and `info` separately
— `TaskResult` is not propagated into `EnvironmentOutput`.

## Migration impact

**All `Task` subclasses** — must change `evaluate()` return from
`return reward, info` to `return TaskResult(reward=reward, checks=[], info=info)`.

**Framework call sites** — `Task.step()` and `server.py` adapt to
read `result.reward` / `result.info`.
