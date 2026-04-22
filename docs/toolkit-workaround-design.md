# Toolkit stable workaround — background + poll exec

## Problem recap

`eai job exec` has a ~6–9% per-call hang rate with hangs clustered in bursts.
All observed hangs show the CLI stuck in a blocking `read()` syscall on the
server-response socket — command completed on the cluster side, response
never arrives. **Retry helps for short commands** (seconds) but **wastes huge
time for long commands** (pytest) because it re-runs the work.

## Design

Replace single long-running `eai job exec <long-cmd>` with three short calls:

1. **Kick off**: `eai job exec <id> -- bash -c "(<cmd> > /tmp/cmd.out 2>&1; echo $? > /tmp/cmd.rc) & disown"`
   — Returns in < 1 s. No ongoing read-response to hang on.

2. **Poll**: `eai job exec <id> -- bash -c "test -f /tmp/cmd.rc && cat /tmp/cmd.rc"`
   — Poll every 15–30 s. Each call is <1 s. Retry-safe.

3. **Collect**: `eai job exec <id> -- bash -c "cat /tmp/cmd.out"`
   — Single short exec for the output. If output is > 10 KB, stream it
   via chunked reads (`tail -c+N` in pieces).

## Idempotent kick (critical — observed in the wild)

An earlier version had the kick directly do:

```bash
(<cmd> > /tmp/cmd.out 2>&1; echo $? > /tmp/cmd.rc) & disown
```

When the kick RPC hangs, ``_run_eai`` retries. If the first kick actually
reached the container (which the bugreport's H4 test confirms it often
does), the retry re-runs the command. For **non-idempotent** commands
like ``git apply`` this corrupts state — the second run fails because
the patch is already applied, and the cube sees rc≠0.

Observed failure: ``swebench_verified-toolkit`` passed with retry when
only pytest used bg+poll but ``git apply`` still used plain exec. On
re-run with the retry hitting during ``git apply``, the retried second
apply failed and reward=0.

Fix — gate the kick with a POSIX-atomic ``mkdir`` lock:

```bash
if mkdir /tmp/cmd.lock 2>/dev/null; then
  (<cmd> > /tmp/cmd.out 2>&1; echo $? > /tmp/cmd.rc) & disown
  echo KICKED
else
  echo ALREADY_STARTED
fi
```

``mkdir`` is atomic across concurrent processes. First kick acquires
the lock and fires the command. Retries see the lock and no-op. The
command runs **exactly once** regardless of how many times the kick
RPC is retried. Cleanup at the end removes lock + marker + output
files together via ``rm -rf``.

## Why this is stable

- **Each exec call is short and cheap to retry**. With 6 % hang rate and 2
  retries, per-call failure = 0.06³ ≈ 0.02 %.
- **Long-running work is decoupled from any RPC channel**. The container
  runs the command with its own lifecycle; the CLI's transient state
  doesn't matter.
- **No data loss on hang**: the `.rc` sentinel + output file mean a failed
  poll call just means "check again later" — no state to recover.

## Proposed abstraction (cube-standard side)

```python
class Container(ABC):
    # Existing
    def exec(self, cmd, timeout=...) -> ExecResult: ...

    # New (default impl for backends that don't need it)
    def exec_long_running(self, cmd, *, timeout: int, poll_interval: int = 30) -> ExecResult:
        """Run `cmd` in the background in the container, polling for completion.

        Designed for commands that take minutes (pytest, heavy git, builds)
        where a single blocking exec would suffer RPC-layer timeouts or
        hangs.  Each underlying exec call is short and retry-safe.

        Default: delegate to self.exec(cmd, timeout=timeout) — fine for
        backends with reliable exec streaming (LocalContainer, DaytonaContainer).
        Toolkit/slow backends override.
        """
        return self.exec(cmd, timeout=timeout)

class ToolkitContainer(Container):
    def exec_long_running(self, cmd, *, timeout, poll_interval=30):
        marker = f"/tmp/cube_lr_{uuid.uuid4().hex[:8]}"
        self.exec(
            f"rm -f {marker}.rc {marker}.out; "
            f"(timeout {timeout}s {cmd} > {marker}.out 2>&1; echo $? > {marker}.rc) & disown",
            timeout=30,
        )
        deadline = time.monotonic() + timeout + 60
        while time.monotonic() < deadline:
            rc_result = self.exec(f"test -f {marker}.rc && cat {marker}.rc || echo PENDING", timeout=30)
            rc_text = rc_result.stdout.strip()
            if rc_text != "PENDING":
                rc = int(rc_text)
                out = self.exec(f"cat {marker}.out", timeout=60).stdout
                self.exec(f"rm -f {marker}.rc {marker}.out", timeout=30)
                return ExecResult(stdout=out, stderr="", exit_code=rc, duration_seconds=0.0)
            time.sleep(poll_interval)
        raise ContainerExecError(f"exec_long_running timed out after {timeout}s")
```

## Where to use it

Cube code that uses `self.tool.bash_unlimited(cmd, timeout=N)` for large `N`:

- `swebench-verified-cube.task._run_tests` (timeout=1800)
- `swebench-live-cube.task._run_test_cmds` (timeout=1800)
- `terminalbench-cube.task.evaluate` → `test.sh` (timeout=900)

The cube-side change is one-liner per call site: replace `self.tool.bash`
with `self.tool.bash_long_running` when the command is expected to take
minutes.

## Tradeoffs

- **Single-process parallelism**: the `& disown` forks a bg job inside the
  container's sleep-infinity process. If many long-running commands fire
  concurrently against one container, they'll race for output files. Not a
  concern in our single-task model.
- **Polling overhead**: 30 s poll interval means up to 30 s delay between
  completion and observation. For a 15-min pytest that's tolerable.
- **Output size**: `cat /tmp/cmd.out` would fail if output > 10 KB (shell
  arg limit comes back on the return path too). Mitigation: stream via
  `dd if=/tmp/cmd.out bs=8192 count=N skip=M` chunks.
- **Not universally needed**: LocalContainer / DaytonaContainer / ModalContainer
  don't need this — their exec streams reliably. Only Toolkit opts in.

## Estimated impact on test matrix

- swebench_live-toolkit: 30-min pytest hung the response channel in 2/2
  attempts. With background+poll: 1 short exec (≈1 s) kicks off; ~60 polls
  × 1 s = 60 s CLI time across 15 min wall; each poll individually retryable.
  Expected success rate: > 99%.
- swebench_verified-toolkit: already borderline (passes with retry). Would
  become rock-solid.
- terminalbench-toolkit: still fails due to test.sh network issue, unrelated.

## Open questions

1. Does `& disown` survive if the parent shell (the `bash -c` from eai exec)
   exits? Yes — `disown` removes the bg job from the shell's job table, and
   the process inherits `sleep infinity` (PID 1 of the container) as parent.
2. Should this be a Container abstraction or a cube-side helper?  The
   Container layer is cleaner, and "long-running" is a legitimate generic
   notion — argues for Container method.
3. Should polling use `inotifywait` instead of busy-polling? Not all images
   ship inotifywait. Polling every 30 s is wasteful but universally available.
