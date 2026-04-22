# `eai job exec` hangs under sustained load — bug report

**Reporter**: Alexandre Lacoste
**Date**: 2026-04-22
**Cluster**: yul101
**`eai` CLI versions tested**: 1.13.1 and 2.0.13 darwin — both reproduce the hang.
**Severity**: Medium. Blocks "interactive agent over long-lived job" workloads; workaround available (see below).
**Root cause summary**: TCP half-close race between CLI and cluster — one of the two HTTPS connections terminates server-side (CLOSE_WAIT) while the CLI's `read()` on the other connection blocks forever waiting for an exit-status message.

---

## Summary

`eai job exec` against a long-lived `sleep infinity` job hangs indefinitely on
roughly **6% of calls**, with hangs arriving in **bursts of 3–5 consecutive
failures**. A fresh `eai` subprocess (SIGKILL the hung one, re-run the exact
same command) recovers after a few seconds to minutes. The target container
is alive and idle during the hang — the wedge is in the CLI's wire-up or
response-delivery path, not in the container's command execution.

This makes `eai job exec` unsuitable as the primary RPC primitive for
integration-test or interactive-agent workloads that make tens of sequential
exec calls — the compound P(no hang in N calls) = `(1 − 0.06)^N` is ~30% for
N=20.

## Reproduction

Python script (minimal):

```python
import os, subprocess, time, pathlib
JOB = pathlib.Path("/tmp/probe.txt").read_text().strip()
EAI = os.path.expanduser("~/bin/eai")

# Pre-req: an EAI job in RUNNING state, e.g.
#   eai --profile yul101 job new --non-preemptable --format json --no-header \
#     -i python:3.13 --cpu 2 --mem 4 -- sleep infinity

hangs = 0
for i in range(100):
    start = time.monotonic()
    try:
        r = subprocess.run(
            [EAI, "--profile", "yul101", "job", "exec", JOB, "--",
             "bash", "-c", f"echo iter_{i}"],
            capture_output=True, timeout=10, check=False,
        )
        d = time.monotonic() - start
        if d > 8:  # never finishes under normal conditions
            print(f"iter {i}: suspiciously slow ({d:.1f}s)")
    except subprocess.TimeoutExpired:
        hangs += 1
        print(f"iter {i}: HANG")

print(f"Hangs: {hangs}/100")
```

Observed across three runs on yul101:

| Run | eai version | Hangs / 100 | Hang iterations (burst structure) |
|---|---|---|---|
| 1 | 1.13.1 | 6 | 23, 24, 27, 80, 85, 92 |
| 2 | 2.0.13 | 9 | 4, 22, 24, 26, 28, 33, 49, 82, 90 |
| 3 | 1.13.1 (sampler run) | 5/50 (10%) | 3, 4, 17, 19, 32 |

Average per-call duration on v2.0.13 is **~2.5× slower** than v1.13.1
(min 2.75 s vs 0.71 s; avg 3.52 s vs 1.38 s) — the newer CLI made things
meaningfully worse on steady-state performance as well.

## Observations

### What works
- Non-zero `eai` exit codes pass through reliably (auth errors, 404s, bad args — none of these hang).
- **Fresh retry after kill**: on `subprocess.TimeoutExpired`, kill the process group (`killpg(..., SIGKILL)`) and retry with a new subprocess — this almost always succeeds within the next 5–15 s (see flakiness probe clustering).
- Payload size as a shell arg up to 10 KB works in ~1 s; 15 KB+ fails fast with `rc=1` (separate issue — CLI arg length limit).

### What's broken

1. **Hangs are clustered, not random.** Observed bursts of 3–5 consecutive
   hangs separated by 50–100+ clean calls. Suggests a transient cluster-side
   or CLI-side state, not a pure per-call dice roll.

2. **Hangs on response delivery for long commands.** A separate failure mode:
   a command that takes ~30 minutes (e.g. `pytest` in a SWE-bench testbed)
   completes in the container — verified by `eai job exec <same job> -- ps -ef`
   showing only `sleep infinity` — but the CLI call that launched the pytest
   never returns. Retry re-runs the whole 30-minute command pointlessly.

3. **`stdin` is not supported.** Piping data into `eai job exec` hangs
   indefinitely at every payload size tested (100 B through 10 KB). This
   forces workloads to encode payloads as shell-args (10 KB ceiling) or write
   to a file in chunks.

4. **Zombie / defunct eai children after timeout.** Without
   `start_new_session=True` + `killpg`, killed `eai` processes leave `(eai)`
   defuncts in `ps`, suggesting internal child processes that weren't
   reaped.

### Hang characterization

While hung:
- Target container is alive (a fresh `eai job exec` for a trivial `echo`
  responds normally from a different terminal — usually).
- No container-side process is running (just `sleep infinity`) — evidence the
  command either completed and the response channel wedged, or never made it
  to the container in the first place.
- The hung `eai` process is in a stuck state that SIGTERM sometimes doesn't
  clear (need SIGKILL on the process group).

### Stack trace (5/5 samples identical)

Captured with `sample <pid> 4` on macOS while `eai job exec` was hung.
Every single sample shows **one thread** stuck in a blocking `read()`:

```
2958 Thread_103785772
+ 2958 runtime.read_trampoline.abi0  (in eai) + 28  [0x102c23c0c]
+   2958 read  (in libsystem_kernel.dylib) + 8  [0x1861b9908]
```

All other Go-runtime threads are parked in `pthread_cond_wait` / `kevent`
(normal idle state). So the hang is exactly one goroutine blocked on a
socket read.

### Socket state during hang (`lsof -p <hung-eai-pid>`)

```
eai  78646  ...  7u  IPv4  ...  TCP 192.168.86.250:51218-><cluster>:https (ESTABLISHED)
eai  78646  ...  8u  IPv4  ...  TCP 192.168.86.250:51224-><cluster>:https (CLOSE_WAIT)
```

**`CLOSE_WAIT` on fd 8** means the cluster sent a FIN on that connection
but the CLI has not called `close()` on its end. Combined with the stack
trace — eai is reading from fd 7 (the other HTTPS connection) and never
returning, while fd 8 is half-closed and orphaned.

Interpretation: the CLI uses two parallel HTTPS connections per exec (plausibly
one for command/control, one for streaming stdout). The server finishes the
command and closes the data channel (fd 8, now CLOSE_WAIT), but the CLI is
still reading on the control channel (fd 7) waiting for a completion message
that is either never sent or was lost.

This is a concrete, reproducible protocol bug — not flakiness in the
networking substrate. Forensic evidence files attached.

## Additional characterization

1. **Hangs can persist through a 120 s fresh-retry window.** In one run (H1,
   30 iterations), one of the three observed hangs did not clear when the
   call was retried with a 120 s deadline. So "transient blip" can last
   minutes. A tight retry loop isn't guaranteed to recover.
2. **Inter-call spacing appears to reduce hang rate.** With 5 s between calls
   (H3, 15 iterations), 0 hangs observed vs a ~6–9 % baseline. Small sample
   — could be noise — but suggests back-to-back exec calls worsen the
   issue.
3. **Output suppression (``>/dev/null``) showed a lower hang rate** (1/30 vs
   baseline 6–9%) but small-sample variance makes this non-definitive.
4. **CRITICAL: container-side execution is unaffected by CLI hangs.**
   H4 test: we kicked off a bg command via ``(sleep 15 && touch /tmp/h4_probe) & disown``
   wrapped in ``eai job exec``. The CLI *hung* (10 s timeout, rc=-1) —
   but the container-side command *ran to completion*: ``/tmp/h4_probe``
   existed after 20 s, confirmed via a subsequent (successful) exec.
   Implication: even when the response-delivery hangs, the command already
   reached the container. This is the foundation for our stable workaround
   (background + poll, see companion doc).

## Impact

CUBE's Toolkit-based integration tests (terminalbench, swebench-verified,
swebench-live cubes) make ~15–25 sequential `eai job exec` calls per task.
With 6% per-call hang rate, P(any hang in 20 calls) = 71% — so without
mitigation, Toolkit as a backend has a ~30% task-level success rate.

With a process-group kill + 2 retries + 5s/10s backoff wrapper (see our
`_run_eai` in cube-standard), short-command success rises to >99% per call.
The retry does **not** help for long-running commands (see observation #2).

## Workarounds we're using

1. `_run_eai` wrapper with `start_new_session=True`, `subprocess.Popen`,
   `killpg(pid, SIGKILL)` on `TimeoutExpired`, and 2 retries with 5s/10s
   exponential backoff.
2. Encoding payloads as shell-args (kept < 10 KB), never stdin.
3. Upcoming: background-spawn + poll for long-running commands (split a
   single 30-min `pytest` call into ~20 short 1-s polls of a
   `/tmp/pytest.done` sentinel).

## Questions for the Toolkit team

1. Are there known issues in the exec-session protocol under sustained load?
   Any rate-limit / connection-pool exhaustion we should be aware of?
2. Is the response-delivery hang on long commands a known issue?
3. Is `stdin` passthrough on the roadmap?
4. Is there a recommended primitive for "interactive agent making many
   sequential exec calls"? Port-forward + SSH is the only alternative we've
   seen in the docs — if so, is the expectation that users install sshd in
   their images?

## Attachments

- `/tmp/eai_flakiness.py` — 100-call probe script (flakiness quantification)
- `/tmp/eai_flakiness.log` — run 1 (v1.13.1) raw output
- `/tmp/eai_flakiness_v2.log` — run 2 (v2.0.13) raw output
- `/tmp/eai_exec_probe.py` — payload-shape sweep (shell-arg sizes, stdin, chunked, cube-wrap)
- `/tmp/eai_probe_results.json` — structured results
- `/tmp/eai_sampled_probe.py` — hang-triggered stack-trace sampler
- `/tmp/sample_eai_*.txt` — 5× macOS `sample` outputs of hung eai processes
- `/tmp/eai_lsof_probe.py` — hang-triggered `lsof` capturer
- `/tmp/lsof_eai_*.txt` — `lsof` output from a hung eai process (shows CLOSE_WAIT)
