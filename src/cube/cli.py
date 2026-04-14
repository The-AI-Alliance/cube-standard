"""cube CLI — entry point for the `cube` command.

Global options (before or after the subcommand): ``--no-color`` disables ANSI
colors (also respects the ``NO_COLOR`` environment variable per no-color.org).

Usage:
    cube list           List all cube benchmarks installed in the current
                        environment (registered under the cube.benchmarks
                        entry-point group).  Shows name, version, task count,
                        tags, and description for each benchmark.
    cube init [NAME]    Copy the new_cube_package template into <cwd>/<NAME>.
                        NAME defaults to "new_cube_package". The template lives
                        at src/cube/_template/new_cube_package/ inside the
                        cube-standard package, so it is always in sync with the
                        rest of the codebase and can be edited directly as normal
                        Python files.
    cube test NAME      Run the debug suite and check compliance (every debug task
                        must reach reward == 1.0).  NAME is either a benchmark
                        entry-point name (e.g. counter-cube) or a dotted module
                        path (e.g. counter_cube.debug).  When an entry-point name
                        is given the debug module is auto-derived from the
                        registered benchmark module.
"""

import base64
import importlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
import tomllib
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from cube import __version__

# ── Console setup ─────────────────────────────────────────────────────────────

# "dim" uses the terminal default foreground + dim — "dim white" was unreadable on light themes (e.g. Solarized Light).
_THEME = Theme(
    {
        "info": "cyan",
        "success": "bold green",
        "warning": "bold yellow",
        "error": "bold red",
        "dim": "dim",
        "file": "green",
        "cmd": "bold cyan",
        "brand": "bold blue",
    }
)

_NO_COLOR = False


def _env_requests_no_color() -> bool:
    """https://no-color.org/ — any non-empty NO_COLOR disables ANSI styling."""
    v = os.environ.get("NO_COLOR")
    return v is not None and v != ""


def _make_console(**kwargs: Any) -> Console:
    kw: dict[str, Any] = {"theme": _THEME, **kwargs}
    if _NO_COLOR:
        kw["no_color"] = True
    return Console(**kw)


console = _make_console()
err_console = _make_console(stderr=True)


def _strip_no_color_flags(argv: list[str]) -> tuple[list[str], bool]:
    """Remove --no-color from argv; returns (rest, True if flag was present)."""
    rest: list[str] = []
    seen = False
    for a in argv:
        if a == "--no-color":
            seen = True
        else:
            rest.append(a)
    return rest, seen


# ── Constants ──────────────────────────────────────────────────────────────────

_TEMPLATE_DIR = Path(__file__).parent / "_template" / "new_cube_package"
_DEFAULT_NAME = "my-benchmark"


# ── Commands ───────────────────────────────────────────────────────────────────


_PLACEHOLDER_NAMES = {"cube_package", "new_cube_package", "new-cube-package"}


def _stress_fill_bar(fraction: float, width: int) -> str:
    """Solid block + light shade (█░) for latency / profiling / efficiency bars."""
    if width <= 0:
        return ""
    frac = min(max(fraction, 0.0), 1.0)
    filled = int(frac * width)
    return "█" * filled + "░" * (width - filled)


# Aggregate episode_time_s below this is treated as timer noise — avoid tasks/min in the billions.
_THROUGHPUT_MIN_TOTAL_S = 0.01
_RESET_DIFF_DISPLAY_MAX = 24_000


def _print_reset_reproducibility_error_block(
    out: Console,
    *,
    reset_ok: bool,
    reset_msg: str,
    reset_diff: str,
    panel_width: int | None,
) -> None:
    """Print reset-repro mismatch before the main stress-test panel (pytest-style ordering)."""
    if reset_ok:
        return
    intro = f"(first task, two fresh Task instances): {reset_msg}".strip()
    parts: list = [
        Text.from_markup(f"{intro} [dim]A mismatch is not always a bug (e.g. time-dependent observations).[/dim]")
    ]
    if reset_diff.strip():
        d = reset_diff.rstrip("\n")
        if len(d) > _RESET_DIFF_DISPLAY_MAX:
            d = d[:_RESET_DIFF_DISPLAY_MAX] + "\n... [diff truncated]\n"
        parts.append(Syntax(d, lexer="text", word_wrap=True, line_numbers=False))
    panel_kw: dict[str, Any] = {
        "title": "[bold]Reset reproducibility error[/bold]",
        "border_style": "yellow",
        "padding": (0, 1),
    }
    if panel_width is not None:
        panel_kw["width"] = panel_width
    out.print(Panel(Group(*parts), **panel_kw))
    out.print()


def _format_small_seconds(sec: float) -> str:
    # Show ms when below 10ms so fast runs do not look like “all zeros”.
    if sec <= 0.0:
        return "0ms"
    if sec < 0.01:
        return f"{sec * 1000:.2f}ms"
    return f"{sec:.3f}s"


def _episode_wall_display_s(report: dict) -> str:
    # Prefer summed step times when rounded episode_time_s is 0 but work clearly ran.
    ep = float(report.get("episode_time_s") or 0.0)
    step_sum = sum(float(x) for x in (report.get("step_times_s") or []))
    sec = max(ep, step_sum)
    if sec == 0.0 and (report.get("steps") or 0) > 0:
        rt = float(report.get("reset_time_s") or 0.0)
        sec = max(sec, rt)
    return _format_small_seconds(sec)


def cmd_init(name: str, cwd: Path) -> None:
    """Copy the template into *cwd*/<name>, refusing to overwrite anything."""
    if name in _PLACEHOLDER_NAMES:
        err_console.print(
            Panel(
                f"[error]'{name}'[/error] conflicts with internal template placeholders.\n"
                "Choose a different name (e.g. [cmd]cube init my-bench[/cmd]).",
                title="[error]Error[/error]",
                border_style="red",
                padding=(0, 1),
            )
        )
        sys.exit(1)

    dest = cwd / name

    if dest.exists():
        err_console.print(
            Panel(
                f"[error]'{dest}'[/error] already exists.\nChoose a different name or remove it first.",
                title="[error]Error[/error]",
                border_style="red",
                padding=(0, 1),
            )
        )
        sys.exit(1)

    module_name = name.replace("-", "_")

    with console.status(f"[info]Scaffolding[/info] [file]{name}[/file]…", spinner="dots"):
        shutil.copytree(_TEMPLATE_DIR, dest)

        # Rename the Python package directory: src/cube_package/ → src/<module_name>/
        old_pkg_dir = dest / "src" / "cube_package"
        new_pkg_dir = dest / "src" / module_name
        if old_pkg_dir.exists():
            old_pkg_dir.rename(new_pkg_dir)

        # Substitute placeholders in all text files (most-specific patterns first)
        replacements = [
            ("new-cube-package", name),
            ("new_cube_package", module_name),
            ("cube_package", module_name),
        ]
        for f in dest.rglob("*"):
            if not f.is_file():
                continue
            try:
                text = f.read_text(encoding="utf-8")
            except (UnicodeDecodeError, PermissionError):
                continue
            new_text = text
            for old, new in replacements:
                new_text = new_text.replace(old, new)
            if new_text != text:
                f.write_text(new_text, encoding="utf-8")

    # ── Created-files table ────────────────────────────────────────────────────
    files = sorted(p for p in dest.rglob("*") if p.is_file())

    table = Table(
        show_header=False,
        box=box.SIMPLE,
        padding=(0, 1),
        show_edge=False,
    )
    table.add_column("file", style="file", no_wrap=True)

    for f in files:
        table.add_row(str(f.relative_to(dest)))

    console.print(
        Panel(
            table,
            title=f"[success]Created[/success] [file]{dest}/[/file]",
            border_style="green",
            padding=(0, 1),
        )
    )

    # ── Next-steps panel ───────────────────────────────────────────────────────
    steps = Text()
    steps.append("  cd ", style="dim")
    steps.append(str(dest), style="cmd")
    steps.append("\n  ", style="dim")
    steps.append("uv sync", style="cmd")
    steps.append("\n  ", style="dim")
    steps.append("# Edit ", style="dim")
    steps.append("tool.py", style="file")
    steps.append(", ", style="dim")
    steps.append("task.py", style="file")
    steps.append(", ", style="dim")
    steps.append("benchmark.py", style="file")
    steps.append("\n  ", style="dim")
    steps.append("pytest tests/", style="cmd")

    console.print(
        Panel(
            steps,
            title="[info]Next steps[/info]",
            border_style="cyan",
            padding=(0, 1),
        )
    )


def cmd_list() -> None:
    """Print all installed cube benchmarks registered under the cube.benchmarks entry point."""
    eps = importlib.metadata.entry_points(group="cube.benchmarks")

    if not eps:
        console.print(
            Panel(
                "No cube benchmarks found in the current environment.\n"
                "Install a cube package (e.g. [cmd]uv sync[/cmd]) and make sure its\n"
                '[file]pyproject.toml[/file] declares a [cmd][project.entry-points."cube.benchmarks"][/cmd] section.',
                title="[brand]cube list[/brand]",
                border_style="yellow",
                padding=(0, 1),
            )
        )
        return

    table = Table(
        show_header=True,
        box=box.SIMPLE,
        padding=(0, 1),
        show_edge=False,
        header_style="bold",
    )
    table.add_column("name", style="file", no_wrap=True)
    table.add_column("version", style="dim", no_wrap=True)
    table.add_column("tasks", justify="right", no_wrap=True)
    table.add_column("tags", style="dim")
    # Default foreground so description stays readable on light backgrounds (plain "white" did not).
    table.add_column("description")

    for ep in sorted(eps, key=lambda e: e.name):
        version = num_tasks = tags = description = ""
        try:
            benchmark_cls = ep.load()
            meta = benchmark_cls.benchmark_metadata
            version = meta.version
            num_tasks = str(meta.num_tasks) if meta.num_tasks else ""
            tags = ", ".join(meta.tags) if meta.tags else ""
            description = meta.description
        except Exception:
            description = "[error](failed to load)[/error]"

        table.add_row(ep.name, version, num_tasks, tags, description)

    console.print(
        Panel(
            table,
            title=f"[brand]cube list[/brand]  [dim]{len(eps)} benchmark(s) installed[/dim]",
            border_style="blue",
            padding=(0, 1),
        )
    )


def _resolve_debug_module(name: str) -> str:
    """Resolve *name* to a fully-qualified debug module path.

    If *name* contains a dot it is assumed to already be a module path and is
    returned unchanged.  Otherwise it is looked up as a ``cube.benchmarks``
    entry-point name and the debug module is derived by replacing the last
    component of the benchmark's module path with ``debug``.

    Example::

        "counter-cube"          → looks up entry point → "counter_cube.debug"
        "counter_cube.debug"    → returned as-is
    """
    if "." in name:
        return name

    eps = importlib.metadata.entry_points(group="cube.benchmarks")
    matched = {ep.name: ep for ep in eps}

    if name not in matched:
        available = ", ".join(sorted(matched)) or "(none installed)"
        err_console.print(
            Panel(
                f"[error]No cube benchmark registered as[/error] [file]{name}[/file].\n"
                f"Available: {available}\n"
                "Or pass the full module path, e.g. [cmd]cube test counter_cube.debug[/cmd].",
                title="[error]Unknown benchmark[/error]",
                border_style="red",
                padding=(0, 1),
            )
        )
        sys.exit(1)

    ep = matched[name]
    # ep.value is "some.module:ClassName" — derive debug module from the package
    benchmark_module = ep.value.split(":")[0]  # e.g. "counter_cube.benchmark"
    package_root = benchmark_module.rsplit(".", 1)[0]  # e.g. "counter_cube"
    return f"{package_root}.debug"


def cmd_test(
    module_name: str,
    *,
    max_steps: int = 20,
    output_path: str | None = None,
    ci_mode: bool = False,
) -> None:
    """Import *module_name* (or resolve an entry-point name) and run the debug compliance suite.

    When *ci_mode* is True (or ``CUBE_CI=1`` is set), the Rich terminal dashboard is suppressed
    and only plain-text compliance results are printed — suitable for GitHub Actions logs.
    """
    ci_mode = ci_mode or bool(os.environ.get("CUBE_CI"))
    from cube.testing import (
        aggregate_profiling,
        build_stress_test_report,
        check_benchmark_metadata,
        check_reset_reproducibility,
        run_debug_suite,
    )

    resolved = _resolve_debug_module(module_name)
    if resolved != module_name:
        console.print(
            Panel(
                f"[info]Resolved[/info] [file]{module_name}[/file] → [file]{resolved}[/file]",
                border_style="cyan",
                padding=(0, 1),
            )
        )
    else:
        console.print(
            Panel(
                f"[info]Importing[/info] [file]{resolved}[/file]…",
                border_style="cyan",
                padding=(0, 1),
            )
        )

    try:
        module = importlib.import_module(
            resolved
        )  # nosemgrep: non-literal-import  # trusted: module path from CLI user who already has local shell access
    except ModuleNotFoundError as exc:
        err_console.print(
            Panel(
                f"[error]Cannot import[/error] [file]{resolved}[/file]: {exc}\n"
                "Make sure the package is installed (e.g. [cmd]uv sync[/cmd]) and "
                "that the module exposes [cmd]get_debug_benchmark()[/cmd] and "
                "[cmd]make_debug_agent()[/cmd].",
                title="[error]Import Error[/error]",
                border_style="red",
                padding=(0, 1),
            )
        )
        sys.exit(1)

    for required in ("get_debug_benchmark", "make_debug_agent"):
        if not callable(getattr(module, required, None)):
            err_console.print(
                Panel(
                    f"[error]Module[/error] [file]{resolved}[/file] does not expose "
                    f"[cmd]{required}()[/cmd].\n"
                    "See the [file]cube_package/debug.py[/file] template for the required interface.",
                    title="[error]Compliance Error[/error]",
                    border_style="red",
                    padding=(0, 1),
                )
            )
            sys.exit(1)

    with console.status(
        f"[info]Running debug suite for[/info] [file]{resolved}[/file]…",
        spinner="dots",
    ):
        results = run_debug_suite(resolved, module, max_steps=max_steps, print_json=False)

    if not results:
        err_console.print(
            Panel(
                "No debug tasks were run.\n"
                "Make sure [cmd]get_debug_benchmark()[/cmd] returns a benchmark whose "
                "[cmd]get_task_configs()[/cmd] yields at least one config.",
                title="[warning]No tasks found[/warning]",
                border_style="yellow",
                padding=(0, 1),
            )
        )
        sys.exit(1)

    failures: list[dict] = []
    for r in results:
        passed = not r["error"] and r["done"] and r["reward"] == 1.0
        if not passed:
            failures.append(r)

    # ── Extra compliance checks (stress_test_specs.md) ─────────────────────────────
    reset_ok, reset_msg, reset_diff = check_reset_reproducibility(module)
    meta_ok, _ = check_benchmark_metadata(module)
    close_idempotent_ok = all(r.get("close_idempotent_ok", False) for r in results)
    tools_list_ok = all(r.get("tools_list_ok", False) for r in results)
    compliance_passed = []
    compliance_failed = []
    if results:
        compliance_passed.append("test_debug_tasks_exist")
        compliance_passed.append("test_debug_agent_exists")
    if not failures:
        compliance_passed.append("test_full_episode")
    else:
        compliance_failed.append("test_full_episode")
    if reset_ok:
        compliance_passed.append("test_reset_reproducibility")
    else:
        compliance_failed.append("test_reset_reproducibility")
    if tools_list_ok:
        compliance_passed.append("test_tools_list")
    else:
        compliance_failed.append("test_tools_list")
    if close_idempotent_ok:
        compliance_passed.append("test_close_idempotent")
    else:
        compliance_failed.append("test_close_idempotent")
    if meta_ok:
        compliance_passed.append("test_benchmark_metadata")
    else:
        compliance_failed.append("test_benchmark_metadata")

    # ── Latency: p50, p95, p99 from step_times_s across all episodes ────────────
    all_step_times: list[float] = []
    for r in results:
        all_step_times.extend(r.get("step_times_s") or [])
    if all_step_times:
        sorted_times = sorted(all_step_times)
        n = len(sorted_times)
        p50_s = sorted_times[int(0.50 * (n - 1))] if n else 0.0
        p95_s = sorted_times[int(0.95 * (n - 1))] if n else 0.0
        p99_s = sorted_times[int(0.99 * (n - 1))] if n else 0.0
    else:
        p50_s = p95_s = p99_s = 0.0

    try:
        _term_width = console.size.width
    except Exception:
        _term_width = 80
    _display_width = min(_term_width, 96)
    _bar_width = min(32, max(14, _display_width - 48))
    _eff_bar_w = min(14, max(8, _display_width // 7))

    profiling_agg = aggregate_profiling(results)

    # ── CI mode: plain-text output, no terminal dashboard ────────────────────
    if ci_mode:
        print(f"cube test  benchmark={resolved}  tasks={len(results)}  failures={len(failures)}")
        for name in compliance_passed:
            print(f"  PASS  {name}")
        for name in compliance_failed:
            print(f"  FAIL  {name}")
        if not reset_ok:
            print("\nReset reproducibility error:")
            print(
                f"  (first task, two fresh Task instances): {reset_msg} "
                "A mismatch is not always a bug (e.g. time-dependent observations)."
            )
            if reset_diff.strip():
                print(reset_diff.rstrip("\n"))
        report = build_stress_test_report(resolved, results, compliance_passed, compliance_failed)
        if output_path:
            report.save(output_path)
            print(f"  JSON  {output_path}")
        if failures:
            print(f"\nFAILED — {len(failures)}/{len(results)} tasks did not reach reward 1.0")
            sys.exit(1)
        print("\nPASSED")
        return

    # ── Stress-test dashboard (header → compliance → latency → throughput → profiling → tasks)
    status_str = "Passed" if not failures else "Failed"
    progress_done = len(results) - len(failures)
    progress_total = len(results)
    workers_avail = max(1, os.cpu_count() or 1)
    workers_active = 1

    bench_label = module_name if len(module_name) <= 40 else resolved
    if len(bench_label) > 40:
        bench_label = bench_label[:37] + "…"

    header_grid = Table(show_header=False, box=None, expand=True, padding=(0, 1))
    header_grid.add_column("left", ratio=1)
    header_grid.add_column("right", justify="right", ratio=1)
    header_grid.add_row(
        Text.from_markup(
            f"Benchmark: [file]{bench_label}[/file]\n"
            # Debug suite runs one task at a time; second number is host CPU count, not pool size.
            f"Suite: [dim]sequential[/dim] · workers [bold]{workers_active}[/bold] · "
            f"host CPUs [dim]{workers_avail}[/dim]"
        ),
        Text.from_markup(
            f"Status: [{'success' if not failures else 'error'}]{status_str}[/]\n"
            f"Progress: [bold]{progress_done}[/bold]/[dim]{progress_total}[/dim]"
        ),
    )

    full_episode_status = "[success]✓[/success]" if not failures else "[error]✗[/error]"
    ok_tasks = "[success]✓[/success]" if results else "[dim]—[/dim]"
    ok_agent = "[success]✓[/success]" if results else "[dim]—[/dim]"
    ok_tools = "[success]✓[/success]" if tools_list_ok else "[error]✗[/error]"
    ok_meta = "[success]✓[/success]" if meta_ok else "[error]✗[/error]"
    ok_reset = "[success]✓[/success]" if reset_ok else "[error]✗[/error]"
    ok_close = "[success]✓[/success]" if close_idempotent_ok else "[error]✗[/error]"

    compliance_header = Text.from_markup("[bold]COMPLIANCE[/bold]")
    compliance_checks_table = Table(
        show_header=False,
        box=box.SIMPLE,
        padding=(0, 1),
        show_edge=False,
    )
    compliance_checks_table.add_column("check", style="dim", no_wrap=True)
    compliance_checks_table.add_column("st", no_wrap=True, width=3)
    compliance_checks_table.add_column("check2", style="dim", no_wrap=True)
    compliance_checks_table.add_column("st2", no_wrap=True, width=3)
    left_col = [
        ("debug_tasks_exist", ok_tasks),
        ("debug_agent_exists", ok_agent),
        ("tools_list", ok_tools),
        ("benchmark_metadata", ok_meta),
    ]
    right_col = [
        ("full_episode", full_episode_status),
        ("reset_reproducibility", ok_reset),
        ("close_idempotent", ok_close),
    ]
    for i in range(len(left_col)):
        ln, ls = left_col[i]
        if i < len(right_col):
            rn, rs = right_col[i]
            compliance_checks_table.add_row(ln, ls, rn, rs)
        else:
            compliance_checks_table.add_row(ln, ls, "", "")

    n_comp_pass = len(compliance_passed)
    n_comp_fail = len(compliance_failed)
    if n_comp_fail:
        compliance_banner = Text.from_markup(
            f"[bold]COMPLIANCE[/bold]  [dim]{n_comp_pass} passed[/dim]  [error]· {n_comp_fail} failed[/error]"
        )
    else:
        compliance_banner = Text.from_markup(f"[bold]COMPLIANCE[/bold]  [dim]{n_comp_pass} passed[/dim]")

    max_lat = max(p50_s, p95_s, p99_s, 0.001)
    latency_section = Table(show_header=False, box=None, padding=(0, 0), show_edge=False)
    latency_section.add_column(no_wrap=True)
    latency_section.add_row(Text.from_markup("[bold]LATENCY (step wall time)[/bold]"))
    for label, sec in (("p50", p50_s), ("p95", p95_s), ("p99", p99_s)):
        bar = _stress_fill_bar(sec / max_lat, _bar_width)
        latency_section.add_row(
            Text.from_markup(f"  {label:3} │{bar}│ [dim]{_format_small_seconds(sec)}[/dim]"),
        )
    reset_times = [r.get("reset_time_s") for r in results if r.get("reset_time_s") is not None]
    mean_setup_s = sum(reset_times) / len(reset_times) if reset_times else None
    if mean_setup_s is not None:
        latency_section.add_row(
            Text.from_markup(f"[dim]Task setup (mean): {_format_small_seconds(mean_setup_s)}[/dim]"),
        )

    total_ep_s = sum(float(r.get("episode_time_s") or 0.0) for r in results)
    n_tasks = len(results)
    throughput_specs = (
        (1, 1.0),
        (2, 0.93),
        (4, 0.84),
    )
    throughput_blocks: list = []
    throughput_blocks.append(Text.from_markup("[bold]THROUGHPUT (tasks/min)[/bold]"))
    if total_ep_s >= _THROUGHPUT_MIN_TOTAL_S:
        rate_1 = n_tasks / (total_ep_s / 60.0)
        throughput_table = Table(
            show_header=True,
            box=box.SIMPLE,
            padding=(0, 1),
            show_edge=False,
            header_style="bold",
        )
        throughput_table.add_column("Workers", justify="right", style="dim")
        # Only the 1-worker row is measured; 2/4 are illustrative scaling (not separate benchmark runs).
        throughput_table.add_column("Measured", justify="right")
        throughput_table.add_column("Illustrative", justify="right", style="dim")
        throughput_table.add_column("Linear", justify="right", style="dim")
        throughput_table.add_column("Efficiency", justify="right")

        for w, eff in throughput_specs:
            linear = rate_1 * w
            projected = linear * eff
            measured_str = f"{rate_1:.1f}" if w == 1 else "—"
            illustrative_str = f"{projected:.1f}" if w > 1 else "—"
            eff_pct = int(round(100 * eff))
            bar = _stress_fill_bar(eff, _eff_bar_w)
            throughput_table.add_row(
                str(w),
                measured_str,
                illustrative_str,
                f"{linear:.1f}",
                Text.from_markup(f"{eff_pct}% {bar}"),
            )
        throughput_blocks.append(throughput_table)
        throughput_blocks.append(
            Text.from_markup(
                "[dim]Measured = sequential 1-worker tasks/min. Illustrative = linear × efficiency factor "
                "(fixed; not measured; real multi-worker needs a parallel harness).[/dim]"
            )
        )
    else:
        # Short lines so narrow terminals do not truncate mid-sentence.
        throughput_blocks.append(
            Text.from_markup(
                "[dim]Skipped: summed episode wall time is under "
                f"{_THROUGHPUT_MIN_TOTAL_S:.2f}s.\n"
                "Tasks/min needs a longer measurable run.[/dim]"
            )
        )

    profiling_section = Table(show_header=False, box=None, padding=(0, 0), show_edge=False)
    profiling_section.add_column(no_wrap=True)
    profiling_section.add_row(Text.from_markup("[bold]PROFILING BREAKDOWN[/bold]"))
    if profiling_agg:
        # Keys that are derived/non-additive: exclude from the step-time total
        independent_keys = {k for k in profiling_agg if k.endswith("/n_actions") or k.endswith("/avg_per_action")}
        additive_keys = {k for k in profiling_agg if k not in independent_keys}
        total_prof = sum(profiling_agg[k] for k in additive_keys) or 1e-9
        for op_name, val in sorted(profiling_agg.items(), key=lambda x: (x[0] not in independent_keys, -x[1])):
            if op_name in independent_keys:
                fmt = f"{val:.0f} action(s) per step" if op_name.endswith("/n_actions") else f"{val:.4f}s per action"
                profiling_section.add_row(
                    Text.from_markup(f"  {op_name:16}\t{fmt}"),
                )
            else:
                frac = val / total_prof
                bar = _stress_fill_bar(frac, _bar_width)
                pct = 100.0 * frac
                profiling_section.add_row(
                    Text.from_markup(f"  {op_name:16} │{bar}│ [dim]{val:.4f}s ({pct:.0f}%)[/dim]"),
                )
    else:
        profiling_section.add_row(
            Text.from_markup(
                "  [dim]No timings: steps must attach [file]info['profiling'][/file].\n"
                "  [dim](Optional dict: op name → (t0, t1) perf counters.)[/dim]"
            )
        )

    task_results_table = Table(
        show_header=True,
        box=box.SIMPLE,
        padding=(0, 1),
        show_edge=False,
        header_style="bold",
    )
    task_results_table.add_column("task", style="file", no_wrap=True)
    task_results_table.add_column("done", justify="center")
    task_results_table.add_column("rwd", justify="right")
    task_results_table.add_column("st", justify="right")
    task_results_table.add_column("wall", justify="right")
    task_results_table.add_column("err", style="error")

    for r in results:
        done_str = "[success]✓[/success]" if r["done"] else "[error]✗[/error]"
        reward_str = (
            f"[success]{r['reward']:.1f}[/success]" if r["reward"] == 1.0 else f"[error]{r['reward']:.1f}[/error]"
        )
        task_results_table.add_row(
            r["task_id"],
            done_str,
            reward_str,
            str(r["steps"]),
            _episode_wall_display_s(r),
            r["error"] or "",
        )

    # Build and optionally save stress-test report
    report = build_stress_test_report(resolved, results, compliance_passed, compliance_failed)
    if output_path:
        report.save(output_path)
        console.print(f"[dim]Baseline saved to [file]{output_path}[/file][/dim]")

    if failures:
        border = "red"
    elif compliance_failed:
        border = "yellow"
    else:
        border = "green"
    comp_sub = ""
    if compliance_failed and not failures:
        comp_sub = f"  [warning]· {len(compliance_failed)} compliance failed[/warning]"
    status_sub = (
        f"[success]{n_tasks} task(s) passed[/success]{comp_sub}"
        if not failures
        else f"[error]{len(failures)} / {n_tasks} failed[/error]"
    )

    body_parts: list = [
        header_grid,
        Rule(style="dim"),
        compliance_header,
        compliance_banner,
        compliance_checks_table,
        Rule(style="dim"),
        latency_section,
        Rule(style="dim"),
        *throughput_blocks,
        Rule(style="dim"),
        profiling_section,
    ]
    body_parts.extend([Rule(style="dim"), task_results_table])

    stress_panel = Panel(
        Group(*body_parts),
        title="[bold]CUBE Stress Test[/bold]",
        subtitle=status_sub,
        border_style=border,
        box=box.HEAVY,
        padding=(0, 1),
    )
    dash_console = _make_console(width=_display_width)
    _print_reset_reproducibility_error_block(
        dash_console,
        reset_ok=reset_ok,
        reset_msg=reset_msg,
        reset_diff=reset_diff,
        panel_width=_display_width,
    )
    dash_console.print(Rule("[bold]CUBE Stress Test[/bold]", style="dim"))
    dash_console.print(stress_panel)

    if failures:
        console.print(
            Panel(
                "\n".join(
                    f"  [file]{r['task_id']}[/file]  "
                    + (
                        f"error: [error]{r['error']}[/error]"
                        if r["error"]
                        else f"reward={r['reward']:.3f}, done={r['done']}"
                    )
                    for r in failures
                ),
                title="[error]Failures[/error]",
                border_style="red",
                padding=(0, 1),
            )
        )
        sys.exit(1)


# ── Help / entrypoint ──────────────────────────────────────────────────────────


# ── cube registry add ────────────────────────────────────────────────────────

_TODO = "<TODO: {}>"
_REGISTRY_DEFAULT = "The-AI-Alliance/cube-registry"


def _guess_display_name(package_name: str) -> str:
    """arithmetic-cube → 'Arithmetic Cube', miniwob-cube → 'Miniwob Cube'."""
    return " ".join(p.capitalize() for p in package_name.replace("_", "-").split("-"))


def _detect_dev_install_url(path: Path) -> str | None:
    """Construct a git+ install URL from the directory's git remote and relative path."""
    r = subprocess.run(["git", "remote", "get-url", "origin"], cwd=path, capture_output=True, text=True)
    if r.returncode != 0:
        return None
    remote = r.stdout.strip()
    if remote.startswith("git@github.com:"):
        remote = "https://github.com/" + remote[len("git@github.com:") :].removesuffix(".git")
    elif remote.endswith(".git"):
        remote = remote.removesuffix(".git")
    if not remote.startswith("https://github.com/"):
        return None
    root_r = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=path, capture_output=True, text=True)
    if root_r.returncode != 0:
        return None
    git_root = Path(root_r.stdout.strip())
    try:
        subdir = path.resolve().relative_to(git_root)
        return f"git+{remote}" if str(subdir) == "." else f"git+{remote}#subdirectory={subdir}"
    except ValueError:
        return None


def _parse_pyproject_license(project: dict) -> str | None:
    """Extract SPDX license string from pyproject.toml [project.license] if possible."""
    lic = project.get("license")
    if lic is None:
        return None
    if isinstance(lic, str):
        return lic  # PEP 639: license = "MIT"
    if isinstance(lic, dict):
        return lic.get("text")  # PEP 517: license = {text = "MIT"}
    return None


def _build_registry_yaml(
    *,
    id: str,
    name: str,
    name_is_guessed: bool,
    version: str,
    description: str,
    package: str,
    dev_install_url: str | None,
    authors: list[dict],
    wrapper_license: str | None,
) -> str:
    lines: list[str] = []

    lines.append(f"id: {id}")

    name_comment = "  # auto-guessed — verify" if name_is_guessed else ""
    lines.append(f'name: "{name}"{name_comment}')

    lines.append(f'version: "{version}"')

    # description as YAML block scalar, wrapped at 78 chars
    wrapped = textwrap.fill(description.strip(), width=78, break_long_words=False, break_on_hyphens=False)
    desc_lines = ["description: >"] + ["  " + ln for ln in wrapped.splitlines()]
    lines.extend(desc_lines)

    lines.append(f"package: {package}")

    if dev_install_url:
        lines.append(f'dev_install_url: "{dev_install_url}"')
    else:
        lines.append(f'# dev_install_url: "{_TODO.format("git+https://github.com/ORG/REPO#subdirectory=PATH")}"')

    lines.append("")
    lines.append("authors:")
    for author in authors:
        github = author.get("github") or _TODO.format("github-handle")
        name_val = author.get("name") or _TODO.format("Full Name")
        lines.append(f"  - github: {github}")
        lines.append(f"    name: {name_val}")

    lines.append("")
    lines.append("legal:")
    lic = wrapper_license or _TODO.format("MIT|Apache-2.0|BSD-3-Clause|...")
    lines.append(f"  wrapper_license: {lic}")
    lines.append("  # benchmark_license:")
    lines.append(f"  #   reported: {_TODO.format('SPDX-id of the underlying benchmark')}")
    lines.append(f'  #   source_url: "{_TODO.format("https://...")}"')

    lines.append("")
    lines.append("tags:")
    lines.append(f"  - {_TODO.format('math|web|gui|desktop')}")

    lines.append("")
    lines.append("# Optional fields — uncomment and fill as needed:")
    lines.append(f'# paper: "{_TODO.format("https://arxiv.org/abs/...")}"')
    lines.append(f'# getting_started_url: "{_TODO.format("https://...")}"')
    lines.append("# parallelization_mode: task-parallel")
    lines.append("# max_concurrent_tasks: 64")
    lines.append("")

    return "\n".join(lines)


# ── gh helpers ────────────────────────────────────────────────────────────────


def _gh_available() -> bool:
    r = subprocess.run(["gh", "auth", "status", "--active"], capture_output=True)
    return r.returncode == 0


def _gh_get(endpoint: str) -> dict | list:
    r = subprocess.run(["gh", "api", endpoint], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip())
    return json.loads(r.stdout) if r.stdout.strip() else {}


def _gh_post(endpoint: str, **fields: str) -> dict:
    cmd = ["gh", "api", "--method", "POST", endpoint]
    for k, v in fields.items():
        cmd.extend(["--field", f"{k}={v}"])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip())
    return json.loads(r.stdout) if r.stdout.strip() else {}


def _gh_put_json(endpoint: str, body: dict) -> dict:
    """PUT with a JSON body via a temp file (avoids stdin/capture_output interaction)."""
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(body, f)
        tmp = f.name
    try:
        r = subprocess.run(["gh", "api", "--method", "PUT", endpoint, "--input", tmp], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr.strip())
        return json.loads(r.stdout) if r.stdout.strip() else {}
    finally:
        Path(tmp).unlink(missing_ok=True)


def _gh_submit_pr(entry_id: str, yaml_text: str, registry: str) -> str:
    """Fork registry, create branch, upload YAML, open PR. Returns PR URL."""
    repo_name = registry.split("/")[1]

    user_info = _gh_get("/user")
    user = user_info["login"]
    user_id = user_info.get("id", "")
    # Use GitHub's noreply email so DCO sign-off matches the API commit author.
    user_email = f"{user_id}+{user}@users.noreply.github.com"

    # Fork (idempotent)
    _gh_post(f"/repos/{registry}/forks")

    # Wait for fork's main branch to be ready (up to 30s)
    sha = None
    for _ in range(15):
        try:
            ref = _gh_get(f"/repos/{user}/{repo_name}/git/refs/heads/main")
            # API returns either a single object or a list
            obj = ref[0] if isinstance(ref, list) else ref
            sha = obj["object"]["sha"]
            break
        except Exception:
            time.sleep(2)
    if sha is None:
        raise RuntimeError(f"Fork {user}/{repo_name} not ready after 30s — try again.")

    branch = f"add/{entry_id}"

    # Create branch (ignore if already exists)
    try:
        _gh_post(f"/repos/{user}/{repo_name}/git/refs", ref=f"refs/heads/{branch}", sha=sha)
    except RuntimeError as e:
        if "already exists" not in str(e):
            raise

    # Upload file — if the file already exists on the branch (e.g. updating an entry),
    # GitHub requires the existing blob sha.
    content_b64 = base64.b64encode(yaml_text.encode()).decode()
    commit_msg = f"feat: add {entry_id} entry\n\nSigned-off-by: {user} <{user_email}>"
    committer = {"name": user, "email": user_email}
    file_body: dict = {
        "message": commit_msg,
        "content": content_b64,
        "branch": branch,
        "author": committer,
        "committer": committer,
    }
    try:
        existing = _gh_get(f"/repos/{user}/{repo_name}/contents/entries/{entry_id}.yaml?ref={branch}")
        if isinstance(existing, dict) and "sha" in existing:
            file_body["sha"] = existing["sha"]
    except Exception:
        pass
    _gh_put_json(f"/repos/{user}/{repo_name}/contents/entries/{entry_id}.yaml", file_body)

    # Open PR
    pr = _gh_post(
        f"/repos/{registry}/pulls",
        title=f"feat: add {entry_id} entry",
        head=f"{user}:{branch}",
        base="main",
        body=f"Adds the `{entry_id}` benchmark entry.\n\n> Generated by `cube registry add --submit`",
    )
    return pr["html_url"]


def _manual_submit_commands(entry_id: str, yaml_path: Path, registry: str) -> str:
    return (
        f"  git clone https://github.com/{registry}\n"
        f"  cd {registry.split('/')[1]}\n"
        f"  git checkout -b add/{entry_id}\n"
        f"  cp {yaml_path} entries/\n"
        f"  git add entries/{entry_id}.yaml\n"
        f"  git commit -s -m 'feat: add {entry_id} entry'\n"
        f"  git push origin add/{entry_id}\n"
        f"  gh pr create --repo {registry} --title 'feat: add {entry_id} entry'"
    )


# ── cmd_registry_add ──────────────────────────────────────────────────────────


def cmd_registry_add(path: Path, submit: bool, registry: str) -> None:
    out_path = path / "cube-registry-entry.yaml"

    if submit and out_path.exists():
        # --submit with an existing file: the user already edited the TODOs — use it as-is.
        yaml_text = out_path.read_text()
        m = re.search(r"^id:\s*(\S+)", yaml_text, re.MULTILINE)
        if not m:
            err_console.print(f"[error]Cannot parse 'id:' from[/error] [file]{out_path}[/file]")
            sys.exit(1)
        entry_id = m.group(1)
        console.print(f"[info]Using existing[/info] [file]{out_path}[/file]")
    else:
        # Generate (or regenerate) from pyproject.toml.
        pyproject_path = path / "pyproject.toml"
        if not pyproject_path.exists():
            err_console.print(f"[error]No pyproject.toml found in[/error] [file]{path}[/file]")
            sys.exit(1)

        with open(pyproject_path, "rb") as f:
            pyproject = tomllib.load(f)

        project = pyproject.get("project", {})
        package = project.get("name", "")
        version = project.get("version", "")
        description = project.get("description", "")

        if not package:
            err_console.print("[error]pyproject.toml is missing [cmd]project.name[/cmd][/error]")
            sys.exit(1)
        if not version:
            err_console.print("[error]pyproject.toml is missing [cmd]project.version[/cmd][/error]")
            sys.exit(1)

        entry_id = package
        raw_authors = project.get("authors", [])
        authors = [{"github": None, "name": a.get("name")} for a in raw_authors] or [{}]

        yaml_text = _build_registry_yaml(
            id=entry_id,
            name=_guess_display_name(package),
            name_is_guessed=True,
            version=version,
            description=description,
            package=package,
            dev_install_url=_detect_dev_install_url(path),
            authors=authors,
            wrapper_license=_parse_pyproject_license(project),
        )

        out_path.write_text(yaml_text)
        console.print(f"[success]✓[/success] Generated [file]{out_path}[/file]")

    todos = [ln.strip() for ln in yaml_text.splitlines() if "<TODO:" in ln and not ln.strip().startswith("#")]

    if todos:
        console.print("")
        console.print("[warning]Fill in the following before submitting:[/warning]")
        for t in todos:
            console.print(f"  [dim]{t}[/dim]")
        console.print("")

    if not submit:
        console.print(
            f"[dim]Edit [file]{out_path}[/file], then run:[/dim]\n"
            f"  [cmd]cube registry add --submit[/cmd]\n"
            f"[dim]or submit the YAML manually as a PR to[/dim] [cmd]{registry}[/cmd]"
        )
        return

    if todos:
        err_console.print("[error]Cannot submit: fill in all <TODO:...> fields first.[/error]")
        sys.exit(1)

    if not _gh_available():
        console.print(
            "[warning]gh CLI not found or not authenticated.[/warning]\n"
            "[dim]Install: https://cli.github.com  then run: gh auth login[/dim]\n\n"
            "[dim]Manual commands:[/dim]\n" + _manual_submit_commands(entry_id, out_path, registry)
        )
        sys.exit(1)

    console.print(f"[info]Submitting PR to[/info] [cmd]{registry}[/cmd] ...")
    try:
        pr_url = _gh_submit_pr(entry_id, yaml_text, registry)
    except Exception as e:
        err_console.print(f"[error]PR creation failed:[/error] {e}")
        sys.exit(1)

    console.print(f"[success]✓ PR opened:[/success] {pr_url}")


def _print_help() -> None:
    """Print a rich-formatted help screen."""
    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2), show_edge=False)
    table.add_column("cmd", style="cmd", no_wrap=True)
    table.add_column("desc")
    table.add_column("example", style="dim")

    table.add_row(
        "cube list",
        "List all installed cube benchmarks",
        "cube list",
    )
    table.add_row(
        "cube init [NAME]",
        f"Scaffold a new cube package (default: [file]{_DEFAULT_NAME}[/file])",
        "cube init my-env",
    )
    table.add_row(
        "cube test NAME",
        "Run the debug compliance suite — NAME is a benchmark entry-point name or a dotted module path. "
        "Options: [cmd]--ci[/cmd] (plain-text CI output, also set via CUBE_CI=1), "
        "[cmd]--output=PATH[/cmd] (save JSON report), [cmd]--max-steps=N[/cmd]",
        "cube test counter-cube --ci --output=results.json",
    )
    table.add_row(
        "cube registry add [PATH]",
        "Generate a cube-registry YAML entry for the cube package at PATH (default: cwd).  Use --submit to open a PR automatically.",
        "cube registry add --submit",
    )

    console.print(
        Panel(
            table,
            title=f"[brand]cube[/brand] [dim]v{__version__}[/dim]",
            subtitle="[dim]Common Unified Benchmark Environments[/dim]\n"
            "[dim]Set NO_COLOR=1 or use --no-color for plain output. Set CUBE_CI=1 for CI mode.[/dim]",
            border_style="blue",
            padding=(0, 1),
        )
    )


def main() -> None:
    global _NO_COLOR, console, err_console

    raw_argv = sys.argv[1:]
    args, no_color_flag = _strip_no_color_flags(raw_argv)
    _NO_COLOR = no_color_flag or _env_requests_no_color()
    console = _make_console()
    err_console = _make_console(stderr=True)

    if not args or args[0] in ("-h", "--help"):
        _print_help()
        sys.exit(0 if not args else 0)

    command = args[0]

    if command == "list":
        cmd_list()
    elif command == "init":
        name = args[1] if len(args) > 1 else _DEFAULT_NAME
        cmd_init(name=name, cwd=Path.cwd())
    elif command == "test":
        if len(args) < 2:
            err_console.print(
                Panel(
                    "[error]Missing argument:[/error] [cmd]cube test NAME[/cmd]\n"
                    "Examples: [cmd]cube test counter-cube[/cmd]  or  [cmd]cube test counter_cube.debug[/cmd]",
                    title="[error]Error[/error]",
                    border_style="red",
                    padding=(0, 1),
                )
            )
            sys.exit(1)
        max_steps = 20
        output_path = None
        ci_mode = False
        remaining = args[2:]
        for opt in remaining:
            if opt.startswith("--max-steps="):
                max_steps = int(opt.split("=", 1)[1])
            elif opt.startswith("--output="):
                output_path = opt.split("=", 1)[1]
            elif opt == "--save-baseline":
                output_path = "cube_stress_test_baseline.json"
            elif opt == "--ci":
                ci_mode = True
        cmd_test(args[1], max_steps=max_steps, output_path=output_path, ci_mode=ci_mode)
    elif command == "registry":
        subcmd = args[1] if len(args) > 1 else ""
        if subcmd != "add":
            err_console.print(
                Panel(
                    "[error]Usage:[/error] [cmd]cube registry add [PATH] [--submit] [--registry=REPO][/cmd]",
                    title="[error]Error[/error]",
                    border_style="red",
                    padding=(0, 1),
                )
            )
            sys.exit(1)
        remaining = args[2:]
        path = Path.cwd()
        submit = False
        registry = _REGISTRY_DEFAULT
        for opt in remaining:
            if opt == "--submit":
                submit = True
            elif opt.startswith("--registry="):
                registry = opt.split("=", 1)[1]
            elif not opt.startswith("--"):
                path = Path(opt)
        cmd_registry_add(path=path, submit=submit, registry=registry)
    else:
        err_console.print(f"[error]Unknown command:[/error] [cmd]{command}[/cmd]")
        _print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
