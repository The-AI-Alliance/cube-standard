"""cube CLI — entry point for the `cube` command.

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

import importlib
import importlib.metadata
import shutil
import sys
from pathlib import Path

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from cube import __version__

# ── Console setup ─────────────────────────────────────────────────────────────

_THEME = Theme(
    {
        "info": "cyan",
        "success": "bold green",
        "warning": "bold yellow",
        "error": "bold red",
        "dim": "dim white",
        "file": "green",
        "cmd": "bold cyan",
        "brand": "bold blue",
    }
)

console = Console(theme=_THEME)
err_console = Console(stderr=True, theme=_THEME)

# ── Constants ──────────────────────────────────────────────────────────────────

_TEMPLATE_DIR = Path(__file__).parent / "_template" / "new_cube_package"
_DEFAULT_NAME = "my-benchmark"


# ── Commands ───────────────────────────────────────────────────────────────────


_PLACEHOLDER_NAMES = {"cube_package", "new_cube_package", "new-cube-package"}


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
    table.add_column("description", style="white")

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
) -> None:
    """Import *module_name* (or resolve an entry-point name) and run the debug compliance suite."""
    from cube.testing import (
        aggregate_profiling,
        build_stress_test_report,
        collect_stress_compliance,
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

    failures = [r for r in results if r.get("error") or not r.get("done") or r.get("reward") != 1.0]
    compliance_passed, compliance_failed = collect_stress_compliance(results, module)
    reset_ok = "test_reset_reproducibility" in compliance_passed
    meta_ok = "test_benchmark_metadata" in compliance_passed
    close_idempotent_ok = "test_close_idempotent" in compliance_passed
    tools_list_ok = "test_tools_list" in compliance_passed

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

    # Constrain width so panel is narrower and taller (pic2-like h/w aspect ratio)
    try:
        _term_width = console.size.width
    except Exception:
        _term_width = 80
    # Narrow layout: compress table columns (short headers + compact values) so it fits
    _display_width = min(_term_width, 62)
    _bar_width = min(26, max(12, _display_width - 42))

    def _latency_bar(sec: float, max_sec: float = 0.2, width: int = _bar_width) -> str:
        if max_sec <= 0:
            return "░" * width
        filled = min(int((sec / max_sec) * width), width)
        return "█" * filled + "░" * (width - filled)

    # ── Stress-test style layout: CUBE Stress Test + COMPLIANCE + LATENCY ────────
    status_str = "Passed" if not failures else "Failed"
    progress_str = f"{len(results) - len(failures)}/{len(results)}"
    header_lines = [
        "[bold]CUBE Stress Test[/bold]",
        "",
        f"Benchmark: [file]{resolved}[/file]    Status: [{'success' if not failures else 'error'}]{status_str}[/]    Progress: {progress_str}",
    ]

    # COMPLIANCE: named checks from stress_test_specs.md
    full_episode_status = "[success]✓[/success]" if not failures else "[error]✗[/error]"
    compliance_checks = [
        ("debug_tasks_exist", "[success]✓[/success]" if results else "[dim]NULL[/dim]"),
        ("debug_agent_exists", "[success]✓[/success]" if results else "[dim]NULL[/dim]"),
        ("full_episode", full_episode_status),
        ("reset_reproducibility", "[success]✓[/success]" if reset_ok else "[error]✗[/error]"),
        ("tools_list", "[success]✓[/success]" if tools_list_ok else "[error]✗[/error]"),
        ("close_idempotent", "[success]✓[/success]" if close_idempotent_ok else "[error]✗[/error]"),
        ("benchmark_metadata", "[success]✓[/success]" if meta_ok else "[error]✗[/error]"),
    ]
    compliance_header = Text.from_markup("[bold]COMPLIANCE[/bold]")
    compliance_checks_table = Table(
        show_header=False,
        box=box.SIMPLE,
        padding=(0, 1),
        show_edge=False,
    )
    compliance_checks_table.add_column("check", style="dim", no_wrap=True)
    compliance_checks_table.add_column("status", no_wrap=True)
    compliance_checks_table.add_column("check2", style="dim", no_wrap=True)
    compliance_checks_table.add_column("status2", no_wrap=True)
    for i in range(0, len(compliance_checks), 2):
        name1, status1 = compliance_checks[i]
        if i + 1 < len(compliance_checks):
            name2, status2 = compliance_checks[i + 1]
            compliance_checks_table.add_row(name1, status1, name2, status2)
        else:
            compliance_checks_table.add_row(name1, status1, "", "")

    # Task-level results table (per-episode); compressed headers/values for narrow width
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
    task_results_table.add_column("t(s)", justify="right")
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
            f"{r['episode_time_s']:.2f}",
            r["error"] or "",
        )

    max_lat = max(p50_s, p95_s, p99_s, 0.001)
    latency_lines = [
        "[bold]LATENCY (seconds)[/bold]",
        f"  p50 │{_latency_bar(p50_s, max_lat)}│ [dim]{p50_s:.3f}s[/dim]",
        f"  p95 │{_latency_bar(p95_s, max_lat)}│ [dim]{p95_s:.3f}s[/dim]",
        f"  p99 │{_latency_bar(p99_s, max_lat)}│ [dim]{p99_s:.3f}s[/dim]",
    ]
    reset_times = [r.get("reset_time_s") for r in results if r.get("reset_time_s") is not None]
    mean_setup_s = sum(reset_times) / len(reset_times) if reset_times else None
    if mean_setup_s is not None:
        latency_lines.append("")
        latency_lines.append(f"[bold]Task setup (mean)[/bold]: [dim]{mean_setup_s:.4f}s[/dim]")
    profiling_agg = aggregate_profiling(results)
    if profiling_agg:
        latency_lines.append("")
        latency_lines.append("[bold]Profiling[/bold]")
        for op_name, dur_s in profiling_agg.items():
            latency_lines.append(f"  {op_name}: [dim]{dur_s:.4f}s[/dim]")

    # Build and optionally save stress-test report
    report = build_stress_test_report(resolved, results, compliance_passed, compliance_failed)
    if output_path:
        report.save(output_path)
        console.print(f"[dim]Baseline saved to [file]{output_path}[/file][/dim]")

    border = "green" if not failures else "red"
    status_text = (
        f"[success]{len(results)} task(s) passed[/success]"
        if not failures
        else f"[error]{len(failures)} / {len(results)} task(s) failed[/error]"
    )

    stress_panel = Panel(
        Group(
            Text.from_markup("\n".join(header_lines)),
            "",
            compliance_header,
            compliance_checks_table,
            "",
            task_results_table,
            "",
            Text.from_markup("\n".join(latency_lines)),
        ),
        title=f"[bold]CUBE Stress Test[/bold]  [file]{module_name}[/file]  —  {status_text}",
        border_style=border,
        padding=(0, 1),
    )
    # Render at fixed narrow width so output shape matches pic2 (taller than wide)
    Console(theme=_THEME, width=_display_width).print(stress_panel)

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


def _print_help() -> None:
    """Print a rich-formatted help screen."""
    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2), show_edge=False)
    table.add_column("cmd", style="cmd", no_wrap=True)
    table.add_column("desc", style="white")
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
        "Run the debug compliance suite — NAME is a benchmark entry-point name or a dotted module path",
        "cube test counter-cube",
    )

    console.print(
        Panel(
            table,
            title=f"[brand]cube[/brand] [dim]v{__version__}[/dim]",
            subtitle="[dim]Common Unified Benchmark Environments[/dim]",
            border_style="blue",
            padding=(0, 1),
        )
    )


def main() -> None:
    args = sys.argv[1:]

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
        remaining = args[2:]
        for opt in remaining:
            if opt.startswith("--max-steps="):
                max_steps = int(opt.split("=", 1)[1])
            elif opt.startswith("--output="):
                output_path = opt.split("=", 1)[1]
            elif opt == "--save-baseline":
                output_path = "cube_stress_test_baseline.json"
        cmd_test(args[1], max_steps=max_steps, output_path=output_path)
    else:
        err_console.print(f"[error]Unknown command:[/error] [cmd]{command}[/cmd]")
        _print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
