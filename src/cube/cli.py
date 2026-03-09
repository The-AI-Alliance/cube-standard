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
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

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
_VERSION = "0.1.0rc1"


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


def cmd_test(module_name: str, *, max_steps: int = 20) -> None:
    """Import *module_name* (or resolve an entry-point name) and run the debug compliance suite."""
    from cube.testing import run_debug_suite

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
        module = importlib.import_module(resolved)
    except ModuleNotFoundError as exc:
        err_console.print(
            Panel(
                f"[error]Cannot import[/error] [file]{resolved}[/file]: {exc}\n"
                "Make sure the package is installed (e.g. [cmd]uv sync[/cmd]) and "
                "that the module exposes [cmd]get_debug_task_configs()[/cmd] and "
                "[cmd]make_debug_agent()[/cmd].",
                title="[error]Import Error[/error]",
                border_style="red",
                padding=(0, 1),
            )
        )
        sys.exit(1)

    for required in ("get_debug_task_configs", "make_debug_agent"):
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
        results = run_debug_suite(resolved, module, max_steps=max_steps)

    if not results:
        err_console.print(
            Panel(
                "No debug tasks were run.\n"
                "Make sure [file]debug.py[/file] has entries in [cmd]_TASK_ACTIONS[/cmd] "
                "and [cmd]get_debug_task_configs()[/cmd] returns at least one config.",
                title="[warning]No tasks found[/warning]",
                border_style="yellow",
                padding=(0, 1),
            )
        )
        sys.exit(1)

    # ── Results table ──────────────────────────────────────────────────────────
    table = Table(
        show_header=True,
        box=box.SIMPLE,
        padding=(0, 1),
        show_edge=False,
        header_style="bold",
    )
    table.add_column("task_id", style="file", no_wrap=True)
    table.add_column("done", justify="center")
    table.add_column("reward", justify="right")
    table.add_column("steps", justify="right")
    table.add_column("time (s)", justify="right")
    table.add_column("error", style="error")

    failures: list[dict] = []
    for r in results:
        passed = not r["error"] and r["done"] and r["reward"] == 1.0
        if not passed:
            failures.append(r)

        done_str = "[success]✓[/success]" if r["done"] else "[error]✗[/error]"
        reward_str = (
            f"[success]{r['reward']:.3f}[/success]" if r["reward"] == 1.0 else f"[error]{r['reward']:.3f}[/error]"
        )
        table.add_row(
            r["task_id"],
            done_str,
            reward_str,
            str(r["steps"]),
            str(r["episode_time_s"]),
            r["error"] or "",
        )

    border = "green" if not failures else "red"
    status_text = (
        f"[success]{len(results)} task(s) passed[/success]"
        if not failures
        else f"[error]{len(failures)} / {len(results)} task(s) failed[/error]"
    )
    console.print(
        Panel(
            table,
            title=f"[brand]cube test[/brand]  [file]{module_name}[/file]  —  {status_text}",
            border_style=border,
            padding=(0, 1),
        )
    )

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
            title=f"[brand]cube[/brand] [dim]v{_VERSION}[/dim]",
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
        remaining = args[2:]
        if remaining and remaining[0].startswith("--max-steps="):
            max_steps = int(remaining[0].split("=", 1)[1])
        cmd_test(args[1], max_steps=max_steps)
    else:
        err_console.print(f"[error]Unknown command:[/error] [cmd]{command}[/cmd]")
        _print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
