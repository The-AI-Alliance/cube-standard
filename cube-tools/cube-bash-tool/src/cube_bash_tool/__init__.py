"""cube-bash-tool — container-backed bash tool for cube-standard benchmarks.

BashToolConfig  — serializable config; call .make(container) to get a BashTool
BashTool        — sync bash tool with optional read_file / write_file actions
"""

from cube_bash_tool.bash import BashTool, BashToolConfig

__all__ = ["BashTool", "BashToolConfig"]
