<img width="3168" height="1344" alt="Untitled" src="https://github.com/user-attachments/assets/68e50a4d-fe15-4d08-b52e-38dae3122493" />

# README for CUBE Standard

<!--
[Published Documentation](https://the-ai-alliance.github.io/cube-standard/)
-->

This repo contains the code and documentation for the **AI Alliance: CUBE Standard** project, which meets a common necessity, to standardize benchmark wrapping so the community can wrap various otherwise-incompatible benchmarks uniformly and use them everywhere.

Principal developer: [ServiceNow AI Research](https://servicenow.com/research).

## Installation

Requires Python 3.12+. Install with [uv](https://docs.astral.sh/uv/):

```sh
uv add cube-standard
```

Or with pip:

```sh
pip install cube-standard
```

To include optional container backends:

```sh
# Docker support
uv add "cube-standard[docker]"

# Modal support
uv add "cube-standard[modal]"

# Daytona support
uv add "cube-standard[daytona]"
```

For development (includes test and lint tools):

```sh
git clone https://github.com/The-AI-Alliance/cube-standard
cd cube-standard
uv sync --extra dev
```

## CLI commands

| Command | What it does |
| --- | --- |
| `cube init [NAME]` | Scaffolds a new benchmark package from the built-in template |
| `cube list` | Lists all installed benchmarks registered under `cube.benchmarks` entry points |
| `cube test NAME` | Runs the debug suite and asserts `reward == 1.0` on every debug task |

## For benchmark contributors

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide. Quick reference:

```sh
cube init my-bench    # scaffold a new benchmark package from the template
cd my-bench
uv sync
cube test my-bench    # run the debug compliance suite
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the five-layer architecture and implementation order.

## Getting Involved

We welcome contributions as PRs. Please see our [Alliance community repo](https://github.com/The-AI-Alliance/community/) for general information about contributing to any of our projects. This section provides some specific details you need to know.

In particular, see the AI Alliance [CONTRIBUTING](https://github.com/The-AI-Alliance/community/blob/main/CONTRIBUTING.md) instructions. You will need to agree with the AI Alliance [Code of Conduct](https://github.com/The-AI-Alliance/community/blob/main/CODE_OF_CONDUCT.md).

All _code_ contributions are licensed under the [Apache 2.0 LICENSE](https://github.com/The-AI-Alliance/community/blob/main/LICENSE.Apache-2.0) (which is also in this repo, [LICENSE.Apache-2.0](LICENSE.Apache-2.0)).

All _documentation_ contributions are licensed under the [Creative Commons Attribution 4.0 International](https://github.com/The-AI-Alliance/community/blob/main/LICENSE.CC-BY-4.0) (which is also in this repo, [LICENSE.CC-BY-4.0](LICENSE.CC-BY-4.0)).

All _data_ contributions are licensed under the [Community Data License Agreement - Permissive - Version 2.0](https://github.com/The-AI-Alliance/community/blob/main/LICENSE.CDLA-2.0) (which is also in this repo, [LICENSE.CDLA-2.0](LICENSE.CDLA-2.0)).

### We use the "Developer Certificate of Origin" (DCO).

> [!WARNING]
> Before you make any git commits with changes, understand what's required for DCO.

See the Alliance contributing guide [section on DCO](https://github.com/The-AI-Alliance/community/blob/main/CONTRIBUTING.md#developer-certificate-of-origin) for details. In practical terms, supporting this requirement means you must use the `-s` flag with your `git commit` commands.

### Pre-commit hooks (recommended)

This repo uses the [`pre-commit`](https://pre-commit.com/) framework to run fast checks locally before you commit, including enforcing the DCO `Signed-off-by` line.

Install the hooks (you only need to do this once per clone):

```sh
pre-commit install --hook-type pre-commit --hook-type commit-msg
```

Run the checks on all files (optional, useful the first time):

```sh
pre-commit run --all-files
```

When committing, include your sign-off:

```sh
git commit -s -m "your message"
```
