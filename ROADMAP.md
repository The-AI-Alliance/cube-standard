# CUBE Standard — Roadmap

> This roadmap reflects current priorities and is updated as the project evolves. Items are roughly ordered by priority within each phase. For detailed proposals, see the [`design/`](design/) folder and the [RFC process](#rfc-process) in [CONTRIBUTING.md](CONTRIBUTING.md).

## Phase 1 — Alpha Stabilization (current)

Goal: stable core protocol, first wave of cubes, compliance tooling.

- [x] Core protocol: `Tool`, `Task`, `Benchmark`, `Observation`, `Action`
- [x] `cube init` / `cube test` CLI
- [x] Reference implementation: `counter-cube`
- [x] Container backends (Docker, Modal, Daytona)
- [x] First cubes landing:
  - *Web agents:* MiniWob ✅, WebArena-Verified ✅ ([cube-harness#214](https://github.com/The-AI-Alliance/cube-harness/pull/214)), WorkArena ✅
  - *Computer use (CUA):* OSWorld ✅
  - *SWE:* SWE-bench Verified + Live ✅, TerminalBench 2 ✅, LiveCodeBench ✅
- [ ] Benchmark metadata schema — `BenchmarkMetadata` fields: homepage, citation, license, task count, modality ([`benchmark.py`](src/cube/benchmark.py))
- [x] CUBE Stress Test — compliance checks and latency suite (`cube test cube-name) — nearly complete, see [PR #22](https://github.com/The-AI-Alliance/cube-standard/pull/22)
- [x] Unified resource provisioning — converged on `InfraConfig` + `VMResourceConfig` (`LocalInfraConfig` for local QEMU/qcow2, `cube-infra-aws` / `cube-infra-azure` for cloud); the standalone `VMBackend` / `VM` abstraction was removed in favour of it ([`resource/spec.md`](openspec/specs/resource/spec.md))
- [ ] Stable `v0.1` API — freeze core interfaces, tag release
- [x] PyPI publication (`cube-standard`)
- [ ] Published documentation site

## Phase 2 — Platform Integrations & Cube Growth

Goal: integrate with major agent frameworks, grow to ~50 cubes.

- [ ] NemoGym integration — bidirectional: run CUBE cubes from NemoGym, expose NemoGym envs as cubes
- [ ] AgentBeats integration — leaderboard and evaluation pipeline connected to CUBE
- [ ] Other platform integrations — ongoing discussions with framework maintainers
- [ ] ~50 cubes, growing across categories
- [ ] RFC: streaming observations ([`design/rfc_core_extensions.md`](design/rfc_core_extensions.md))
- [ ] RFC: better async task execution ([`design/rfc_core_extensions.md`](design/rfc_core_extensions.md))
- [ ] RFC: multi-agent support ([`design/rfc_core_extensions.md`](design/rfc_core_extensions.md))
- [ ] RFC: multi-dimensional rewards ([`design/rfc_core_extensions.md`](design/rfc_core_extensions.md))

## Phase 3 — Broad Ecosystem

Goal: CUBE becomes the default interoperability layer for agent benchmarks. Exact scope TBD — to be discussed with the community.

- [ ] Large-scale cube registry — community-maintained index of CUBE-compatible benchmarks
- [ ] Cube discovery and install (`cube add <benchmark>`)
- [ ] Broader platform integrations (beyond Phase 2)
- [ ] Number of cubes: open-ended, driven by community adoption

> Phase 3 priorities will be shaped by what the community builds in Phase 2. Join the [discussions](https://github.com/The-AI-Alliance/cube-standard/discussions) to help define it.

## RFC Process

Have an idea that changes the core protocol? Open a GitHub Discussion or file a PR against `design/`. See [CONTRIBUTING.md](CONTRIBUTING.md#rfc-process) for the full process.

## How to Influence the Roadmap

- Comment on existing [GitHub Issues](https://github.com/The-AI-Alliance/cube-standard/issues) or open a new one
- Start a [GitHub Discussion](https://github.com/The-AI-Alliance/cube-standard/discussions)
- Submit an RFC draft in `design/` via PR
- [Propose a benchmark for wrapping](https://docs.google.com/forms/d/e/1FAIpQLSddMFyRXZJPpD0I2K27OEmIPUpj57w--u2NuMscrjNlkqy8rQ/viewform) — flag a benchmark you'd like to see as a CUBE, or contribute one yourself
- [Apply as a core contributor](https://forms.gle/JFiBi4ynfVLMghAH8) to help shape priorities directly
