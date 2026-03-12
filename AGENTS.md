# Repository Guidelines

## Agent Execution Principles
- Always reply in Chinese unless the user explicitly asks for another language.
- Prefer direct execution over long planning for clear, bounded edits.
- Be conservative with behavior changes; prioritize correctness and backward compatibility.
- Never run destructive git commands (`reset --hard`, `checkout --`) unless explicitly requested.
- For ML/AI Infra/HPC changes, include correctness impact and performance impact in the summary.

## Project Structure & Module Organization
- `python/sglang/`: core runtime, API servers, schedulers, model backends, and distributed logic.
- `test/`: runtime/frontend tests (`test/srt`, `test/lang`, plus registered/manual suites).
- `sgl-kernel/`: CUDA/C++ and Triton kernels for performance-critical paths.
- `sgl-model-gateway/`: Rust model gateway, routing, and protocol adaptation.
- `docs/`: Sphinx docs, guides, and notebooks.
- `examples/`, `benchmark/`: deployment examples and performance scripts.
- `scripts/`, `docker/`, `assets/`, `3rdparty/`: project tooling and dependencies.

## Build, Test, and Development Commands
- `pip install -e "python"`: editable install for Python runtime development.
- `python3 -m sglang.launch_server --model <path>`: local serving sanity check.
- `make format`: run formatting (`black` + `isort`) on Python changes.
- `pre-commit run --all-files`: full style/lint checks before PR.
- `python3 test/srt/run_suite.py --suite per-commit`: CI-aligned runtime suite.

### Focused Validation (Preferred During Iteration)
- Python unit/integration change: run only impacted tests first, then expand.
- Kernel change (`sgl-kernel/`): run related kernel tests/benchmarks before broad suites.
- Router/gateway change (`sgl-model-gateway/`): run relevant Rust tests for touched modules.
- API behavior change: add or update endpoint-level tests under `test/srt`.

## Safe Change Workflow
1. Scope the minimal affected modules and interfaces.
2. Implement the smallest correct fix first; avoid opportunistic refactors.
3. Run targeted tests near the change.
4. Run at least one broader suite (`per-commit` or equivalent) before finishing.
5. Document risks, limitations, and untested paths in the final summary.

## Coding Style & Naming Conventions
- Follow `black` + `isort`; keep imports and formatting deterministic.
- Prefer small pure functions; avoid hidden in-place mutation unless required for performance.
- Keep files cohesive; split only when it improves ownership/readability.
- Name hardware-specific modules explicitly (e.g., `allocator_ascend.py`, `*_xpu.py`).
- For distributed code, make rank/world-size assumptions explicit and validated.

## Testing Guidelines
- Frameworks: primarily `unittest` and `pytest`.
- Place tests by responsibility: runtime in `test/srt`, language/frontend in `test/lang`.
- Register new suites in the proper runner (`run_suite.py`) so CI can discover them.
- Add regression tests for every bug fix when feasible.
- For performance-sensitive changes, include benchmark evidence when behavior/perf may shift.

## ML/AI Infra/HPC Guardrails
- Avoid implicit CPU/GPU sync points in hot paths unless necessary.
- Watch tensor shape/dtype/device transitions explicitly, especially across distributed boundaries.
- Validate fallback behavior when optional backends (FlashInfer/Triton/etc.) are unavailable.
- Treat memory footprint changes as first-class: mention expected KV/cache/activation impact.
- Prefer incremental rollout flags for risky runtime behavior changes.

## Commit & Pull Request Guidelines
- Use short imperative commit messages (`fix:`, `docs:`, `chore:`, `[BUGFIX]`).
- Never commit directly to `main`; use a feature branch.
- Include in PR description:
  - What changed and why
  - Test commands executed
  - Performance impact (if applicable)
  - Compatibility or migration notes (if any)
- Check `.github/CODEOWNERS` and request required reviewers.

## Documentation Workflow
- Keep docs in `docs/`; prefer executable notebooks for runnable examples.
- Build docs from `docs/` via `make html` or preview locally with `make serve`.
- Update docs when changing public API, CLI flags, config semantics, or deployment flow.

# 个人开发偏好

## 通用指令
- 始终使用中文回复（代码和专有名词除外）。
- 关注 ML/AI Infra/HPC 领域问题。
- 优先详细分析，避免快速猜测。
- 简单明确的修改请求，直接动手，不进入冗长规划流程。

## 代码修改原则
- 宁可保守也不要引入 bug。
- 优先最小改动原则，避免无关重构。
- 对高风险改动先补测试，再扩展优化。
