# Repository Guidelines

## Project Structure & Module Organization
- `python/sglang/`: core Python runtime, APIs, and model backends.
- `test/`: unit and integration tests (notably `test/srt` and `test/lang`).
- `docs/`: documentation sources (Sphinx, notebooks, and guides).
- `examples/` and `benchmark/`: usage examples and performance scripts.
- `sgl-kernel/` and `sgl-router/`: related subprojects with their own code.
- `assets/`, `scripts/`, `docker/`, `3rdparty/`: supporting resources and tooling.

## Build, Test, and Development Commands
- `pip install -e "python"`: install the editable Python package for local development.
- `python3 -m sglang.launch_server --model <path>`: start a local server for manual testing.
- `make format`: run `isort` and `black` on modified Python files.
- `pre-commit run --all-files`: run the full formatting/linting suite.
- `python3 test/srt/run_suite.py --suite per-commit`: run a CI-aligned test suite.

## Coding Style & Naming Conventions
- Follow `black` + `isort`; run `make format` or `pre-commit` before pushing.
- Prefer small, pure functions and avoid in-place argument mutation.
- Keep files concise (split if they grow too large) and avoid duplicating logic.
- Name new hardware-specific modules as separate files (e.g., `allocator_ascend.py`).

## Testing Guidelines
- Primary framework: Python `unittest`; some tests use `pytest`.
- Place tests under `test/srt` (runtime) or `test/lang` (frontend language).
- Ensure new tests are registered in the relevant `run_suite.py` for CI pickup.
- Favor short, focused tests and reuse server launches to reduce runtime.

## Commit & Pull Request Guidelines
- Commit messages are short and imperative; common prefixes include `fix:`, `docs:`, `chore:`, or bracketed tags like `[BUGFIX]`.
- Do not commit directly to `main`; work on a branch (e.g., `feature/...`).
- Run `pre-commit run --all-files` and include relevant tests in the PR description.
- Check `.github/CODEOWNERS` for required reviewers; request the `run-ci` label to trigger CI.

## Documentation Workflow
- Docs live in `docs/`; prefer notebooks for executable examples when possible.
- Build docs with `make html` or serve locally via `make serve` from `docs/`.

# 个人开发偏好

## 通用指令
- 始终使用中文回复（代码和专有名词除外）
- 关注 ML/AI Infra/HPC 领域问题
- 优先使用详细分析而非快速猜测

## 代码修改原则
- 宁可保守也不要引入 bug
