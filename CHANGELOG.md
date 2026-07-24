# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [1.0.0] - 2026-07-25

### Added

- Comprehensive quantitative analysis suite: ablation analysis, weighting analysis, sensitivity analysis, statistical validation, error analysis, benchmark analysis, model analysis, reproducibility analysis
- Complete LaTeX manuscript sections: introduction, related work, methodology, results, discussion, conclusion, extended related work, threats to validity, limitations, practical significance
- 38 publication-quality figures and 9 LaTeX tables traceable to actual experiment data
- GitHub health files: `CODE_OF_CONDUCT.md`, `SECURITY.md`, `CONTRIBUTING.md`, `CHANGELOG.md`, `CITATION.cff`
- Runnable examples: `examples/basic_experiment.py`, `examples/cli_usage.sh`, `examples/docker_usage.md`
- Pre-commit hooks configuration (`.pre-commit-config.yaml`) with trailing whitespace, YAML/TOML validation, ruff, black, isort
- Dependabot configuration for automated dependency updates
- Reproducibility documentation (`docs/reproducibility.md`)

### Changed

- Rewrote `README.md` as comprehensive project landing page with badges, table of contents, project structure, and documentation links
- Updated `pyproject.toml` version to `1.0.0` with expanded dependencies and package metadata
- Upgraded `.gitignore` with entries for cache directories, binaries, archives, and generated files
- Improved `docs/architecture.md` with updated component descriptions
- Replaced paired t-test with independent t-test in statistical analysis pipeline
- Made bootstrap count consistent at 10,000 across all analyses

### Removed

- Deleted synthetic publication artifacts: `fig1_ranking_bump_chart.*`, `fig2_success_vs_reliability_scatter.*`, `table1_agent_reliability_matrix.tex`
- Moved synthetic pipeline `scripts/generate_paper_artifacts.py` to `legacy/` with deprecation header
- Removed tracked binaries and archives (`ChatGPT Installer.exe`, `*.zip`, `LLM___Reliability.pdf`)
- Removed unused `demo.py` and `interfaces/__init__.py`
- Untracked generated `requirements_lock.txt`

### Fixed

- Corrected manuscript claims: MockAgent accuracy (0.5), model count (6), run count (9), "small differences" language in methodology, results, discussion, and conclusion sections
- Removed spurious hardware sensitivity string-match in `hardware_profile.py`
- Group metric computation by (benchmark, agent) for heterogeneous records
- Aligned fault fallback record names for consistent aggregation
- Added perturbation aliases for robust metric grouping

## [0.1.0] - 2026-07-25

### Added

- Initial framework release with core benchmark evaluation pipeline
- Expanded benchmark adapters: GAIA, AgentBoard, SWE-Bench Lite, ARC, GSM8K, MMLU, Hellaswag, PIQA, WinoGrande, TruthfulQA, MBPP, HumanEval
- Runtime support for Ollama, OpenAI, Anthropic, Gemini, DeepSeek, Qwen, Hugging Face, vLLM, llama.cpp
- Statistical validation engine with Spearman/Kendall correlations, bootstrap confidence intervals, and hypothesis testing
- Publication package with manuscript, results, methodology, and audit report
- Plugin-based benchmark discovery via `BenchmarkRegistry` with auto-import and decorator registration
- Plugin-based runtime registry (`RuntimeRegistry`) with dynamic discovery
- Command-line interface (`llm-reliability`) with `run`, `list`, `validate`, and `clear-cache` subcommands
- Experiment caching layer with `ExperimentCache` and `FileSystemCacheBackend`
- Structured logging with `LogContext` and thread-local context variables
- Docker support with `Dockerfile` and `docker-compose.yml`
- CI/CD pipeline configuration
- Reproducibility toolkit: manifest generator, environment capture, archive builder, and reproducibility checklist
- Multi-format report export (Markdown, LaTeX, HTML) with figures (PNG, SVG, PDF)

### Changed

- Migrated from hard-coded agent registry to plugin-based `RuntimeRegistry`
- Replaced inline benchmark discovery with `BenchmarkRegistry` plugin architecture
- Modernized project structure with standardized CLI, caching, and logging infrastructure

### Fixed

- Group metric computation by (benchmark, agent) to handle heterogeneous perturbation/fault records
- Aligned fault fallback record names for consistent aggregation
- Added perturbation aliases for robust metric grouping
