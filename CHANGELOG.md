# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [2.0.0] - 2026-07-25

### Added

- **Multi-Runtime Support**: Enhanced `Runtime` interface with optional capability methods (`load_model`, `unload_model`, `health_check`, `count_tokens`, `measure_latency`, `measure_memory`, `runtime_metadata`) — all with safe defaults preserving backward compatibility
- **Runtime Metadata**: `RuntimeMetadata` and `RuntimeCapabilities` models for standardized runtime info collection across all backends
- **7 Runtime Adapters**: `OllamaRuntime`, `VLLMRuntime`, `LlamaCppRuntime`, `TGIRuntime`, `LMStudioRuntime`, `MLXRuntime`, `OpenAICompatRuntime` — fully capable implementations registered with `RuntimeRegistry`
- **Publication-Ready Reporting**: `save_publication_artifacts()` generates `experiment_summary.json`, `runtime_summary.json`, `hardware_summary.json`, `benchmark_summary.json`, `ranking_summary.json`, `statistics_summary.json`, LaTeX tables, Markdown tables, CSV summaries
- **Reproducibility Manifests**: `ReproducibilityManifest` model capturing git commit, framework version, Python/OS info, hardware profile, seeds, env vars, artifact checksums
- **Enhanced Configuration**: `ExperimentRunConfig` with runtime selection, hardware profile selection, parameter sweeps (`SweepConfig`), model groups (`ModelGroup`), resource limits (`ResourceLimits`), checkpoint frequency, execution limits
- **Extended CLI**: 10 new commands — `resume`, `checkpoint`, `compare`, `report`, `export`, `discover-models`, `discover-runtimes`, `hardware-info`, `system-info`, `statistics`
- **Publication Artifacts**: `generate_latex_table()`, `generate_markdown_table()`, `generate_csv()` for LaTeX/Markdown/CSV export of ranking data
- **Stateless Health Checks**: All runtime adapters verify server availability without side effects
- **Plugin Developer Documentation**: Comprehensive `docs/plugin_development.md` with examples for runtimes, benchmarks, reports, visualizations, and configuration schemas
- **Migration Guide**: `docs/migration_v2.md` documenting all changes and migration steps
- **69 New Tests**: Runtime metadata (12), runtime adapters (22), extended models (11), publication reporting (17), CLI extended (18) — all pass alongside 938 existing tests

### Changed

- Enhanced `Runtime` base class with 7 new optional capability methods (backward compatible defaults)
- Enhanced `Runtime._detect_capabilities()` auto-detects which optional methods are overridden in subclasses
- Enhanced `ExperimentSpec` support for runtime selection via `ExperimentRunConfig` wrapper
- Updated CLI `checkpoint` command to validate directory existence before checking checkpoint state
- Added `metadata()` required method to all new runtime adapters for `Agent` ABC compliance

### Fixed

- `OllamaRuntime.health_check()` correctly extracts boolean from `check_ollama_server()` tuple return
- `checkpoint` CLI command now exits with error on non-existent directory
- Config validation CLI test uses correct `ExperimentSpec` format (removed `benchmark`/`agent` extras)

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
