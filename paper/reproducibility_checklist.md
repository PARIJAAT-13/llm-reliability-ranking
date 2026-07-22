# Reproducibility Checklist & Publication Readiness Audit

## 1. Codebase & Software Dependencies

- [x] All source code hosted in modular repository (`src/llm_reliability/`).
- [x] Exact dependencies specified in `pyproject.toml`.
- [x] 100% type-annotated, Google-style docstrings throughout all packages.
- [x] Full test suite (333 tests) passing with **0 failures and 0 warnings**.

## 2. Experimental Data & Provenance

- [x] Canonical SHA-256 configuration hashing (`hash_configuration()`) for all runs.
- [x] `SeedManager` enforcing deterministic seeds across `random`, `numpy`, and `torch`.
- [x] Environment metadata (Python version, hardware specs, OS, git commit hash) logged in `ExperimentManifest`.
- [x] All data logged in immutable Pydantic records (`ExecutionRecord`, `EvaluationRecord`, `MetricRecord`, `RankingRecord`).

## 3. Statistical Validity & Artifacts

- [x] Spearman $\rho$, Kendall $\tau$, Mann-Whitney U, Cliff's Delta, and Bootstrap CIs implemented in `StatisticalEngine`.
- [x] Pairwise rank overlap and rank displacement metrics implemented in `analyze_ranking_divergence`.
- [x] Vector PDF, SVG, and high-DPI PNG figures generated in `paper/figures/`.
- [x] Publication-ready LaTeX tables generated in `paper/tables/`.

## 4. Preflight & Execution Verification

- [x] `scripts/preflight_check.py` inspector verifying dependencies, keys, disk space, and network.
- [x] Pilot experiment verified (`scripts/run_pilot_experiment.py`).
- [x] Production runner with checkpoint/resume support verified (`scripts/run_large_scale_experiment.py`).
