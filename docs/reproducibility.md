# Reproducibility

The framework provides built-in tools to make every experiment fully reproducible.

## Reproducibility Modules

Located in `src/llm_reliability/reproducibility/`:

| Module       | Purpose                                           |
|--------------|---------------------------------------------------|
| `manifest.py` | Generates `manifest.json` with artifact SHA-256 hashes |
| `environment.py` | Captures Python version, packages, OS, hardware |
| `archive.py` | Assembles complete results directory tree |
| `checklist.py` | Automated reproducibility checklist (pass/fail) |
| `citation.py` | CITATION.cff generation |

## Experiment Manifests

Every experiment produces a `manifest.json` containing:
- Experiment ID, name, and timestamps
- Git commit hash
- SHA-256 hashes of all executions, evaluations, metrics, and rankings
- Configuration snapshot hash

## Environment Validation

`EnvironmentCapture.capture()` snapshots:
- Python interpreter version
- All installed package versions (via `importlib.metadata`)
- OS name, version, and architecture
- CPU count and available memory
- Hostname (anonymisable)

## Archive Structure

The archive builder produces a self-contained directory:

```
results/<experiment_id>/
├── figures/          # PNG, SVG, PDF plots
├── tables/           # CSV, JSON, Markdown, LaTeX tables
├── reports/          # report.md, report.tex, report.html
├── manifest.json     # Artifact hashes
├── environment.json  # Environment snapshot
├── CITATION.cff      # Citation metadata
└── CHECKLIST.md      # Reproducibility checklist
```

## Reproducing Experiments

1. Ensure the same Python version and dependencies (check `environment.json`)
2. Use the same seed values from the original config
3. Verify artifact hashes match `manifest.json`
4. Run the reproducibility checklist: `ReproducibilityChecklist.run(summary, archive_dir)`

## Seed Management

`SeedManager` sets deterministic seeds for `random`, `numpy`, and `torch` (if installed). Seeds are derived per (base_seed, benchmark, agent, run_index) so execution order never affects reproducibility.
