"""
Reproduce an experiment from a saved configuration.

Purpose
-------
Provide a CLI command and Python API to re-run an experiment from its
``ExperimentSpec`` JSON file, validate the environment, execute the
experiment, produce a reproducibility manifest, and package everything
into a ZIP archive.

Usage examples
--------------
.. code-block:: bash

    python -m llm_reliability.reproducibility.reproduce \\
        --config experiments/my_exp.json \\
        --output-dir reproduced \\
        --no-cache

.. code-block:: python

    from pathlib import Path
    from llm_reliability.reproducibility.reproduce import reproduce_experiment
    archive_path = reproduce_experiment(
        config_path=Path("experiments/my_exp.json"),
        output_dir=Path("reproduced"),
        use_cache=False,
    )
"""

from __future__ import annotations

import argparse
import logging
import sys
import zipfile
from pathlib import Path
from typing import Any

from llm_reliability.cache import ExperimentCache
from llm_reliability.experiments.experiment_models import ExperimentSpec
from llm_reliability.experiments.experiment_runner import ExperimentRunner
from llm_reliability.reporting.summary import ExperimentSummary
from llm_reliability.reproducibility.archive import ArchiveBuilder

logger = logging.getLogger(__name__)


def reproduce_experiment(
    config_path: Path,
    output_dir: Path | None = None,
    use_cache: bool = True,
) -> Path:
    """Re-run an experiment from its configuration JSON.

    Parameters
    ----------
    config_path : Path
        Path to the ``ExperimentSpec`` JSON file.
    output_dir : Path | None
        Directory for the reproduced archive.  Defaults to ``results/reproduced``.
    use_cache : bool
        Whether to use the experiment cache for repeated runs.

    Returns
    -------
    Path
        Path to the ZIP archive containing the full reproducibility package.
    """
    # ---- Validate environment -------------------------------------------------
    logger.info("Validating execution environment…")
    _validate_environment()

    # ---- Load configuration ---------------------------------------------------
    logger.info("Loading experiment config from %s", config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = config_path.read_text(encoding="utf-8")
    spec = ExperimentSpec.from_canonical_json(raw)

    # ---- Resolve output directory ---------------------------------------------
    if output_dir is None:
        output_dir = Path("results") / "reproduced"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Run experiment -------------------------------------------------------
    logger.info(
        "Running experiment '%s' (%s)…",
        spec.experiment_name,
        spec.experiment_id,
    )
    cache = ExperimentCache(enabled=use_cache)
    runner = ExperimentRunner(spec, cache=cache)
    status = runner.run()

    if status.failed_runs > 0:
        logger.warning(
            "%d run(s) failed during reproduction.",
            status.failed_runs,
        )

    # ---- Prepare summary ------------------------------------------------------
    summary = ExperimentSummary(
        experiment_id=spec.experiment_id,
        experiment_name=spec.experiment_name,
        metrics=list(runner.metrics),
        rankings=list(runner.rankings),
        executions=list(runner.executions),
        evaluations=list(runner.evaluations),
        config_snapshot=_config_to_snapshot(spec),
    )

    # ---- Build archive directory ----------------------------------------------
    builder = ArchiveBuilder()
    archive_dir = builder.build(summary, root_dir=str(output_dir))

    # ---- Package as ZIP -------------------------------------------------------
    archive_zip = output_dir / f"{spec.experiment_id}.zip"
    logger.info("Packaging archive into %s", archive_zip)

    with zipfile.ZipFile(archive_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in archive_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(archive_dir)
                zf.write(file_path, arcname)

    logger.info("Reproduction complete → %s", archive_zip)
    return archive_zip


def _validate_environment() -> None:
    """Quick sanity check that required imports and Python version are available."""
    import platform

    major, minor, *_ = platform.python_version_tuple()
    if int(major) < 3 or (int(major) == 3 and int(minor) < 10):
        raise RuntimeError(f"Python >= 3.10 required, found {platform.python_version()}")


def _config_to_snapshot(spec: ExperimentSpec) -> dict[str, Any]:
    """Extract a JSON-serialisable snapshot from an ExperimentSpec."""
    return {
        "experiment_id": spec.experiment_id,
        "experiment_name": spec.experiment_name,
        "benchmarks": [b.model_dump() for b in spec.benchmarks],
        "agents": [a.model_dump() for a in spec.agents],
        "seeds": spec.seeds,
        "repetitions": spec.repetitions,
        "perturbations": spec.perturbations,
        "fault_injection": spec.fault_injection,
        "parallel": spec.parallel,
        "max_workers": spec.max_workers,
        "llm": spec.llm,
        "prompt_version": spec.prompt_version,
        "dataset_version": spec.dataset_version,
        "metadata": spec.metadata,
    }


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for ``python -m llm_reliability.reproducibility.reproduce``."""
    parser = argparse.ArgumentParser(
        prog="reproduce",
        description="Reproduce an experiment from a config JSON and package as a ZIP archive.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for the reproduced archive (default: results/reproduced)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the experiment cache",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
    )

    config_path = Path(args.config)
    output_dir = Path(args.output_dir) if args.output_dir else None

    try:
        archive_path = reproduce_experiment(
            config_path=config_path,
            output_dir=output_dir,
            use_cache=not args.no_cache,
        )
        print(f"Reproduction archive created at: {archive_path}")
    except Exception as exc:
        print(f"Reproduction failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
