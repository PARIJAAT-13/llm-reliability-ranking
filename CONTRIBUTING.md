# Contributing to LLM Reliability Ranking

Thank you for your interest in contributing!

## Repository Overview

```
src/llm_reliability/     # Framework source code
tests/                   # Pytest test suite (361+ tests)
paper/                   # Research publication artifacts
configs/                 # Experiment configuration templates
scripts/                 # Large-scale experiment runners
docs/                    # Documentation
```

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

The `[dev]` extra installs `pytest`.

## Code Style

This project uses `ruff` for linting and `black` + `isort` for formatting:

```bash
ruff check src tests
black --check src tests
isort --check src tests
```

Configuration is in `pyproject.toml`:
- Line length: 100
- Target: Python 3.10+
- Ruff rules: E, F, W, I, N, UP

## Testing

Run the test suite with pytest:

```bash
pytest                          # All tests
pytest tests/unit               # Unit tests only
pytest tests/integration        # Integration tests only
pytest -v                       # Verbose output
```

Tests are auto-discovered from the `tests/` directory.

## Pull Request Process

1. Fork the repository and create a feature branch from `main`
2. Make your changes, keeping them focused and atomic
3. Run `ruff check src tests` and `pytest` to verify nothing is broken
4. Write or update tests for any new functionality
5. Update documentation if public APIs or behavior changed
6. Open a pull request with a clear title and description

## Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <short description>

<optional body>
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`

Examples:
```
feat: add GAIA benchmark adapter
fix: correct metric grouping by (benchmark, agent)
docs: update CLI usage examples
```

## Issue Reporting

- **Bug reports**: Include the full error, Python version, OS, and steps to reproduce
- **Feature requests**: Describe the use case and expected behavior
- **Questions**: Use GitHub Discussions for open-ended questions

For security issues, please contact the maintainers directly.
