"""
Purpose
-------
Provide utility functions specifically designed for processing SWE-bench outputs.

Responsibilities
----------------
- Normalize generated patches for resilient evaluations
"""


def normalize_patch(patch: str) -> str:
    """
    Normalize a code patch to ensure consistent comparison.
    Strips trailing whitespace line by line and removes empty lines to
    handle minor spacing discrepancies in generated diffs.
    """
    if not isinstance(patch, str):
        return ""

    lines = [line.rstrip() for line in patch.splitlines() if line.strip()]
    return "\n".join(lines)
