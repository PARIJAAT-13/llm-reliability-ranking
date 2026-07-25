"""
Purpose
-------
Provide utility functions specifically designed for processing AgentBoard outputs.

Responsibilities
----------------
- Normalize agent outputs for resilient evaluation against expected answers
- Handle whitespace, case, and punctuation discrepancies
- Support optional sub-task progress scoring where the dataset provides it

Design notes
------------
AgentBoard tasks span diverse categories (web browsing, tool use, coding, QA).
A single normalisation pass — lowercase + strip whitespace/punctuation — is
sufficient for exact-match evaluation without introducing category-specific
branching that would complicate the pipeline.  Partial-progress scores (when
``progress_rate`` is present in the task) are handled by ``score_agentboard``
independently of the normalised string match.
"""

from __future__ import annotations

import string


def normalize_agentboard_answer(answer: str) -> str:
    """Normalise an AgentBoard answer string for evaluation.

    Converts to lowercase, strips surrounding whitespace, and removes leading /
    trailing punctuation to improve robustness against minor formatting differences
    between agent outputs and expected answers.

    Parameters
    ----------
    answer:
        Raw string produced by the agent or stored as the expected answer.

    Returns
    -------
    str
        Normalised string suitable for equality comparison.

    Examples
    --------
    >>> normalize_agentboard_answer("  Yes! ")
    'yes'
    >>> normalize_agentboard_answer("SUCCESS.")
    'success'
    """
    if not isinstance(answer, str):
        return ""

    # Lowercase and strip surrounding whitespace
    ans = answer.lower().strip()

    # Remove leading / trailing punctuation
    ans = ans.strip(string.punctuation)

    return ans


def score_agentboard(
    *,
    expected: str,
    agent_output: str,
    progress_rate: float | None = None,
) -> tuple[bool, float]:
    """Compute success and score for one AgentBoard task evaluation.

    Parameters
    ----------
    expected:
        Ground-truth expected output as stored in the dataset.
    agent_output:
        Raw string returned by the agent.
    progress_rate:
        Optional sub-task progress rate in [0, 1] provided by the dataset for
        partial-credit scoring.  When *None*, scoring is binary (0.0 or 1.0).

    Returns
    -------
    (success, score):
        *success* is ``True`` when the normalised outputs match exactly.
        *score* mirrors *success* as 1.0 / 0.0 unless ``progress_rate`` is
        provided, in which case *score* equals ``progress_rate`` while
        *success* still requires an exact normalised match.

    Notes
    -----
    The AgentBoard paper reports both exact-match success and a progress-rate
    metric for tasks that have multiple sub-goals.  This function supports both
    modes so the adapter can forward the richer signal to ``EvaluationRecord``
    without redesigning the record schema.
    """
    expected_norm = normalize_agentboard_answer(expected)
    output_norm = normalize_agentboard_answer(agent_output)

    success = expected_norm == output_norm

    if progress_rate is not None:
        # Clamp to [0, 1] defensively; dataset values should already be valid.
        score = float(max(0.0, min(1.0, progress_rate)))
    else:
        score = 1.0 if success else 0.0

    return success, score
