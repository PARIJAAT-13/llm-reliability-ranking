"""
Lightweight tokenizer utilities for the LLM Agent Adapter Framework.

Provides a simple whitespace-based approximation of token counts.
Providers should replace this with their own tokenizer when available.
"""


def approximate_token_count(text: str) -> int:
    """Estimate token count using whitespace splitting (≈ 1 token per word).

    This is intentionally naive.  Providers should override with their
    own tokenizer (e.g., tiktoken for OpenAI, SentencePiece for others)
    when accurate counts are required.

    Args:
        text: The text to tokenize.

    Returns:
        An integer approximation of the token count.
    """
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(text.split())
