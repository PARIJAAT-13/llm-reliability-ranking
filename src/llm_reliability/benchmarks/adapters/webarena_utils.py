from __future__ import annotations

import string


def normalize_webarena_answer(answer: str) -> str:
    if not isinstance(answer, str):
        return ""
    return answer.lower().strip().strip(string.punctuation)
