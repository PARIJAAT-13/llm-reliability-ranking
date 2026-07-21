"""
Purpose
-------
Provide concrete prompt perturbation strategies for LLM reliability testing.

Responsibilities
----------------
- Implement WhitespacePerturbationStrategy
- Implement FormattingPerturbationStrategy
- Implement InstructionReorderingPerturbationStrategy
- Implement SynonymSubstitutionPerturbationStrategy
- Implement PromptWrapperPerturbationStrategy

Design notes
------------
All strategies inherit from PerturbationStrategy and operate on task dictionaries.
They preserve task identity, ground truth, and semantic meaning while introducing
controlled variations to prompt texts.
"""

from __future__ import annotations

import re
import random
from typing import Any

from llm_reliability.reliability.perturbation.base import PerturbationStrategy


def _extract_prompt(task: dict[str, Any]) -> tuple[str, str]:
    """Extract primary prompt string and the key it was stored under."""
    for key in ("prompt", "instruction", "query", "task"):
        val = task.get(key)
        if isinstance(val, str) and val.strip():
            return val, key
    return "", "prompt"


def _create_perturbed_task(
    task: dict[str, Any],
    new_prompt: str,
    strategy_name: str,
    seed: int | None = None,
) -> dict[str, Any]:
    """Create a shallow copy of task dictionary with perturbed prompt and metadata."""
    task_copy = task.copy()
    if "metadata" in task_copy and isinstance(task_copy["metadata"], dict):
        task_copy["metadata"] = task_copy["metadata"].copy()
    else:
        task_copy["metadata"] = {}

    _, key = _extract_prompt(task)
    task_copy[key] = new_prompt

    task_id = str(task_copy.get("task_id", "unknown_task"))
    perturbation_id = f"{strategy_name}_{seed if seed is not None else '0'}"

    task_copy["metadata"]["perturbation"] = {
        "strategy": strategy_name,
        "perturbation_id": perturbation_id,
        "original_task_id": task_id,
    }
    return task_copy


class WhitespacePerturbationStrategy(PerturbationStrategy):
    """Whitespace normalization and controlled spacing variation."""

    @property
    def name(self) -> str:
        return "whitespace"

    @property
    def description(self) -> str:
        return "Normalize or introduce controlled variation in whitespace and line breaks."

    def apply(self, task: dict[str, Any], seed: int | None = None) -> dict[str, Any]:
        prompt, _ = _extract_prompt(task)
        if not prompt:
            return task.copy()

        rng = random.Random(seed if seed is not None else 42)
        mode = rng.choice(["double_space", "extra_newlines", "strip_lines", "tabs"])

        if mode == "double_space":
            # Add double spaces after sentences or words
            lines = prompt.splitlines()
            perturbed_lines = [re.sub(r"(\.|\?|\!)\s+", r"\1  ", line) for line in lines]
            new_prompt = "\n".join(perturbed_lines)
            if new_prompt == prompt:
                new_prompt = re.sub(r" ", "  ", prompt, count=5)
        elif mode == "extra_newlines":
            # Add extra newlines between non-empty lines
            lines = [line.strip() for line in prompt.splitlines() if line.strip()]
            new_prompt = "\n\n".join(lines)
        elif mode == "strip_lines":
            # Normalize internal spaces to single spaces
            new_prompt = re.sub(r"[ \t]+", " ", prompt).strip()
        else:
            # Replace some spaces with tab-stop spacing or indented lines
            lines = prompt.splitlines()
            new_prompt = "\n".join(["\t" + line if line.strip() else line for line in lines])

        return _create_perturbed_task(task, new_prompt, self.name, seed)


class FormattingPerturbationStrategy(PerturbationStrategy):
    """List style toggle (bullets <-> numbers) and paragraph formatting variation."""

    @property
    def name(self) -> str:
        return "formatting"

    @property
    def description(self) -> str:
        return "Toggle list formats (bullets/numbered) and alter paragraph spacing."

    def apply(self, task: dict[str, Any], seed: int | None = None) -> dict[str, Any]:
        prompt, _ = _extract_prompt(task)
        if not prompt:
            return task.copy()

        rng = random.Random(seed if seed is not None else 42)
        lines = prompt.splitlines()

        # Check if text contains numbered or bulleted list items
        has_numbered = any(re.match(r"^\s*\d+[\.\)]\s+", line) for line in lines)
        has_bulleted = any(re.match(r"^\s*[\*\-\+]\s+", line) for line in lines)

        new_lines = []
        if has_numbered:
            # Convert numbered list items to bullets
            for line in lines:
                m = re.match(r"^(\s*)\d+[\.\)]\s+(.*)$", line)
                if m:
                    new_lines.append(f"{m.group(1)}* {m.group(2)}")
                else:
                    new_lines.append(line)
            new_prompt = "\n".join(new_lines)
        elif has_bulleted:
            # Convert bullet list items to numbered items
            num = 1
            for line in lines:
                m = re.match(r"^(\s*)[\*\-\+]\s+(.*)$", line)
                if m:
                    new_lines.append(f"{m.group(1)}{num}. {m.group(2)}")
                    num += 1
                else:
                    new_lines.append(line)
            new_prompt = "\n".join(new_lines)
        else:
            # Alter paragraph spacing or wrap sentences into a formatted bullet list
            paragraphs = [p.strip() for p in prompt.split("\n\n") if p.strip()]
            if len(paragraphs) > 1:
                join_delim = "\n---\n" if rng.choice([True, False]) else "\n\n\n"
                new_prompt = join_delim.join(paragraphs)
            else:
                sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", prompt) if s.strip()]
                if len(sentences) > 1:
                    bullet_items = [f"- {s}" for s in sentences]
                    new_prompt = "Task breakdown:\n" + "\n".join(bullet_items)
                else:
                    new_prompt = f"### Task Prompt\n\n{prompt}"

        return _create_perturbed_task(task, new_prompt, self.name, seed)


class InstructionReorderingPerturbationStrategy(PerturbationStrategy):
    """Reorder independent instructions or sentences without altering dependencies."""

    @property
    def name(self) -> str:
        return "reordering"

    @property
    def description(self) -> str:
        return "Reorder independent instructions or sentences within the prompt."

    def apply(self, task: dict[str, Any], seed: int | None = None) -> dict[str, Any]:
        prompt, _ = _extract_prompt(task)
        if not prompt:
            return task.copy()

        rng = random.Random(seed if seed is not None else 42)
        lines = prompt.splitlines()

        # Check for list items first
        list_item_indices = [
            i for i, line in enumerate(lines)
            if re.match(r"^\s*([\*\-\+]|\d+[\.\)])\s+", line)
        ]

        if len(list_item_indices) >= 2:
            items = [lines[i] for i in list_item_indices]
            shuffled = items.copy()
            rng.shuffle(shuffled)
            # Guarantee at least a swap if shuffle produced same order
            if shuffled == items and len(shuffled) >= 2:
                shuffled[0], shuffled[1] = shuffled[1], shuffled[0]

            new_lines = list(lines)
            for idx, new_item in zip(list_item_indices, shuffled):
                new_lines[idx] = new_item
            new_prompt = "\n".join(new_lines)
        else:
            # Shuffle independent sentences
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", prompt) if s.strip()]
            if len(sentences) >= 2:
                shuffled_sentences = sentences.copy()
                rng.shuffle(shuffled_sentences)
                if shuffled_sentences == sentences:
                    shuffled_sentences[0], shuffled_sentences[1] = (
                        shuffled_sentences[1],
                        shuffled_sentences[0],
                    )
                new_prompt = " ".join(shuffled_sentences)
            else:
                new_prompt = prompt

        return _create_perturbed_task(task, new_prompt, self.name, seed)


class SynonymSubstitutionPerturbationStrategy(PerturbationStrategy):
    """Substitute non-technical words with appropriate English synonyms."""

    SYNONYM_MAP: dict[str, list[str]] = {
        "solve": ["resolve", "work out", "settle"],
        "find": ["determine", "locate", "discover"],
        "calculate": ["compute", "figure out", "estimate"],
        "provide": ["supply", "furnish", "give"],
        "answer": ["response", "solution", "reply"],
        "question": ["problem", "query", "task"],
        "task": ["assignment", "exercise", "job"],
        "describe": ["explain", "detail", "outline"],
        "create": ["generate", "produce", "build"],
        "complete": ["finish", "fulfill", "carry out"],
        "check": ["verify", "validate", "confirm"],
        "select": ["choose", "pick", "identify"],
        "obtain": ["retrieve", "acquire", "get"],
        "show": ["display", "demonstrate", "indicate"],
        "evaluate": ["assess", "judge", "rate"],
    }

    TECHNICAL_RESERVED: set[str] = {
        "def", "class", "import", "return", "True", "False", "None",
        "int", "str", "float", "list", "dict", "bool", "json", "http", "https",
    }

    @property
    def name(self) -> str:
        return "synonym"

    @property
    def description(self) -> str:
        return "Substitute selected non-technical words with appropriate synonyms."

    def apply(self, task: dict[str, Any], seed: int | None = None) -> dict[str, Any]:
        prompt, _ = _extract_prompt(task)
        if not prompt:
            return task.copy()

        rng = random.Random(seed if seed is not None else 42)

        def replace_word(match: re.Match[str]) -> str:
            word = match.group(0)
            lower_word = word.lower()
            if lower_word in self.TECHNICAL_RESERVED:
                return word
            if lower_word in self.SYNONYM_MAP:
                synonyms = self.SYNONYM_MAP[lower_word]
                chosen = rng.choice(synonyms)
                if word.istitle():
                    return chosen.capitalize()
                elif word.isupper():
                    return chosen.upper()
                return chosen
            return word

        pattern = r"\b[a-zA-Z]+\b"
        new_prompt = re.sub(pattern, replace_word, prompt)

        return _create_perturbed_task(task, new_prompt, self.name, seed)


class PromptWrapperPerturbationStrategy(PerturbationStrategy):
    """Wrap task prompts in neutral instruction containers."""

    WRAPPERS: list[str] = [
        "Please answer the following task:\n\n{prompt}",
        "Complete the task below:\n\n{prompt}",
        "Kindly review and fulfill the request below:\n\n{prompt}",
        "Task Instructions:\n{prompt}",
        "Here is the task to complete:\n\n{prompt}",
    ]

    @property
    def name(self) -> str:
        return "prompt_wrapper"

    @property
    def description(self) -> str:
        return "Wrap prompt text with neutral task instruction wrappers."

    def apply(self, task: dict[str, Any], seed: int | None = None) -> dict[str, Any]:
        prompt, _ = _extract_prompt(task)
        if not prompt:
            return task.copy()

        rng = random.Random(seed if seed is not None else 42)
        wrapper_template = rng.choice(self.WRAPPERS)
        new_prompt = wrapper_template.format(prompt=prompt)

        return _create_perturbed_task(task, new_prompt, self.name, seed)
