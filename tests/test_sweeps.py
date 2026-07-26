from __future__ import annotations

from typing import Any

import pytest


def _sweep(param_name: str, values: list[Any]) -> list[dict[str, Any]]:
    return [{param_name: v} for v in values]


def _grid_sweep(**params: list[Any]) -> list[dict[str, Any]]:
    import itertools

    keys = list(params.keys())
    combos = list(itertools.product(*params.values()))
    return [dict(zip(keys, combo)) for combo in combos]


def _to_configs(
    sweep_results: list[dict[str, Any]], base: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    base = base or {}
    return [{**base, **cfg} for cfg in sweep_results]


class TestBasicSweep:
    def test_sweep_over_seeds(self):
        configs = _to_configs(_sweep("seed", [0, 1, 2, 42, 123]))
        assert len(configs) == 5
        assert configs[0]["seed"] == 0
        assert configs[4]["seed"] == 123

    def test_sweep_returns_list_of_dicts(self):
        result = _sweep("seed", [1, 2, 3])
        assert isinstance(result, list)
        assert all(isinstance(c, dict) for c in result)

    def test_sweep_over_models(self):
        models = ["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet", "gemini-2.5-pro"]
        configs = _to_configs(_sweep("llm", models))
        assert len(configs) == 4
        assert configs[0]["llm"] == "gpt-4o"
        assert configs[2]["llm"] == "claude-3-5-sonnet"

    def test_sweep_over_benchmarks(self):
        benchmarks = ["agentboard", "swebench", "gaia"]
        configs = _to_configs(_sweep("benchmark", benchmarks))
        assert len(configs) == 3
        assert configs[1]["benchmark"] == "swebench"

    def test_sweep_over_repetitions(self):
        configs = _to_configs(_sweep("repetitions", [3, 5, 10]))
        assert len(configs) == 3
        assert configs[-1]["repetitions"] == 10

    def test_sweep_single_value(self):
        configs = _to_configs(_sweep("seed", [42]))
        assert len(configs) == 1
        assert configs[0]["seed"] == 42

    def test_sweep_with_base_config(self):
        base = {"benchmark": "agentboard", "agent": "mock_agent"}
        configs = _to_configs(_sweep("seed", [1, 2]), base)
        assert len(configs) == 2
        assert configs[0]["benchmark"] == "agentboard"
        assert configs[0]["agent"] == "mock_agent"
        assert configs[0]["seed"] == 1
        assert configs[1]["seed"] == 2


class TestGridSweep:
    def test_grid_over_two_params(self):
        configs = _grid_sweep(seed=[1, 42], llm=["gpt-4o", "claude-3-5-sonnet"])
        assert len(configs) == 4
        assert configs[0] == {"seed": 1, "llm": "gpt-4o"}
        assert configs[3] == {"seed": 42, "llm": "claude-3-5-sonnet"}

    def test_grid_over_three_params(self):
        configs = _grid_sweep(
            seed=[1, 2],
            llm=["gpt-4o"],
            benchmark=["agentboard", "swebench"],
        )
        assert len(configs) == 4
        for c in configs:
            assert c["llm"] == "gpt-4o"

    def test_grid_single_param_is_same_as_sweep(self):
        grid = _grid_sweep(seed=[1, 2, 3])
        simple = _sweep("seed", [1, 2, 3])
        assert len(grid) == len(simple)
        assert grid == simple

    def test_grid_with_base_config(self):
        base = {"agent": "mock_agent", "prompt_version": "v1"}
        configs = _to_configs(
            _grid_sweep(seed=[1], llm=["gpt-4o", "gpt-4o-mini"]),
            base,
        )
        assert len(configs) == 2
        for c in configs:
            assert c["agent"] == "mock_agent"
            assert c["prompt_version"] == "v1"

    def test_grid_on_models_and_seeds(self):
        configs = _grid_sweep(
            llm=["gpt-4o", "claude-3-5-sonnet", "gemini-2.5-pro"],
            seed=[0, 42, 123],
        )
        assert len(configs) == 9
        seed_values = {c["seed"] for c in configs}
        assert seed_values == {0, 42, 123}

    def test_grid_preserves_order(self):
        configs = _grid_sweep(a=[1, 2], b=["x", "y"])
        assert configs[0] == {"a": 1, "b": "x"}
        assert configs[1] == {"a": 1, "b": "y"}
        assert configs[2] == {"a": 2, "b": "x"}
        assert configs[3] == {"a": 2, "b": "y"}


class TestSweepEdgeCases:
    def test_empty_sweep_list(self):
        configs = _to_configs(_sweep("seed", []))
        assert len(configs) == 0

    def test_empty_grid(self):
        configs = _grid_sweep()
        assert len(configs) == 1
        assert configs[0] == {}

    def test_overwrite_base_with_sweep(self):
        base = {"seed": 0}
        configs = _to_configs(_sweep("seed", [1, 2]), base)
        assert configs[0]["seed"] == 1
        assert configs[1]["seed"] == 2

    def test_grid_with_multiple_agents(self):
        configs = _grid_sweep(
            agent=["mock_agent", "gpt_agent"],
            llm=["gpt-4o", "gpt-4o-mini"],
            seed=[42],
        )
        assert len(configs) == 4
        for c in configs:
            assert c["seed"] == 42

    def test_sweep_by_prompt_version(self):
        configs = _to_configs(_sweep("prompt_version", ["v1", "v2", "v3"]))
        assert len(configs) == 3
        assert configs[0]["prompt_version"] == "v1"

    def test_sweep_by_dataset_version(self):
        configs = _to_configs(_sweep("dataset_version", ["1.0", "2.0"]))
        assert len(configs) == 2
        assert configs[1]["dataset_version"] == "2.0"
