"""Tests for benchmark plugin discovery and decorator registration."""

import importlib
import sys
import tempfile
from pathlib import Path

import pytest

from llm_reliability.benchmarks import BenchmarkPlugin
from llm_reliability.benchmarks.adapters.base_adapter import BaseBenchmarkAdapter
from llm_reliability.benchmarks.adapters.registry import BenchmarkRegistry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_discovery_state():
    """Reset the discovery tracking set so tests are isolated."""
    saved = BenchmarkRegistry._discovered_module_names.copy()
    yield
    BenchmarkRegistry._discovered_module_names.clear()
    BenchmarkRegistry._discovered_module_names.update(saved)
    # Don't clear _adapters — it holds the real adapters loaded at import time.


# ---------------------------------------------------------------------------
# Helpers: a minimal concrete adapter for testing
# ---------------------------------------------------------------------------


class MiniAdapter(BaseBenchmarkAdapter):
    def _load_tasks(self):
        self._tasks = {"t1": {"id": "t1"}}

    def run(self, agent, task):
        pass

    def evaluate(self, execution):
        pass


class OtherMiniAdapter(BaseBenchmarkAdapter):
    def _load_tasks(self):
        self._tasks = {"t2": {"id": "t2"}}

    def run(self, agent, task):
        pass

    def evaluate(self, execution):
        pass


# ---------------------------------------------------------------------------
# Plugin interface
# ---------------------------------------------------------------------------


class TestBenchmarkPlugin:
    def test_plugin_is_abstract(self):
        """BenchmarkPlugin cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BenchmarkPlugin()

    def test_base_adapter_is_plugin(self):
        """BaseBenchmarkAdapter is a subclass of BenchmarkPlugin."""
        assert issubclass(BaseBenchmarkAdapter, BenchmarkPlugin)

    def test_concrete_adapter_is_plugin(self):
        """Concrete adapters are recognised as BenchmarkPlugin instances."""
        assert issubclass(MiniAdapter, BenchmarkPlugin)


# ---------------------------------------------------------------------------
# Decorator registration
# ---------------------------------------------------------------------------


class TestDecoratorRegistration:
    def test_decorator_registers_adapter(self):
        BenchmarkRegistry._adapters.clear()

        @BenchmarkRegistry.register("DecoratedBench")
        class DecoratedBench(BaseBenchmarkAdapter):
            def _load_tasks(self):
                self._tasks = {}

            def run(self, agent, task):
                pass

            def evaluate(self, execution):
                pass

        assert BenchmarkRegistry.exists("DecoratedBench")
        assert BenchmarkRegistry.get("DecoratedBench") is DecoratedBench

    def test_decorator_rejects_duplicate(self):
        with pytest.raises(ValueError, match="is already registered"):
            BenchmarkRegistry.register("DecoratedBench", MiniAdapter)

    def test_direct_call_still_works(self):
        BenchmarkRegistry._adapters.clear()
        BenchmarkRegistry.register("DirectBench", MiniAdapter)
        assert BenchmarkRegistry.exists("DirectBench")


# ---------------------------------------------------------------------------
# Duplicate registration
# ---------------------------------------------------------------------------


class TestDuplicateRegistration:
    def test_duplicate_name_raises(self):
        BenchmarkRegistry._adapters.clear()
        BenchmarkRegistry.register("dup_test", MiniAdapter)
        with pytest.raises(ValueError, match="is already registered"):
            BenchmarkRegistry.register("dup_test", OtherMiniAdapter)

    def test_duplicate_name_decorator_raises(self):
        BenchmarkRegistry._adapters.clear()
        BenchmarkRegistry.register("dup_deco", MiniAdapter)

        with pytest.raises(ValueError, match="is already registered"):

            @BenchmarkRegistry.register("dup_deco")
            class _Dup(BaseBenchmarkAdapter):
                def _load_tasks(self):
                    pass

                def run(self, agent, task):
                    pass

                def evaluate(self, execution):
                    pass


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


class TestDiscovery:
    def test_discover_imports_new_module(self):
        """A new module added to the adapters package is discovered."""
        import llm_reliability.benchmarks.adapters as adapters_pkg

        adapters_dir = Path(adapters_pkg.__file__).parent
        module_path = adapters_dir / "_test_discovery_plugin.py"
        modname = "llm_reliability.benchmarks.adapters._test_discovery_plugin"

        module_code = """
from llm_reliability.benchmarks.adapters.base_adapter import BaseBenchmarkAdapter
from llm_reliability.benchmarks.adapters.registry import BenchmarkRegistry

class DiscoveredBench(BaseBenchmarkAdapter):
    def _load_tasks(self):
        self._tasks = {"d1": {"id": "d1"}}
    def run(self, agent, task):
        pass
    def evaluate(self, execution):
        pass

BenchmarkRegistry.register("DiscoveredBench", DiscoveredBench)
"""
        try:
            module_path.write_text(module_code, encoding="utf-8")

            # Discover again — our module is new and not yet imported
            BenchmarkRegistry.discover()

            # It should now be registered
            assert BenchmarkRegistry.exists("DiscoveredBench")
        finally:
            # Cleanup
            if module_path.exists():
                module_path.unlink()
            if modname in sys.modules:
                del sys.modules[modname]
            BenchmarkRegistry._discovered_module_names.discard(modname)
            if BenchmarkRegistry.exists("DiscoveredBench"):
                BenchmarkRegistry._adapters.pop("DiscoveredBench", None)

    def test_discover_scans_already_imported(self):
        """After a registry clear, discover re-registers adapters."""
        # First ensure the real adapters are present (they were imported at session start)
        BenchmarkRegistry.discover()
        assert BenchmarkRegistry.exists("AgentBoard")
        assert BenchmarkRegistry.exists("MMLU")

        # Simulate test fixture clearing the registry
        saved = BenchmarkRegistry._adapters.copy()
        BenchmarkRegistry._adapters.clear()

        # _ensure_discovered should restore them
        BenchmarkRegistry._ensure_discovered()

        assert BenchmarkRegistry.exists("AgentBoard")
        assert BenchmarkRegistry.exists("MMLU")

        # Restore for other tests
        BenchmarkRegistry._adapters.update(saved)

    def test_discover_no_double_registration(self):
        """Calling discover multiple times does not error."""
        BenchmarkRegistry.discover()
        BenchmarkRegistry.discover()
        BenchmarkRegistry.discover()
        # We just need it to not raise


# ---------------------------------------------------------------------------
# Loading multiple plugins
# ---------------------------------------------------------------------------


class TestMultiplePlugins:
    def test_multiple_adapters_loaded(self):
        adapters = BenchmarkRegistry.list()
        assert len(adapters) >= 12  # All built-in adapters
        assert "AgentBoard" in adapters
        assert "MMLU" in adapters
        assert "HumanEval" in adapters


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_existing_adapters_still_discoverable(self):
        """All built-in adapters are registered and retrievable."""
        expected = [
            "AgentBoard",
            "ARC",
            "GAIA",
            "GSM8K",
            "HellaSwag",
            "HumanEval",
            "MBPP",
            "MMLU",
            "PIQA",
            "SWEBenchLite",
            "TruthfulQA",
            "Winogrande",
        ]
        for name in expected:
            assert BenchmarkRegistry.exists(name), f"Missing: {name}"

    def test_existing_api_get_works(self):
        adapter = BenchmarkRegistry.get("AgentBoard")
        from llm_reliability.benchmarks.adapters.agentboard_adapter import (
            AgentBoardAdapter,
        )

        assert adapter is AgentBoardAdapter

    def test_existing_api_list_works(self):
        adapters = BenchmarkRegistry.list()
        assert isinstance(adapters, list)
        assert adapters == sorted(adapters)

    def test_module_imports_still_work(self):
        """Importing adapters directly from the package still works."""
        from llm_reliability.benchmarks.adapters import (
            AgentBoardAdapter,
            ARCAdapter,
            GAIAAdapter,
        )

        assert issubclass(AgentBoardAdapter, BaseBenchmarkAdapter)
        assert issubclass(ARCAdapter, BaseBenchmarkAdapter)
        assert issubclass(GAIAAdapter, BaseBenchmarkAdapter)
