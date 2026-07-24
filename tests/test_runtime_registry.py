"""Tests for RuntimeRegistry — registration, discovery, lookup."""

import pytest

from llm_reliability.runtime.interface import Runtime
from llm_reliability.runtime.registry import RuntimeRegistry


class _TestRuntime(Runtime):
    def initialize(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def metadata(self) -> dict:
        return {"name": "test"}

    def run(self, task: dict) -> str:
        return f"ran: {task}"


class _OtherRuntime(Runtime):
    def initialize(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def metadata(self) -> dict:
        return {"name": "other"}

    def run(self, task: dict) -> str:
        return f"other: {task}"


@pytest.fixture(autouse=True)
def clean_registry() -> None:
    RuntimeRegistry._runtimes.clear()
    RuntimeRegistry._initialised = True
    RuntimeRegistry._discovered_module_names.clear()
    yield
    RuntimeRegistry._runtimes.clear()
    RuntimeRegistry._initialised = False
    RuntimeRegistry._discovered_module_names.clear()


class TestRuntimeRegistry:
    def test_register_runtime(self):
        RuntimeRegistry.register("test_rt", _TestRuntime)
        assert RuntimeRegistry.exists("test_rt")

    def test_register_duplicate_raises(self):
        RuntimeRegistry.register("dup", _TestRuntime)
        with pytest.raises(ValueError, match="already registered"):
            RuntimeRegistry.register("dup", _OtherRuntime)

    def test_register_non_runtime_raises(self):
        with pytest.raises(TypeError, match="must be a subclass"):
            RuntimeRegistry.register("bad", object)  # type: ignore

    def test_get_runtime(self):
        RuntimeRegistry.register("test_rt", _TestRuntime)
        cls = RuntimeRegistry.get("test_rt")
        assert cls is _TestRuntime

    def test_get_missing_runtime_raises(self):
        with pytest.raises(ValueError, match="not found in registry"):
            RuntimeRegistry.get("nonexistent")

    def test_list_returns_sorted(self):
        RuntimeRegistry.register("z_runtime", _TestRuntime)
        RuntimeRegistry.register("a_runtime", _OtherRuntime)
        names = RuntimeRegistry.list()
        assert names == ["a_runtime", "z_runtime"]

    def test_list_empty_when_nothing_registered(self):
        assert RuntimeRegistry.list() == []

    def test_exists_true_for_registered(self):
        RuntimeRegistry.register("exists_rt", _TestRuntime)
        assert RuntimeRegistry.exists("exists_rt")

    def test_exists_false_for_unregistered(self):
        assert not RuntimeRegistry.exists("no_such_rt")

    def test_unregister_removes_runtime(self):
        RuntimeRegistry.register("removable", _TestRuntime)
        RuntimeRegistry.unregister("removable")
        assert not RuntimeRegistry.exists("removable")

    def test_unregister_missing_raises(self):
        with pytest.raises(ValueError, match="not registered"):
            RuntimeRegistry.unregister("not_there")

    def test_double_unregister_raises(self):
        RuntimeRegistry.register("double", _TestRuntime)
        RuntimeRegistry.unregister("double")
        with pytest.raises(ValueError, match="not registered"):
            RuntimeRegistry.unregister("double")

    def test_register_lifecycle(self):
        RuntimeRegistry.register("lifecycle", _TestRuntime)
        assert RuntimeRegistry.exists("lifecycle")
        RuntimeRegistry.unregister("lifecycle")
        assert not RuntimeRegistry.exists("lifecycle")

    def test_decorator_registration(self):
        @RuntimeRegistry.register("deco_rt")
        class DecoratedRuntime(Runtime):
            def initialize(self) -> None:
                pass

            def reset(self) -> None:
                pass

            def shutdown(self) -> None:
                pass

            def metadata(self) -> dict:
                return {"name": "deco"}

            def run(self, task: dict) -> str:
                return f"deco: {task}"

        assert RuntimeRegistry.exists("deco_rt")
        cls = RuntimeRegistry.get("deco_rt")
        assert cls is DecoratedRuntime

    def test_discover_discovers_known_runtimes(self):
        RuntimeRegistry.discover()
        names = RuntimeRegistry.list()
        assert "mock" in names
        assert "gpt" in names
        assert "anthropic" in names

    def test_get_returns_callable_class(self):
        RuntimeRegistry.register("callable", _TestRuntime)
        cls = RuntimeRegistry.get("callable")
        instance = cls()
        result = instance.run("hello")
        assert result == "ran: hello"
