"""
Purpose
-------
Provide a global registry for all benchmark adapters.

Responsibilities
----------------
- Register adapters by name (direct call or decorator)
- Unregister adapters
- Retrieve adapters by name safely
- List all available adapters
- Prevent duplicate registrations
- Auto-discover adapters from a package
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
from collections.abc import Callable

from llm_reliability.benchmarks.adapters.base_adapter import BaseBenchmarkAdapter


class BenchmarkRegistry:
    """Registry to manage and discover benchmark adapters dynamically.

    Can auto-discover plugins by scanning a package directory and/or
    by inspecting already-imported modules for ``BaseBenchmarkAdapter``
    subclasses.  Safe to call multiple times — the registry recovers
    even after being cleared (e.g. by test fixtures).
    """

    _adapters: dict[str, type[BaseBenchmarkAdapter]] = {}
    _initialised: bool = False
    _discovered_module_names: set[str] = set()

    @classmethod
    def register(
        cls,
        name: str,
        adapter_cls: type[BaseBenchmarkAdapter] | None = None,
    ) -> (
        type[BaseBenchmarkAdapter]
        | Callable[[type[BaseBenchmarkAdapter]], type[BaseBenchmarkAdapter]]
    ):
        """Register a benchmark adapter by name.

        Can be used as a direct call::

            BenchmarkRegistry.register("MyBench", MyAdapter)

        Or as a decorator::

            @BenchmarkRegistry.register("MyBench")
            class MyAdapter(BaseBenchmarkAdapter):
                ...
        """

        def _do_register(adapter_cls: type[BaseBenchmarkAdapter]) -> type[BaseBenchmarkAdapter]:
            if name in cls._adapters:
                raise ValueError(f"Adapter '{name}' is already registered.")
            if not issubclass(adapter_cls, BaseBenchmarkAdapter):
                raise TypeError("adapter_cls must be a subclass of BaseBenchmarkAdapter.")
            cls._adapters[name] = adapter_cls
            # Stash every name so _scan_imported_modules can recover
            # registrations after the registry dict is cleared.
            if not hasattr(adapter_cls, "_benchmark_registry_names"):
                adapter_cls._benchmark_registry_names = []  # type: ignore[attr-defined]
            adapter_cls._benchmark_registry_names.append(name)  # type: ignore[attr-defined]
            return adapter_cls

        if adapter_cls is not None:
            return _do_register(adapter_cls)
        return _do_register

    @classmethod
    def unregister(cls, name: str) -> None:
        """Unregister a benchmark adapter by name."""
        if name not in cls._adapters:
            raise ValueError(f"Adapter '{name}' is not registered.")
        del cls._adapters[name]

    @classmethod
    def get(cls, name: str) -> type[BaseBenchmarkAdapter]:
        """Retrieve a registered benchmark adapter by name."""
        cls._ensure_discovered()
        if name not in cls._adapters:
            raise ValueError(f"Adapter '{name}' not found in registry.")
        return cls._adapters[name]

    @classmethod
    def list(cls) -> list[str]:
        """List all available benchmark adapter names."""
        cls._ensure_discovered()
        return sorted(cls._adapters.keys())

    @classmethod
    def exists(cls, name: str) -> bool:
        """Check if an adapter is registered."""
        cls._ensure_discovered()
        return name in cls._adapters

    @classmethod
    def _ensure_discovered(cls) -> None:
        """If the registry is empty, try to restore adapters from
        already-imported modules.  This handles the case where a test
        fixture (e.g. ``clean_registry``) cleared the ``_adapters``
        dict after the initial discovery."""
        if not cls._adapters and cls._initialised:
            cls._scan_imported_modules()

    @classmethod
    def _scan_imported_modules(cls) -> None:
        """Scan discovered modules for ``BaseBenchmarkAdapter`` subclasses
        and register any that are not already in the registry.

        Only modules that were originally imported by ``discover()``
        are scanned.  This prevents test fixtures or user code that
        clears the registry from accidentally re-registering
        dynamically created test adapter classes.
        """
        if not cls._discovered_module_names:
            return
        for modname in cls._discovered_module_names:
            module = sys.modules.get(modname)
            if module is None:
                continue
            for attr_name in dir(module):
                attr = getattr(module, attr_name, None)
                if not isinstance(attr, type):
                    continue
                if attr is BaseBenchmarkAdapter:
                    continue
                try:
                    is_sub = issubclass(attr, BaseBenchmarkAdapter)
                except TypeError:
                    continue
                if not is_sub:
                    continue
                # Check if this class is already registered
                already = any(v is attr for v in cls._adapters.values())
                if not already:
                    names = getattr(attr, "_benchmark_registry_names", None) or [attr.__name__]
                    for n in names:
                        if n not in cls._adapters:
                            cls._adapters[n] = attr

    @classmethod
    def discover(cls, package: object | None = None) -> None:
        """Auto-discover and import all benchmark adapter modules.

        Scans the given package (defaults to
        ``llm_reliability.benchmarks.adapters``) for Python modules,
        imports each one, and thereby triggers any module-level
        registration calls or decorators.

        New adapters added to the package are automatically discovered
        without requiring edits to any core framework file.

        Safe to call multiple times — modules already imported are
        skipped.  After discovery the registry scans all loaded
        modules so that any cleared registrations are restored.
        """
        if package is None:
            from llm_reliability.benchmarks import adapters as _adapters_pkg

            package = _adapters_pkg

        package_name = package.__name__ if hasattr(package, "__name__") else str(package)

        for _importer, modname, _ispkg in pkgutil.iter_modules(
            package.__path__,  # type: ignore[attr-defined]
            prefix=f"{package_name}.",
        ):
            if modname not in sys.modules:
                importlib.import_module(modname)
            cls._discovered_module_names.add(modname)

        cls._initialised = True
        cls._scan_imported_modules()
