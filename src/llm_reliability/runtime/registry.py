"""
RuntimeRegistry — plugin-based registry for ``Runtime`` implementations.

Allows agent/runtime classes to register themselves by name via
direct call, decorator, or automatic discovery.  The registry is
used by ``AgentFactory`` instead of the previously hard-coded
``_REGISTRY`` dict, enabling third-party runtimes to be added
without editing framework core files.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
from collections.abc import Callable

from llm_reliability.runtime.interface import Runtime


class RuntimeRegistry:
    """Registry for ``Runtime`` implementations with discovery support."""

    _runtimes: dict[str, type[Runtime]] = {}
    _initialised: bool = False
    _discovered_module_names: set[str] = set()

    @classmethod
    def register(
        cls,
        name: str,
        runtime_cls: type[Runtime] | None = None,
    ) -> type[Runtime] | Callable[[type[Runtime]], type[Runtime]]:
        """Register a runtime by name.

        Direct call::

            RuntimeRegistry.register("my_runtime", MyRuntime)

        Decorator::

            @RuntimeRegistry.register("my_runtime")
            class MyRuntime(Runtime):
                ...
        """

        def _do_register(runtime_cls: type[Runtime]) -> type[Runtime]:
            if name in cls._runtimes:
                raise ValueError(f"Runtime '{name}' is already registered.")
            if not issubclass(runtime_cls, Runtime):
                raise TypeError("runtime_cls must be a subclass of Runtime.")
            cls._runtimes[name] = runtime_cls
            if not hasattr(runtime_cls, "_runtime_registry_names"):
                runtime_cls._runtime_registry_names = []  # type: ignore[attr-defined]
            runtime_cls._runtime_registry_names.append(name)  # type: ignore[attr-defined]
            return runtime_cls

        if runtime_cls is not None:
            return _do_register(runtime_cls)
        return _do_register

    @classmethod
    def unregister(cls, name: str) -> None:
        """Unregister a runtime by name."""
        if name not in cls._runtimes:
            raise ValueError(f"Runtime '{name}' is not registered.")
        del cls._runtimes[name]

    @classmethod
    def get(cls, name: str) -> type[Runtime]:
        """Retrieve a registered runtime class by name."""
        cls._ensure_discovered()
        if name not in cls._runtimes:
            raise ValueError(f"Runtime '{name}' not found in registry.")
        return cls._runtimes[name]

    @classmethod
    def list(cls) -> list[str]:
        """Return all registered runtime names, sorted."""
        cls._ensure_discovered()
        return sorted(cls._runtimes.keys())

    @classmethod
    def exists(cls, name: str) -> bool:
        """Return ``True`` if a runtime is registered under *name*."""
        cls._ensure_discovered()
        return name in cls._runtimes

    @classmethod
    def _ensure_discovered(cls) -> None:
        """Restore registrations after registry was cleared (e.g. by tests)."""
        if not cls._runtimes and cls._initialised:
            cls._scan_imported_modules()

    @classmethod
    def _scan_imported_modules(cls) -> None:
        """Re-register ``Runtime`` subclasses from discovered modules."""
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
                if attr is Runtime:
                    continue
                try:
                    is_sub = issubclass(attr, Runtime)
                except TypeError:
                    continue
                if not is_sub:
                    continue
                already = any(v is attr for v in cls._runtimes.values())
                if not already:
                    names = getattr(attr, "_runtime_registry_names", None) or [attr.__name__]
                    for n in names:
                        if n not in cls._runtimes:
                            cls._runtimes[n] = attr

    @classmethod
    def discover(cls, package: object | None = None) -> None:
        """Auto-discover and import all runtime modules in *package*.

        Defaults to ``llm_reliability.agents``.
        """
        if package is None:
            import llm_reliability.agents as _agents_pkg

            package = _agents_pkg

        pkg_name = package.__name__ if hasattr(package, "__name__") else str(package)

        for _importer, modname, _ispkg in pkgutil.iter_modules(
            package.__path__,  # type: ignore[attr-defined]
            prefix=f"{pkg_name}.",
        ):
            if modname not in sys.modules:
                importlib.import_module(modname)
            cls._discovered_module_names.add(modname)

        cls._initialised = True
        cls._scan_imported_modules()
