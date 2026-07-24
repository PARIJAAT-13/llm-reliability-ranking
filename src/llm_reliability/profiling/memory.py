"""Per-execution memory profiling.

Captures RAM and optional GPU VRAM before and after execution.
Gracefully degrades when psutil or torch CUDA are unavailable.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class MemoryProfiler:
    """Captures memory usage deltas around experiment execution.

    Usage::

        profiler = MemoryProfiler()
        profiler.snapshot_before()
        ...  # execution
        profiler.snapshot_after()
        delta = profiler.delta()
    """

    def __init__(self) -> None:
        self._ram_before: float | None = None
        self._ram_after: float | None = None
        self._vram_before: float | None = None
        self._vram_after: float | None = None
        self._peak_ram: float | None = None
        self._peak_vram: float | None = None

    @staticmethod
    def _get_ram_gb() -> float | None:
        """Return current process RAM usage in GB, or None."""
        try:
            import os

            import psutil

            proc = psutil.Process(os.getpid())
            return round(proc.memory_info().rss / (1024**3), 3)
        except Exception:
            return None

    @staticmethod
    def _get_vram_gb() -> float | None:
        """Return current GPU VRAM usage in GB, or None."""
        try:
            import torch

            if torch.cuda.is_available():
                return round(torch.cuda.memory_allocated() / (1024**3), 3)
        except Exception:
            pass
        return None

    @staticmethod
    def _get_peak_vram_gb() -> float | None:
        """Return peak GPU VRAM since last reset, or None."""
        try:
            import torch

            if torch.cuda.is_available():
                return round(torch.cuda.max_memory_allocated() / (1024**3), 3)
        except Exception:
            pass
        return None

    def snapshot_before(self) -> None:
        """Capture memory state before execution."""
        self._ram_before = self._get_ram_gb()
        self._vram_before = self._get_vram_gb()
        logger.debug("Memory before — RAM: %s GB, VRAM: %s GB", self._ram_before, self._vram_before)

    def snapshot_after(self) -> None:
        """Capture memory state after execution and compute peaks."""
        self._ram_after = self._get_ram_gb()
        self._vram_after = self._get_vram_gb()
        self._peak_vram = self._get_peak_vram_gb()
        if self._ram_before is not None and self._ram_after is not None:
            self._peak_ram = max(self._ram_before, self._ram_after)
        logger.debug("Memory after — RAM: %s GB, VRAM: %s GB", self._ram_after, self._vram_after)

    def delta(self) -> dict[str, Any]:
        """Return a snapshot of memory deltas.

        Returns
        -------
        dict
            Keys: ``ram_before_gb``, ``ram_after_gb``, ``ram_delta_gb``,
            ``vram_before_gb``, ``vram_after_gb``, ``vram_delta_gb``,
            ``peak_ram_gb``, ``peak_vram_gb``.
            Missing/unavailable values are ``None``.
        """

        def _diff(a: float | None, b: float | None) -> float | None:
            if a is not None and b is not None:
                return round(b - a, 3)
            return None

        return {
            "ram_before_gb": self._ram_before,
            "ram_after_gb": self._ram_after,
            "ram_delta_gb": _diff(self._ram_before, self._ram_after),
            "vram_before_gb": self._vram_before,
            "vram_after_gb": self._vram_after,
            "vram_delta_gb": _diff(self._vram_before, self._vram_after),
            "peak_ram_gb": self._peak_ram,
            "peak_vram_gb": self._peak_vram,
        }
