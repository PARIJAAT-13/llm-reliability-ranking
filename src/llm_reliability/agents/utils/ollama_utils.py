"""
Ollama API & Runtime Diagnostics Utility.

Provides pre-execution validation, model tag listing, memory estimation,
automatic model unloading (keep_alive=0), and user-friendly error formatting.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"


def normalize_ollama_url(base_url: str) -> str:
    """Normalize base URL to host:port format without /v1 trailing path."""
    url = base_url.strip().rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3]
    return url


def check_ollama_server(base_url: str = DEFAULT_OLLAMA_BASE_URL, timeout: float = 3.0) -> tuple[bool, str]:
    """Check if the local Ollama server is reachable."""
    host_url = normalize_ollama_url(base_url)
    tags_url = f"{host_url}/api/tags"
    try:
        req = urllib.request.Request(tags_url, headers={"User-Agent": "llm-reliability-ranking"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                return True, "Ollama server is running."
            return False, f"Ollama server returned HTTP status {resp.status}."
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", str(exc))
        return False, f"Ollama server not reachable at {host_url}: {reason}"
    except Exception as exc:  # noqa: BLE001
        return False, f"Connection error: {exc}"


def list_local_models(base_url: str = DEFAULT_OLLAMA_BASE_URL, timeout: float = 3.0) -> list[str]:
    """Retrieve list of installed models from local Ollama via GET /api/tags."""
    host_url = normalize_ollama_url(base_url)
    tags_url = f"{host_url}/api/tags"
    try:
        req = urllib.request.Request(tags_url, headers={"User-Agent": "llm-reliability-ranking"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                payload = json.loads(resp.read().decode("utf-8"))
                models_list = payload.get("models", [])
                names = []
                for m in models_list:
                    name = m.get("name") or m.get("model")
                    if name:
                        names.append(name)
                return names
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to query Ollama /api/tags at %s: %s", tags_url, exc)
    return []


def model_matches(requested: str, available_models: list[str]) -> bool:
    """Check if requested model matches an available model tag."""
    req = requested.lower().strip()
    avail_lower = [m.lower().strip() for m in available_models]
    if req in avail_lower:
        return True
    if f"{req}:latest" in avail_lower:
        return True
    if req.endswith(":latest") and req[:-7] in avail_lower:
        return True
    for avail in avail_lower:
        if avail.endswith(":latest") and avail[:-7] == req:
            return True
    return False


def validate_models_exist(models: list[str], base_url: str = DEFAULT_OLLAMA_BASE_URL) -> tuple[bool, list[str]]:
    """Validate that Ollama server is running and all requested models exist locally."""
    server_ok, msg = check_ollama_server(base_url)
    if not server_ok:
        return False, [
            f"\n[CRITICAL ERROR] Ollama server is unavailable!\n{msg}\n"
            f"Suggestions:\n"
            f"  • Ensure Ollama is installed and running (`ollama serve`)\n"
            f"  • Verify base_url setting: {base_url}\n"
        ]

    installed = list_local_models(base_url)
    missing = []
    for m in models:
        # Extract model name if compound string provider:model or dict
        model_name = m
        if isinstance(m, str) and m.startswith("ollama:"):
            model_name = m.split(":", 1)[1]
        elif isinstance(m, dict):
            model_name = m.get("model", "")

        if model_name and not model_matches(model_name, installed):
            missing.append(model_name)

    if missing:
        err_msg = format_model_not_found_error(missing, installed)
        return False, [err_msg]

    return True, []


def estimate_model_memory(model_name: str, base_url: str = DEFAULT_OLLAMA_BASE_URL, timeout: float = 3.0) -> dict[str, Any]:
    """Query model details from POST /api/show to estimate size and memory requirements."""
    host_url = normalize_ollama_url(base_url)
    show_url = f"{host_url}/api/show"
    payload = json.dumps({"name": model_name}).encode("utf-8")
    try:
        req = urllib.request.Request(
            show_url,
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "llm-reliability-ranking"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                details = data.get("details", {})
                size_bytes = data.get("size", 0)
                size_gb = round(size_bytes / (1024 ** 3), 2) if size_bytes else None
                return {
                    "model": model_name,
                    "size_gb": size_gb,
                    "family": details.get("family"),
                    "parameter_size": details.get("parameter_size"),
                    "quantization_level": details.get("quantization_level"),
                }
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to query /api/show for model %s: %s", model_name, exc)
    return {"model": model_name, "size_gb": None}


def get_available_memory_gb() -> float | None:
    """Get system available memory in GB."""
    try:
        import psutil  # noqa: PLC0415
        return round(psutil.virtual_memory().available / (1024 ** 3), 2)
    except ImportError:
        pass

    # Windows OS fallback
    if sys.platform == "win32":
        try:
            import ctypes  # noqa: PLC0415

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return round(stat.ullAvailPhys / (1024 ** 3), 2)
        except Exception:  # noqa: BLE001
            pass

    # Linux /proc/meminfo fallback
    if os.path.exists("/proc/meminfo"):
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        kb = int(line.split()[1])
                        return round(kb / (1024 ** 2), 2)
        except Exception:  # noqa: BLE001
            pass

    return None


def unload_ollama_model(model_name: str, base_url: str = DEFAULT_OLLAMA_BASE_URL, timeout: float = 5.0) -> bool:
    """Unload model from Ollama VRAM/RAM by sending keep_alive: 0 to /api/generate."""
    host_url = normalize_ollama_url(base_url)
    gen_url = f"{host_url}/api/generate"
    payload = json.dumps({"model": model_name, "keep_alive": 0}).encode("utf-8")
    try:
        req = urllib.request.Request(
            gen_url,
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "llm-reliability-ranking"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status in (200, 204):
                logger.info("Unloaded Ollama model '%s' from RAM/VRAM (keep_alive=0).", model_name)
                return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not automatically unload Ollama model '%s': %s", model_name, exc)
    return False


def format_model_not_found_error(missing_models: list[str], installed_models: list[str]) -> str:
    """Format user-friendly model not found error with actionable suggestions."""
    installed_str = ", ".join(sorted(installed_models)) if installed_models else "None"
    lines = [
        "\n[OLLAMA CONFIGURATION ERROR] Model(s) not found on local Ollama server!",
        f"Missing model(s): {', '.join(missing_models)}",
        f"Installed model(s): {installed_str}",
        "\nSuggestions:",
    ]
    for m in missing_models:
        lines.append(f"  • Run: `ollama pull {m}` to download this model")
    lines.extend([
        "  • Or try using an installed model like phi3:mini or tinyllama",
        "  • Verify Ollama is running (`ollama list`)",
    ])
    return "\n".join(lines)


def format_memory_error(model_name: str, req_gb: float | None, avail_gb: float | None) -> str:
    """Format user-friendly memory error with actionable suggestions."""
    lines = [
        f"\n[OLLAMA MEMORY ERROR] Unable to load model '{model_name}' due to insufficient system memory.",
    ]
    if req_gb:
        lines.append(f"Estimated required memory: {req_gb:.1f} GB")
    if avail_gb:
        lines.append(f"Available system memory : {avail_gb:.1f} GB")

    lines.extend([
        "\nSuggestions:",
        "  • Close other applications to free system RAM and GPU VRAM",
        f"  • Use a smaller quantized model (e.g. {model_name.split(':')[0]}:1.8b or phi3:mini)",
        "  • Use a smaller context window size (max_tokens)",
    ])
    return "\n".join(lines)
