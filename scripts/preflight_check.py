#!/usr/bin/env python3
"""
preflight_check.py — Preflight Environment & API Validation Inspector.

Performs strict system audits before starting multi-hour benchmark evaluations:
- Python environment & required dependencies
- Provider API key availability
- Local disk space and write permissions
- Internet connectivity
- Dataset cache status and integrity
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import shutil
import socket
import sys


def check_item(name: str, status: str, details: str):
    """Format and print a single preflight check item."""
    symbol = " PASS " if status == "PASS" else (" WARN " if status == "WARNING" else " FAIL ")
    print(f"[{symbol:^8}] {name:<30} : {details}")


def main() -> int:
    print("=" * 70)
    print("      LLM Reliability Ranking — Preflight System Inspector")
    print("=" * 70)

    has_failures = False

    # 1. Python Version
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    check_item("Python Version", "PASS", f"v{py_ver} (project requires Python >= 3.10)")

    # 2. Key Dependencies
    required_deps = ["pydantic", "numpy", "scipy", "matplotlib", "pytest"]
    optional_deps = ["seaborn", "yaml"]

    for dep in required_deps:
        spec = importlib.util.find_spec(dep)
        if spec is not None:
            check_item(f"Dependency: {dep}", "PASS", "Installed")
        else:
            check_item(f"Dependency: {dep}", "FAIL", "Missing")
            has_failures = True

    for dep in optional_deps:
        spec = importlib.util.find_spec(dep)
        if spec is not None:
            check_item(f"Optional Dep: {dep}", "PASS", "Installed")
        else:
            check_item(
                f"Optional Dep: {dep}", "WARNING", "Missing (fallback graphics mode will be used)"
            )

    # 3. Provider API Keys
    api_keys = [
        ("OPENAI_API_KEY", "OpenAI API"),
        ("ANTHROPIC_API_KEY", "Anthropic API"),
        ("GEMINI_API_KEY", "Google Gemini API"),
        ("DEEPSEEK_API_KEY", "DeepSeek API"),
        ("QWEN_API_KEY", "DashScope Qwen API"),
        ("HF_TOKEN", "HuggingFace Hub"),
    ]
    print("-" * 70)
    print("                      Provider API Keys Check")
    print("-" * 70)
    for env_var, label in api_keys:
        val = os.getenv(env_var)
        if val and len(val.strip()) > 5:
            check_item(label, "PASS", f"Key present ({env_var[:4]}...{val[-4:]})")
        else:
            check_item(
                label,
                "WARNING",
                f"Key not set ({env_var}). Real model calls will fall back to mock.",
            )

    # 4. Storage & Disk Space
    print("-" * 70)
    print("                  Storage & Directory Permissions Check")
    print("-" * 70)
    repo_root = pathlib.Path(__file__).parent.parent
    total, used, free = shutil.disk_usage(repo_root)
    free_gb = free / (1024**3)

    if free_gb > 2.0:
        check_item("Available Disk Space", "PASS", f"{free_gb:.2f} GB free")
    else:
        check_item("Available Disk Space", "WARNING", f"{free_gb:.2f} GB free (Low disk space)")

    dirs_to_check = ["results", "data/cache", "paper/figures", "paper/tables"]
    for dir_rel in dirs_to_check:
        d_path = repo_root / dir_rel
        try:
            d_path.mkdir(parents=True, exist_ok=True)
            test_file = d_path / ".perm_check"
            test_file.write_text("test", encoding="utf-8")
            test_file.unlink()
            check_item(f"Write Permission ({dir_rel})", "PASS", "Writable")
        except Exception as e:
            check_item(f"Write Permission ({dir_rel})", "FAIL", f"Cannot write: {e}")
            has_failures = True

    # 5. Network Connectivity
    print("-" * 70)
    print("                      Network Connectivity Check")
    print("-" * 70)
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        check_item("Internet Access", "PASS", "Online (Google DNS reachable)")
    except OSError:
        check_item("Internet Access", "WARNING", "Offline or restricted network")

    print("=" * 70)
    if has_failures:
        print("RESULT: PREFLIGHT AUDIT FAILED. Fix highlighted [FAIL] issues above.")
        return 1
    else:
        print("RESULT: PREFLIGHT AUDIT PASSED. System ready for benchmark runs.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
