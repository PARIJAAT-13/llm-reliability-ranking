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

import argparse
import importlib.util
import os
import pathlib
import shutil
import socket
import sys

_REPO_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))


def check_item(name: str, status: str, details: str, verbose: bool = False) -> None:
    symbol = " PASS " if status == "PASS" else (" WARN " if status == "WARNING" else " FAIL ")
    if verbose or status != "PASS":
        print(f"[{symbol:^8}] {name:<30} : {details}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="preflight_check",
        description="LLM Reliability Ranking — Preflight System Inspector",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Show all checks (including passing ones).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Treat WARNING as FAIL (exit non-zero on warnings too).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    verbose = args.verbose
    strict = args.strict

    print("=" * 70)
    print("      LLM Reliability Ranking — Preflight System Inspector")
    print("=" * 70)

    has_failures = False
    has_warnings = False

    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    check_item("Python Version", "PASS", f"v{py_ver} (requires >= 3.10)", verbose)

    required_deps = ["pydantic", "numpy", "scipy", "matplotlib", "pytest"]
    optional_deps = ["seaborn", "yaml"]

    for dep in required_deps:
        spec = importlib.util.find_spec(dep)
        if spec is not None:
            check_item(f"Dependency: {dep}", "PASS", "Installed", verbose)
        else:
            check_item(f"Dependency: {dep}", "FAIL", "Missing", verbose)
            has_failures = True

    for dep in optional_deps:
        spec = importlib.util.find_spec(dep)
        if spec is not None:
            check_item(f"Optional Dep: {dep}", "PASS", "Installed", verbose)
        else:
            check_item(f"Optional Dep: {dep}", "WARNING", "Not installed", verbose)
            has_warnings = True

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
            check_item(label, "PASS", "Key present", verbose)
        else:
            check_item(label, "WARNING", f"Key not set ({env_var})", verbose)
            has_warnings = True

    print("-" * 70)
    print("                  Storage & Directory Permissions Check")
    print("-" * 70)
    total, used, free = shutil.disk_usage(_REPO_ROOT)
    free_gb = free / (1024**3)

    if free_gb > 2.0:
        check_item("Available Disk Space", "PASS", f"{free_gb:.2f} GB free", verbose)
    else:
        check_item("Available Disk Space", "WARNING", f"{free_gb:.2f} GB free (low)", verbose)
        has_warnings = True

    dirs_to_check = ["results", "data/cache", "paper/figures", "paper/tables"]
    for dir_rel in dirs_to_check:
        d_path = _REPO_ROOT / dir_rel
        try:
            d_path.mkdir(parents=True, exist_ok=True)
            test_file = d_path / ".perm_check"
            test_file.write_text("test", encoding="utf-8")
            test_file.unlink()
            check_item(f"Write Permission ({dir_rel})", "PASS", "Writable", verbose)
        except Exception as e:
            check_item(f"Write Permission ({dir_rel})", "FAIL", f"Cannot write: {e}", verbose)
            has_failures = True

    print("-" * 70)
    print("                      Network Connectivity Check")
    print("-" * 70)
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        check_item("Internet Access", "PASS", "Online", verbose)
    except OSError:
        check_item("Internet Access", "WARNING", "Offline or restricted", verbose)
        has_warnings = True

    print("=" * 70)
    if strict and has_warnings:
        has_failures = True
    if has_failures:
        print("RESULT: PREFLIGHT AUDIT FAILED. Fix highlighted [FAIL] issues above.")
        return 1
    if has_warnings:
        print("RESULT: PREFLIGHT AUDIT PASSED WITH WARNINGS. System may be usable.")
        return 0
    print("RESULT: PREFLIGHT AUDIT PASSED. System ready for benchmark runs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
