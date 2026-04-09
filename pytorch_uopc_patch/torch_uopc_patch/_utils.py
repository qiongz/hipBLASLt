from __future__ import annotations

import os
import sys


_TRUE_VALUES = {"1", "on", "true", "yes"}


def env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in _TRUE_VALUES


def env_csv(name: str, default: str = "") -> tuple[str, ...]:
    raw = os.environ.get(name, default)
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def log(message: str) -> None:
    print(f"[torch_uopc_patch] {message}", file=sys.stderr)


def warn(message: str) -> None:
    log(message)


def debug_enabled() -> bool:
    return env_flag("TORCH_UOPC_PATCH_DEBUG")


def debug(message: str) -> None:
    if debug_enabled():
        log(message)


def fail_open_enabled() -> bool:
    if env_flag("TORCH_UOPC_STRICT_AUTO_ENABLE"):
        return False
    return env_flag("TORCH_UOPC_FAIL_OPEN", "1")


def torch_version_supported(version: str) -> bool:
    prefixes = env_csv("TORCH_UOPC_SUPPORTED_TORCH_PREFIXES", "2.9")
    if not prefixes:
        return True
    return any(version.startswith(prefix) for prefix in prefixes)
