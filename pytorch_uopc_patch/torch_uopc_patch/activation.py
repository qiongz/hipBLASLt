from __future__ import annotations

from ._utils import debug, env_flag, fail_open_enabled, torch_version_supported, warn


_AUTO_ENABLE_ATTEMPTED = False
_AUTO_ENABLE_BACKEND: str | None = None


def auto_enable_from_env() -> str | None:
    global _AUTO_ENABLE_ATTEMPTED, _AUTO_ENABLE_BACKEND

    if _AUTO_ENABLE_ATTEMPTED:
        return _AUTO_ENABLE_BACKEND
    _AUTO_ENABLE_ATTEMPTED = True

    if not env_flag("TORCH_UOPC_AUTO_ENABLE"):
        debug("auto-enable skipped because TORCH_UOPC_AUTO_ENABLE is disabled")
        return None

    try:
        import torch

        version = getattr(torch, "__version__", "unknown")
        if not torch_version_supported(version):
            message = (
                "torch version gate rejected "
                f"{version}; set TORCH_UOPC_SUPPORTED_TORCH_PREFIXES to override"
            )
            if fail_open_enabled():
                warn(message)
                return None
            raise RuntimeError(message)

        from .backend import enable

        _AUTO_ENABLE_BACKEND = enable()
        return _AUTO_ENABLE_BACKEND
    except Exception as exc:
        if fail_open_enabled():
            warn(f"auto-enable failed: {exc}")
            return None
        raise
