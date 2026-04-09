from __future__ import annotations

import contextlib
from functools import wraps
from typing import Any, Callable

import torch.distributed as dist

from . import _C
from ._utils import debug, env_flag
from .adapters import (
    enable_deepspeed_adapter as _enable_deepspeed_adapter,
    enable_fsdp_adapter as _enable_fsdp_adapter,
    enable_megatron_adapter as _enable_megatron_adapter,
)


BACKEND_NAME = "uopc_nccl"

_REGISTERED = False
_PATCHED_INIT = False
_PATCHED_NEW_GROUP = False
_PATCHED_FSDP = False
_PATCHED_DEEPSPEED = False
_PATCHED_MEGATRON = False
_ORIGINAL_INIT_PROCESS_GROUP: Callable[..., Any] | None = None
_ORIGINAL_NEW_GROUP: Callable[..., Any] | None = None


def _create_uopc_nccl_backend(store, rank: int, world_size: int, timeout):
    opts = dist.ProcessGroupNCCL.Options(
        is_high_priority_stream=env_flag("TORCH_UOPC_NCCL_HIGH_PRIORITY_STREAM")
    )
    backend = dist.ProcessGroupNCCL(store, rank, world_size, opts)
    try:
        backend._set_default_timeout(timeout)
    except AttributeError:
        pass
    return _C.create_process_group_uopc_nccl(backend)


def register_backend() -> str:
    global _REGISTERED
    if _REGISTERED:
        return BACKEND_NAME

    if BACKEND_NAME not in dist.Backend.backend_list:
        dist.Backend.register_backend(
            BACKEND_NAME,
            _create_uopc_nccl_backend,
            devices=["cuda"],
        )
        debug(f"registered backend {BACKEND_NAME}")
    _REGISTERED = True
    return BACKEND_NAME


def _rewrite_backend_name(backend: Any) -> Any:
    if isinstance(backend, str) and backend.lower() == "nccl":
        return BACKEND_NAME
    return backend


def enable_nccl_backend_patch() -> None:
    global _PATCHED_INIT, _PATCHED_NEW_GROUP

    register_backend()

    if not _PATCHED_INIT:
        global _ORIGINAL_INIT_PROCESS_GROUP
        _ORIGINAL_INIT_PROCESS_GROUP = dist.init_process_group

        @wraps(_ORIGINAL_INIT_PROCESS_GROUP)
        def _wrapped_init_process_group(*args, **kwargs):
            backend = kwargs.get("backend")
            if backend is None and args:
                backend = args[0]
            rewritten = _rewrite_backend_name(backend)
            if rewritten is not backend:
                if "backend" in kwargs:
                    kwargs["backend"] = rewritten
                else:
                    args = (rewritten, *args[1:])
            return _ORIGINAL_INIT_PROCESS_GROUP(*args, **kwargs)

        _wrapped_init_process_group.__torch_uopc_wrapped__ = True
        dist.init_process_group = _wrapped_init_process_group
        _PATCHED_INIT = True
        debug("patched torch.distributed.init_process_group")

    if not _PATCHED_NEW_GROUP:
        global _ORIGINAL_NEW_GROUP
        _ORIGINAL_NEW_GROUP = dist.new_group

        @wraps(_ORIGINAL_NEW_GROUP)
        def _wrapped_new_group(*args, **kwargs):
            if "backend" in kwargs:
                kwargs["backend"] = _rewrite_backend_name(kwargs["backend"])
            elif len(args) >= 3:
                args = (*args[:2], _rewrite_backend_name(args[2]), *args[3:])
            return _ORIGINAL_NEW_GROUP(*args, **kwargs)

        _wrapped_new_group.__torch_uopc_wrapped__ = True
        dist.new_group = _wrapped_new_group
        _PATCHED_NEW_GROUP = True
        debug("patched torch.distributed.new_group")


def push_scope(
    *,
    parallel_axis: int,
    phase: int,
    process_group_kind: int,
    overlap_window_type: int = 0,
    bucket_or_microbatch_id: int = 0,
    flags: int = 0,
) -> None:
    _C.push_collective_scope(
        int(parallel_axis),
        int(phase),
        int(process_group_kind),
        int(overlap_window_type),
        int(bucket_or_microbatch_id),
        int(flags),
    )


def pop_scope() -> None:
    _C.pop_collective_scope()


@contextlib.contextmanager
def collective_scope(
    *,
    parallel_axis: int,
    phase: int,
    process_group_kind: int,
    overlap_window_type: int = 0,
    bucket_or_microbatch_id: int = 0,
    flags: int = 0,
):
    push_scope(
        parallel_axis=parallel_axis,
        phase=phase,
        process_group_kind=process_group_kind,
        overlap_window_type=overlap_window_type,
        bucket_or_microbatch_id=bucket_or_microbatch_id,
        flags=flags,
    )
    try:
        yield
    finally:
        pop_scope()


def enable_fsdp_scope_patch() -> None:
    global _PATCHED_FSDP
    if _PATCHED_FSDP:
        return
    _enable_fsdp_adapter()
    _PATCHED_FSDP = True
    debug("enabled FSDP adapter hooks")


def enable_deepspeed_adapter() -> None:
    global _PATCHED_DEEPSPEED
    if _PATCHED_DEEPSPEED:
        return
    _enable_deepspeed_adapter()
    _PATCHED_DEEPSPEED = True
    debug("enabled DeepSpeed adapter hooks")


def enable_megatron_adapter() -> None:
    global _PATCHED_MEGATRON
    if _PATCHED_MEGATRON:
        return
    _enable_megatron_adapter()
    _PATCHED_MEGATRON = True
    debug("enabled Megatron adapter hooks")


def enable() -> str:
    register_backend()
    if env_flag("TORCH_UOPC_WRAP_NCCL", "1"):
        enable_nccl_backend_patch()
    if env_flag("TORCH_UOPC_ENABLE_FSDP_SCOPES", "1"):
        enable_fsdp_scope_patch()
    if env_flag("TORCH_UOPC_ENABLE_DEEPSPEED_ADAPTER", "1"):
        enable_deepspeed_adapter()
    if env_flag("TORCH_UOPC_ENABLE_MEGATRON_ADAPTER", "1"):
        enable_megatron_adapter()
    return BACKEND_NAME
