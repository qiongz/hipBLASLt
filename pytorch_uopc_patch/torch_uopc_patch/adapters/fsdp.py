from __future__ import annotations

from types import ModuleType

from .._importer import register_post_import_hook
from .._utils import debug
from ..constants import (
    AXIS_FSDP,
    PG_DP_SHARD,
    PHASE_FORWARD,
    PHASE_POST_BACKWARD,
    PHASE_PRE_BACKWARD,
    WINDOW_BACKWARD_PREFETCH_AG_VS_CURRENT_GRAD,
    WINDOW_BACKWARD_RS_VS_NEXT_GRAD,
    WINDOW_FORWARD_AG_VS_CURRENT_GEMM,
)
from .common import replace_callable, static_scope, wrap_with_dynamic_scope


_PATCHED_MODULES: set[str] = set()


def _is_backward_training_state(value) -> bool:
    name = getattr(value, "name", "")
    return name in {"BACKWARD_PRE", "BACKWARD_POST", "PRE_BACKWARD", "POST_BACKWARD"}


def _resolve_unshard_scope(*args, **kwargs) -> dict[str, int]:
    handle = kwargs.get("handle")
    if handle is None and len(args) >= 2:
        handle = args[1]

    training_state = getattr(handle, "_training_state", None)
    if _is_backward_training_state(training_state):
        return static_scope(
            parallel_axis=AXIS_FSDP,
            phase=PHASE_PRE_BACKWARD,
            process_group_kind=PG_DP_SHARD,
            overlap_window_type=WINDOW_BACKWARD_PREFETCH_AG_VS_CURRENT_GRAD,
        )

    return static_scope(
        parallel_axis=AXIS_FSDP,
        phase=PHASE_FORWARD,
        process_group_kind=PG_DP_SHARD,
        overlap_window_type=WINDOW_FORWARD_AG_VS_CURRENT_GEMM,
    )


def _resolve_param_group_unshard_scope(self, *args, **kwargs) -> dict[str, int]:
    training_state = getattr(self, "_training_state", None)
    if _is_backward_training_state(training_state):
        return static_scope(
            parallel_axis=AXIS_FSDP,
            phase=PHASE_PRE_BACKWARD,
            process_group_kind=PG_DP_SHARD,
            overlap_window_type=WINDOW_BACKWARD_PREFETCH_AG_VS_CURRENT_GRAD,
        )

    return static_scope(
        parallel_axis=AXIS_FSDP,
        phase=PHASE_FORWARD,
        process_group_kind=PG_DP_SHARD,
        overlap_window_type=WINDOW_FORWARD_AG_VS_CURRENT_GEMM,
    )


def _patch_runtime_utils(module: ModuleType) -> None:
    if module.__name__ in _PATCHED_MODULES:
        return

    changed = False
    changed |= replace_callable(
        module,
        "_unshard",
        lambda func: wrap_with_dynamic_scope(func, _resolve_unshard_scope),
    )
    changed |= replace_callable(
        module,
        "_reduce_grad",
        lambda func: wrap_with_dynamic_scope(
            func,
            lambda *args, **kwargs: static_scope(
                parallel_axis=AXIS_FSDP,
                phase=PHASE_POST_BACKWARD,
                process_group_kind=PG_DP_SHARD,
                overlap_window_type=WINDOW_BACKWARD_RS_VS_NEXT_GRAD,
            ),
        ),
    )
    if changed:
        _PATCHED_MODULES.add(module.__name__)
        debug(f"patched FSDP runtime scopes: {module.__name__}")


def _patch_param_group(module: ModuleType) -> None:
    if module.__name__ in _PATCHED_MODULES:
        return

    param_group_cls = getattr(module, "FSDPParamGroup", None)
    if param_group_cls is None:
        return

    changed = False
    changed |= replace_callable(
        param_group_cls,
        "unshard",
        lambda func: wrap_with_dynamic_scope(func, _resolve_param_group_unshard_scope),
    )
    changed |= replace_callable(
        param_group_cls,
        "post_backward",
        lambda func: wrap_with_dynamic_scope(
            func,
            lambda self, *args, **kwargs: static_scope(
                parallel_axis=AXIS_FSDP,
                phase=PHASE_POST_BACKWARD,
                process_group_kind=PG_DP_SHARD,
                overlap_window_type=WINDOW_BACKWARD_RS_VS_NEXT_GRAD,
            ),
        ),
    )
    if changed:
        _PATCHED_MODULES.add(module.__name__)
        debug(f"patched FSDP param-group scopes: {module.__name__}")


def enable() -> None:
    register_post_import_hook("torch.distributed.fsdp._runtime_utils", _patch_runtime_utils)
    register_post_import_hook(
        "torch.distributed.fsdp._fully_shard._fsdp_param_group",
        _patch_param_group,
    )
