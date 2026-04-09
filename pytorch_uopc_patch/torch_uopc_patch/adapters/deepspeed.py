from __future__ import annotations

from types import ModuleType

from .._importer import register_post_import_hook
from .._utils import debug
from ..constants import (
    AXIS_DP,
    AXIS_EP,
    AXIS_ZERO,
    PG_DP_REPLICA,
    PG_DP_SHARD,
    PG_EP,
    PHASE_FORWARD,
    PHASE_POST_BACKWARD,
    WINDOW_A2A_VS_EXPERT_GEMM,
    WINDOW_BACKWARD_AR_VS_NEXT_GEMM,
    WINDOW_BACKWARD_RS_VS_NEXT_GRAD,
    WINDOW_FORWARD_AG_VS_CURRENT_GEMM,
)
from .common import replace_callable, wrap_with_static_scope


_PATCHED_MODULES: set[str] = set()


def _patch_comm(module: ModuleType) -> None:
    if module.__name__ in _PATCHED_MODULES:
        return

    changed = False
    changed |= replace_callable(
        module,
        "all_reduce",
        lambda func: wrap_with_static_scope(
            func,
            parallel_axis=AXIS_DP,
            phase=PHASE_POST_BACKWARD,
            process_group_kind=PG_DP_REPLICA,
            overlap_window_type=WINDOW_BACKWARD_AR_VS_NEXT_GEMM,
        ),
    )
    changed |= replace_callable(
        module,
        "all_reduce_coalesced",
        lambda func: wrap_with_static_scope(
            func,
            parallel_axis=AXIS_DP,
            phase=PHASE_POST_BACKWARD,
            process_group_kind=PG_DP_REPLICA,
            overlap_window_type=WINDOW_BACKWARD_AR_VS_NEXT_GEMM,
        ),
    )
    changed |= replace_callable(
        module,
        "all_gather",
        lambda func: wrap_with_static_scope(
            func,
            parallel_axis=AXIS_ZERO,
            phase=PHASE_FORWARD,
            process_group_kind=PG_DP_SHARD,
            overlap_window_type=WINDOW_FORWARD_AG_VS_CURRENT_GEMM,
        ),
    )
    changed |= replace_callable(
        module,
        "all_gather_into_tensor",
        lambda func: wrap_with_static_scope(
            func,
            parallel_axis=AXIS_ZERO,
            phase=PHASE_FORWARD,
            process_group_kind=PG_DP_SHARD,
            overlap_window_type=WINDOW_FORWARD_AG_VS_CURRENT_GEMM,
        ),
    )
    changed |= replace_callable(
        module,
        "all_gather_coalesced",
        lambda func: wrap_with_static_scope(
            func,
            parallel_axis=AXIS_ZERO,
            phase=PHASE_FORWARD,
            process_group_kind=PG_DP_SHARD,
            overlap_window_type=WINDOW_FORWARD_AG_VS_CURRENT_GEMM,
        ),
    )
    changed |= replace_callable(
        module,
        "reduce_scatter_fn",
        lambda func: wrap_with_static_scope(
            func,
            parallel_axis=AXIS_ZERO,
            phase=PHASE_POST_BACKWARD,
            process_group_kind=PG_DP_SHARD,
            overlap_window_type=WINDOW_BACKWARD_RS_VS_NEXT_GRAD,
        ),
    )
    changed |= replace_callable(
        module,
        "reduce_scatter",
        lambda func: wrap_with_static_scope(
            func,
            parallel_axis=AXIS_ZERO,
            phase=PHASE_POST_BACKWARD,
            process_group_kind=PG_DP_SHARD,
            overlap_window_type=WINDOW_BACKWARD_RS_VS_NEXT_GRAD,
        ),
    )
    changed |= replace_callable(
        module,
        "reduce_scatter_tensor",
        lambda func: wrap_with_static_scope(
            func,
            parallel_axis=AXIS_ZERO,
            phase=PHASE_POST_BACKWARD,
            process_group_kind=PG_DP_SHARD,
            overlap_window_type=WINDOW_BACKWARD_RS_VS_NEXT_GRAD,
        ),
    )
    changed |= replace_callable(
        module,
        "all_to_all_single",
        lambda func: wrap_with_static_scope(
            func,
            parallel_axis=AXIS_EP,
            phase=PHASE_FORWARD,
            process_group_kind=PG_EP,
            overlap_window_type=WINDOW_A2A_VS_EXPERT_GEMM,
        ),
    )
    if changed:
        _PATCHED_MODULES.add(module.__name__)
        debug(f"patched DeepSpeed comm scopes: {module.__name__}")


def enable() -> None:
    register_post_import_hook("deepspeed.comm.comm", _patch_comm)
