from __future__ import annotations

from types import ModuleType

from .._importer import register_post_import_hook
from .._utils import debug
from ..constants import (
    AXIS_EP,
    AXIS_PP,
    AXIS_TP,
    PG_EP,
    PG_PP,
    PG_TP,
    PHASE_FORWARD,
    PHASE_P2P_BOUNDARY,
    PHASE_POST_BACKWARD,
    WINDOW_A2A_VS_EXPERT_GEMM,
    WINDOW_BACKWARD_RS_VS_NEXT_GRAD,
    WINDOW_FORWARD_AG_VS_CURRENT_GEMM,
    WINDOW_P2P_VS_PIPELINE_GEMM,
    WINDOW_UNKNOWN,
)
from .common import replace_callable, wrap_with_static_scope


_PATCHED_MODULES: set[str] = set()


def _patch_tensor_parallel_mappings(module: ModuleType) -> None:
    if module.__name__ in _PATCHED_MODULES:
        return

    changed = False
    changed |= replace_callable(
        module,
        "_reduce",
        lambda func: wrap_with_static_scope(
            func,
            parallel_axis=AXIS_TP,
            phase=PHASE_POST_BACKWARD,
            process_group_kind=PG_TP,
            overlap_window_type=WINDOW_UNKNOWN,
        ),
    )
    changed |= replace_callable(
        module,
        "_gather_along_last_dim",
        lambda func: wrap_with_static_scope(
            func,
            parallel_axis=AXIS_TP,
            phase=PHASE_FORWARD,
            process_group_kind=PG_TP,
            overlap_window_type=WINDOW_FORWARD_AG_VS_CURRENT_GEMM,
        ),
    )
    changed |= replace_callable(
        module,
        "_gather_along_first_dim",
        lambda func: wrap_with_static_scope(
            func,
            parallel_axis=AXIS_TP,
            phase=PHASE_FORWARD,
            process_group_kind=PG_TP,
            overlap_window_type=WINDOW_FORWARD_AG_VS_CURRENT_GEMM,
        ),
    )
    changed |= replace_callable(
        module,
        "_reduce_scatter_along_last_dim",
        lambda func: wrap_with_static_scope(
            func,
            parallel_axis=AXIS_TP,
            phase=PHASE_POST_BACKWARD,
            process_group_kind=PG_TP,
            overlap_window_type=WINDOW_BACKWARD_RS_VS_NEXT_GRAD,
        ),
    )
    changed |= replace_callable(
        module,
        "_reduce_scatter_along_first_dim",
        lambda func: wrap_with_static_scope(
            func,
            parallel_axis=AXIS_TP,
            phase=PHASE_POST_BACKWARD,
            process_group_kind=PG_TP,
            overlap_window_type=WINDOW_BACKWARD_RS_VS_NEXT_GRAD,
        ),
    )
    changed |= replace_callable(
        module,
        "all_to_all",
        lambda func: wrap_with_static_scope(
            func,
            parallel_axis=AXIS_EP,
            phase=PHASE_FORWARD,
            process_group_kind=PG_EP,
            overlap_window_type=WINDOW_A2A_VS_EXPERT_GEMM,
        ),
    )
    changed |= replace_callable(
        module,
        "all_to_all_sp2hp",
        lambda func: wrap_with_static_scope(
            func,
            parallel_axis=AXIS_EP,
            phase=PHASE_FORWARD,
            process_group_kind=PG_EP,
            overlap_window_type=WINDOW_A2A_VS_EXPERT_GEMM,
        ),
    )
    changed |= replace_callable(
        module,
        "all_to_all_hp2sp",
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
        debug(f"patched Megatron tensor-parallel scopes: {module.__name__}")


def _patch_pipeline_p2p(module: ModuleType) -> None:
    if module.__name__ in _PATCHED_MODULES:
        return

    changed = False
    changed |= replace_callable(
        module,
        "_communicate_shapes",
        lambda func: wrap_with_static_scope(
            func,
            parallel_axis=AXIS_PP,
            phase=PHASE_P2P_BOUNDARY,
            process_group_kind=PG_PP,
            overlap_window_type=WINDOW_P2P_VS_PIPELINE_GEMM,
        ),
    )
    changed |= replace_callable(
        module,
        "_batched_p2p_ops",
        lambda func: wrap_with_static_scope(
            func,
            parallel_axis=AXIS_PP,
            phase=PHASE_P2P_BOUNDARY,
            process_group_kind=PG_PP,
            overlap_window_type=WINDOW_P2P_VS_PIPELINE_GEMM,
        ),
    )
    changed |= replace_callable(
        module,
        "_p2p_ops",
        lambda func: wrap_with_static_scope(
            func,
            parallel_axis=AXIS_PP,
            phase=PHASE_P2P_BOUNDARY,
            process_group_kind=PG_PP,
            overlap_window_type=WINDOW_P2P_VS_PIPELINE_GEMM,
        ),
    )
    if changed:
        _PATCHED_MODULES.add(module.__name__)
        debug(f"patched Megatron pipeline scopes: {module.__name__}")


def enable() -> None:
    register_post_import_hook(
        "megatron.core.tensor_parallel.mappings",
        _patch_tensor_parallel_mappings,
    )
    register_post_import_hook(
        "megatron.core.pipeline_parallel.p2p_communication",
        _patch_pipeline_p2p,
    )
