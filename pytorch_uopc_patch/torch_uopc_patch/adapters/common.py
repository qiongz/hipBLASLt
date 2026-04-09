from __future__ import annotations

from functools import wraps
from typing import Any, Callable


def static_scope(
    *,
    parallel_axis: int,
    phase: int,
    process_group_kind: int,
    overlap_window_type: int,
    bucket_or_microbatch_id: int = 0,
    flags: int = 0,
) -> dict[str, int]:
    return {
        "parallel_axis": parallel_axis,
        "phase": phase,
        "process_group_kind": process_group_kind,
        "overlap_window_type": overlap_window_type,
        "bucket_or_microbatch_id": bucket_or_microbatch_id,
        "flags": flags,
    }


def wrap_with_dynamic_scope(
    target: Callable[..., Any],
    scope_resolver: Callable[..., dict[str, int] | None],
) -> Callable[..., Any]:
    if getattr(target, "__torch_uopc_scoped__", False):
        return target

    @wraps(target)
    def _wrapped(*args, **kwargs):
        scope_kwargs = scope_resolver(*args, **kwargs)
        if not scope_kwargs:
            return target(*args, **kwargs)

        from ..backend import collective_scope

        with collective_scope(**scope_kwargs):
            return target(*args, **kwargs)

    _wrapped.__torch_uopc_scoped__ = True
    return _wrapped


def wrap_with_static_scope(
    target: Callable[..., Any],
    *,
    parallel_axis: int,
    phase: int,
    process_group_kind: int,
    overlap_window_type: int,
    bucket_or_microbatch_id: int = 0,
    flags: int = 0,
) -> Callable[..., Any]:
    return wrap_with_dynamic_scope(
        target,
        lambda *args, **kwargs: static_scope(
            parallel_axis=parallel_axis,
            phase=phase,
            process_group_kind=process_group_kind,
            overlap_window_type=overlap_window_type,
            bucket_or_microbatch_id=bucket_or_microbatch_id,
            flags=flags,
        ),
    )


def replace_callable(
    owner: Any,
    name: str,
    wrapper_factory: Callable[[Callable[..., Any]], Callable[..., Any]],
) -> bool:
    current = getattr(owner, name, None)
    if current is None or getattr(current, "__torch_uopc_scoped__", False):
        return False
    setattr(owner, name, wrapper_factory(current))
    return True
