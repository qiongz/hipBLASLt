from .backend import (
    BACKEND_NAME,
    collective_scope,
    enable,
    enable_deepspeed_adapter,
    enable_fsdp_scope_patch,
    enable_megatron_adapter,
    enable_nccl_backend_patch,
    pop_scope,
    push_scope,
    register_backend,
)
from .activation import auto_enable_from_env

__all__ = [
    "BACKEND_NAME",
    "auto_enable_from_env",
    "collective_scope",
    "enable",
    "enable_deepspeed_adapter",
    "enable_fsdp_scope_patch",
    "enable_megatron_adapter",
    "enable_nccl_backend_patch",
    "pop_scope",
    "push_scope",
    "register_backend",
]
