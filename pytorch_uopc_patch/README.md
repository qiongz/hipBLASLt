# `torch-uopc-patch`

最小独立 PyTorch patch 包原型，用来把 UOPC sideband 挂到现有安装的 `torch 2.9.1+rocm7.1.0` 上，而不重编 PyTorch 本体。

## 设计

- `torch_uopc_patch._C`:
  - 提供一个 `ProcessGroupNCCL` 包装器。
  - 在 `allreduce / reduce_scatter / allgather / alltoall` 进入底层 NCCL 前调用 `uopcSubmitRcclLaunch()`.
- `torch_uopc_patch.backend`:
  - 注册 `uopc_nccl` 自定义 backend。
  - 可把 `init_process_group(backend="nccl")` 自动改写为 `uopc_nccl`。
  - 为 FSDP 模块提供轻量 scope proxy。
- `sitecustomize.py`:
  - 配合 `PYTHONPATH`，实现不改 `train.py` 的自动启用。
  - 支持 fail-open 与版本门控，默认 patch 失败时自动回落。
- `torch_uopc_patch.adapters.*`:
  - 为 `FSDP`、`DeepSpeed`、`Megatron-LM` 提供外置 Python adapter。
  - 只补 `axis / pg / phase / window` 语义，不改它们原生通信调度。

## 构建

```bash
cd /apps/qiongzhu/hipBLASLt/pytorch_uopc_patch
UOPC_INSTALL_ROOT=/apps/qiongzhu/rocm-overlap-policy/install-uopc-four-bucket python3 setup.py build_ext --inplace
```

## 使用

显式导入：

```bash
PYTHONPATH=/apps/qiongzhu/hipBLASLt/pytorch_uopc_patch python3 -c "import torch_uopc_patch; print(torch_uopc_patch.enable())"
```

自动启用且不改训练脚本：

```bash
export PYTHONPATH=/apps/qiongzhu/hipBLASLt/pytorch_uopc_patch:${PYTHONPATH}
export TORCH_UOPC_AUTO_ENABLE=1
export TORCH_UOPC_WRAP_NCCL=1
export TORCH_UOPC_FAIL_OPEN=1
export TORCH_UOPC_SUPPORTED_TORCH_PREFIXES=2.9
```

如果需要给 FSDP 增加更丰富的 phase 标注：

```bash
export TORCH_UOPC_ENABLE_FSDP_SCOPES=1
```

如果希望同时支持外置框架 adapter：

```bash
export TORCH_UOPC_ENABLE_DEEPSPEED_ADAPTER=1
export TORCH_UOPC_ENABLE_MEGATRON_ADAPTER=1
```

## 当前原型限制

- `streamUid` 目前固定为 `0`，优先验证无侵入 backend 路线。
- `commHash` 目前使用包装 backend 地址，先依赖 `seqNumber` 对齐。
- FSDP / DeepSpeed / Megatron 目前仍是 Python overlay adapter，不是最终正式 ABI。
- `run_fremont_0306_uopc_bringup.sh` 当前默认带 `UOPC_EVENT_STOP_GRACE_TTL_US=5000`，但它只是 `DDP` bring-up 的 bootstrap 默认值，不应被当成可推广到所有模型/collective 的固定最优参数。
- 下一阶段 controller 主线会把 `DDP backward allreduce` 的 `grace TTL` 做成 warmup 自收敛策略；这一步仍然不要求用户修改训练脚本。
