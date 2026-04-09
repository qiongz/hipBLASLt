#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GEMM + all_reduce overlap microbench")
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--measure-iters", type=int, default=20)
    parser.add_argument("--gemm-repeat", type=int, default=64)
    parser.add_argument("--comm-numel", type=int, default=33554432)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--trans-a", choices=["N", "T"], default="N")
    parser.add_argument("--trans-b", choices=["N", "T"], default="T")
    parser.add_argument("--bias-vector", type=int, choices=[0, 1], default=0)
    parser.add_argument("--skip-overlap", action="store_true")
    parser.add_argument("--seed", type=int, default=20260408)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def get_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int((len(ordered) - 1) * ratio)
    return ordered[index]


def build_tensors(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    a_shape = (args.m, args.k) if args.trans_a == "N" else (args.k, args.m)
    b_shape = (args.k, args.n) if args.trans_b == "N" else (args.n, args.k)
    a = torch.randn(a_shape, device=device, dtype=dtype)
    b = torch.randn(b_shape, device=device, dtype=dtype)
    bias = torch.randn((args.n,), device=device, dtype=dtype) if args.bias_vector else None
    return a, b, bias


def run_gemm(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None, trans_a: str, trans_b: str) -> torch.Tensor:
    a_mat = a if trans_a == "N" else a.transpose(0, 1)
    b_mat = b if trans_b == "N" else b.transpose(0, 1)
    if trans_a == "N" and trans_b == "T":
        return F.linear(a, b, bias)
    if bias is not None:
        return torch.addmm(bias, a_mat, b_mat)
    return torch.matmul(a_mat, b_mat)


def run_mode(
    mode: str,
    *,
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None,
    comm_tensor: torch.Tensor,
    warmup_iters: int,
    measure_iters: int,
    gemm_repeat: int,
    trans_a: str,
    trans_b: str,
) -> dict[str, float]:
    times_ms: list[float] = []

    for iteration in range(warmup_iters + measure_iters):
        dist.barrier()
        torch.cuda.synchronize()
        start = time.perf_counter()

        if mode == "gemm_only":
            for _ in range(gemm_repeat):
                out = run_gemm(a, b, bias, trans_a, trans_b)
            del out
        elif mode == "allreduce_only":
            work = dist.all_reduce(comm_tensor, async_op=True)
            work.wait()
        elif mode == "overlap":
            work = dist.all_reduce(comm_tensor, async_op=True)
            for _ in range(gemm_repeat):
                out = run_gemm(a, b, bias, trans_a, trans_b)
            del out
            work.wait()
        elif mode == "serialized":
            work = dist.all_reduce(comm_tensor, async_op=True)
            work.wait()
            for _ in range(gemm_repeat):
                out = run_gemm(a, b, bias, trans_a, trans_b)
            del out
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if iteration >= warmup_iters:
            times_ms.append(elapsed_ms)

    return {
        "mean_ms": sum(times_ms) / len(times_ms),
        "p50_ms": percentile(times_ms, 0.50),
        "p95_ms": percentile(times_ms, 0.95),
        "p99_ms": percentile(times_ms, 0.99),
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
    }


def gather_rank_results(local_result: dict[str, object]) -> list[dict[str, object]]:
    world_size = dist.get_world_size()
    gathered: list[dict[str, object] | None] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_result)
    return [item for item in gathered if item is not None]


def aggregate_mode(gathered: list[dict[str, object]], mode: str) -> dict[str, float]:
    means = [entry["modes"][mode]["mean_ms"] for entry in gathered]
    p50s = [entry["modes"][mode]["p50_ms"] for entry in gathered]
    p95s = [entry["modes"][mode]["p95_ms"] for entry in gathered]
    p99s = [entry["modes"][mode]["p99_ms"] for entry in gathered]
    mins = [entry["modes"][mode]["min_ms"] for entry in gathered]
    maxs = [entry["modes"][mode]["max_ms"] for entry in gathered]
    return {
        "rank_mean_avg_ms": sum(means) / len(means),
        "rank_mean_max_ms": max(means),
        "rank_p50_avg_ms": sum(p50s) / len(p50s),
        "rank_p50_max_ms": max(p50s),
        "rank_p95_avg_ms": sum(p95s) / len(p95s),
        "rank_p95_max_ms": max(p95s),
        "rank_p99_avg_ms": sum(p99s) / len(p99s),
        "rank_p99_max_ms": max(p99s),
        "rank_min_min_ms": min(mins),
        "rank_max_max_ms": max(maxs),
    }


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")

    dtype = get_dtype(args.dtype)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    a, b, bias = build_tensors(args, device, dtype)
    comm_tensor = torch.randn((args.comm_numel,), device=device, dtype=dtype)

    modes = ["gemm_only", "allreduce_only", "serialized"]
    if not args.skip_overlap:
        modes.append("overlap")

    local_modes: dict[str, dict[str, float]] = {}
    for mode in modes:
        local_modes[mode] = run_mode(
            mode,
            a=a,
            b=b,
            bias=bias,
            comm_tensor=comm_tensor,
            warmup_iters=args.warmup_iters,
            measure_iters=args.measure_iters,
            gemm_repeat=args.gemm_repeat,
            trans_a=args.trans_a,
            trans_b=args.trans_b,
        )

    result = {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "modes": local_modes,
    }
    gathered = gather_rank_results(result)

    if rank == 0:
        summary: dict[str, object] = {
            "world_size": world_size,
            "gemm_repeat": args.gemm_repeat,
            "comm_numel": args.comm_numel,
            "dtype": args.dtype,
            "signature": {
                "m": args.m,
                "n": args.n,
                "k": args.k,
                "trans_a": args.trans_a,
                "trans_b": args.trans_b,
                "bias_vector": args.bias_vector,
            },
            "environment": {
                "uopc_force_bucket": os.environ.get("UOPC_FORCE_BUCKET", ""),
                "comm_cu": os.environ.get("COMM_CU", ""),
                "nccl_min_nchannels": os.environ.get("NCCL_MIN_NCHANNELS", ""),
                "nccl_max_nchannels": os.environ.get("NCCL_MAX_NCHANNELS", ""),
                "hipblaslt_tuning_override_file": os.environ.get(
                    "HIPBLASLT_TUNING_OVERRIDE_FILE", ""
                ),
            },
            "ranks": gathered,
        }

        aggregate: dict[str, dict[str, float]] = {}
        for mode in modes:
            aggregate[mode] = aggregate_mode(gathered, mode)
        summary["aggregate"] = aggregate

        if "overlap" in aggregate:
            overlap = aggregate["overlap"]["rank_mean_max_ms"]
            serialized = aggregate["serialized"]["rank_mean_max_ms"]
            gemm_only = aggregate["gemm_only"]["rank_mean_max_ms"]
            comm_only = aggregate["allreduce_only"]["rank_mean_max_ms"]
            summary["derived"] = {
                "overlap_over_gemm_only": overlap / gemm_only if gemm_only else 0.0,
                "overlap_over_serialized": overlap / serialized if serialized else 0.0,
                "overlap_minus_max_component_ms": overlap - max(gemm_only, comm_only),
            }

        print("OVERLAP_BENCH_RESULT_BEGIN", flush=True)
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        print("OVERLAP_BENCH_RESULT_END", flush=True)

        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
