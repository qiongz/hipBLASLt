#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run overlap-aware GEMM+allreduce microbench for auto272 targets"
    )
    parser.add_argument("--targets-csv", type=Path, required=True, help="uopc_overlap_targets.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for per-bucket JSON results")
    parser.add_argument("--nproc-per-node", type=int, default=2)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--measure-iters", type=int, default=10)
    parser.add_argument("--gemm-repeat", type=int, default=64)
    parser.add_argument("--comm-numel", type=int, default=33554432)
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--buckets", type=int, nargs="+", default=[272, 304])
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    return parser.parse_args()


def load_targets(path: Path, limit: int) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row.get("score_target", 0)) == 1]
    rows.sort(
        key=lambda row: (
            -int(row.get("calls", 0)),
            int(row["m"]),
            int(row["n"]),
            int(row["k"]),
        )
    )
    if limit >= 0:
        rows = rows[:limit]
    return rows


def result_name(target: Dict[str, str], bucket: int) -> str:
    return (
        f"force{bucket}_ta{target['ta']}_tb{target['tb']}"
        f"_m{target['m']}_n{target['n']}_k{target['k']}_bias{target['bias_vector']}.json"
    )


def build_command(args: argparse.Namespace, target: Dict[str, str], bucket: int, output_json: Path) -> List[str]:
    script_path = Path(__file__).with_name("gemm_allreduce_overlap_bench.py")
    return [
        args.python_exe,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(args.nproc_per_node),
         "--",
        str(script_path),
        "--warmup-iters",
        str(args.warmup_iters),
        "--measure-iters",
        str(args.measure_iters),
        "--gemm-repeat",
        str(args.gemm_repeat),
        "--comm-numel",
        str(args.comm_numel),
        "--dtype",
        args.dtype,
        "--m",
        target["m"],
        "--n",
        target["n"],
        "--k",
        target["k"],
        "--trans-a",
        target["ta"],
        "--trans-b",
        target["tb"],
        "--bias-vector",
        target["bias_vector"],
        "--output-json",
        str(output_json),
    ]


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    targets = load_targets(args.targets_csv, args.limit)
    manifest: List[Dict[str, object]] = []
    for target in targets:
        for bucket in args.buckets:
            output_json = args.output_dir / result_name(target, bucket)
            if args.skip_existing and output_json.exists():
                manifest.append(
                    {
                        "status": "skipped_existing",
                        "bucket": bucket,
                        "output_json": str(output_json),
                        "target": {
                            "ta": target["ta"],
                            "tb": target["tb"],
                            "m": int(target["m"]),
                            "n": int(target["n"]),
                            "k": int(target["k"]),
                            "bias_vector": int(target["bias_vector"]),
                        },
                    }
                )
                continue

            env = os.environ.copy()
            env["UOPC_FORCE_BUCKET"] = str(bucket)
            command = build_command(args, target, bucket, output_json)
            subprocess.run(command, check=True, env=env)
            manifest.append(
                {
                    "status": "completed",
                    "bucket": bucket,
                    "output_json": str(output_json),
                    "target": {
                        "ta": target["ta"],
                        "tb": target["tb"],
                        "m": int(target["m"]),
                        "n": int(target["n"]),
                        "k": int(target["k"]),
                        "bias_vector": int(target["bias_vector"]),
                    },
                }
            )

    manifest_path = args.output_dir / "bench_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "targets": len(targets),
                "buckets": args.buckets,
                "manifest": str(manifest_path),
                "output_dir": str(args.output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
