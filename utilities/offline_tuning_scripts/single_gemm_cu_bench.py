#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import time
from collections import Counter
from pathlib import Path

RE_TIME_1 = re.compile(r"avg[\s_]*time[^0-9]*([0-9]+\.?[0-9]*)\s*(us|µs|ms|s)", re.I)
RE_TIME_2 = re.compile(r"\b([0-9]+\.?[0-9]*)\s*ms\b", re.I)
RE_TF_G = re.compile(r"([0-9]+\.?[0-9]*)\s*(TFLOPS|GFLOPS)", re.I)

SHAPE_FIELDS = ("m", "n", "k", "transA", "transB", "bias", "lda", "ldb", "ldc", "ldd")


def parse_kv(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"Expected KEY=VALUE, got: {spec}")
    key, value = spec.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key or not value:
        raise ValueError(f"Invalid KEY=VALUE spec: {spec}")
    return key, value


def get_flag(parts: list[str], flag: str, default: str | None = None) -> str | None:
    if flag not in parts:
        return default
    index = parts.index(flag)
    if index + 1 >= len(parts):
        return default
    return parts[index + 1]


def strip_flag_with_value(parts: list[str], flag: str) -> list[str]:
    out: list[str] = []
    skip_next = False
    for token in parts:
        if skip_next:
            skip_next = False
            continue
        if token == flag:
            skip_next = True
            continue
        out.append(token)
    return out


def shape_key_from_parts(parts: list[str]) -> tuple[int, int, int, str, str, int, int, int, int, int]:
    return (
        int(get_flag(parts, "-m", "0")),
        int(get_flag(parts, "-n", "0")),
        int(get_flag(parts, "-k", "0")),
        str(get_flag(parts, "--transA", "N")).upper(),
        str(get_flag(parts, "--transB", "N")).upper(),
        1 if "--bias_vector" in parts else 0,
        int(get_flag(parts, "--lda", "0")),
        int(get_flag(parts, "--ldb", "0")),
        int(get_flag(parts, "--ldc", "0")),
        int(get_flag(parts, "--ldd", "0")),
    )


def shape_key_from_row(row: dict[str, str]) -> tuple[int, int, int, str, str, int, int, int, int, int]:
    return (
        int(row["m"]),
        int(row["n"]),
        int(row["k"]),
        row["transA"].upper(),
        row["transB"].upper(),
        int(row["bias_vector"]),
        int(row["lda"]),
        int(row["ldb"]),
        int(row["ldc"]),
        int(row["ldd"]),
    )


def shape_key_to_dict(shape_key: tuple[int, int, int, str, str, int, int, int, int, int]) -> dict[str, int | str]:
    return dict(zip(SHAPE_FIELDS, shape_key, strict=True))


def shape_key_to_string(shape_key: tuple[int, int, int, str, str, int, int, int, int, int]) -> str:
    info = shape_key_to_dict(shape_key)
    return (
        f"m{info['m']}_n{info['n']}_k{info['k']}"
        f"_ta{info['transA']}_tb{info['transB']}_bias{info['bias']}"
        f"_lda{info['lda']}_ldb{info['ldb']}_ldc{info['ldc']}_ldd{info['ldd']}"
    )


def load_commands(path: Path) -> list[dict[str, object]]:
    commands: list[dict[str, object]] = []
    seen: set[tuple[int, int, int, str, str, int, int, int, int, int]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or not stripped.startswith("hipblaslt-bench"):
            continue
        parts = shlex.split(stripped)
        key = shape_key_from_parts(parts)
        if key in seen:
            continue
        seen.add(key)
        commands.append(
            {
                "shape_key": key,
                "shape_label": shape_key_to_string(key),
                "raw_command": stripped,
                "parts": parts,
            }
        )
    return commands


def load_shape_weights(path: Path | None) -> Counter:
    if path is None:
        return Counter()
    counts: Counter = Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("hipblaslt-bench"):
            continue
        counts[shape_key_from_parts(shlex.split(stripped))] += 1
    return counts


def load_tuning_rows(path: Path) -> dict[tuple[tuple[int, int, int, str, str, int, int, int, int, int], int], dict[str, str]]:
    rows: dict[tuple[tuple[int, int, int, str, str, int, int, int, int, int], int], dict[str, str]] = {}
    header: list[str] | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("Git Version:"):
            continue
        if "transA,transB" in line and "solution_index" in line:
            header = [column.strip() for column in line.split(",")]
            continue
        if header is None:
            continue
        values = [column.strip() for column in raw_line.split(",")]
        if len(values) != len(header):
            continue
        row = dict(zip(header, values, strict=True))
        shape_key = shape_key_from_row(row)
        cu = int(row["CUs"])
        rows[(shape_key, cu)] = row
    return rows


def build_case_specs(target_cus: list[int], reference_cu: int) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = [
        {
            "case_id": f"ref{reference_cu}_mask{reference_cu}",
            "kind": "reference",
            "solution_cu": reference_cu,
            "physical_cu": reference_cu,
        }
    ]
    for target_cu in target_cus:
        cases.append(
            {
                "case_id": f"baseline{reference_cu}_mask{target_cu}",
                "kind": "baseline",
                "solution_cu": reference_cu,
                "physical_cu": target_cu,
            }
        )
        cases.append(
            {
                "case_id": f"tuned{target_cu}_mask{target_cu}",
                "kind": "tuned",
                "solution_cu": target_cu,
                "physical_cu": target_cu,
            }
        )
    return cases


def build_mask(mask_scope: str, physical_cu: int) -> str:
    return f"{mask_scope}:0-{physical_cu - 1}"


def build_index_command(parts: list[str], solution_index: int, iters: int, cold_iters: int) -> list[str]:
    sanitized = list(parts)
    for flag in ("--algo_method", "--solution_index", "--requested_solution", "--iters", "--cold_iters"):
        sanitized = strip_flag_with_value(sanitized, flag)
    sanitized.extend(
        [
            "--algo_method",
            "index",
            "--solution_index",
            str(solution_index),
            "--cold_iters",
            str(cold_iters),
            "--iters",
            str(iters),
        ]
    )
    return sanitized


def choose_iterations(
    solution_us: float,
    default_iters: int,
    default_cold_iters: int,
    max_case_seconds: float,
    estimate_safety_factor: float,
    min_iters: int,
) -> tuple[int, int, float]:
    estimated_seconds = (default_iters + default_cold_iters) * solution_us / 1e6 * estimate_safety_factor
    if estimated_seconds <= max_case_seconds:
        return default_iters, default_cold_iters, estimated_seconds
    scale = max_case_seconds / estimated_seconds
    iters = max(min_iters, int(default_iters * scale))
    cold_iters = max(min_iters, int(default_cold_iters * scale))
    estimated_seconds = (iters + cold_iters) * solution_us / 1e6 * estimate_safety_factor
    return iters, cold_iters, estimated_seconds


def parse_perf(stdout: str, m: int, n: int, k: int) -> tuple[float | None, float | None]:
    avg_ms: float | None = None
    tflops: float | None = None
    for line in stdout.splitlines():
        if line.count(",") >= 38 and avg_ms is None:
            columns = [column.strip() for column in line.split(",")]
            try:
                gflops = float(columns[-3])
                microseconds = float(columns[-1])
            except (IndexError, ValueError):
                pass
            else:
                avg_ms = microseconds / 1000.0
                tflops = gflops / 1000.0
        tf_match = RE_TF_G.search(line)
        if tf_match:
            value = float(tf_match.group(1))
            unit = tf_match.group(2).upper()
            tflops = value / 1000.0 if unit.startswith("G") else value
        time_match = RE_TIME_1.search(line)
        if time_match:
            value = float(time_match.group(1))
            unit = time_match.group(2).lower()
            if unit in ("us", "µs"):
                avg_ms = value / 1000.0
            elif unit == "ms":
                avg_ms = value
            else:
                avg_ms = value * 1000.0
        elif avg_ms is None:
            fallback_match = RE_TIME_2.search(line)
            if fallback_match:
                avg_ms = float(fallback_match.group(1))
    if tflops is None and avg_ms is not None and avg_ms > 0:
        flops = 2.0 * m * n * k
        tflops = flops / (avg_ms / 1000.0) / 1e12
    return avg_ms, tflops


def format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_case(
    bench_path: Path,
    command_parts: list[str],
    env: dict[str, str],
    timeout_seconds: float,
    raw_log_path: Path,
) -> tuple[str, float, str]:
    start = time.time()
    try:
        proc = subprocess.run(
            [str(bench_path), *command_parts[1:]],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
            env=env,
            check=False,
        )
        stdout = proc.stdout
        elapsed = time.time() - start
        raw_log_path.write_text(stdout, encoding="utf-8")
        if proc.returncode != 0:
            return "error", elapsed, stdout
        return "ok", elapsed, stdout
    except subprocess.TimeoutExpired as exc:
        elapsed = time.time() - start
        stdout = exc.stdout or ""
        raw_log_path.write_text(stdout, encoding="utf-8")
        return "timeout", elapsed, stdout


def main() -> int:
    parser = argparse.ArgumentParser(description="Run single-GEMM CU-index benchmarks across tuned CU buckets")
    parser.add_argument("--bench_path", type=Path, required=True)
    parser.add_argument("--dedup_cmd_file", type=Path, required=True)
    parser.add_argument("--raw_cmd_file", type=Path, default=None)
    parser.add_argument("--reference_tuning_file", type=Path, required=True)
    parser.add_argument("--cu272_tuning_file", type=Path, default=None)
    parser.add_argument("--cu302_tuning_file", type=Path, default=None)
    parser.add_argument("--cu248_tuning_file", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--targets", type=int, nargs="+", default=[302, 272, 248])
    parser.add_argument("--reference_cu", type=int, default=304)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--cold_iters", type=int, default=500)
    parser.add_argument("--min_iters", type=int, default=50)
    parser.add_argument("--max_case_seconds", type=float, default=120.0)
    parser.add_argument("--estimate_safety_factor", type=float, default=8.0)
    parser.add_argument("--mask_scope", type=str, default="0-7")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--set_env", action="append", default=[], help="Extra KEY=VALUE env entries")
    args = parser.parse_args()

    extra_env: dict[str, str] = {}
    for spec in args.set_env:
        key, value = parse_kv(spec)
        extra_env[key] = value

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_log_dir = args.output_dir / "raw_logs"
    raw_log_dir.mkdir(parents=True, exist_ok=True)

    commands = load_commands(args.dedup_cmd_file)
    shape_weights = load_shape_weights(args.raw_cmd_file)

    tuning_rows: dict[tuple[tuple[int, int, int, str, str, int, int, int, int, int], int], dict[str, str]] = {}
    for path in (
        args.reference_tuning_file,
        args.cu272_tuning_file,
        args.cu302_tuning_file,
        args.cu248_tuning_file,
    ):
        if path is None:
            continue
        tuning_rows.update(load_tuning_rows(path))

    cases = build_case_specs(args.targets, args.reference_cu)
    total_weight = sum(shape_weights[command["shape_key"]] for command in commands) or len(commands)

    result_rows: list[dict[str, object]] = []
    for command in commands:
        shape_key = command["shape_key"]
        shape_label = command["shape_label"]
        parts = command["parts"]
        shape_info = shape_key_to_dict(shape_key)
        weight = shape_weights[shape_key] or 1

        for case in cases:
            solution_cu = int(case["solution_cu"])
            physical_cu = int(case["physical_cu"])
            row = tuning_rows.get((shape_key, solution_cu))
            result: dict[str, object] = {
                "shape_label": shape_label,
                "weight": weight,
                "weight_norm": weight / total_weight,
                "case_id": case["case_id"],
                "kind": case["kind"],
                "solution_cu": solution_cu,
                "physical_cu": physical_cu,
                "mask": build_mask(args.mask_scope, physical_cu),
                "status": "missing_tuning",
                "solution_index": "",
                "tuning_us": "",
                "estimated_seconds": "",
                "used_iters": "",
                "used_cold_iters": "",
                "avg_time_ms": "",
                "tflops": "",
                "wall_seconds": "",
                "raw_log": "",
                **shape_info,
            }
            if row is None:
                result_rows.append(result)
                continue

            solution_index = int(row["solution_index"])
            solution_us = float(row["us"])
            iters, cold_iters, estimated_seconds = choose_iterations(
                solution_us,
                args.iters,
                args.cold_iters,
                args.max_case_seconds,
                args.estimate_safety_factor,
                args.min_iters,
            )

            env = os.environ.copy()
            env.update(extra_env)
            env["HIP_VISIBLE_DEVICES"] = args.device
            env["HSA_CU_MASK"] = build_mask(args.mask_scope, physical_cu)

            command_parts = build_index_command(parts, solution_index, iters, cold_iters)
            raw_log_path = raw_log_dir / f"{case['case_id']}__{shape_label}.log"
            status, wall_seconds, stdout = run_case(args.bench_path, command_parts, env, args.max_case_seconds, raw_log_path)

            if status == "timeout" and iters > args.min_iters and cold_iters > args.min_iters:
                iters = max(args.min_iters, iters // 2)
                cold_iters = max(args.min_iters, cold_iters // 2)
                command_parts = build_index_command(parts, solution_index, iters, cold_iters)
                raw_log_path = raw_log_dir / f"{case['case_id']}__{shape_label}__retry.log"
                retry_status, wall_seconds, stdout = run_case(
                    args.bench_path,
                    command_parts,
                    env,
                    args.max_case_seconds,
                    raw_log_path,
                )
                status = "ok_retry" if retry_status == "ok" else retry_status

            avg_time_ms, tflops = parse_perf(stdout, int(shape_info["m"]), int(shape_info["n"]), int(shape_info["k"]))
            if status in {"ok", "ok_retry"} and avg_time_ms is None:
                status = "parse_error"

            result.update(
                {
                    "status": status,
                    "solution_index": solution_index,
                    "tuning_us": format_float(solution_us),
                    "estimated_seconds": format_float(estimated_seconds),
                    "used_iters": iters,
                    "used_cold_iters": cold_iters,
                    "avg_time_ms": format_float(avg_time_ms),
                    "tflops": format_float(tflops),
                    "wall_seconds": format_float(wall_seconds),
                    "raw_log": raw_log_path.name,
                }
            )
            result_rows.append(result)

    results_csv = args.output_dir / "results.csv"
    write_rows(
        results_csv,
        [
            *SHAPE_FIELDS,
            "shape_label",
            "weight",
            "weight_norm",
            "case_id",
            "kind",
            "solution_cu",
            "physical_cu",
            "mask",
            "status",
            "solution_index",
            "tuning_us",
            "estimated_seconds",
            "used_iters",
            "used_cold_iters",
            "avg_time_ms",
            "tflops",
            "wall_seconds",
            "raw_log",
        ],
        result_rows,
    )

    by_shape_case = {(row["shape_label"], row["case_id"]): row for row in result_rows if row["avg_time_ms"]}

    shape_summary_rows: list[dict[str, object]] = []
    aggregate_source: dict[int, list[dict[str, object]]] = {target: [] for target in args.targets}
    for command in commands:
        shape_label = command["shape_label"]
        weight = shape_weights[command["shape_key"]] or 1
        ref_row = by_shape_case.get((shape_label, f"ref{args.reference_cu}_mask{args.reference_cu}"))
        if ref_row is None:
            continue
        ref_ms = float(ref_row["avg_time_ms"])
        for target in args.targets:
            baseline_row = by_shape_case.get((shape_label, f"baseline{args.reference_cu}_mask{target}"))
            tuned_row = by_shape_case.get((shape_label, f"tuned{target}_mask{target}"))
            if baseline_row is None or tuned_row is None:
                continue
            baseline_ms = float(baseline_row["avg_time_ms"])
            tuned_ms = float(tuned_row["avg_time_ms"])
            linear_expected_ms = ref_ms * args.reference_cu / target
            baseline_over_linear_pct = (baseline_ms / linear_expected_ms - 1.0) * 100.0
            tuned_over_linear_pct = (tuned_ms / linear_expected_ms - 1.0) * 100.0
            tuned_gain_vs_baseline_pct = (baseline_ms / tuned_ms - 1.0) * 100.0
            summary_row = {
                "shape_label": shape_label,
                "weight": weight,
                "weight_norm": weight / total_weight,
                "target_cu": target,
                "ref304_ms": format_float(ref_ms),
                "linear_expected_ms": format_float(linear_expected_ms),
                "baseline304_mask_ms": format_float(baseline_ms),
                "tuned_mask_ms": format_float(tuned_ms),
                "baseline_over_linear_pct": format_float(baseline_over_linear_pct),
                "tuned_over_linear_pct": format_float(tuned_over_linear_pct),
                "tuned_gain_vs_baseline_pct": format_float(tuned_gain_vs_baseline_pct),
            }
            shape_summary_rows.append(summary_row)
            aggregate_source[target].append(
                {
                    "weight": weight,
                    "ref304_ms": ref_ms,
                    "linear_expected_ms": linear_expected_ms,
                    "baseline304_mask_ms": baseline_ms,
                    "tuned_mask_ms": tuned_ms,
                }
            )

    write_rows(
        args.output_dir / "shape_summary.csv",
        [
            "shape_label",
            "weight",
            "weight_norm",
            "target_cu",
            "ref304_ms",
            "linear_expected_ms",
            "baseline304_mask_ms",
            "tuned_mask_ms",
            "baseline_over_linear_pct",
            "tuned_over_linear_pct",
            "tuned_gain_vs_baseline_pct",
        ],
        shape_summary_rows,
    )

    aggregate_rows: list[dict[str, object]] = []
    for target, rows in aggregate_source.items():
        if not rows:
            continue
        weight_sum = sum(row["weight"] for row in rows)
        aggregate_rows.append(
            {
                "target_cu": target,
                "shape_count": len(rows),
                "weight_sum": weight_sum,
                "unweighted_ref304_ms": format_float(sum(row["ref304_ms"] for row in rows) / len(rows)),
                "unweighted_linear_expected_ms": format_float(sum(row["linear_expected_ms"] for row in rows) / len(rows)),
                "unweighted_baseline304_mask_ms": format_float(sum(row["baseline304_mask_ms"] for row in rows) / len(rows)),
                "unweighted_tuned_mask_ms": format_float(sum(row["tuned_mask_ms"] for row in rows) / len(rows)),
                "unweighted_tuned_gain_vs_baseline_pct": format_float(
                    (sum(row["baseline304_mask_ms"] for row in rows) / sum(row["tuned_mask_ms"] for row in rows) - 1.0) * 100.0
                ),
                "weighted_ref304_ms": format_float(sum(row["ref304_ms"] * row["weight"] for row in rows) / weight_sum),
                "weighted_linear_expected_ms": format_float(
                    sum(row["linear_expected_ms"] * row["weight"] for row in rows) / weight_sum
                ),
                "weighted_baseline304_mask_ms": format_float(
                    sum(row["baseline304_mask_ms"] * row["weight"] for row in rows) / weight_sum
                ),
                "weighted_tuned_mask_ms": format_float(sum(row["tuned_mask_ms"] * row["weight"] for row in rows) / weight_sum),
                "weighted_tuned_gain_vs_baseline_pct": format_float(
                    (
                        sum(row["baseline304_mask_ms"] * row["weight"] for row in rows)
                        / sum(row["tuned_mask_ms"] * row["weight"] for row in rows)
                        - 1.0
                    )
                    * 100.0
                ),
            }
        )

    write_rows(
        args.output_dir / "aggregate_summary.csv",
        [
            "target_cu",
            "shape_count",
            "weight_sum",
            "unweighted_ref304_ms",
            "unweighted_linear_expected_ms",
            "unweighted_baseline304_mask_ms",
            "unweighted_tuned_mask_ms",
            "unweighted_tuned_gain_vs_baseline_pct",
            "weighted_ref304_ms",
            "weighted_linear_expected_ms",
            "weighted_baseline304_mask_ms",
            "weighted_tuned_mask_ms",
            "weighted_tuned_gain_vs_baseline_pct",
        ],
        aggregate_rows,
    )

    metadata = {
        "bench_path": str(args.bench_path),
        "dedup_cmd_file": str(args.dedup_cmd_file),
        "raw_cmd_file": str(args.raw_cmd_file) if args.raw_cmd_file else "",
        "reference_tuning_file": str(args.reference_tuning_file),
        "cu272_tuning_file": str(args.cu272_tuning_file) if args.cu272_tuning_file else "",
        "cu302_tuning_file": str(args.cu302_tuning_file) if args.cu302_tuning_file else "",
        "cu248_tuning_file": str(args.cu248_tuning_file) if args.cu248_tuning_file else "",
        "reference_cu": args.reference_cu,
        "targets": args.targets,
        "iters": args.iters,
        "cold_iters": args.cold_iters,
        "min_iters": args.min_iters,
        "max_case_seconds": args.max_case_seconds,
        "estimate_safety_factor": args.estimate_safety_factor,
        "mask_scope": args.mask_scope,
        "device": args.device,
        "shape_count": len(commands),
        "weight_total": total_weight,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
