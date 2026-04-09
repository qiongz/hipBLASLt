#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def parse_kv(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        value = spec.strip()
        if not value:
            raise ValueError("Empty profile value is not allowed")
        return value, value

    key, value = spec.split("=", 1)
    key = key.strip()
    value = value.strip()
    if not key or not value:
        raise ValueError(f"Invalid KEY=VALUE spec: {spec}")
    return key, value


def merge_tuning_files(merged_path: Path, profile_outputs: list[Path]) -> None:
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    with merged_path.open("w", encoding="utf-8") as merged:
        for profile_output in profile_outputs:
            tuning_file = profile_output / "tuning.txt"
            if not tuning_file.exists():
                raise FileNotFoundError(f"Missing tuning file: {tuning_file}")

            content = tuning_file.read_text(encoding="utf-8")
            if not content:
                continue
            merged.write(content)
            if not content.endswith("\n"):
                merged.write("\n")


def infer_cu_count(profile_label: str) -> int | None:
    match = re.fullmatch(r"(?:cu)?(\d+)", profile_label.strip(), flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def rewrite_tuning_file_cu_count(tuning_file: Path, cu_count: int) -> None:
    lines = tuning_file.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise ValueError(f"Unexpected tuning file format: {tuning_file}")

    header = [column.strip() for column in lines[1].split(",")]
    try:
        cu_index = header.index("CUs")
    except ValueError as exc:
        raise ValueError(f"Missing CUs column in tuning file: {tuning_file}") from exc

    rewritten_lines = lines[:2]
    for line in lines[2:]:
        if "transA" in line and "solution_index" in line:
            rewritten_lines.append(line)
            continue
        columns = line.split(",")
        if len(columns) <= cu_index:
            rewritten_lines.append(line)
            continue
        columns[cu_index] = str(cu_count)
        rewritten_lines.append(",".join(columns))

    tuning_file.write_text("\n".join(rewritten_lines) + "\n", encoding="utf-8")


def build_gemm_tuning_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    command = [
        args.python_exe,
        str(args.tuning_script),
        "--input_file",
        str(args.input_file),
        "--output_path",
        str(output_dir),
        "--requested_solution",
        str(args.requested_solution),
        "--cold_iters",
        str(args.cold_iters),
        "--iters",
        str(args.iters),
        "--gpu_id",
        str(args.gpu_id),
        "--max_gemms",
        str(args.max_gemms),
    ]

    if args.bench_path:
        command.extend(["--bench_path", args.bench_path])
    if args.bench_library_dir:
        command.extend(["--bench_library_dir", args.bench_library_dir])
    if args.tensile_libpath:
        command.extend(["--tensile_libpath", args.tensile_libpath])
    if args.swizzleA:
        command.append("--swizzleA")
    if args.swizzleB:
        command.append("--swizzleB")
    if args.stablize_gpu:
        command.append("--stablize_gpu")

    return command


def main() -> int:
    parser = argparse.ArgumentParser(description="Run hipBLASLt GEMM tuning across multiple CU profiles")
    parser.add_argument("--input_file", type=Path, required=True, help="Path to hipBLASLt GEMM log")
    parser.add_argument("--output_path", type=Path, required=True, help="Root output directory")
    parser.add_argument(
        "--cu_profile",
        action="append",
        required=True,
        help="CU profile spec. Accepts LABEL or LABEL=ENV_VALUE. Repeat this flag for multiple profiles.",
    )
    parser.add_argument(
        "--cu_env_var",
        type=str,
        default="HSA_CU_MASK",
        help="Environment variable used to expose a CU profile",
    )
    parser.add_argument(
        "--set_env",
        action="append",
        default=[],
        help="Extra KEY=VALUE environment entries propagated to every tuning run",
    )
    parser.add_argument(
        "--merged_tuning_file",
        type=Path,
        default=None,
        help="Optional merged tuning file path produced by concatenating every per-profile tuning.txt",
    )
    parser.add_argument(
        "--tuning_script",
        type=Path,
        default=Path(__file__).with_name("gemm_tuning.py"),
        help="Path to gemm_tuning.py",
    )
    parser.add_argument(
        "--python_exe",
        type=str,
        default=sys.executable,
        help="Python executable used to launch gemm_tuning.py",
    )
    parser.add_argument("--bench_path", type=str, default="", help="Path to hipblaslt-bench")
    parser.add_argument("--bench_library_dir", type=str, default="", help="Path to libhipblaslt directory used by bench")
    parser.add_argument("--tensile_libpath", type=str, default="", help="Path to Tensile logic directory used by the local build")
    parser.add_argument("--requested_solution", type=int, default=128)
    parser.add_argument("--cold_iters", type=int, default=-1)
    parser.add_argument("--iters", type=int, default=-1)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--max_gemms", type=int, default=-1)
    parser.add_argument("--swizzleA", action="store_true")
    parser.add_argument("--swizzleB", action="store_true")
    parser.add_argument("--stablize_gpu", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without executing them")
    args = parser.parse_args()

    extra_env = {}
    for spec in args.set_env:
        key, value = parse_kv(spec)
        extra_env[key] = value

    args.output_path.mkdir(parents=True, exist_ok=True)

    profile_outputs: list[Path] = []
    for spec in args.cu_profile:
        profile_label, profile_value = parse_kv(spec)
        profile_output = args.output_path / f"cu{profile_label}"
        profile_output.mkdir(parents=True, exist_ok=True)
        profile_outputs.append(profile_output)

        env = os.environ.copy()
        env.update(extra_env)
        env[args.cu_env_var] = profile_value
        command = build_gemm_tuning_command(args, profile_output)

        print(
            f"[cu-sweep] profile={profile_label} {args.cu_env_var}={profile_value} "
            f"output={profile_output}"
        )
        print("[cu-sweep] command:", " ".join(command))

        if args.dry_run:
            continue

        subprocess.run(command, check=True, env=env)
        inferred_cu_count = infer_cu_count(profile_label)
        if inferred_cu_count is not None:
            tuning_file = profile_output / "tuning.txt"
            rewrite_tuning_file_cu_count(tuning_file, inferred_cu_count)
            print(f"[cu-sweep] rewrote {tuning_file} CUs={inferred_cu_count}")

    if args.merged_tuning_file is not None and not args.dry_run:
        merge_tuning_files(args.merged_tuning_file, profile_outputs)
        print(f"[cu-sweep] merged tuning file: {args.merged_tuning_file}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
