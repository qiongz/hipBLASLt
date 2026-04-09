#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score 272-vs-304 overlap bench results and recommend a conservative bucket"
    )
    parser.add_argument("--targets-csv", type=Path, required=True, help="uopc_overlap_targets.csv")
    parser.add_argument(
        "--results",
        type=Path,
        nargs="+",
        required=True,
        help="Overlap bench JSON files, or directories that contain them",
    )
    parser.add_argument("--output-csv", type=Path, required=True, help="Scored candidate CSV output")
    parser.add_argument(
        "--mean-tolerance",
        type=float,
        default=0.02,
        help="Allowable relative regression for overlap rank_mean_max_ms",
    )
    parser.add_argument(
        "--tail-tolerance",
        type=float,
        default=0.05,
        help="Allowable relative regression for overlap rank_p95_max_ms and rank_max_max_ms",
    )
    return parser.parse_args()


def candidate_key(record: Dict[str, object]) -> Tuple[object, ...]:
    return (
        record["ta"],
        record["tb"],
        int(record["m"]),
        int(record["n"]),
        int(record["k"]),
        int(record["bias_vector"]),
    )


def load_targets(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if int(row.get("score_target", 0)) == 1]


def collect_result_files(paths: List[Path]) -> List[Path]:
    result_files: List[Path] = []
    for path in paths:
        if path.is_dir():
            result_files.extend(sorted(path.glob("*.json")))
        else:
            result_files.append(path)
    return result_files


def load_results(paths: List[Path]) -> Dict[Tuple[object, ...], Dict[int, Dict[str, object]]]:
    grouped: Dict[Tuple[object, ...], Dict[int, Dict[str, object]]] = {}
    for path in collect_result_files(paths):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(summary, dict) or "signature" not in summary or "environment" not in summary:
            continue
        signature = summary["signature"]
        bucket = int(summary["environment"].get("uopc_force_bucket") or 0)
        key = (
            signature["trans_a"],
            signature["trans_b"],
            int(signature["m"]),
            int(signature["n"]),
            int(signature["k"]),
            int(signature["bias_vector"]),
        )
        grouped.setdefault(key, {})[bucket] = summary
    return grouped


def extract_metric(summary: Dict[str, object], mode: str, metric: str) -> float:
    return float(summary["aggregate"][mode][metric])


def score_candidate(
    target: Dict[str, object],
    result_272: Dict[str, object],
    result_304: Dict[str, object],
    mean_tolerance: float,
    tail_tolerance: float,
) -> Dict[str, object]:
    overlap_mean_272 = extract_metric(result_272, "overlap", "rank_mean_max_ms")
    overlap_mean_304 = extract_metric(result_304, "overlap", "rank_mean_max_ms")
    overlap_p95_272 = extract_metric(result_272, "overlap", "rank_p95_max_ms")
    overlap_p95_304 = extract_metric(result_304, "overlap", "rank_p95_max_ms")
    overlap_max_272 = extract_metric(result_272, "overlap", "rank_max_max_ms")
    overlap_max_304 = extract_metric(result_304, "overlap", "rank_max_max_ms")

    mean_ratio = overlap_mean_272 / overlap_mean_304 if overlap_mean_304 else 0.0
    p95_ratio = overlap_p95_272 / overlap_p95_304 if overlap_p95_304 else 0.0
    max_ratio = overlap_max_272 / overlap_max_304 if overlap_max_304 else 0.0

    status = "pass"
    gate_reason = "mean/p95/max within conservative overlap gate"
    if mean_ratio > 1.0 + mean_tolerance:
        status = "fail"
        gate_reason = "overlap_mean_regressed"
    elif p95_ratio > 1.0 + tail_tolerance:
        status = "fail"
        gate_reason = "overlap_p95_regressed"
    elif max_ratio > 1.0 + tail_tolerance:
        status = "fail"
        gate_reason = "overlap_max_regressed"

    scored = dict(target)
    scored.update(
        {
            "status": status,
            "gate_reason": gate_reason,
            "recommended_cu": 272 if status == "pass" else 304,
            "overlap_mean_272_ms": round(overlap_mean_272, 6),
            "overlap_mean_304_ms": round(overlap_mean_304, 6),
            "overlap_p95_272_ms": round(overlap_p95_272, 6),
            "overlap_p95_304_ms": round(overlap_p95_304, 6),
            "overlap_max_272_ms": round(overlap_max_272, 6),
            "overlap_max_304_ms": round(overlap_max_304, 6),
            "mean_ratio_272_over_304": round(mean_ratio, 6),
            "p95_ratio_272_over_304": round(p95_ratio, 6),
            "max_ratio_272_over_304": round(max_ratio, 6),
            "serialized_mean_272_ms": round(
                extract_metric(result_272, "serialized", "rank_mean_max_ms"), 6
            ),
            "serialized_mean_304_ms": round(
                extract_metric(result_304, "serialized", "rank_mean_max_ms"), 6
            ),
            "gemm_only_mean_272_ms": round(
                extract_metric(result_272, "gemm_only", "rank_mean_max_ms"), 6
            ),
            "gemm_only_mean_304_ms": round(
                extract_metric(result_304, "gemm_only", "rank_mean_max_ms"), 6
            ),
            "allreduce_only_mean_272_ms": round(
                extract_metric(result_272, "allreduce_only", "rank_mean_max_ms"), 6
            ),
            "allreduce_only_mean_304_ms": round(
                extract_metric(result_304, "allreduce_only", "rank_mean_max_ms"), 6
            ),
            "overlap_over_serialized_272": round(
                float(result_272.get("derived", {}).get("overlap_over_serialized", 0.0)), 6
            ),
            "overlap_over_serialized_304": round(
                float(result_304.get("derived", {}).get("overlap_over_serialized", 0.0)), 6
            ),
        }
    )
    return scored


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    targets = load_targets(args.targets_csv)
    results = load_results(args.results)

    scored_rows: List[Dict[str, object]] = []
    for target in targets:
        key = candidate_key(target)
        per_bucket = results.get(key, {})
        result_272 = per_bucket.get(272)
        result_304 = per_bucket.get(304)
        if result_272 is None or result_304 is None:
            incomplete = dict(target)
            incomplete.update(
                {
                    "status": "incomplete",
                    "gate_reason": "missing_bucket_results",
                    "recommended_cu": 304,
                }
            )
            scored_rows.append(incomplete)
            continue
        scored_rows.append(
            score_candidate(
                target,
                result_272,
                result_304,
                args.mean_tolerance,
                args.tail_tolerance,
            )
        )

    scored_rows.sort(
        key=lambda row: (
            row["status"] != "pass",
            -int(row.get("calls", 0)),
            int(row.get("m", 0)),
            int(row.get("n", 0)),
            int(row.get("k", 0)),
        )
    )
    write_csv(args.output_csv, scored_rows)

    summary = {
        "targets": len(targets),
        "pass": sum(1 for row in scored_rows if row["status"] == "pass"),
        "fail": sum(1 for row in scored_rows if row["status"] == "fail"),
        "incomplete": sum(1 for row in scored_rows if row["status"] == "incomplete"),
        "output_csv": str(args.output_csv),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
