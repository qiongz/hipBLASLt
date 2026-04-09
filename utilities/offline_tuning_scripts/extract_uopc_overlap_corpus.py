#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


QUERY_RE = re.compile(
    r"\[uopc\] queryHint status=(?P<status>\w+) msg=(?P<msg>.*?) "
    r"device=(?P<device>-?\d+) stream=(?P<stream>\d+) backend=(?P<backend>\d+) "
    r"problem=(?P<problem>\d+) m=(?P<m>\d+) n=(?P<n>\d+) k=(?P<k>\d+) "
    r"valid=(?P<valid>\d+) bucket=(?P<bucket>\d+) upper=(?P<upper>\d+) "
    r"pressure=(?P<pressure>\d+) source=(?P<source>\d+) related_seq=(?P<related_seq>\d+) "
    r"ts_ns=(?P<ts_ns>\d+)"
)

OVERRIDE_KEY_RE = re.compile(
    r"\[hipblaslt\]\[override-summary\] rank=(?P<rank>\S+) local_rank=(?P<local_rank>\S+) "
    r"key=arch=(?P<arch>[^|]*)\|cu=(?P<cu>\d+)\|ta=(?P<ta>[NT])\|tb=(?P<tb>[NT])"
    r"\|m=(?P<m>\d+)\|n=(?P<n>\d+)\|k=(?P<k>\d+)\|b=(?P<b>\d+)\|act=(?P<act>[^|]+)\|bias=(?P<bias>\d+) "
    r"calls=(?P<calls>\d+) cache_hits=(?P<cache_hits>\d+) cache_misses=(?P<cache_misses>\d+) "
    r"success=(?P<success>\d+) failure=(?P<failure>\d+) exact=(?P<exact>\d+) "
    r"arch_default=(?P<arch_default>\d+) legacy=(?P<legacy>\d+) no_match=(?P<no_match>\d+) "
    r"candidate_attempts=(?P<candidate_attempts>\d+) candidate_rejects=(?P<candidate_rejects>\d+)"
)


def normalize_arch(raw_arch: str) -> str:
    return raw_arch.split(":", 1)[0].strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract overlap-aware 272/304 corpus from a UOPC bring-up combined log"
    )
    parser.add_argument("--log", type=Path, required=True, help="Path to uopc_bringup_combined.log")
    parser.add_argument(
        "--override-file",
        type=Path,
        required=True,
        help="Path to merged_tuning.txt used to enrich exact keys with type metadata",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for corpus outputs")
    parser.add_argument(
        "--top-n-targets",
        type=int,
        default=8,
        help="Limit overlap bench targets to the hottest N auto272 keys",
    )
    return parser.parse_args()


def tuning_row_key(row: Dict[str, str]) -> Tuple[object, ...]:
    return (
        normalize_arch(row["gcnArchName"]),
        int(row["CUs"]),
        row["transA"],
        row["transB"],
        int(row["batch_count"]),
        int(row["m"]),
        int(row["n"]),
        int(row["k"]),
        row["a_type"],
        row["b_type"],
        row["c_type"],
        row["d_type"],
        row["compute_type"],
        row["activation_type"],
        int(row["bias_vector"]),
        row["bias_type"],
        row["aux_type"],
    )


def summary_signature_key(record: Dict[str, object]) -> Tuple[object, ...]:
    return (
        record["arch"],
        record["cu"],
        record["ta"],
        record["tb"],
        record["batch_count"],
        record["m"],
        record["n"],
        record["k"],
        record["activation_type"],
        record["bias_vector"],
    )


def reduced_signature_key(record: Dict[str, object]) -> Tuple[object, ...]:
    return (
        record["arch"],
        record["ta"],
        record["tb"],
        record["batch_count"],
        record["m"],
        record["n"],
        record["k"],
        record["activation_type"],
        record["bias_vector"],
    )


def iter_tuning_rows(path: Path) -> Iterable[Dict[str, str]]:
    header: List[str] | None = None
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("Git Version:"):
                continue
            columns = next(csv.reader([line]))
            if columns[0].strip() == "transA":
                header = [column.strip() for column in columns]
                continue
            if header is None:
                continue
            if len(columns) != len(header):
                continue
            yield {key: value.strip() for key, value in zip(header, columns)}


def load_tuning_index(path: Path) -> Dict[Tuple[object, ...], List[Dict[str, str]]]:
    index: Dict[Tuple[object, ...], List[Dict[str, str]]] = defaultdict(list)
    for row in iter_tuning_rows(path):
        index[tuning_row_key(row)].append(row)
    return index


def parse_query_counts(path: Path) -> Dict[Tuple[int, int, int], Dict[str, int]]:
    counts: Dict[Tuple[int, int, int], Dict[str, int]] = defaultdict(
        lambda: {
            "query_success_272": 0,
            "query_success_other": 0,
            "query_success_controller": 0,
            "query_success_provisional": 0,
            "query_unavailable": 0,
        }
    )
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = QUERY_RE.search(line)
            if not match:
                continue
            m = int(match.group("m"))
            n = int(match.group("n"))
            k = int(match.group("k"))
            dims_key = (m, n, k)
            bucket = int(match.group("bucket"))
            status = match.group("status")
            msg = match.group("msg")
            entry = counts[dims_key]
            if status == "success":
                if bucket == 272:
                    entry["query_success_272"] += 1
                else:
                    entry["query_success_other"] += 1
                if "controller snapshot" in msg:
                    entry["query_success_controller"] += 1
                if "provisional" in msg:
                    entry["query_success_provisional"] += 1
            elif status == "unavailable":
                entry["query_unavailable"] += 1
    return counts


def parse_override_summary(path: Path) -> Dict[Tuple[str, Tuple[object, ...]], Dict[str, object]]:
    last_seen: Dict[Tuple[str, Tuple[object, ...]], Dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = OVERRIDE_KEY_RE.search(line)
            if not match:
                continue
            record: Dict[str, object] = {
                "rank": match.group("rank"),
                "local_rank": match.group("local_rank"),
                "arch": match.group("arch"),
                "cu": int(match.group("cu")),
                "ta": match.group("ta"),
                "tb": match.group("tb"),
                "m": int(match.group("m")),
                "n": int(match.group("n")),
                "k": int(match.group("k")),
                "batch_count": int(match.group("b")),
                "activation_type": match.group("act"),
                "bias_vector": int(match.group("bias")),
                "calls": int(match.group("calls")),
                "cache_hits": int(match.group("cache_hits")),
                "cache_misses": int(match.group("cache_misses")),
                "successes": int(match.group("success")),
                "failures": int(match.group("failure")),
                "exact": int(match.group("exact")),
                "arch_default": int(match.group("arch_default")),
                "legacy": int(match.group("legacy")),
                "no_match": int(match.group("no_match")),
                "candidate_attempts": int(match.group("candidate_attempts")),
                "candidate_rejects": int(match.group("candidate_rejects")),
            }
            keyed = (record["rank"], summary_signature_key(record))
            last_seen[keyed] = record
    return last_seen


def enrich_record(
    record: Dict[str, object],
    tuning_index: Dict[Tuple[object, ...], List[Dict[str, str]]],
    query_counts: Dict[Tuple[int, int, int], Dict[str, int]],
) -> Dict[str, object]:
    exact_key = (
        record["arch"],
        record["cu"],
        record["ta"],
        record["tb"],
        record["batch_count"],
        record["m"],
        record["n"],
        record["k"],
        record["activation_type"],
        record["bias_vector"],
    )

    matching_rows = []
    for candidate_key, rows in tuning_index.items():
        (
            arch,
            cu,
            ta,
            tb,
            batch_count,
            m,
            n,
            k,
            _a_type,
            _b_type,
            _c_type,
            _d_type,
            _compute_type,
            activation_type,
            bias_vector,
            _bias_type,
            _aux_type,
        ) = candidate_key
        if (
            arch,
            cu,
            ta,
            tb,
            batch_count,
            m,
            n,
            k,
            activation_type,
            bias_vector,
        ) == exact_key:
            matching_rows.extend(rows)

    enriched = dict(record)
    dims_counts = query_counts.get((record["m"], record["n"], record["k"]), {})
    enriched.update(dims_counts)
    enriched.setdefault("query_success_272", 0)
    enriched.setdefault("query_success_other", 0)
    enriched.setdefault("query_success_controller", 0)
    enriched.setdefault("query_success_provisional", 0)
    enriched.setdefault("query_unavailable", 0)
    enriched["tuning_match_count"] = len(matching_rows)
    if matching_rows:
        field_names = (
            "a_type",
            "b_type",
            "c_type",
            "d_type",
            "compute_type",
            "bias_type",
            "aux_type",
            "gcnArchName",
            "solution_index",
        )
        for field_name in field_names:
            values = sorted({row[field_name] for row in matching_rows})
            enriched[field_name] = ";".join(values)
    else:
        enriched["a_type"] = ""
        enriched["b_type"] = ""
        enriched["c_type"] = ""
        enriched["d_type"] = ""
        enriched["compute_type"] = ""
        enriched["bias_type"] = ""
        enriched["aux_type"] = ""
        enriched["gcnArchName"] = ""
        enriched["solution_index"] = ""
    return enriched


def aggregate_override_records(
    last_seen: Dict[Tuple[str, Tuple[object, ...]], Dict[str, object]]
) -> Dict[Tuple[object, ...], Dict[str, object]]:
    aggregated: Dict[Tuple[object, ...], Dict[str, object]] = {}
    for (_, signature), record in last_seen.items():
        if signature not in aggregated:
            aggregated[signature] = {**record, "rank_count": 1}
            continue
        target = aggregated[signature]
        target["rank_count"] += 1
        for field_name in (
            "calls",
            "cache_hits",
            "cache_misses",
            "successes",
            "failures",
            "exact",
            "arch_default",
            "legacy",
            "no_match",
            "candidate_attempts",
            "candidate_rejects",
        ):
            target[field_name] += record[field_name]
    return aggregated


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
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
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tuning_index = load_tuning_index(args.override_file)
    query_counts = parse_query_counts(args.log)
    last_seen = parse_override_summary(args.log)
    aggregated = aggregate_override_records(last_seen)

    reduced_signatures_with_272 = {
        reduced_signature_key(record) for record in aggregated.values() if int(record["cu"]) == 272
    }

    enriched_rows: List[Dict[str, object]] = []
    for record in aggregated.values():
        enriched = enrich_record(record, tuning_index, query_counts)
        reduced_key = reduced_signature_key(enriched)
        enriched["has_272_pair"] = int(reduced_key in reduced_signatures_with_272)

        paired_304_rows = [
            row
            for row in tuning_index.get(
                (
                    enriched["arch"],
                    304,
                    enriched["ta"],
                    enriched["tb"],
                    enriched["batch_count"],
                    enriched["m"],
                    enriched["n"],
                    enriched["k"],
                    enriched["a_type"],
                    enriched["b_type"],
                    enriched["c_type"],
                    enriched["d_type"],
                    enriched["compute_type"],
                    enriched["activation_type"],
                    enriched["bias_vector"],
                    enriched["bias_type"],
                    enriched["aux_type"],
                ),
                [],
            )
        ]
        enriched["has_304_pair"] = int(bool(paired_304_rows))

        if int(enriched["cu"]) == 272:
            enriched["category"] = "auto272_hot"
        elif reduced_key in reduced_signatures_with_272:
            enriched["category"] = "paired304_hot"
        else:
            enriched["category"] = "always304_hot"

        enriched["score_target"] = int(
            enriched["category"] == "auto272_hot"
            and enriched["tuning_match_count"] == 1
            and enriched["has_304_pair"] == 1
        )
        enriched_rows.append(enriched)

    unavailable_rows: List[Dict[str, object]] = []
    dims_with_exact_rows = defaultdict(list)
    for row in enriched_rows:
        dims_with_exact_rows[(row["m"], row["n"], row["k"])].append(row)

    for (m, n, k), counts in query_counts.items():
        if counts["query_unavailable"] == 0 or counts["query_success_272"] > 0:
            continue
        matching_exact_rows = dims_with_exact_rows.get((m, n, k), [])
        if matching_exact_rows:
            for row in matching_exact_rows:
                unavailable_rows.append(
                    {
                        **row,
                        "category": "unavailable_hot",
                    }
                )
            continue
        unavailable_rows.append(
            {
                "category": "unavailable_hot",
                "arch": "",
                "cu": "",
                "ta": "",
                "tb": "",
                "batch_count": 1,
                "m": m,
                "n": n,
                "k": k,
                "activation_type": "",
                "bias_vector": "",
                "calls": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "successes": 0,
                "failures": 0,
                "exact": 0,
                "arch_default": 0,
                "legacy": 0,
                "no_match": 0,
                "candidate_attempts": 0,
                "candidate_rejects": 0,
                "rank_count": 0,
                "query_success_272": 0,
                "query_success_other": counts["query_success_other"],
                "query_success_controller": counts["query_success_controller"],
                "query_success_provisional": counts["query_success_provisional"],
                "query_unavailable": counts["query_unavailable"],
                "tuning_match_count": 0,
                "a_type": "",
                "b_type": "",
                "c_type": "",
                "d_type": "",
                "compute_type": "",
                "bias_type": "",
                "aux_type": "",
                "gcnArchName": "",
                "solution_index": "",
                "has_272_pair": 0,
                "has_304_pair": 0,
                "score_target": 0,
            }
        )

    exact_rows = sorted(
        enriched_rows,
        key=lambda row: (-int(row["calls"]), row["category"], int(row["m"]), int(row["n"]), int(row["k"])),
    )
    unavailable_rows = sorted(
        unavailable_rows,
        key=lambda row: (-int(row["query_unavailable"]), int(row.get("m", 0)), int(row.get("n", 0)), int(row.get("k", 0))),
    )
    corpus_rows = exact_rows + unavailable_rows
    target_rows = [
        row
        for row in exact_rows
        if int(row["score_target"]) == 1
    ][: args.top_n_targets]

    summary = {
        "log": str(args.log),
        "override_file": str(args.override_file),
        "query_dims": len(query_counts),
        "exact_keys": len(exact_rows),
        "auto272_hot": sum(1 for row in exact_rows if row["category"] == "auto272_hot"),
        "always304_hot": sum(1 for row in exact_rows if row["category"] == "always304_hot"),
        "paired304_hot": sum(1 for row in exact_rows if row["category"] == "paired304_hot"),
        "unavailable_hot": len(unavailable_rows),
        "score_targets": len(target_rows),
        "top_auto272": [
            {
                "ta": row["ta"],
                "tb": row["tb"],
                "m": row["m"],
                "n": row["n"],
                "k": row["k"],
                "calls": row["calls"],
                "bias_vector": row["bias_vector"],
            }
            for row in target_rows
        ],
    }

    write_csv(args.output_dir / "uopc_overlap_corpus.csv", corpus_rows)
    write_csv(args.output_dir / "uopc_overlap_targets.csv", target_rows)
    write_csv(args.output_dir / "uopc_overlap_unavailable.csv", unavailable_rows)
    (args.output_dir / "uopc_overlap_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
