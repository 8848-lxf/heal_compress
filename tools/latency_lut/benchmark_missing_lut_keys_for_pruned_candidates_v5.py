from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from opencood.tools.compression.latency_lut.tensorRT_benchmark import SUCCESS_STATUSES, benchmark_key
from tools.latency_lut.benchmark_layer_width_precision_lut import _key as _row_to_lut_key
from tools.latency_lut.v5_pruned_mixed_common import append_jsonl, load_json, read_jsonl, stable_hash, write_json


def _missing_rows(decomp_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(decomp_dir.glob("*.json")):
        data = load_json(path)
        for missing in data.get("missing_keys") or []:
            row = {
                "unit_id": missing["layer_name"],
                "module_path": missing["layer_name"],
                "onnx_node": missing["layer_name"],
                "unit_type": "conv_bn_act",
                "parent_stage": "",
                "parent_block": "",
                "H": 16,
                "W": 16,
                "kernel_size": (missing.get("kernel") or [1])[0] if isinstance(missing.get("kernel"), list) else 1,
                "stride": 1,
                "groups": int(missing.get("groups") or 1),
                "C_in_original": int(missing.get("C_in") or missing.get("C_in_aligned8") or 1),
                "C_out_original": int(missing.get("C_out") or missing.get("C_out_aligned8") or 1),
                "sampled_width": int(missing.get("C_out_aligned8") or missing.get("C_out") or 1),
                "C_in_aligned8": int(missing.get("C_in_aligned8") or missing.get("C_in") or 1),
                "C_out_aligned8": int(missing.get("C_out_aligned8") or missing.get("C_out") or 1),
                "precision": str(missing["precision"]),
                "activation_scale_required": str(missing["precision"]) == "INT8_QDQ",
                "activation_scale_available": str(missing["precision"]) != "INT8_QDQ",
                "is_shape_legal": True,
                "illegal_reason": "",
                "synthetic_width": False,
            }
            row["lut_key"] = stable_hash(row)
            rows.append(row)
    by_key = {row["lut_key"]: row for row in rows}
    return list(by_key.values())


def run(args: argparse.Namespace) -> dict[str, Any]:
    missing = _missing_rows(Path(args.decomposition_dir))
    existing = read_jsonl(args.output)
    done = {row.get("lut_key") for row in existing if row.get("valid", row.get("status") in SUCCESS_STATUSES)}
    cache = load_json(args.scale_cache, {})
    selected = [row for row in missing if row["lut_key"] not in done]
    failures: list[dict[str, Any]] = []
    measured = 0
    for idx, row in enumerate(selected, start=1):
        if row["precision"] == "INT8_QDQ":
            # Do not use dummy scale. If no explicit scale exists for this
            # pruned tensor name, the key remains unsupported until calib200
            # collection maps that pruned ONNX tensor.
            units = cache.get("units") or {}
            if row["unit_id"] not in units and not any(row["unit_id"] in key or key in row["unit_id"] for key in units):
                failure = {**row, "valid": False, "source": "failed", "status": "missing_activation_scale", "invalid_reason": "missing_activation_scale"}
                append_jsonl(args.output, failure)
                failures.append(failure)
                continue
        print(f"[missing-lut-v5] {idx}/{len(selected)} {row['unit_id']} {row['precision']} C={row['C_in_aligned8']}->{row['C_out_aligned8']}")
        record = benchmark_key(
            _row_to_lut_key(row, cache),
            output_dir=args.work_dir,
            onnx_dir=args.onnx_dir,
            engine_dir=args.engine_dir,
            warmup=int(args.warmup),
            repeat=int(args.repeat),
            dry_run=False,
            trtexec_path=args.trtexec,
            device=int(args.device),
            iterations=args.iterations,
            min_repeat_ms=args.min_repeat_ms,
            plugin_path=args.plugin,
        )
        measurement = {
            **row,
            "benchmark_backend": "trt_subgraph",
            "engine_mode": "subgraph_engine",
            "warmup": int(args.warmup),
            "repeat": int(args.repeat),
            "latency_p50_ms": float(record.latency_p50_ms),
            "latency_p90_ms": float(record.latency_p90_ms),
            "latency_mean_ms": float(record.latency_mean_ms),
            "latency_std_ms": float(record.latency_std_ms),
            "source": "measured" if record.status in SUCCESS_STATUSES else "failed",
            "valid": record.status in SUCCESS_STATUSES,
            "invalid_reason": "" if record.status in SUCCESS_STATUSES else (record.error_message or record.status),
            "status": record.status,
            "key_hash": record.key_hash,
            "engine_hash": record.engine_hash,
            "onnx_hash": record.onnx_hash,
        }
        append_jsonl(args.output, measurement)
        if measurement["valid"]:
            measured += 1
        else:
            failures.append(measurement)
    all_rows = read_jsonl(args.output)
    stats = {
        "missing_keys_before_on_demand": len(missing),
        "selected_this_run": len(selected),
        "new_valid_measurements": measured,
        "total_v5_records": len([r for r in all_rows if r.get("valid")]),
        "failures": len(failures),
        "failure_by_reason": dict(Counter(r.get("invalid_reason") or r.get("status") for r in failures)),
    }
    write_json(args.report_json, stats)
    lines = ["# Missing LUT Key Benchmark v5", "", *[f"- {k}: {v}" for k, v in stats.items()]]
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decomposition-dir", "--decomposition_dir", dest="decomposition_dir", default="outputs/latency_lut/pruned_candidate_lut_decomposition_v5")
    parser.add_argument("--output", default="outputs/latency_lut/layer_width_precision_lut_measurements_v5.jsonl")
    parser.add_argument("--scale-cache", "--scale_cache", dest="scale_cache", default="outputs/latency_lut/activation_scale_cache_calib200.json")
    parser.add_argument("--work-dir", "--work_dir", dest="work_dir", default="outputs/latency_lut/layer_width_v5_missing_subgraphs")
    parser.add_argument("--onnx-dir", "--onnx_dir", dest="onnx_dir", default="outputs/latency_lut/layer_width_v5_missing_onnx")
    parser.add_argument("--engine-dir", "--engine_dir", dest="engine_dir", default="outputs/latency_lut/layer_width_v5_missing_engine")
    parser.add_argument("--trtexec", default="${TENSORRT_ROOT}/targets/x86_64-linux-gnu/bin/trtexec")
    parser.add_argument("--plugin", default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--min-repeat-ms", "--min_repeat_ms", dest="min_repeat_ms", type=int, default=None)
    parser.add_argument("--report", default="outputs/latency_lut/missing_lut_key_benchmark_v5_report.md")
    parser.add_argument("--report-json", "--report_json", dest="report_json", default="outputs/latency_lut/missing_lut_key_benchmark_v5_report.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
