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

from opencood.tools.compression.latency_lut.schema import DEPLOY_MODE, FIXED_K, LatencyLUTKey
from opencood.tools.compression.latency_lut.tensorRT_benchmark import SUCCESS_STATUSES, benchmark_key


PROFILE = {"FP32": "TRT_FP32", "FP16": "TRT_FP16", "INT8_QDQ": "TRT_INT8_QDQ"}
WEIGHT = {"FP32": "FP32", "FP16": "FP16", "INT8_QDQ": "INT8"}


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _scale_entry(unit_id: str, cache: dict[str, Any]) -> dict[str, Any] | None:
    units = cache.get("units") or {}
    if unit_id in units:
        return units[unit_id]
    for key, value in units.items():
        if key == unit_id or key.startswith(unit_id + ".") or unit_id in key:
            return value
    return None


def _scale_values(unit_id: str, cache: dict[str, Any]) -> tuple[float | None, float | None, str | None]:
    entry = _scale_entry(unit_id, cache)
    if not entry:
        return None, None, None
    act = None
    for tensors in (entry.get("input_activation_tensors") or {}, entry.get("output_activation_tensors") or {}):
        for value in tensors.values():
            if isinstance(value, dict) and value.get("scale") is not None:
                act = float(value["scale"])
                break
        if act is not None:
            break
    weight = None
    for value in (entry.get("weight_tensors") or {}).values():
        scale = value.get("scale") if isinstance(value, dict) else None
        if isinstance(scale, list) and scale:
            weight = max(float(x) for x in scale)
            break
        if scale is not None:
            weight = float(scale)
            break
    return act, weight, str(entry.get("scale_source") or "train_calib200")


def _module_block(row: dict[str, Any]) -> tuple[str, str]:
    unit_type = str(row["unit_type"])
    stage = str(row.get("parent_stage") or "")
    if unit_type == "gemm":
        return "pfn", "pfn_block"
    if unit_type == "residual_block":
        return "backbone", "residual_block"
    if unit_type == "shrink":
        return "shrink", "compression_1x1"
    if unit_type == "fusion":
        return "pyramid_fusion", "fusion_block"
    if unit_type == "head":
        return "detection_head", "head_branch"
    if "shrink" in stage:
        return "shrink", "compression_1x1"
    if "fusion" in stage:
        return "pyramid_fusion", "fusion_block"
    if "head" in stage:
        return "detection_head", "head_branch"
    return "backbone", "conv_block"


def _key(row: dict[str, Any], cache: dict[str, Any]) -> LatencyLUTKey:
    module, block = _module_block(row)
    precision = str(row["precision"])
    act, wscale, source = _scale_values(str(row["unit_id"]), cache)
    metadata = {
        "atomic_unit_id": row["unit_id"],
        "layer_width_key": row["lut_key"],
        "sampled_width": row["sampled_width"],
        "parent_stage": row.get("parent_stage"),
    }
    if precision == "INT8_QDQ":
        metadata.update({"activation_scale": act, "weight_scale": wscale, "scale_source": source})
    return LatencyLUTKey(
        deploy_mode=DEPLOY_MODE,
        fixed_K=FIXED_K,
        module_name=module,
        block_name=str(row["unit_id"]),
        block_type=block,
        H=row.get("H"),
        W=row.get("W"),
        C_in=row.get("C_in_aligned8"),
        C_mid=row.get("C_out_aligned8") if block == "residual_block" else None,
        C_out=row.get("C_out_aligned8"),
        kernel_size=row.get("kernel_size"),
        stride=row.get("stride"),
        padding=(int(row.get("kernel_size") or 1) // 2 if int(row.get("kernel_size") or 1) > 1 else 0),
        groups=row.get("groups"),
        precision_profile=PROFILE[precision],
        weight_precision=WEIGHT[precision],
        activation_precision=WEIGHT[precision],
        compute_precision=WEIGHT[precision],
        metadata=metadata,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    key_space = json.loads(Path(args.key_space).read_text(encoding="utf-8"))
    rows = [row for row in key_space.get("keys", []) if row.get("is_shape_legal")]
    cache = json.loads(Path(args.scale_cache).read_text(encoding="utf-8")) if Path(args.scale_cache).is_file() else {}
    existing = _read_jsonl(args.output)
    valid_existing = [row for row in existing if row.get("valid")]
    if args.limit is not None and int(args.limit) < 500 and len(valid_existing) < 500:
        raise SystemExit("limit < 500 is forbidden until valid measured records >= 500")
    done = {row.get("lut_key") for row in valid_existing} if args.resume else set()
    selected = [row for row in rows if row.get("lut_key") not in done]
    if args.precision_filter:
        allowed = set(args.precision_filter)
        selected = [row for row in selected if row.get("precision") in allowed]
    if args.limit is not None:
        selected = selected[: int(args.limit)]
    failures: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        if row["precision"] == "INT8_QDQ" and not row.get("activation_scale_available"):
            failure = {"lut_key": row["lut_key"], "unit_id": row["unit_id"], "precision": row["precision"], "valid": False, "invalid_reason": "missing_activation_scale"}
            failures.append(failure)
            _append_jsonl(args.output, {**row, **failure, "source": "invalid"})
            continue
        print(f"[layer-width-v4] {index}/{len(selected)} {row['unit_id']} width={row['sampled_width']} {row['precision']}")
        record = benchmark_key(
            _key(row, cache),
            output_dir=args.work_dir,
            onnx_dir=args.onnx_dir,
            engine_dir=args.engine_dir,
            warmup=int(args.warmup),
            repeat=int(args.repeat),
            dry_run=False,
            trtexec_path=args.trtexec,
            device=args.device,
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
        _append_jsonl(args.output, measurement)
        if not measurement["valid"]:
            failures.append(measurement)
    all_rows = _read_jsonl(args.output)
    valid = [row for row in all_rows if row.get("valid")]
    stats = {
        "selected_this_run": len(selected),
        "valid_measured_records": len(valid),
        "fp32_measured_records": sum(1 for row in valid if row.get("precision") == "FP32"),
        "fp16_measured_records": sum(1 for row in valid if row.get("precision") == "FP16"),
        "int8_qdq_measured_records": sum(1 for row in valid if row.get("precision") == "INT8_QDQ"),
        "residual_block_measured_records": sum(1 for row in valid if row.get("unit_type") == "residual_block"),
        "failure_count": len(failures),
        "failure_by_reason": dict(Counter(row.get("invalid_reason") for row in failures)),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("# Layer Width Precision LUT Measurements v4 Report\n\n" + "\n".join(f"- {k}: {v}" for k, v in stats.items()) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key-space", "--key_space", dest="key_space", default="outputs/latency_lut/layer_width_precision_key_space_v4.json")
    parser.add_argument("--scale-cache", "--scale_cache", dest="scale_cache", default="outputs/latency_lut/activation_scale_cache_calib200.json")
    parser.add_argument("--output", default="outputs/latency_lut/layer_width_precision_lut_measurements_v4.jsonl")
    parser.add_argument("--report", default="outputs/latency_lut/layer_width_precision_lut_measurements_v4_report.md")
    parser.add_argument("--work-dir", "--work_dir", dest="work_dir", default="outputs/latency_lut/layer_width_v4_subgraphs")
    parser.add_argument("--onnx-dir", "--onnx_dir", dest="onnx_dir", default="outputs/latency_lut/layer_width_v4_onnx")
    parser.add_argument("--engine-dir", "--engine_dir", dest="engine_dir", default="outputs/latency_lut/layer_width_v4_engine")
    parser.add_argument("--trtexec", default="/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118/targets/x86_64-linux-gnu/bin/trtexec")
    parser.add_argument("--plugin", default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--min-repeat-ms", "--min_repeat_ms", dest="min_repeat_ms", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--precision-filter", "--precision_filter", dest="precision_filter", nargs="*", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
