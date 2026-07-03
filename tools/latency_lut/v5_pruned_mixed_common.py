from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import onnx
from onnx import numpy_helper


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "latency_lut"
BASELINE_ONNX = ROOT / (
    "tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/"
    "artifacts/onnx/fixedK29696/dynamic_agent_single_engine_maxK/"
    "lidar_pyramid_dynamic_agent_single_engine_maxK.onnx"
)
V4_LUT = OUT / "layer_width_precision_lut_measurements_v4.jsonl"
V5_LUT = OUT / "layer_width_precision_lut_measurements_v5.jsonl"


def load_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.is_file():
        return {} if default is None else default
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stable_hash(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def precision_values(profile: dict[str, Any]) -> list[str]:
    cfg = dict(profile or {})
    values = [str(cfg.get("default", "FP16")).upper()]
    values.extend(str(v).upper() for v in dict(cfg.get("overrides") or {}).values())
    return values


def is_mixed_precision(profile: dict[str, Any]) -> bool:
    return len(set(precision_values(profile))) > 1


def has_int8(profile: dict[str, Any]) -> bool:
    return any(v in {"INT8", "TRT_INT8_QDQ"} for v in precision_values(profile))


def has_all_three(profile: dict[str, Any]) -> bool:
    vals = {"INT8" if v == "TRT_INT8_QDQ" else v for v in precision_values(profile)}
    return {"FP32", "FP16", "INT8"}.issubset(vals)


def initializer_map(model: Any) -> dict[str, Any]:
    return {init.name: init for init in model.graph.initializer}


def producer_map(model: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for node in model.graph.node:
        for tensor in node.output:
            out[tensor] = node
    return out


def _find_initializer_name(name: str, initializers: dict[str, Any], producers: dict[str, Any], depth: int = 0) -> str | None:
    if name in initializers:
        return name
    if depth > 12:
        return None
    node = producers.get(name)
    if node is None or not node.input:
        return None
    if node.op_type in {"DequantizeLinear", "QuantizeLinear", "Cast", "Identity"}:
        return _find_initializer_name(node.input[0], initializers, producers, depth + 1)
    return None


def _attr_int(node: Any, name: str, default: int) -> int:
    for attr in node.attribute:
        if attr.name == name:
            return int(attr.i)
    return default


def conv_gemm_shapes(path: str | Path) -> tuple[dict[str, dict[str, Any]], int]:
    p = Path(path)
    if not p.is_file():
        return {}, 0
    model = onnx.load(str(p))
    initializers = initializer_map(model)
    producers = producer_map(model)
    param_count = 0
    for init in model.graph.initializer:
        param_count += int(numpy_helper.to_array(init).size)
    shapes: dict[str, dict[str, Any]] = {}
    for node in model.graph.node:
        if node.op_type == "Conv" and len(node.input) >= 2:
            init_name = _find_initializer_name(node.input[1], initializers, producers)
            if not init_name:
                continue
            arr = numpy_helper.to_array(initializers[init_name])
            if arr.ndim < 4:
                continue
            groups = _attr_int(node, "group", 1)
            shapes[node.name or node.output[0]] = {
                "op_type": "Conv",
                "weight": init_name,
                "C_out": int(arr.shape[0]),
                "C_in": int(arr.shape[1]) * int(groups),
                "kernel": [int(v) for v in arr.shape[2:]],
                "groups": int(groups),
                "weight_shape": [int(v) for v in arr.shape],
            }
        elif node.op_type in {"Gemm", "MatMul"} and len(node.input) >= 2:
            init_name = _find_initializer_name(node.input[1], initializers, producers)
            if not init_name:
                continue
            arr = numpy_helper.to_array(initializers[init_name])
            if arr.ndim < 2:
                continue
            shapes[node.name or node.output[0]] = {
                "op_type": node.op_type,
                "weight": init_name,
                "C_out": int(arr.shape[0]),
                "C_in": int(arr.shape[1]),
                "weight_shape": [int(v) for v in arr.shape],
            }
    return shapes, param_count


def changed_layers(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in sorted(set(baseline) | set(candidate)):
        b = baseline.get(name)
        c = candidate.get(name)
        if b != c:
            rows.append({"layer_name": name, "baseline": b, "candidate": c})
    return rows


def audit_width_changed_onnx(candidate_id: str, onnx_path: str | Path, *, candidate: dict[str, Any] | None = None) -> dict[str, Any]:
    base_shapes, base_params = conv_gemm_shapes(BASELINE_ONNX)
    cand_shapes, cand_params = conv_gemm_shapes(onnx_path)
    changes = changed_layers(base_shapes, cand_shapes)
    pruning = dict((candidate or {}).get("pruning") or {})
    uses_pruning = bool(pruning.get("enabled"))
    return {
        "candidate_id": candidate_id,
        "onnx_path": str(onnx_path),
        "is_baseline_topology": len(changes) == 0,
        "is_width_changed_subnet": len(changes) > 0,
        "is_pruned": uses_pruning,
        "uses_group_mask": bool((candidate or {}).get("group_mask") or (candidate or {}).get("pruning_config", {}).get("group_mask")),
        "uses_channel_resolver": True,
        "uses_coupled_groups": True,
        "uses_physical_pruning": uses_pruning,
        "baseline_param_count": base_params,
        "candidate_param_count": cand_params,
        "param_keep_ratio": float(cand_params / base_params) if base_params else None,
        "num_changed_conv_layers": len(changes),
        "changed_conv_layers": changes,
    }


def load_lut_rows(*paths: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            key = str(row.get("lut_key") or row.get("key_hash") or stable_hash(row))
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    return rows


def lut_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if r.get("valid", r.get("status") == "success")]
    return {
        "total": len(valid),
        "by_precision": dict(Counter(str(r.get("precision") or r.get("precision_profile") or "") for r in valid)),
        "by_width": dict(Counter(str(r.get("sampled_width") or r.get("C_out_aligned8") or r.get("C_out") or "") for r in valid)),
    }


def safe_id(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")
