from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import onnx
from onnx import numpy_helper


OUT_DIR = Path("outputs/latency_lut")
DATASET = OUT_DIR / "full_engine_calibration_dataset_v4.jsonl"
BASELINE_ONNX = Path(
    "tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/"
    "artifacts/onnx/fixedK29696/dynamic_agent_single_engine_maxK/"
    "lidar_pyramid_dynamic_agent_single_engine_maxK.onnx"
)
RESULT_DIR = OUT_DIR / "full_engine_results_v4"
CID_DIR = OUT_DIR / "full_engine_candidates_v4"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _result_for(candidate_id: str) -> dict[str, Any]:
    for suffix in (".result.json", ".debug2.result.json", ".debug.result.json"):
        data = _read_json(RESULT_DIR / f"{candidate_id}{suffix}")
        if data:
            return data
    return {}


def _candidate_for(candidate_id: str) -> dict[str, Any]:
    return _read_json(CID_DIR / f"{candidate_id}.json")


def _initializer_map(model: Any) -> dict[str, Any]:
    return {init.name: init for init in model.graph.initializer}


def _producer_map(model: Any) -> dict[str, Any]:
    prod: dict[str, Any] = {}
    for node in model.graph.node:
        for out in node.output:
            prod[out] = node
    return prod


def _find_initializer_name(name: str, initializers: dict[str, Any], producers: dict[str, Any], depth: int = 0) -> str | None:
    if name in initializers:
        return name
    if depth > 8:
        return None
    node = producers.get(name)
    if node is None:
        return None
    # Q/DQ and Cast wrap a tensor without changing its logical weight shape.
    if node.op_type in {"DequantizeLinear", "QuantizeLinear", "Cast", "Identity"} and node.input:
        return _find_initializer_name(node.input[0], initializers, producers, depth + 1)
    return None


def _attr_int(node: Any, name: str, default: int) -> int:
    for attr in node.attribute:
        if attr.name == name:
            return int(attr.i)
    return default


def _conv_gemm_shapes(path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    model = onnx.load(str(path))
    initializers = _initializer_map(model)
    producers = _producer_map(model)
    param_count = 0
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init)
        param_count += int(arr.size)
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


def _changed_layers(base: dict[str, Any], cand: dict[str, Any]) -> list[dict[str, Any]]:
    changed: list[dict[str, Any]] = []
    for name in sorted(set(base) | set(cand)):
        b = base.get(name)
        c = cand.get(name)
        if b != c:
            changed.append({"layer_name": name, "baseline": b, "candidate": c})
    return changed


def _precision_config(candidate: dict[str, Any]) -> dict[str, Any]:
    return dict(candidate.get("precision_config") or {})


def _precision_lists(candidate: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    cfg = _precision_config(candidate)
    default = str(cfg.get("default", "FP16")).upper()
    overrides = {str(k): str(v).upper() for k, v in dict(cfg.get("overrides") or {}).items()}
    fp32 = [k for k, v in overrides.items() if v == "FP32"]
    int8 = [k for k, v in overrides.items() if v == "INT8"]
    fp16 = [k for k, v in overrides.items() if v == "FP16"]
    if default == "FP16":
        fp16.append("__default__")
    elif default == "FP32":
        fp32.append("__default__")
    elif default == "INT8":
        int8.append("__default__")
    return sorted(fp32), sorted(fp16), sorted(int8)


def _read_report(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    return _read_json(Path(path))


def _route2_reports(result: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    route = dict(result.get("route2_prepare") or {})
    typed = _read_report(route.get("typed_report"))
    qdq = _read_report(route.get("qdq_report"))
    resolved = _read_report(Path(route.get("typed_report", "")).with_name("resolved_precision_profile.json") if route.get("typed_report") else None)
    return typed, qdq, resolved


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = _read_jsonl(DATASET)
    baseline_shapes, baseline_params = _conv_gemm_shapes(BASELINE_ONNX)
    labels: list[dict[str, Any]] = []
    mixed: list[dict[str, Any]] = []
    for row in rows:
        cid = str(row.get("candidate_id"))
        result = _result_for(cid)
        candidate = _candidate_for(cid)
        onnx_path = Path(row.get("onnx_path") or result.get("onnx_path") or "")
        if not onnx_path.is_file():
            # Fall back to the exported pre-rewrite ONNX if the dataset points
            # at a missing typed/QDQ graph.
            work_onnx = RESULT_DIR / f"{cid}.work/quant_deploy/artifacts/onnx/fixedK29696/dynamic_agent_single_engine_maxK/lidar_pyramid_dynamic_agent_single_engine_maxK.onnx"
            onnx_path = work_onnx if work_onnx.is_file() else onnx_path
        cand_shapes, cand_params = _conv_gemm_shapes(onnx_path) if onnx_path.is_file() else ({}, 0)
        changed = _changed_layers(baseline_shapes, cand_shapes)
        pruning_cfg = dict(candidate.get("pruning") or {})
        uses_group_mask = bool(candidate.get("group_mask"))
        uses_physical_pruning = bool(pruning_cfg.get("enabled")) or bool(result.get("pruned_checkpoint") or result.get("prune_replay"))
        precision_cfg = _precision_config(candidate)
        overrides = dict(precision_cfg.get("overrides") or {})
        mixed_precision = bool(overrides) or str(precision_cfg.get("default", "")).upper() not in {"", "FP16"}
        typed, qdq, resolved = _route2_reports(result)
        route = dict(result.get("route2_prepare") or {})
        pv = dict(result.get("precision_verification") or {})
        route2_typed = bool(route.get("route2_onnx_path") or route.get("typed_report")) and Path(route.get("typed_report", "")).is_file()
        route2_qdq = bool(route.get("qdq_report")) and Path(route.get("qdq_report", "")).is_file()
        uses_qdq = bool(result.get("uses_qdq") or qdq.get("uses_qdq"))
        is_route2 = bool(route2_typed and pv.get("success", row.get("precision_verification_failures") == []))
        is_trt_auto = bool(mixed_precision and not route2_typed)
        label = {
            "candidate_id": cid,
            "engine_path": row.get("engine_path") or result.get("engine_path"),
            "onnx_path": str(onnx_path),
            "is_baseline_topology": len(changed) == 0,
            "is_width_changed_subnet": len(changed) > 0,
            "is_pruned": uses_physical_pruning or uses_group_mask,
            "is_pruned_mixed": bool((uses_physical_pruning or uses_group_mask) and mixed_precision),
            "baseline_param_count": baseline_params,
            "candidate_param_count": cand_params,
            "param_keep_ratio": float(cand_params / baseline_params) if baseline_params else None,
            "baseline_conv_shapes": baseline_shapes,
            "candidate_conv_shapes": cand_shapes,
            "changed_conv_layers": changed,
            "num_changed_conv_layers": len(changed),
            "uses_group_mask": uses_group_mask,
            "uses_channel_resolver": bool(candidate.get("channel_resolver") or result.get("channel_resolver")),
            "uses_coupled_groups": bool(candidate.get("coupled_groups") or result.get("coupled_groups")),
            "uses_physical_pruning": uses_physical_pruning,
            "precision_config_requested": precision_cfg,
            "precision_profile_resolved": resolved.get("resolved_precision_config", {}),
            "route2_typed_onnx_generated": route2_typed,
            "route2_qdq_onnx_generated": route2_qdq,
            "uses_explicit_qdq": uses_qdq,
            "observed_fp32_layers": int(pv.get("observed_fp32_layers") or row.get("observed_fp32_layers") or 0),
            "observed_fp16_layers": int(pv.get("observed_fp16_layers") or row.get("observed_fp16_layers") or 0),
            "observed_int8_layers": int(pv.get("observed_int8_layers") or row.get("observed_int8_layers") or 0),
            "precision_verification_failures": pv.get("failures", row.get("precision_verification_failures", [])),
            "is_trt_auto_precision_only": is_trt_auto,
            "is_route2_explicit_precision": is_route2,
        }
        labels.append(label)
        fp32_req, fp16_req, int8_req = _precision_lists(candidate)
        obs_by_unit = dict(pv.get("observed_by_unit") or {})
        observed_fp32 = []
        observed_fp16 = []
        observed_int8 = []
        for unit, entries in obs_by_unit.items():
            for entry in entries:
                rowp = {"unit": unit, **entry}
                if entry.get("precision") == "FP32":
                    observed_fp32.append(rowp)
                elif entry.get("precision") == "FP16":
                    observed_fp16.append(rowp)
                elif entry.get("precision") == "INT8":
                    observed_int8.append(rowp)
        requested_units = set(fp32_req + int8_req + [u for u in fp16_req if u != "__default__"])
        observed_units = set(obs_by_unit)
        mismatched = sorted(requested_units - observed_units)
        mixed.append(
            {
                "candidate_id": cid,
                "requested_fp32_units": fp32_req,
                "requested_fp16_units": fp16_req,
                "requested_int8_units": int8_req,
                "resolved_fp32_units": sorted([k for k, v in dict(resolved.get("resolved_precision_config") or {}).items() if str(v).upper() in {"FP32", "TRT_FP32"}]),
                "resolved_fp16_units": sorted([k for k, v in dict(resolved.get("resolved_precision_config") or {}).items() if str(v).upper() in {"FP16", "TRT_FP16"}]),
                "resolved_int8_units": sorted([k for k, v in dict(resolved.get("resolved_precision_config") or {}).items() if str(v).upper() in {"INT8", "TRT_INT8_QDQ"}]),
                "typed_cast_nodes_inserted": int(typed.get("num_cast_inserted") or len(typed.get("inserted_cast_nodes") or [])),
                "qdq_nodes_inserted": int(qdq.get("num_qdq_nodes_inserted") or 0),
                "observed_fp32_layers": observed_fp32,
                "observed_fp16_layers": observed_fp16,
                "observed_int8_layers": observed_int8,
                "requested_vs_observed_match": bool(not mismatched and not label["precision_verification_failures"]),
                "mismatched_units": mismatched,
                "precision_verification_failures": label["precision_verification_failures"],
            }
        )
    summary = {
        "total_full_engine_labels": len(labels),
        "baseline_topology_labels": sum(1 for x in labels if x["is_baseline_topology"]),
        "width_changed_subnet_labels": sum(1 for x in labels if x["is_width_changed_subnet"]),
        "pruned_labels": sum(1 for x in labels if x["is_pruned"]),
        "pruned_mixed_labels": sum(1 for x in labels if x["is_pruned_mixed"]),
        "route2_explicit_precision_labels": sum(1 for x in labels if x["is_route2_explicit_precision"]),
        "trt_auto_precision_only_labels": sum(1 for x in labels if x["is_trt_auto_precision_only"]),
        "fp32_fp16_mixed_labels": sum(1 for x in labels if x["observed_fp32_layers"] > 0 and x["observed_fp16_layers"] > 0 and x["observed_int8_layers"] == 0),
        "int8_containing_labels": sum(1 for x in labels if x["observed_int8_layers"] > 0),
        "fp32_fp16_int8_labels": sum(1 for x in labels if x["observed_fp32_layers"] > 0 and x["observed_fp16_layers"] > 0 and x["observed_int8_layers"] > 0),
    }
    payload = {
        "baseline_onnx": str(BASELINE_ONNX),
        "summary": summary,
        "labels": labels,
        "mixed_precision_candidates": mixed,
    }
    (OUT_DIR / "full_engine_structure_and_precision_audit_v4.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md = ["# Full-Engine Structure and Precision Audit v4", "", "## Summary", "", "```json", json.dumps(summary, indent=2, ensure_ascii=False), "```", ""]
    if summary["width_changed_subnet_labels"] == 0:
        md.append("当前 full-engine calibration dataset 还没有包含 layer 宽度变化后的完整子网引擎。")
        md.append("")
    if summary["pruned_mixed_labels"] == 0:
        md.append("当前还没有剪枝结构 + 混合精度量化的 full-engine calibration label。")
        md.append("")
    md.append("## Per-Label Audit")
    for item in labels:
        md.append(f"- `{item['candidate_id']}`: baseline_topology={item['is_baseline_topology']}, width_changed={item['is_width_changed_subnet']}, pruned={item['is_pruned']}, route2={item['is_route2_explicit_precision']}, qdq={item['uses_explicit_qdq']}, observed=FP32:{item['observed_fp32_layers']} FP16:{item['observed_fp16_layers']} INT8:{item['observed_int8_layers']}, changed_layers={item['num_changed_conv_layers']}")
    (OUT_DIR / "full_engine_structure_and_precision_audit_v4.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    if summary["width_changed_subnet_labels"] == 0:
        gap = {
            "can_current_dataset_train_width_changed_latency_proxy": False,
            "reason": "no_width_changed_full_engine_labels",
            "required_next_step": "connect ChannelResolver/coupled groups replay to physical pruning export and Route2 mixed full-engine batch",
        }
    else:
        gap = {
            "can_current_dataset_train_width_changed_latency_proxy": True,
            "reason": "width_changed_full_engine_labels_present",
            "required_next_step": "expand width-changed/pruned-mixed label count before calibration",
        }
    gap_md = ["# Pruned / Width-Changed Full-Engine Gap Report", "", "```json", json.dumps(gap, indent=2, ensure_ascii=False), "```", ""]
    if not gap["can_current_dataset_train_width_changed_latency_proxy"]:
        gap_md.append("Current v4 labels must not be used as structure-pruned latency labels. Required flow:")
        gap_md.extend([
            "1. Connect ChannelResolver / coupled groups replay.",
            "2. Generate true width-changed structure candidates.",
            "3. Export physically pruned candidate ONNX.",
            "4. Apply Route2 Cast/QDQ rewrite to the pruned ONNX.",
            "5. Build TensorRT full-engine.",
            "6. Generate pruned+mixed labels.",
        ])
    (OUT_DIR / "pruned_width_changed_full_engine_gap_report.md").write_text("\n".join(gap_md) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
