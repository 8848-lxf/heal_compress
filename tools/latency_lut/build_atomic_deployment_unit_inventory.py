from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


MERGE_OPS = {"Add", "Sub", "Mul", "Div"}


def _scope_for_node(name: str) -> tuple[str, str, str]:
    low = name.lower()
    if "pointpillarscattertrt" in low:
        return "scatter", "scatter", "plugin"
    if "shrink_conv" in low:
        return "shrink", "shrink", "shrink"
    if "cls_head" in low or "reg_head" in low or "dir_head" in low:
        return "detection_head", "detection_head", "detection_head"
    if "gridsample" in low or "single_head" in low:
        return "pyramid_fusion", "pyramid_fusion", "pyramid_fusion"
    if "layer0" in low:
        return "backbone.stage1", "backbone", "backbone.stage1"
    if "layer1" in low:
        return "backbone.stage2", "backbone", "backbone.stage2"
    if "layer2" in low:
        return "backbone.stage3", "backbone", "backbone.stage3"
    if "pillar_vfe" in low or "pfn" in low or "matmul" in low:
        return "pfn", "pfn", "pfn"
    return "other", "other", "other"


def _unit_type(op_type: str, name: str) -> str:
    if op_type == "Conv":
        return "conv"
    if op_type in {"Gemm", "MatMul"}:
        return "gemm"
    if op_type == "PointPillarScatterTRT":
        return "plugin"
    if op_type == "GridSample":
        return "grid_sample"
    if op_type in MERGE_OPS:
        return "merge"
    if op_type == "Concat":
        return "concat"
    return "other"


def _supported_precision(op_type: str) -> tuple[list[str], bool, str]:
    if op_type in {"Conv", "Gemm", "MatMul"}:
        return ["FP32", "FP16", "INT8"], True, "Conv/GEMM atomic unit can be Q/DQ quantized when calib200 scales are available"
    if op_type == "PointPillarScatterTRT":
        return ["FP32", "FP16"], False, "PointPillarScatterTRT INT8 support is not verified"
    if op_type in {"Add", "Sub", "Mul", "Div", "Concat", "GridSample"}:
        return ["FP32", "FP16"], False, f"{op_type} merge/boundary is kept floating point; INT8 merge is not implemented"
    return ["FP32", "FP16"], False, "non-Conv/GEMM op is not an INT8 search unit"


def build_inventory(onnx_path: str | Path) -> list[dict[str, Any]]:
    import onnx

    model = onnx.load(str(onnx_path))
    producer: dict[str, str] = {}
    for node in model.graph.node:
        node_name = str(node.name or (node.output[0] if node.output else ""))
        for output in node.output:
            producer[str(output)] = node_name

    units: list[dict[str, Any]] = []
    for idx, node in enumerate(model.graph.node):
        if node.op_type not in {"Conv", "Gemm", "MatMul", "PointPillarScatterTRT", "GridSample", "Add", "Sub", "Mul", "Div", "Concat"}:
            continue
        node_name = str(node.name or (node.output[0] if node.output else f"node_{idx}"))
        parent_stage, parent_block, module_path = _scope_for_node(node_name)
        unit_type = _unit_type(node.op_type, node_name)
        supported, int8_supported, reason = _supported_precision(node.op_type)
        unit_id = module_path if module_path in {"shrink", "detection_head", "pyramid_fusion", "scatter", "pfn"} and node.op_type in {"PointPillarScatterTRT"} else node_name.strip("/").replace("/", ".")
        if module_path == "shrink" and node.op_type == "Conv":
            unit_id = "shrink" if ".double_conv.0." in unit_id else f"shrink.{unit_id.split('.')[-2]}"
        requires_same = [producer[name] for name in node.input if name in producer and node.op_type in {"Add", "Sub", "Mul", "Div", "Concat", "GridSample"}]
        units.append(
            {
                "unit_id": unit_id,
                "module_path": module_path,
                "unit_type": "conv_bn_act" if node.op_type == "Conv" else unit_type,
                "parent_stage": parent_stage,
                "parent_block": parent_block,
                "covered_pytorch_modules": [],
                "covered_onnx_nodes": [node_name],
                "covered_trt_layers": [node_name],
                "input_tensors": list(node.input),
                "output_tensors": list(node.output),
                "weight_tensors": [name for name in node.input[1:2]],
                "supported_precision": supported,
                "int8_supported": int8_supported,
                "int8_reason": reason,
                "requires_same_dtype_with": requires_same,
                "merge_op": node.op_type in MERGE_OPS,
                "plugin_boundary": node.op_type == "PointPillarScatterTRT",
                "grid_sample_boundary": node.op_type == "GridSample",
                "can_be_independently_mixed": node.op_type in {"Conv", "Gemm", "MatMul"},
            }
        )
    return units


def write_report(units: list[dict[str, Any]], inventory_path: Path, report_path: Path) -> None:
    fp_units = [u for u in units if {"FP32", "FP16"}.issubset(set(u["supported_precision"]))]
    int8_units = [u for u in units if u["int8_supported"]]
    fixed = [u for u in units if not u["int8_supported"]]
    lines = [
        "# Atomic Deployment Unit Inventory Report",
        "",
        f"- inventory: `{inventory_path}`",
        f"- total_atomic_deployment_units: {len(units)}",
        f"- supports_FP32_FP16: {len(fp_units)}",
        f"- initial_INT8_supported: {len(int8_units)}",
        f"- fixed_or_float_only_units: {len(fixed)}",
        "",
        "INT8 is only enabled for Conv/GEMM/MatMul units with activation and weight scales. Add/Concat/GridSample/fusion merge and plugin units stay FP16/FP32 in this first route2 implementation.",
        "",
        "## Float-only / boundary examples",
    ]
    for unit in fixed[:40]:
        lines.append(f"- `{unit['unit_id']}` ({unit['unit_type']}): {unit['int8_reason']}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--output", default="outputs/latency_lut/atomic_deployment_unit_inventory.json")
    parser.add_argument("--report", default="outputs/latency_lut/atomic_deployment_unit_inventory_report.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    units = build_inventory(args.onnx)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"success": True, "units": units}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(units, out, Path(args.report))
    print(json.dumps({"success": True, "num_units": len(units), "output": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
