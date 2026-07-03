from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import onnx

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.latency_lut.validate_onnx_dtype_closure_v6 import infer_tensor_dtypes


def _load_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.is_file():
        return {} if default is None else default
    return json.loads(p.read_text(encoding="utf-8"))


def _tail(text: str, n: int = 4000) -> str:
    return text[-n:] if text else ""


def _candidate_id_to_profile(candidate_id: str) -> tuple[str, str]:
    if "__" in candidate_id:
        return candidate_id.split("__", 1)
    return candidate_id, ""


def _producer_map(model: Any) -> dict[str, Any]:
    return {out: node for node in model.graph.node for out in node.output}


def _consumer_map(model: Any) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for node in model.graph.node:
        for inp in node.input:
            out.setdefault(inp, []).append(node.name)
    return out


def _mismatched_nodes(onnx_path: str | Path) -> list[dict[str, Any]]:
    p = Path(onnx_path)
    if not p.is_file():
        return []
    model = onnx.load(str(p))
    dtypes = infer_tensor_dtypes(model)
    producers = _producer_map(model)
    consumers = _consumer_map(model)
    rows: list[dict[str, Any]] = []
    for node in model.graph.node:
        if node.op_type not in {"Conv", "Gemm", "MatMul"}:
            continue
        input_tensor = node.input[0] if node.input else ""
        weight_tensor = node.input[1] if len(node.input) > 1 else ""
        bias_tensor = node.input[2] if len(node.input) > 2 else ""
        input_dtype = dtypes.get(input_tensor, "UNKNOWN")
        weight_dtype = dtypes.get(weight_tensor, "UNKNOWN")
        bias_dtype = dtypes.get(bias_tensor, input_dtype) if bias_tensor else ""
        if input_dtype == weight_dtype and (not bias_tensor or bias_dtype in {input_dtype, weight_dtype}):
            continue
        if input_dtype == "FLOAT" and weight_dtype == "FLOAT16":
            reason = "input_float_kernel_half"
        elif input_dtype == "FLOAT16" and weight_dtype == "FLOAT":
            reason = "input_half_kernel_float"
        else:
            reason = "cast_boundary_missing"
        rows.append(
            {
                "onnx_node_name": node.name,
                "op_type": node.op_type,
                "requested_precision": "unknown_from_graph",
                "input_tensor": input_tensor,
                "input_dtype": input_dtype,
                "weight_tensor": weight_tensor,
                "weight_dtype": weight_dtype,
                "bias_tensor": bias_tensor,
                "bias_dtype": bias_dtype or "UNKNOWN",
                "producer_node": producers.get(input_tensor).name if producers.get(input_tensor) else "",
                "consumer_nodes": consumers.get(node.output[0], []) if node.output else [],
                "reason": reason,
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    failures = _load_json(args.failures, [])
    diagnostics: list[dict[str, Any]] = []
    for failure in failures:
        if failure.get("status") != "engine_build_failed":
            continue
        cid = str(failure.get("candidate_id") or "")
        base_id, profile_id = _candidate_id_to_profile(cid)
        result = _load_json(failure.get("result_path", ""))
        route = result.get("route2_prepare") or {}
        build_report = _load_json((result.get("mixed_build_report") or (result.get("build_report") or {}).get("build_report") or ""))
        qdq_onnx = route.get("qdq_onnx_path") or route.get("route2_onnx_path") or (result.get("route2_prepare") or {}).get("route2_onnx_path")
        typed_onnx = route.get("typed_onnx_path") or ""
        width_changed = Path(args.export_dir) / base_id / "width_changed.onnx"
        tail = failure.get("batch_stdout_tail") or json.dumps(build_report, ensure_ascii=False)
        diagnostics.append(
            {
                "candidate_id": cid,
                "precision_profile_id": profile_id,
                "width_changed_onnx": str(width_changed) if width_changed.is_file() else "",
                "typed_onnx": str(typed_onnx),
                "qdq_onnx": str(qdq_onnx or ""),
                "failed_stage": "engine_build_failed",
                "trt_error_tail": _tail(str(tail)),
                "mismatched_nodes": _mismatched_nodes(qdq_onnx) if qdq_onnx else [],
            }
        )
    payload = {
        "num_failures_diagnosed": len(diagnostics),
        "diagnostics": diagnostics,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Pruned QDQ Dtype Failure Diagnostics v6", "", f"- failures diagnosed: {len(diagnostics)}"]
    for item in diagnostics[:20]:
        lines.append(f"- {item['candidate_id']}: mismatched_nodes={len(item['mismatched_nodes'])}")
        for node in item["mismatched_nodes"][:5]:
            lines.append(f"  - {node['onnx_node_name']}: {node['reason']} ({node['input_dtype']} vs {node['weight_dtype']})")
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"num_failures_diagnosed": len(diagnostics)}, indent=2))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failures", default="outputs/latency_lut/pruned_route2_full_engine_failures_v5.json")
    parser.add_argument("--export-dir", "--export_dir", dest="export_dir", default="outputs/latency_lut/pruned_width_changed_onnx_v5")
    parser.add_argument("--output", default="outputs/latency_lut/pruned_qdq_dtype_failure_diagnostics_v6.json")
    parser.add_argument("--report", default="outputs/latency_lut/pruned_qdq_dtype_failure_diagnostics_v6.md")
    return parser.parse_args()


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
