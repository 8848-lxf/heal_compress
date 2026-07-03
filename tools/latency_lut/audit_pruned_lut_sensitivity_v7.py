from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from tools.latency_lut.v5_pruned_mixed_common import conv_gemm_shapes, stable_hash
except ModuleNotFoundError:  # Direct script execution from tools/latency_lut.
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from tools.latency_lut.v5_pruned_mixed_common import conv_gemm_shapes, stable_hash


def _load_json(path: str | Path, default: Any = None) -> Any:
    p = Path(path)
    if not p.is_file():
        return {} if default is None else default
    return json.loads(p.read_text(encoding="utf-8"))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize_lut_sensitivity(labels: list[dict[str, Any]]) -> dict[str, Any]:
    unique_t = {round(float(x.get("T_lut_raw") or 0.0), 12) for x in labels}
    unique_compute = {round(float(x.get("T_compute_covered") or x.get("T_lut_raw") or 0.0), 12) for x in labels}
    unique_matched = {str(x.get("matched_lut_key_hash")) for x in labels}
    unique_shapes = {str(x.get("candidate_conv_shape_hash")) for x in labels}
    unique_units = {str(x.get("resolved_unit_hash")) for x in labels if x.get("resolved_unit_hash")}
    t_constant = len(unique_t) <= 1
    matched_constant = len(unique_matched) <= 1
    shapes_constant = len(unique_shapes) <= 1
    if t_constant and matched_constant and not shapes_constant:
        root = "lut_matching_too_coarse_or_constant_components"
    elif t_constant and shapes_constant:
        root = "candidate_shapes_constant_or_sampling_collapse"
    elif t_constant:
        root = "constant_non_compute_components_or_lut_match"
    else:
        root = "not_constant"
    return {
        "num_labels": len(labels),
        "num_unique_T_lut_raw": len(unique_t),
        "num_unique_T_compute_covered": len(unique_compute),
        "num_unique_matched_lut_key_hash": len(unique_matched),
        "num_unique_candidate_conv_shape_hash": len(unique_shapes),
        "num_unique_resolved_unit_hash": len(unique_units),
        "T_lut_raw_constant": t_constant,
        "T_compute_covered_constant": len(unique_compute) <= 1,
        "matched_lut_keys_constant": matched_constant,
        "candidate_shapes_constant": shapes_constant,
        "decomposition_uses_pruned_widths": not shapes_constant,
        "decomposition_falls_back_to_baseline_widths": False,
        "decomposition_missing_candidate_shapes": any(not x.get("uses_candidate_conv_shapes") for x in labels),
        "root_cause": root,
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    dataset = _read_jsonl(args.dataset)
    rows: list[dict[str, Any]] = []
    for row in dataset:
        cid = str(row.get("candidate_id"))
        base = str(row.get("base_structure_candidate_id") or cid.split("__", 1)[0])
        decomp = _load_json(row.get("lut_decomposition_path") or Path(args.decomposition_dir) / f"{base}_{row.get('precision_profile_id')}.json", {})
        matched = decomp.get("matched_lut_keys") or []
        shapes, _params = conv_gemm_shapes(Path(args.export_dir) / base / "width_changed.onnx")
        matched_hash = stable_hash([m.get("matched_lut_key") for m in matched])
        shape_hash = stable_hash(shapes)
        components = {k: float(decomp.get(k) or 0.0) for k in [
            "T_compute_covered", "T_boundary_cast", "T_boundary_qdq", "T_plugin_or_scatter",
            "T_grid_sample_or_geometry", "T_elementwise_merge", "T_memory_reformat", "T_fixed_overhead",
        ]}
        item = {
            "candidate_id": cid,
            "precision_profile_id": row.get("precision_profile_id"),
            "T_lut_raw": float(row.get("T_lut_raw") or decomp.get("T_lut_raw") or 0.0),
            **components,
            "num_matched_lut_keys": len(matched),
            "num_unique_matched_lut_keys": len({m.get("matched_lut_key") for m in matched}),
            "matched_lut_key_hash": matched_hash,
            "num_coarse_keys": len(decomp.get("coarse_keys") or []),
            "coarse_key_ratio": len(decomp.get("coarse_keys") or []) / len(matched) if matched else 0.0,
            "uses_candidate_conv_shapes": bool(shapes),
            "uses_resolved_units": bool(row.get("resolved_units")),
            "num_candidate_conv_shapes": len(shapes),
            "num_resolved_units": len(row.get("resolved_units") or []),
            "num_width_changed_layers_seen_by_decomposition": int(row.get("num_changed_conv_layers") or 0),
            "candidate_conv_shape_hash": shape_hash,
            "resolved_unit_hash": stable_hash(row.get("resolved_units") or []),
        }
        rows.append(item)
    summary = summarize_lut_sensitivity(rows)
    for item in rows:
        item["same_matched_keys_as_other_labels"] = summary["matched_lut_keys_constant"]
        item["same_component_breakdown_as_other_labels"] = summary["T_compute_covered_constant"]
    out = {"labels": rows, "summary": summary}
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.output_md).write_text("# Pruned LUT Sensitivity v7\n\n```json\n" + json.dumps(summary, indent=2) + "\n```\n", encoding="utf-8")
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="outputs/latency_lut/full_engine_calibration_dataset_v6.jsonl")
    p.add_argument("--decomposition-dir", default="outputs/latency_lut/pruned_candidate_lut_decomposition_v6")
    p.add_argument("--export-dir", default="outputs/latency_lut/pruned_width_changed_onnx_v5")
    p.add_argument("--output-json", default="outputs/latency_lut/pruned_lut_sensitivity_v7.json")
    p.add_argument("--output-md", default="outputs/latency_lut/pruned_lut_sensitivity_v7.md")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    out = audit(parse_args(argv))
    print(json.dumps(out["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
