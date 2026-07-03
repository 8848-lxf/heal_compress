from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import onnx

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


def _sha_file(path: Path) -> str:
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: str | Path, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _candidate_sampling_name(cid: str) -> str:
    for name in ("keep97", "keep875", "keep75", "keep625"):
        if name in cid:
            return f"local_taylor_{name}"
    return "unknown"


def _target_from_name(name: str) -> float | None:
    return {"local_taylor_keep97": 0.97, "local_taylor_keep875": 0.875, "local_taylor_keep75": 0.75, "local_taylor_keep625": 0.625}.get(name)


def summarize_collapse(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keep_values = {str(r.get("requested_target_keep_ratio")) for r in rows}
    struct_hashes = {str(r.get("structure_changes_hash")) for r in rows if r.get("structure_changes_hash")}
    actual_keep = {round(float(r.get("actual_param_keep_ratio") or 0.0), 6) for r in rows if r.get("actual_param_keep_ratio") is not None}
    by_keep: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_keep[str(row.get("requested_target_keep_ratio"))].add(str(row.get("structure_changes_hash")))
    return {
        "num_candidates": len(rows),
        "num_export_success": sum(1 for r in rows if r.get("export_success")),
        "num_unique_structure_changes_hash": len(struct_hashes),
        "num_unique_channel_keep_map_hash": len({str(r.get("channel_keep_map_hash")) for r in rows if r.get("channel_keep_map_hash")}),
        "num_unique_onnx_conv_shape_hash": len({str(r.get("onnx_conv_shape_hash")) for r in rows if r.get("onnx_conv_shape_hash")}),
        "num_unique_actual_param_keep_ratio": len(actual_keep),
        "requested_keep_to_actual_keep": {k: sorted({round(float(r.get("actual_param_keep_ratio") or 0.0), 6) for r in rows if str(r.get("requested_target_keep_ratio")) == k}) for k in keep_values},
        "requested_keep_to_unique_structure_hashes": {k: len(v) for k, v in by_keep.items()},
        "requested_keep_to_export_success": dict(Counter(str(r.get("requested_target_keep_ratio")) for r in rows if r.get("export_success"))),
        "requested_keep_to_engine_success": dict(Counter(str(r.get("requested_target_keep_ratio")) for r in rows for _ in range(int(r.get("engine_success_count") or 0)))),
        "requested_keep_to_admitted_labels": dict(Counter(str(r.get("requested_target_keep_ratio")) for r in rows for _ in range(int(r.get("admitted_label_count") or 0)))),
        "all_requested_keep_ratios_collapsed_to_same_structure": len(keep_values) > 1 and len(struct_hashes) == 1,
        "all_successful_labels_from_same_actual_keep_ratio": len(actual_keep) == 1,
        "different_structures_generated_but_filtered_by_engine": len(struct_hashes) > 1 and len(actual_keep) == 1,
        "requested_keep_ratio_not_propagated_to_pruner": len(keep_values) > 1 and len(actual_keep) == 1 and len(struct_hashes) == 1,
        "root_node_local_domain_not_used": False,
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    payload = _load_json(args.candidates, {"candidates": []})
    dataset = [json.loads(l) for l in Path(args.dataset).read_text().splitlines() if l.strip()] if Path(args.dataset).is_file() else []
    admitted = Counter(str(r.get("base_structure_candidate_id") or str(r.get("candidate_id", "")).split("__", 1)[0]) for r in dataset)
    result_success = Counter()
    failed_stages: dict[str, list[str]] = defaultdict(list)
    for path in Path(args.results_dir).glob("*.result.json"):
        data = _load_json(path, {})
        base = str(data.get("candidate_id", "")).split("__", 1)[0]
        if data.get("success"):
            result_success[base] += 1
        elif data.get("failed_stage"):
            failed_stages[base].append(str(data.get("failed_stage")))
    rows: list[dict[str, Any]] = []
    for cand in payload.get("candidates", []):
        cid = str(cand.get("candidate_id"))
        cdir = Path(args.export_dir) / cid
        pruning = dict(cand.get("pruning") or {})
        sampling = _candidate_sampling_name(cid)
        audit_json = _load_json(cdir / "structure_audit.json", {})
        shapes, _params = conv_gemm_shapes(cdir / "width_changed.onnx")
        shape_hash = stable_hash(shapes) if shapes else ""
        row = {
            "candidate_id": cid,
            "requested_sampling_name": sampling,
            "requested_target_keep_ratio": pruning.get("target_keep_ratio", _target_from_name(sampling)),
            "requested_min_keep_ratio": pruning.get("min_keep_ratio"),
            "requested_align": pruning.get("align"),
            "requested_scope": pruning.get("scope"),
            "requested_local_scope": pruning.get("local_scope"),
            "requested_importance": pruning.get("importance"),
            "requested_selection_mode": cand.get("selection_mode"),
            "actual_pruner_cli": "",
            "actual_pruner_config": _load_json(cdir / "export_pruned_model_report.json", {}).get("args", {}),
            "actual_scope": _load_json(cdir / "selection_summary.json", {}).get("scope", ""),
            "actual_local_scope": _load_json(cdir / "selection_summary.json", {}).get("local_scope", ""),
            "actual_importance": _load_json(cdir / "selection_summary.json", {}).get("importance", ""),
            "actual_global_ranking": _load_json(cdir / "selection_summary.json", {}).get("global_ranking"),
            "actual_module_stage_based_domain": _load_json(cdir / "selection_summary.json", {}).get("module_stage_based_domain"),
            "actual_root_node_local_domain": bool((cand.get("pruning_config") or {}).get("root_node_domains")),
            "export_success": bool((cdir / "width_changed.onnx").is_file()),
            "width_changed_onnx_exists": bool((cdir / "width_changed.onnx").is_file()),
            "actual_param_keep_ratio": audit_json.get("param_keep_ratio"),
            "actual_num_changed_conv_layers": audit_json.get("num_changed_conv_layers", 0),
            "actual_changed_conv_layers": [x.get("layer_name") for x in audit_json.get("changed_conv_layers", [])],
            "actual_changed_layer_prefix_coverage": dict(Counter(str(x.get("layer_name", "")).strip("/").split("/")[0] for x in audit_json.get("changed_conv_layers", []))),
            "prune_replay_hash": _sha_file(cdir / "prune_replay.json"),
            "channel_keep_map_hash": _sha_file(cdir / "channel_keep_map.json"),
            "structure_changes_hash": _sha_file(cdir / "structure_changes.json") or _sha_file(cdir / "structure_changes.csv"),
            "onnx_conv_shape_hash": shape_hash,
            "changed_layer_set_hash": stable_hash(sorted([x.get("layer_name") for x in audit_json.get("changed_conv_layers", [])])),
            "engine_success_count": result_success[cid],
            "admitted_label_count": admitted[cid],
            "failed_stages": failed_stages.get(cid, []),
            "admitted_to_v6_dataset": admitted[cid] > 0,
        }
        rows.append(row)
    summary = summarize_collapse(rows)
    if summary["requested_keep_ratio_not_propagated_to_pruner"]:
        root = "requested_keep_ratio_not_propagated"
    elif summary["all_requested_keep_ratios_collapsed_to_same_structure"]:
        root = "constraint_quantization_collapse"
    elif summary["different_structures_generated_but_filtered_by_engine"]:
        root = "engine_admission_bias"
    else:
        root = "unknown"
    summary["sampling_collapse_root_cause"] = {"sampling_collapse_root_cause": root, "evidence": []}
    out = {"candidates": rows, "summary": summary}
    _write_json(args.output_json, out)
    Path(args.output_md).write_text("# Pruning Sampling Collapse v7\n\n```json\n" + json.dumps(summary, indent=2) + "\n```\n", encoding="utf-8")
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--candidates", default="outputs/latency_lut/pruned_width_changed_candidates_v5.json")
    p.add_argument("--export-dir", default="outputs/latency_lut/pruned_width_changed_onnx_v5")
    p.add_argument("--dataset", default="outputs/latency_lut/full_engine_calibration_dataset_v6.jsonl")
    p.add_argument("--results-dir", default="outputs/latency_lut/pruned_route2_results_v6")
    p.add_argument("--output-json", default="outputs/latency_lut/pruning_sampling_collapse_v7.json")
    p.add_argument("--output-md", default="outputs/latency_lut/pruning_sampling_collapse_v7.md")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    out = audit(parse_args(argv))
    print(json.dumps(out["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
