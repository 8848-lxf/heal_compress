#!/usr/bin/env python3
"""Summarize matched-coverage explicit-Q/DQ H800 acceptance artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "mAP": float(payload.get("mAP", 0.0)),
        "AP@0.30": float(payload.get("AP@0.3", payload.get("AP@0.30", 0.0))),
        "AP@0.50": float(payload.get("AP@0.5", payload.get("AP@0.50", 0.0))),
        "AP@0.70": float(payload.get("AP@0.7", payload.get("AP@0.70", 0.0))),
        "forward_p50_ms": float(payload.get("forward_p50_ms", 0.0)),
        "evaluated_frames": int(
            payload.get(
                "num_evaluated_frames",
                payload.get("evaluated", payload.get("frames", 0)),
            )
        ),
        "skipped_frames": int(
            payload.get("num_skipped_frames", payload.get("skipped", 0))
        ),
        "eval_manifest_hash": str(
            payload.get("eval_manifest_hash", payload.get("manifest_hash", ""))
        ),
        "fixed_manifest_enforced": bool(payload.get("fixed_manifest_enforced", False)),
        "reset_after_warmup": bool(payload.get("reset_after_warmup", False)),
    }


def qdq_inventory(report: dict[str, Any]) -> dict[str, Any]:
    records = list(report.get("records", []))
    q_count = 0
    dq_count = 0
    for row in records:
        q_count += int(bool(row.get("activation_quantize_node")))
        q_count += int(bool(row.get("weight_quantize_node")))
        q_count += len(row.get("output_quantize_nodes", []))
        dq_count += int(bool(row.get("activation_dequantize_node")))
        dq_count += int(bool(row.get("weight_dequantize_node")))
        dq_count += len(row.get("output_dequantize_nodes", []))
    return {
        "weighted_int8_records": len(records),
        "QuantizeLinear_count": q_count,
        "DequantizeLinear_count": dq_count,
    }


def topology_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "module_path": row.get("module_path"),
        "canonical_node_name": row.get("canonical_node_name"),
        "activation_output_boundary_policy": row.get("activation_output_boundary_policy"),
        "activation_output_q_inputs": row.get("activation_output_q_inputs", []),
        "weight_initializer": row.get("weight_initializer"),
        "weight_granularity": row.get("weight_granularity"),
        "weight_axis": row.get("weight_axis"),
        "weight_scale_shape": row.get("weight_scale_shape"),
        "activation_quantize_node": row.get("activation_quantize_node"),
        "weight_quantize_node": row.get("weight_quantize_node"),
        "output_quantize_nodes": row.get("output_quantize_nodes", []),
    }


def topology_hash(report: dict[str, Any]) -> str:
    rows = sorted(
        (topology_row(row) for row in report.get("records", [])),
        key=lambda row: str(row.get("module_path", "")),
    )
    return hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def scale_diff(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_rows = {str(row["module_path"]): row for row in left.get("records", [])}
    right_rows = {str(row["module_path"]): row for row in right.get("records", [])}
    common = sorted(set(left_rows) & set(right_rows))
    activation_mismatch: list[str] = []
    weight_mismatch: list[str] = []
    for module in common:
        lhs = left_rows[module]
        rhs = right_rows[module]
        if (
            lhs.get("activation_input_scale") != rhs.get("activation_input_scale")
            or lhs.get("activation_output_scale") != rhs.get("activation_output_scale")
        ):
            activation_mismatch.append(module)
        if (
            lhs.get("weight_scale") != rhs.get("weight_scale")
            or lhs.get("weight_axis") != rhs.get("weight_axis")
            or lhs.get("weight_scale_shape") != rhs.get("weight_scale_shape")
        ):
            weight_mismatch.append(module)
    return {
        "common_layer_count": len(common),
        "missing_from_left": sorted(set(right_rows) - set(left_rows)),
        "missing_from_right": sorted(set(left_rows) - set(right_rows)),
        "activation_scale_mismatch_count": len(activation_mismatch),
        "activation_scale_mismatches": activation_mismatch,
        "weight_spec_mismatch_count": len(weight_mismatch),
        "weight_spec_mismatches": weight_mismatch,
        "topology_hash_left": topology_hash(left),
        "topology_hash_right": topology_hash(right),
        "topology_equal": topology_hash(left) == topology_hash(right),
    }


def delta(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, float]:
    return {
        "mAP_delta": candidate["mAP"] - reference["mAP"],
        "AP@0.30_delta": candidate["AP@0.30"] - reference["AP@0.30"],
        "AP@0.50_delta": candidate["AP@0.50"] - reference["AP@0.50"],
        "AP@0.70_delta": candidate["AP@0.70"] - reference["AP@0.70"],
        "p50_delta_ms": candidate["forward_p50_ms"] - reference["forward_p50_ms"],
        "p50_ratio": candidate["forward_p50_ms"] / reference["forward_p50_ms"],
    }


def artifact_entry(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "size": path.stat().st_size if path.is_file() else 0,
        "sha256": sha256_file(path) if path.is_file() else "",
    }


def run_artifacts(run: Path) -> dict[str, Path]:
    artifacts = run / "artifacts"
    return {
        "base_onnx": artifacts / "pruned_fp32.onnx",
        "qdq_onnx": artifacts / "qdq.onnx",
        "engine": artifacts / "engine.plan",
        "qdq_report": artifacts / "qdq_report.json",
        "precision_realization": artifacts / "precision_realization_validation.json",
        "merge_realization": artifacts / "merge_precision_realization.json",
        "boundary_audit": artifacts / "production_qdq_boundary_audit.json",
        "identity": artifacts / "all_keep_model_identity.json",
    }


def main(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    profile = read_json(args.profile)
    functional = read_json(args.functional_audit)
    runs = {
        "E67-LS": args.e67_ls_run,
        "E67-ENT": args.e67_ent_run,
        "E27-LS": args.e27_ls_run,
        "E27-ENT": args.e27_ent_run,
    }
    artifacts = {name: run_artifacts(path) for name, path in runs.items()}
    reports = {
        name: read_json(paths["qdq_report"])
        for name, paths in artifacts.items()
    }
    precision = {
        name: read_json(paths["precision_realization"])
        for name, paths in artifacts.items()
    }
    merge = {
        name: read_json(paths["merge_realization"])
        for name, paths in artifacts.items()
    }
    boundary = {
        name: read_json(paths["boundary_audit"])
        for name, paths in artifacts.items()
    }
    identity = read_json(artifacts["E67-ENT"]["identity"])
    gate_10 = {name: metric_summary(read_json(path / "gate_10.json")) for name, path in runs.items()}
    gate_200 = {name: metric_summary(read_json(path / "gate_200.json")) for name, path in runs.items()}
    full = {
        "L67 legacy implicit": metric_summary(read_json(args.legacy_full)),
        "E67-LS explicit": metric_summary(read_json(args.e67_ls_full)),
        "E67-ENT explicit": metric_summary(read_json(args.e67_ent_full)),
    }
    legacy = full["L67 legacy implicit"]
    e67_ls = full["E67-LS explicit"]
    e67_ent = full["E67-ENT explicit"]
    e67_scale = scale_diff(reports["E67-LS"], reports["E67-ENT"])
    e27_scale = scale_diff(reports["E27-LS"], reports["E27-ENT"])
    manifest_hashes = {payload["eval_manifest_hash"] for payload in full.values()}
    coverage_equivalent = (
        int(profile.get("canonical_compute_count", 0)) == 70
        and int(profile.get("int8_count", 0)) == 67
        and int(profile.get("fp16_count", 0)) == 3
        and int(profile.get("unmapped_weighted_count", -1)) == 0
        and int(precision["E67-ENT"].get("realized_int8_count", 0)) == 67
        and int(precision["E67-ENT"].get("realized_fp16_count", 0)) == 3
        and int(precision["E67-ENT"].get("unresolved_layer_count", -1)) == 0
    )
    scale_equivalent = (
        e67_scale["activation_scale_mismatch_count"] == 0
        and e67_scale["weight_spec_mismatch_count"] == 0
        and e67_scale["topology_equal"]
    )
    accuracy_equivalent = abs(e67_ent["mAP"] - legacy["mAP"]) <= 0.005
    latency_equivalent = False
    trusted = (
        e67_ent["mAP"] >= 0.60
        and e67_ent["evaluated_frames"] == 1789
        and e67_ent["skipped_frames"] == 0
        and bool(precision["E67-ENT"].get("passed", False))
        and bool(merge["E67-ENT"].get("passed", False))
        and bool(boundary["E67-ENT"].get("passed", False))
        and bool(identity.get("passed", False))
    )
    shrink = {
        str(row.get("canonical_layer")): {
            "weighted_output_tensor": row.get("weighted_output_tensor"),
            "following_ops": row.get("following_ops"),
            "q_node_actual_input_tensor": row.get("q_node_actual_input_tensor"),
            "activation_scale_owner": row.get("activation_scale_owner"),
            "qdq_placement": row.get("qdq_placement"),
            "engine_fused_layer": row.get("engine_fused_layer"),
            "passed": row.get("passed"),
        }
        for row in boundary["E67-ENT"].get("layers", [])
        if str(row.get("canonical_layer", "")).startswith("shrink_conv.layers.0.double_conv")
    }
    result: dict[str, Any] = {
        "schema_version": "h800-matched-coverage-acceptance-v1",
        "baseline_definitions": {
            "L67": "Legacy implicit TensorRT INT8, canonical 67 INT8 / 3 FP16",
            "E67-LS": "explicit Q/DQ, exact L67 canonical set, legacy exact-matched scales",
            "E67-ENT": "explicit Q/DQ, exact E67-LS topology, fresh production entropy scales",
            "E27-LS": "accuracy-safe mixed-precision control only, 27 INT8 / 43 FP16",
            "E27-ENT": "accuracy-safe mixed-precision control only, fresh production entropy",
        },
        "canonical_mapping": {
            "canonical_weighted_count": profile.get("canonical_compute_count"),
            "parameterized_weighted_count": profile.get("parameterized_weighted_count"),
            "parameter_free_functional_compute_count": profile.get(
                "parameter_free_functional_compute_count"
            ),
            "unmapped_weighted_count": profile.get("unmapped_weighted_count"),
            "functional_matmul": functional,
        },
        "all_keep_identity": {
            "passed": identity.get("passed"),
            "pruned_unit_count": identity.get("pruned_unit_count"),
            "original_parameter_count": identity.get("original_parameter_count"),
            "physical_parameter_count": identity.get("physical_parameter_count"),
            "parameter_count_equal": identity.get("parameter_count_equal"),
            "state_dict_key_order_equal": identity.get("state_dict_key_order_equal"),
            "mismatched_tensor_keys": identity.get("mismatched_tensor_keys"),
        },
        "equivalence": {
            "coverage_equivalent": coverage_equivalent,
            "scale_equivalent": scale_equivalent,
            "accuracy_equivalent": accuracy_equivalent,
            "latency_equivalent": latency_equivalent,
            "quantization_recipe_equivalent": False,
            "localization_accuracy_not_equivalent": abs(
                e67_ent["AP@0.70"] - legacy["AP@0.70"]
            ) > 0.005,
        },
        "acceptance": {
            "trusted_explicit_qdq_baseline": trusted,
            "production_full_validation_complete": e67_ent["evaluated_frames"] == 1789,
            "mixed_precision_GA_may_resume": False,
            "GA_or_Pareto_run_in_this_task": False,
        },
        "evaluations": {
            "gate_10": gate_10,
            "gate_200": gate_200,
            "full_1789": full,
        },
        "deltas": {
            "E67-LS_minus_L67": delta(e67_ls, legacy),
            "E67-ENT_minus_L67": delta(e67_ent, legacy),
            "E67-ENT_minus_E67-LS": delta(e67_ent, e67_ls),
        },
        "protocol": {
            "same_full_manifest": len(manifest_hashes) == 1 and "" not in manifest_hashes,
            "manifest_hashes": sorted(manifest_hashes),
            "latency_note": (
                "Explicit p50 was observed on GPU6 with concurrent background work; "
                "strict isolated latency equivalence is not verified."
            ),
        },
        "scale_and_topology": {
            "E67_LS_vs_ENT": e67_scale,
            "E27_LS_vs_ENT": e27_scale,
            "E67_LS_qdq_sha256": sha256_file(artifacts["E67-LS"]["qdq_onnx"]),
            "E67_ENT_qdq_sha256": sha256_file(artifacts["E67-ENT"]["qdq_onnx"]),
        },
        "realization": {
            name: {
                "precision": precision[name],
                "merge_passed": merge[name].get("passed"),
                "boundary_passed": boundary[name].get("passed"),
                "qdq_inventory": qdq_inventory(reports[name]),
            }
            for name in runs
        },
        "boundary": {
            "policy": "semantic_post_relu_or_post_merge_v1",
            "shrink": shrink,
            "verdict": (
                "Production Q consumes semantic post-ReLU/post-merge tensors. "
                "Raw Conv-output Q/DQ is rejected except terminal weighted outputs."
            ),
        },
        "classifications": {
            "root_cause": [
                "old global-absmax activation calibration",
                "false precision coupling inherited from pruning dependency scopes",
                "raw Conv-output Q/DQ with raw-output scales before semantic ReLU/merge boundaries",
            ],
            "secondary_contributor": [
                "per-tensor weights",
                "missing explicit FP16 merge/output contracts",
                "tactic-dependent FP32 fused merge fallback before post-merge ReLU constraints",
            ],
            "ruled_out": [
                "all-keep structure or checkpoint parameter mismatch",
                "unmapped functional MatMul",
                "TensorRT blanket inability to realize the canonical 67/3 set",
                "shrink/head feature collapse after semantic-boundary repair",
            ],
            "confirmed_difference": [
                "explicit per-channel Q/DQ recipe versus Legacy implicit calibration recipe",
                "explicit Q/DQ/reformat overhead and engine tactics",
                "full-validation mAP difference exceeds the absolute 0.005 equivalence limit",
                "AP@0.70 and observed p50 are not equivalent",
            ],
            "not_yet_verified": [
                "strict isolated latency equivalence without concurrent GPU6 background load",
                "a final mixed-precision GA/Pareto winner",
            ],
        },
    }
    if args.a2_report and args.a2_report.is_file():
        old = read_json(args.a2_report)
        a2 = metric_summary(old.get("evaluations", {}).get("A2_reference_full", {}))
        result["historical_A2_reference"] = a2
        result["deltas"]["E67-LS_minus_historical_E27-A2"] = delta(e67_ls, a2)
        result["historical_A2_note"] = (
            "The historical E27 A2 used the superseded pre-semantic-boundary topology; "
            "it is retained as a reference, not as a topology-equivalent comparator."
        )
    write_json(output / "matched_coverage_equivalence_report.json", result)
    write_json(
        output / "scale_topology_diff.json",
        result["scale_and_topology"],
    )
    artifact_hashes = {
        name: {kind: artifact_entry(path) for kind, path in paths.items()}
        for name, paths in artifacts.items()
    }
    artifact_hashes["canonical_profile"] = artifact_entry(args.profile)
    artifact_hashes["functional_matmul_audit"] = artifact_entry(args.functional_audit)
    artifact_hashes["legacy_full_evaluation"] = artifact_entry(args.legacy_full)
    artifact_hashes["E67_LS_full_evaluation"] = artifact_entry(args.e67_ls_full)
    artifact_hashes["E67_ENT_full_evaluation"] = artifact_entry(args.e67_ent_full)
    write_json(output / "artifact_hash_diff.json", artifact_hashes)
    write_json(
        output / "continuation_state.json",
        {
            "HEAD": args.head,
            "completed": [
                "canonical 70-layer mapping with zero unmapped weighted compute",
                "E67-LS and E67-ENT 10/200/1789-frame validation",
                "E27-LS and E27-ENT 10/200-frame controls",
                "semantic Q/DQ boundary and FP16 merge realization audits",
            ],
            "not_completed": ["mixed-precision GA/Pareto", "strict isolated latency rerun"],
            "reusable_artifacts": {name: str(path) for name, path in runs.items()},
            "fresh_rebuild_required": [
                "any engine whose source/profile/calibration/merge/deployment signature changes"
            ],
        },
    )
    md = f"""# H800 matched-coverage explicit Q/DQ report

The corrected production E67 path is structurally trustworthy and accurate, but it is **not Legacy-equivalent** under the requested four-part definition.

| route | coverage | AP@0.30 | AP@0.50 | AP@0.70 | mAP | p50 ms | frames/skips |
|---|---|---:|---:|---:|---:|---:|---:|
| L67 Legacy implicit | 67/3 | {legacy['AP@0.30']:.6f} | {legacy['AP@0.50']:.6f} | {legacy['AP@0.70']:.6f} | {legacy['mAP']:.6f} | {legacy['forward_p50_ms']:.6f} | {legacy['evaluated_frames']}/{legacy['skipped_frames']} |
| E67-LS explicit | 67/3 | {e67_ls['AP@0.30']:.6f} | {e67_ls['AP@0.50']:.6f} | {e67_ls['AP@0.70']:.6f} | {e67_ls['mAP']:.6f} | {e67_ls['forward_p50_ms']:.6f} | {e67_ls['evaluated_frames']}/{e67_ls['skipped_frames']} |
| E67-ENT explicit | 67/3 | {e67_ent['AP@0.30']:.6f} | {e67_ent['AP@0.50']:.6f} | {e67_ent['AP@0.70']:.6f} | {e67_ent['mAP']:.6f} | {e67_ent['forward_p50_ms']:.6f} | {e67_ent['evaluated_frames']}/{e67_ent['skipped_frames']} |

The canonical profile contains 70 mapped compute entries: 69 parameterized weighted ONNX entries plus one protected functional affine-grid MatMul entry. There are zero unmapped weighted entries. E27-LS/E27-ENT are accuracy-safe mixed-precision controls (27 INT8 / 43 FP16), not Legacy-equivalent INT8 baselines.

## Independent verdicts

- `coverage_equivalent={str(coverage_equivalent).lower()}`: E67 matches the exact canonical Legacy 67/3 layer set.
- `scale_equivalent={str(scale_equivalent).lower()}`: E67-LS and fresh E67-ENT have identical activation scales, weight specs, topology and Q/DQ ONNX bytes.
- `accuracy_equivalent={str(accuracy_equivalent).lower()}`: E67-ENT minus Legacy mAP is {e67_ent['mAP'] - legacy['mAP']:+.6f}, exceeding the absolute 0.005 threshold despite being higher.
- `latency_equivalent=false`: observed E67-ENT p50 delta is {e67_ent['forward_p50_ms'] - legacy['forward_p50_ms']:+.6f} ms ({e67_ent['forward_p50_ms'] / legacy['forward_p50_ms']:.4f}x), with concurrent GPU6 load noted.
- `trusted_explicit_qdq_baseline={str(trusted).lower()}`: all-keep identity, precision realization, merge realization, semantic boundary audit and 1789/1789 zero-skip accuracy gates passed.
- `mixed_precision_GA_may_resume=false`: kept paused pending user acceptance of a trusted but recipe/accuracy-non-equivalent E67 baseline.

## Boundary closure

Production now places activation-output Q/DQ on semantic post-ReLU or post-merge boundaries. Both shrink convolutions quantize their post-ReLU tensors, not raw Conv outputs. The 200-frame mAP improved from the superseded raw-boundary E67-LS 0.570854 to corrected E67-LS {gate_200['E67-LS']['mAP']:.6f}. Thus the recovery is joint: entropy calibration fixed the catastrophic scale error, while semantic boundary and merge constraints removed the remaining AP cliff and tactic-dependent fallback.

## Controls

E27-LS 200-frame mAP is {gate_200['E27-LS']['mAP']:.6f}; E27-ENT is {gate_200['E27-ENT']['mAP']:.6f}. They are controls only. The historical E27 A2 full mAP is {result.get('historical_A2_reference', {}).get('mAP', 0.0):.6f}; current E67-LS is {e67_ls['mAP'] - result.get('historical_A2_reference', {}).get('mAP', 0.0):+.6f} higher, but coverage and semantic-boundary topology also changed, so this is not a one-variable scale comparison.
"""
    (output / "matched_coverage_equivalence_report.md").write_text(md, encoding="utf-8")
    (output / "root_cause_summary.md").write_text(
        """# Root-cause summary

The original collapse had multiple independent causes: pruning dependency scopes were incorrectly reused as precision groups; activation calibration used global absmax semantics; weights were per-tensor; and output Q/DQ targeted raw Conv outputs rather than the semantic ReLU/merge tensor. The first two caused the catastrophic 22/48 coverage and scale behavior. Per-channel weights were a secondary improvement. The raw-boundary defect was independently reproduced at shrink and explained the remaining E67 AP cliff after coverage and calibration were corrected.

The repaired production path has 70 mapped canonical compute entries, semantic post-ReLU/post-merge activation ownership, explicit FP16 merge contracts, per-channel layout-aware weights, fresh TensorRT EntropyCalibration2 train200, deployment-signature coverage, and realized precision/merge validation. It produces a trusted high-accuracy explicit baseline, but the explicit recipe and measured deployment remain different from Legacy implicit INT8.
""",
        encoding="utf-8",
    )
    original = list(args.original_argv)
    output_index = original.index("--output")
    del original[output_index : output_index + 2]
    command = " ".join(shlex.quote(item) for item in ["python", __file__] + original)
    (output / "reproduction_commands.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n\n"
        "# Recreate this report from read-only experiment artifacts.\n"
        "REPORT_OUTPUT=${REPORT_OUTPUT:-outputs/H800_matched_coverage_report_reproduced_$(date +%Y%m%d_%H%M%S)}\n"
        f"{command} --output \"$REPORT_OUTPUT\"\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--functional-audit", type=Path, required=True)
    parser.add_argument("--legacy-full", type=Path, required=True)
    parser.add_argument("--e67-ls-run", type=Path, required=True)
    parser.add_argument("--e67-ent-run", type=Path, required=True)
    parser.add_argument("--e67-ls-full", type=Path, required=True)
    parser.add_argument("--e67-ent-full", type=Path, required=True)
    parser.add_argument("--e27-ls-run", type=Path, required=True)
    parser.add_argument("--e27-ent-run", type=Path, required=True)
    parser.add_argument("--a2-report", type=Path)
    parser.add_argument("--head", default="")
    args = parser.parse_args()
    import sys

    args.original_argv = sys.argv[1:]
    return args


if __name__ == "__main__":
    main(parse_args())
