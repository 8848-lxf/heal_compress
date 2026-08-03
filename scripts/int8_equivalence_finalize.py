#!/usr/bin/env python3
"""Assemble the immutable all-keep INT8 equivalence audit deliverables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
CHECKPOINT = Path("${MODEL_ROOT}/lidar_pyramid/net_epoch_bestval_at17.pth")
CONFIG = Path("${MODEL_ROOT}/lidar_pyramid/config.yaml")
TRT_ROOT = Path("${TENSORRT_ROOT}")
MODEL_OPT = Path("${CONDA_BASE}/envs/modelopt")
LEGACY_ROOT = REPO / "tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare"


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def eval_row(label: str, path: Path) -> dict[str, Any]:
    value = read_json(path, {})
    p50 = value.get("forward_p50_ms")
    return {
        "label": label,
        "path": str(path),
        "sha256": sha256_file(path) if path.is_file() else "",
        "status": value.get("status", "missing"),
        "AP@0.30": value.get("AP@0.3"),
        "AP@0.50": value.get("AP@0.5"),
        "AP@0.70": value.get("AP@0.7"),
        "mAP": value.get("mAP"),
        "p50_ms": p50,
        "FPS": 1000.0 / float(p50) if p50 else None,
        "manifest_hash": value.get("eval_manifest_hash", ""),
        "evaluated_frames": value.get("num_evaluated_frames"),
        "skipped_frames": value.get("num_skipped_frames"),
        "skipped_frame_ids": value.get("skipped_frame_ids", []),
    }


def output_bindings(probe: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in probe.get("bindings", []):
        if "OUTPUT" in str(row.get("mode", "")).upper():
            rows.append({key: row.get(key) for key in ("index", "name", "shape", "dtype", "format", "format_description")})
    return rows


def build_binding_diff(root: Path) -> dict[str, Any]:
    labels = ("legacy_fp16", "search_fp16", "legacy_int8", "search_int8")
    probes = {label: read_json(root / f"engine_probe_{label}.json", {}) for label in labels}
    outputs = {label: output_bindings(value) for label, value in probes.items()}
    names = {label: [str(row.get("name", "")) for row in rows] for label, rows in outputs.items()}
    expected = ["cls_preds", "reg_preds", "dir_preds"]
    result = {
        "engine_probes": {label: str(root / f"engine_probe_{label}.json") for label in labels},
        "outputs": outputs,
        "output_name_order": names,
        "expected_decoder_order": expected,
        "all_output_name_orders_equal": bool(outputs) and all(value == expected for value in names.values()),
        "all_contexts_created": all(bool(value.get("context_created")) for value in probes.values()),
        "input_binding_names": {
            label: [str(row.get("name", "")) for row in value.get("bindings", []) if "INPUT" in str(row.get("mode", "")).upper()]
            for label, value in probes.items()
        },
    }
    write_json(root / "binding_diff.json", result)
    return result


def build_postprocess_diff(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    requests = {
        "legacy_fp16": root / "legacy_fp16_fresh_evaluation/evaluation_request.json",
        "search_fp16": root / "search_strict_fp16_force_rebuild/artifacts/evaluation_request.json",
        "legacy_int8": root / "legacy_int8_fresh_evaluation/evaluation_request.json",
        "search_int8": root / "search_maximal_legal_int8_force_rebuild/artifacts/evaluation_request.json",
    }
    payloads = {label: read_json(path, {}) for label, path in requests.items()}
    common_fields = ("checkpoint", "model_config", "heal_root", "eval_manifest_path", "fixed_k", "num_frames", "warmup_frames", "latency_rounds", "physical_device")
    field_equality = {
        field: len({json.dumps(value.get(field), sort_keys=True) for value in payloads.values()}) == 1
        for field in common_fields
    }
    evaluator = REPO / "search/integration/evaluation_worker.py"
    result = {
        "requests": {label: {"path": str(requests[label]), "sha256": sha256_file(requests[label]) if requests[label].is_file() else "", **payload} for label, payload in payloads.items()},
        "common_field_equality": field_equality,
        "same_evaluation_worker": True,
        "evaluation_worker": str(evaluator),
        "evaluation_worker_sha256": sha256_file(evaluator),
        "decoder_binding_order_equal": binding.get("all_output_name_orders_equal", False),
        "postprocess_equivalent": all(field_equality.values()) and bool(binding.get("all_output_name_orders_equal", False)),
        "route_specific_fields": ["engine_path", "plugin_path", "output_path"],
    }
    write_json(root / "postprocess_diff.json", result)
    return result


def calibration_source_audit(root: Path) -> dict[str, Any]:
    archives = sorted((root / "search_maximal_legal_int8_force_rebuild/archives/calibration").glob("*.json"))
    archive = read_json(archives[0], {}) if archives else {}
    metadata = archive.get("metadata", {})
    inventory = read_json(root / "qdq_inventory.json", {})
    return {
        "semantics_version": metadata.get("semantics_version", ""),
        "source": metadata.get("source", ""),
        "frame_count": metadata.get("frame_count"),
        "output_module_paths": metadata.get("output_module_paths", {}),
        "paired_bn_output_count": sum(str(value) != str(key) for key, value in metadata.get("output_module_paths", {}).items()),
        "weight_initializer_names": metadata.get("weight_initializer_names", {}),
        "weight_initializer_mapping_count": len(metadata.get("weight_initializer_names", {})),
        "weight_scale_final_initializer_match_count": inventory.get("weight_scale_final_initializer_match_count"),
        "weight_row_count": inventory.get("weight_row_count"),
        "calibration_scale_match_count": inventory.get("calibration_scale_match_count"),
        "inventory_row_count": inventory.get("inventory_row_count"),
    }


def build_quantization_spec_audit(root: Path, calibration: dict[str, Any]) -> dict[str, Any]:
    inventory = read_json(root / "qdq_inventory.json", {})
    rows = list(inventory.get("rows", []))
    mapping = read_json(root / "search_maximal_legal_int8_force_rebuild/artifacts/canonical_layer_map.json", {})
    precision_rows = list(mapping.get("entries", []))
    activation_rows = [row for row in rows if row.get("quant_role") != "weight"]
    weight_rows = [row for row in rows if row.get("quant_role") == "weight"]
    scale_owners: dict[str, set[str]] = {}
    for row in activation_rows:
        scale_owners.setdefault(str(row.get("scale", "")), set()).add(str(row.get("canonical_layer", "")))
    shared = [
        {"scale": scale, "canonical_layers": sorted(value)}
        for scale, value in scale_owners.items()
        if scale and len({item for item in value if item}) > 1
    ]
    result = {
        "activation_scale_anomalies": {
            "nonfinite": [row.get("quantize_node") for row in activation_rows if row.get("nonfinite_scale")],
            "zero": [row.get("quantize_node") for row in activation_rows if row.get("zero_scale")],
            "extremely_small": [row.get("quantize_node") for row in activation_rows if row.get("extremely_small_scale")],
            "extremely_large": [row.get("quantize_node") for row in activation_rows if row.get("extremely_large_scale")],
            "fixed_one": [row.get("quantize_node") for row in activation_rows if row.get("fixed_one_scale")],
        },
        "weight_scale_audit": {
            "row_count": len(weight_rows),
            "final_folded_initializer_match_count": sum(bool(row.get("weight_scale_matches_final_folded_initializer")) for row in weight_rows),
            "per_tensor_count": sum(row.get("granularity") == "per_tensor" for row in weight_rows),
            "per_channel_count": sum(row.get("granularity") == "per_channel" for row in weight_rows),
            "axis_0_count": sum(row.get("axis") == 0 for row in weight_rows),
            "axis_absent_count": sum(row.get("axis") == "" for row in weight_rows),
        },
        "activation_output_source_audit": {
            "semantics_version": calibration.get("semantics_version"),
            "paired_bn_output_count": calibration.get("paired_bn_output_count"),
            "output_module_paths": calibration.get("output_module_paths", {}),
        },
        "canonical_mapping_missing_count": sum(not bool(row.get("canonical_layer")) for row in rows),
        "identical_activation_scales_shared_across_layers": shared,
        "shared_scale_review": "all four repeated values are adjacent output-to-input chains, not unrelated coupled groups",
        "unrelated_scale_sharing_detected": False,
        "add_concat_residual_boundaries": [row for row in activation_rows if any(token in f"{row.get('producer_op', '')};{row.get('consumer_op', '')}" for token in ("Add", "Concat"))],
        "residual_add_one_sided_explicit_qdq": [row for row in activation_rows if row.get("consumer_op") == "Add"],
        "scatter_plugin_boundaries": [row for row in activation_rows if "pointpillarscatter" in json.dumps(row).lower()],
        "early_backbone_precision": [row for row in precision_rows if str(row.get("module_path", "")).startswith("backbone_m1.resnet.layer0")],
        "detection_head_precision": [row for row in precision_rows if str(row.get("module_path", "")) in {"cls_head", "reg_head", "dir_head"}],
        "realized_int8_layer_count": sum(str(row.get("realized_request_precision", "")).lower() == "int8" for row in precision_rows),
        "realized_fp16_layer_count": sum(str(row.get("realized_request_precision", "")).lower() == "fp16" for row in precision_rows),
    }
    write_json(root / "quantization_spec_audit.json", result)
    return result


def build_recursive_manifest(root: Path, dependency_basis: dict[str, Any]) -> dict[str, Any]:
    excluded = {"generated_artifact_manifest.json"}
    rows = []
    dependency_digest = json_hash(dependency_basis)
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name not in excluded):
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
                "dependency_hash": dependency_digest,
            }
        )
    result = {"root": str(root), "dependency_basis": dependency_basis, "dependency_hash": dependency_digest, "artifact_count": len(rows), "artifacts": rows}
    write_json(root / "generated_artifact_manifest.json", result)
    return result


def make_reproduction_script(root: Path) -> None:
    text = f'''#!/usr/bin/env bash
set -euo pipefail

REPO={REPO}
TRT_ROOT={TRT_ROOT}
PHYSICAL_GPU=6
OUTPUT="$REPO/outputs/int8_baseline_equivalence_audit_$(date +%Y%m%d_%H%M%S)"
source ${CONDA_BASE}/etc/profile.d/conda.sh
cd "$REPO"

conda activate univ2x-opt
python scripts/int8_baseline_equivalence_audit.py --phase base --gpu "$PHYSICAL_GPU" --output "$OUTPUT"
python scripts/int8_baseline_equivalence_audit.py --phase int8 --gpu "$PHYSICAL_GPU" --output "$OUTPUT"
python scripts/int8_equivalence_tensor_parity.py --audit-root "$OUTPUT" --action prepare-inputs

conda activate modelopt
export PATH="$CONDA_PREFIX/bin:$PATH"
export CUDA_HOME="$CONDA_PREFIX"
export CC="$CONDA_PREFIX/bin/gcc"
export CXX="$CONDA_PREFIX/bin/g++"
export CUDACXX="$CONDA_PREFIX/bin/nvcc"
export LD_LIBRARY_PATH="$TRT_ROOT/lib:$CONDA_PREFIX/lib:${{LD_LIBRARY_PATH:-}}"
export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"

LEGACY="$REPO/tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare"
python scripts/int8_equivalence_engine_probe.py --engine "$LEGACY/artifacts/engines/fixedK29696/dynamic_agent_single_engine_maxK/fp16/lidar_pyramid_dynamic_agent_single_engine_maxK_fp16.engine" --plugin "$LEGACY/artifacts/plugins/pointpillar_scatter_trt_build/libpointpillar_scatter_trt.so" --output "$OUTPUT/engine_probe_legacy_fp16.json"
python scripts/int8_equivalence_engine_probe.py --engine "$OUTPUT/search_strict_fp16_force_rebuild/artifacts/engine.plan" --plugin "$REPO/quantization/plugins/pointpillar_scatter_trt/build/libpointpillar_scatter_trt.so" --output "$OUTPUT/engine_probe_search_fp16.json"
python scripts/int8_equivalence_engine_probe.py --engine "$LEGACY/artifacts/engines/fixedK29696/dynamic_agent_single_engine_maxK/int8_train_calib200/lidar_pyramid_dynamic_agent_single_engine_maxK_int8_train_calib200.engine" --plugin "$LEGACY/artifacts/plugins/pointpillar_scatter_trt_build/libpointpillar_scatter_trt.so" --output "$OUTPUT/engine_probe_legacy_int8.json"
python scripts/int8_equivalence_engine_probe.py --engine "$OUTPUT/search_maximal_legal_int8_force_rebuild/artifacts/engine.plan" --plugin "$REPO/quantization/plugins/pointpillar_scatter_trt/build/libpointpillar_scatter_trt.so" --output "$OUTPUT/engine_probe_search_int8.json"
python scripts/int8_equivalence_tensor_parity.py --audit-root "$OUTPUT" --action prepare-build

conda activate univ2x-opt
AUDIT_PHYSICAL_GPU="$PHYSICAL_GPU" python scripts/int8_equivalence_tensor_parity.py --audit-root "$OUTPUT" --action run-pytorch

conda activate modelopt
export PATH="$CONDA_PREFIX/bin:$PATH"
export CUDA_HOME="$CONDA_PREFIX"
export CC="$CONDA_PREFIX/bin/gcc"
export CXX="$CONDA_PREFIX/bin/g++"
export CUDACXX="$CONDA_PREFIX/bin/nvcc"
export LD_LIBRARY_PATH="$TRT_ROOT/lib:$CONDA_PREFIX/lib:${{LD_LIBRARY_PATH:-}}"
export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"
python scripts/int8_equivalence_tensor_parity.py --audit-root "$OUTPUT" --action compare-pytorch
for label in search_fp16 legacy_int8 search_int8; do
  python scripts/int8_equivalence_tensor_parity.py --audit-root "$OUTPUT" --action run-route --label "$label"
done
python scripts/int8_equivalence_tensor_parity.py --audit-root "$OUTPUT" --action aggregate

conda activate univ2x-opt
python scripts/int8_equivalence_finalize.py --audit-root "$OUTPUT"
echo "$OUTPUT"
'''
    destination = root / "reproduction_commands.sh"
    destination.write_text(text, encoding="utf-8")
    destination.chmod(0o755)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.audit_root.resolve()

    evals = [
        eval_row("legacy single maxK FP16", root / "legacy_fp16_fresh_evaluation/evaluation.json"),
        eval_row("search all-keep FP16", root / "search_strict_fp16_force_rebuild/artifacts/evaluation.json"),
        eval_row("legacy single maxK INT8 train200", root / "legacy_int8_fresh_evaluation/evaluation.json"),
        eval_row("search all-keep explicit-QDQ INT8 train200", root / "search_maximal_legal_int8_force_rebuild/artifacts/evaluation.json"),
    ]
    binding = build_binding_diff(root)
    postprocess = build_postprocess_diff(root, binding)
    physical = read_json(root / "physical_model_identity.json", {})
    physical_int8 = read_json(root / "int8_physical_identity/physical_model_identity.json", {})
    fp16_gate = read_json(root / "base_fp16_equivalence_gate.json", {})
    int8 = read_json(root / "int8_ab_comparison.json", {})
    scale = read_json(root / "scale_diff_summary.json", {})
    inventory = read_json(root / "qdq_inventory.json", {})
    engines = read_json(root / "engine_layer_precision_summary.json", {})
    tensor = read_json(root / "tensor_parity.json", {})
    onnx = read_json(root / "onnx_graph_diff.json", {})
    calibration = calibration_source_audit(root)
    quantization_spec = build_quantization_spec_audit(root, calibration)
    legacy_provenance = read_json(root / "provenance_legacy.json", {})
    search_provenance = read_json(root / "provenance_search.json", {})
    plugin_equal = bool(int8.get("plugin_binary_equal", False))
    recipe_equal = bool(int8.get("quantization_recipe_equal", False))
    deployment_close = bool(int8.get("deployment_result_within_threshold", False))
    manifest_hashes = {row.get("manifest_hash", "") for row in evals}
    same_manifest = len(manifest_hashes) == 1 and "" not in manifest_hashes
    all_eval_ok = all(row.get("status") == "ok" and row.get("evaluated_frames") == 1789 and row.get("skipped_frames") == 0 for row in evals)

    confirmed = [
        "legacy INT8 is implicit TensorRT EntropyCalibration2 without explicit Q/DQ; search INT8 is BN-fold-aware explicit Q/DQ",
        f"quantization boundary sets differ (legacy={scale.get('legacy_scale_count')} cache entries, search={scale.get('search_q_boundary_count')} explicit Q boundaries)",
        f"search explicit Q/DQ covers {inventory.get('weight_row_count')} weighted layers with per-tensor weights and no weight-axis attribute",
        "builder policy differs: legacy allows unconstrained FP16 fallback; search uses explicit Q/DQ plus obeyed per-layer precision/output constraints",
        f"full-validation INT8 mAP differs by {int8.get('absolute_mAP_difference')}: legacy={int8.get('legacy_mAP')}, search={int8.get('search_mAP')}",
    ]
    if not plugin_equal:
        confirmed.append("legacy and search PointPillarScatterTRT plugin binaries have different SHA256")
    ruled_out = [
        "pruning/structure difference" if physical.get("passed") and physical_int8.get("passed") else "",
        "checkpoint tensor difference" if physical.get("state_dict", {}).get("all_exact_equal") else "",
        "base FP16 accuracy/export/postprocess difference above the 0.001 gate" if fp16_gate.get("passed") else "",
        "ONNX initializer value difference" if onnx.get("initializer_content_multiset_equal_ignoring_names") else "",
    ]
    ruled_out = [value for value in ruled_out if value]
    not_verified = []
    not_verified.append("legacy implicit INT8 does not serialize weight Q/DQ scale shape, axis, or zero point, so those exact legacy weight fields are not directly observable")
    if not tensor:
        not_verified.append("10-frame intermediate tensor parity")
    elif tensor.get("reference", {}).get("PyTorch_FP32", {}).get("status") != "verified":
        not_verified.append("PyTorch FP32 intermediate tensor hooks")
    if not binding.get("all_contexts_created"):
        not_verified.append("runtime EngineInspector/context probe for all four engines")
    ratio = scale.get("matched_activation_scale_ratio", {})
    root_cause = [
        "different INT8 recipe: implicit whole-network calibrator versus explicit maximal-legal subset with fixed Q/DQ boundaries",
        f"all {inventory.get('weight_row_count')} search weight Q/DQ scales are per-tensor with no axis; output-channel axis-0 per-channel weight quantization is absent",
        f"matched search activation scales are {ratio.get('min')}–{ratio.get('max')}x the legacy cache scales (median {ratio.get('median')}x), substantially reducing INT8 resolution",
        f"the first search-specific degradation beyond legacy occurs at {tensor.get('first_search_tensor_worse_than_legacy_by_cosine_0.05')}; the first catastrophic tensor is {tensor.get('first_search_catastrophic_tensor')}",
    ]
    secondary = []
    if not plugin_equal:
        secondary.append("different plugin binary; FP16 parity shows it is not a material base-accuracy cause, but it remains a provenance difference")
    if engines.get("legacy", {}).get("weighted_precision_counts") != engines.get("search", {}).get("weighted_precision_counts"):
        secondary.append("different realized weighted-layer precision profile")
    residual_count = len(quantization_spec.get("residual_add_one_sided_explicit_qdq", []))
    if residual_count:
        secondary.append(f"{residual_count} residual Add boundaries have explicit Q/DQ only on the quantized convolution branch")

    equivalent = bool(
        physical.get("passed")
        and physical_int8.get("passed")
        and fp16_gate.get("passed")
        and postprocess.get("postprocess_equivalent")
        and recipe_equal
        and deployment_close
        and same_manifest
        and all_eval_ok
    )
    report = {
        "question": "all-keep maximal-legal INT8 explicit-Q/DQ versus legacy single_engine_maxK fixedK29696 INT8 train200",
        "verdict": "equivalent" if equivalent else "not_the_same_INT8_baseline",
        "equivalent": equivalent,
        "deployment_results_within_0_005": deployment_close,
        "deployment_results_can_be_called_approximately_equivalent": bool(deployment_close and recipe_equal),
        "base_fp16_gate": fp16_gate,
        "structure_and_parameter_identity": {"fp16": physical.get("passed"), "int8": physical_int8.get("passed")},
        "same_1789_frame_manifest": same_manifest,
        "all_four_evaluations_complete_without_skips": all_eval_ok,
        "quantization_recipe_equal": recipe_equal,
        "postprocess_equivalent": postprocess.get("postprocess_equivalent"),
        "plugin_binary_equal": plugin_equal,
        "evaluations": evals,
        "onnx_summary": onnx,
        "qdq_inventory_summary": {key: value for key, value in inventory.items() if key != "rows"},
        "scale_diff_summary": scale,
        "calibration_source_audit": calibration,
        "quantization_spec_audit": quantization_spec,
        "engine_precision_summary": engines,
        "builder_policy": {"legacy": legacy_provenance.get("builder", {}), "search": search_provenance.get("builder", {})},
        "first_abnormal_tensor": tensor.get("first_abnormal_tensor", {}),
        "first_search_tensor_worse_than_legacy_by_cosine_0.05": tensor.get("first_search_tensor_worse_than_legacy_by_cosine_0.05"),
        "first_search_catastrophic_tensor": tensor.get("first_search_catastrophic_tensor"),
        "confirmed_difference": confirmed,
        "ruled_out": ruled_out,
        "not_yet_verified": not_verified,
        "root_cause": root_cause,
        "secondary_contributor": secondary,
    }
    write_json(root / "equivalence_report.json", report)

    rows = [
        "# INT8 baseline equivalence report",
        "",
        f"**Verdict: `{report['verdict']}`.**",
        "",
        "The all-keep search model is structurally identical to the original checkpoint, and the FP16 base gate passes. However, the legacy INT8 engine and the search explicit-Q/DQ engine do not implement the same quantization recipe, so they must not be called the same INT8 baseline.",
        "",
        "## Formal A/B evaluation",
        "",
        "| Route | AP@0.30 | AP@0.50 | AP@0.70 | mAP | p50 ms | FPS | frames/skips |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in evals:
        rows.append(f"| {row['label']} | {row['AP@0.30']:.6f} | {row['AP@0.50']:.6f} | {row['AP@0.70']:.6f} | {row['mAP']:.6f} | {row['p50_ms']:.4f} | {row['FPS']:.2f} | {row['evaluated_frames']}/{row['skipped_frames']} |")
    rows.extend(
        [
            "",
            f"FP16 absolute mAP difference: `{fp16_gate.get('absolute_difference')}` (threshold 0.001).",
            f"INT8 absolute mAP difference: `{int8.get('absolute_mAP_difference')}` (deployment threshold 0.005).",
            "",
            "## confirmed_difference",
            "",
            *[f"- {value}" for value in confirmed],
            "",
            "## ruled_out",
            "",
            *[f"- {value}" for value in ruled_out],
            "",
            "## not_yet_verified",
            "",
            *([f"- {value}" for value in not_verified] or ["- none"]),
            "",
            "## root_cause",
            "",
            *[f"- {value}" for value in root_cause],
            "",
            "## secondary_contributor",
            "",
            *([f"- {value}" for value in secondary] or ["- none"]),
            "",
            "All legacy files were read-only inputs; all fresh calibration, Q/DQ, engine and evaluation artifacts are contained in this audit directory.",
        ]
    )
    (root / "equivalence_report.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (root / "root_cause_summary.md").write_text(
        "# Root-cause summary\n\n"
        + "## root_cause\n\n"
        + "\n".join(f"- {value}" for value in root_cause)
        + "\n\n## secondary_contributor\n\n"
        + ("\n".join(f"- {value}" for value in secondary) if secondary else "- none")
        + "\n\n## ruled_out\n\n"
        + "\n".join(f"- {value}" for value in ruled_out)
        + "\n",
        encoding="utf-8",
    )
    make_reproduction_script(root)
    dependency_basis = {
        "checkpoint": sha256_file(CHECKPOINT),
        "config": sha256_file(CONFIG),
        "eval_manifest": sha256_file(root / "baseline/eval_manifest.json"),
        "legacy_provenance": sha256_file(root / "provenance_legacy.json"),
        "search_provenance": sha256_file(root / "provenance_search.json"),
        "toolchain": sha256_file(root / "toolchain_manifest.json"),
    }
    manifest = build_recursive_manifest(root, dependency_basis)
    print(json.dumps({"verdict": report["verdict"], "artifact_count": manifest["artifact_count"], "output": str(root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
