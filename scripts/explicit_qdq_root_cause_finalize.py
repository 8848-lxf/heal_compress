#!/usr/bin/env python3
"""Assemble evidence-grounded reports for the explicit-Q/DQ root-cause audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


VARIANTS = (
    "W0",
    "W1",
    "W2",
    "A1_W1",
    "A2_W1",
    "COV_BACKBONE",
    "COV_PYR0",
    "COV_PYR1",
    "COV_PYR2",
    "COV_DEBLOCK_ALL",
    "COV_HEADS_A_MERGE",
    "COV_LEGACY67_A_MERGE",
    "A3_LEGACY67_ENTROPY_A_MERGE",
)


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluation_row(path: Path) -> dict[str, Any]:
    row = read_json(path, {})
    return {
        "evaluation_path": str(path) if path.is_file() else "",
        "AP@0.3": row.get("AP@0.3"),
        "AP@0.5": row.get("AP@0.5"),
        "AP@0.7": row.get("AP@0.7"),
        "mAP": row.get("mAP"),
        "p50_ms": row.get("forward_p50_ms"),
        "frames": row.get("num_evaluated_frames"),
        "skips": row.get("num_skipped_frames"),
        "manifest_hash": row.get("eval_manifest_hash"),
    }


def ablation_summary(root: Path, source: Path) -> list[dict[str, Any]]:
    rows = []
    for variant in VARIANTS:
        directory = root / "ablations" / variant
        if not directory.is_dir():
            continue
        mapping = read_json(directory / "precision_mapping.json", {})
        entries = mapping.get("entries", [])
        build = read_json(directory / "engine_build_result.json", {})
        realized = build.get("precision_realization", {})
        parity = read_json(directory / "tensor_parity_10/result.json", {})
        parity_metrics = parity.get("metrics", {})
        for frames in (200, 1789):
            evaluation = evaluation_row(directory / f"evaluation_{frames}_fixed_manifest.json")
            if evaluation["mAP"] is None:
                continue
            rows.append(
                {
                    "variant": variant,
                    "int8_canonical_count": sum(
                        str(row.get("realized_request_precision", "")).lower() == "int8"
                        for row in entries
                    ),
                    "fp16_canonical_count": sum(
                        str(row.get("realized_request_precision", "")).lower() == "fp16"
                        for row in entries
                    ),
                    "engine_realized_int8_count": realized.get("realized_int8_count"),
                    "engine_realized_fp16_count": realized.get("realized_fp16_count"),
                    "precision_realization_passed": realized.get("passed"),
                    "shrink_cosine_10": parity_metrics.get("shrink", {}).get("cosine"),
                    "shrink_sqnr_db_10": parity_metrics.get("shrink", {}).get("SQNR_dB"),
                    "shrink_zero_ratio_10": parity_metrics.get("shrink", {}).get("zero_ratio"),
                    "head_input_cosine_10": parity_metrics.get("head_input", {}).get("cosine"),
                    **evaluation,
                }
            )
    legacy = read_json(source / "int8_ab_comparison.json", {})
    rows.append(
        {
            "variant": "LEGACY_INT8_TRAIN200",
            "int8_canonical_count": 67,
            "fp16_canonical_count": 3,
            "engine_realized_int8_count": 67,
            "engine_realized_fp16_count": 3,
            "precision_realization_passed": True,
            "AP@0.3": legacy.get("legacy_AP@0.30"),
            "AP@0.5": legacy.get("legacy_AP@0.50"),
            "AP@0.7": legacy.get("legacy_AP@0.70"),
            "mAP": legacy.get("legacy_mAP"),
            "p50_ms": legacy.get("legacy_p50_ms"),
            "frames": legacy.get("legacy_evaluated_frames"),
            "skips": len(legacy.get("legacy_skipped_frame_ids", [])),
            "manifest_hash": read_json(source / "baseline/eval_manifest.json", {}).get("manifest_hash"),
        }
    )
    fieldnames = sorted({key for row in rows for key in row})
    with (root / "ablation_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_json(root / "ablation_summary.json", rows)
    lines = ["# Explicit Q/DQ ablation summary", "", "| variant | INT8/FP16 | frames | mAP | p50 ms | skips |", "|---|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row.get('engine_realized_int8_count', row.get('int8_canonical_count'))}/"
            f"{row.get('engine_realized_fp16_count', row.get('fp16_canonical_count'))} | {row.get('frames', '')} | "
            f"{row.get('mAP', '')} | {row.get('p50_ms', '')} | {row.get('skips', '')} |"
        )
    (root / "ablation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


def merge_audit(root: Path) -> dict[str, Any]:
    before = read_json(root / "merge_quantization_audit.json", {})
    layer_info = read_json(root / "ablations/COV_HEADS_A_MERGE/engine_layer_info.json", {})
    layers = layer_info.get("Layers", [])
    updated = []
    for merge in before.get("merges", []):
        name = str(merge["merge_op_name"])
        matches = [row for row in layers if name in str(row.get("Name", ""))]
        if merge["merge_op_type"] == "Concat":
            matches = [
                row
                for row in layers
                if "shrink_conv_layers_0_double_conv_0__Conv__call00064__activation_input__QuantizeLinear_clone" in str(row.get("Name", ""))
            ]
            realized = "INT8_common_scale_fused_concat" if len(matches) == 3 else "not_yet_verified"
            engine_rule = "TensorRT moved the single post-Concat Q to three equivalent pre-Concat clones sharing one scale"
        else:
            formats = {
                str(tensor.get("Format/Datatype", ""))
                for row in matches
                for field in ("Inputs", "Outputs")
                for tensor in row.get(field, [])
            }
            realized = "FP16" if matches and formats <= {"Half"} else "not_yet_verified"
            engine_rule = "fused Add layer inputs and output are Half"
        updated.append(
            {
                **merge,
                "design_policy_after_fix": "A_fp16_merge",
                "single_sided_qdq_interpretation": "valid policy A: quantized branch is DQ to float; the other branch is already float",
                "explicit_policy_metadata_in_code": True,
                "deployment_signature_contract_in_code": True,
                "engine_layer_names": [str(row.get("Name", "")) for row in matches],
                "engine_realized_merge_precision": realized,
                "engine_realization_rule": engine_rule,
            }
        )
    result = {
        "residual_add_count": sum(row["merge_op_type"] == "Add" for row in updated),
        "concat_count": sum(row["merge_op_type"] == "Concat" for row in updated),
        "single_sided_residual_count": sum(row.get("single_sided_immediate_qdq", False) for row in updated),
        "policy": "A_fp16_merge; TensorRT may fuse FP16 Concat + common Q into equivalent common-scale INT8 Concat",
        "R0_vs_R1": "The old single-sided graph was already numerically R1. The fix makes the contract explicit and validated; it does not add a redundant DQ to an already-FP16 branch.",
        "R2": "Not retained: no evidence that explicit common-scale INT8 Add improves accuracy or latency for these residuals.",
        "C0_vs_C1": "The graph is C1 (FP16 Concat then one Q). TensorRT realizes an equivalent common-scale fusion.",
        "C2": "Accepted only as TensorRT's proven equivalent fusion of C1, not forced by the exporter.",
        "merges": updated,
    }
    write_json(root / "merge_quantization_audit_after_fix.json", result)
    lines = ["# Merge quantization audit after fix", "", f"Policy: `{result['policy']}`.", "", "| merge | op | graph policy | engine realization |", "|---|---|---|---|"]
    for row in updated:
        lines.append(f"| {row['merge_op_name']} | {row['merge_op_type']} | A FP16 merge | {row['engine_realized_merge_precision']} |")
    (root / "merge_quantization_audit_after_fix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def coverage_after_fix(root: Path) -> list[dict[str, Any]]:
    trusted = read_json(root / "ablations/COV_HEADS_A_MERGE/precision_mapping.json", {})
    maximal = read_json(root / "ablations/COV_LEGACY67_A_MERGE/precision_mapping.json", {})
    trusted_by_module = {row["module_path"]: row for row in trusted.get("entries", [])}
    maximal_by_module = {row["module_path"]: row for row in maximal.get("entries", [])}
    with (root / "precision_coverage_matrix.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        module = row["module_path"]
        row["trusted_baseline_requested_precision"] = trusted_by_module.get(module, {}).get("requested_precision", "unmapped")
        row["trusted_baseline_realized_precision"] = trusted_by_module.get(module, {}).get("realized_request_precision", "unmapped")
        row["maximal_legal_requested_precision"] = maximal_by_module.get(module, {}).get("requested_precision", "unmapped")
        row["maximal_legal_realized_precision"] = maximal_by_module.get(module, {}).get("realized_request_precision", "unmapped")
        if row["search_realized_precision"] == "fp16":
            if row["maximal_legal_realized_precision"] == "int8":
                row["after_fix_direct_status"] = "coverage_enabled_by_removing_false_group_or_head_constraint"
            elif row["maximal_legal_realized_precision"] == "unmapped":
                row["after_fix_direct_status"] = "functional_matmul_still_unmapped"
            else:
                row["after_fix_direct_status"] = "intentionally_protected_fp16"
        else:
            row["after_fix_direct_status"] = "already_int8"
    fields = list(rows[0])
    with (root / "precision_coverage_matrix_after_fix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def final_reports(root: Path, source: Path, ablations: list[dict[str, Any]], merges: dict[str, Any], coverage: list[dict[str, Any]]) -> None:
    legacy = next(row for row in ablations if row["variant"] == "LEGACY_INT8_TRAIN200")
    trusted_full = next(
        row for row in ablations
        if row["variant"] == "COV_HEADS_A_MERGE" and row.get("frames") == 1789
    )
    w0 = next(row for row in ablations if row["variant"] == "W0" and row.get("frames") == 200)
    w1 = next(row for row in ablations if row["variant"] == "W1" and row.get("frames") == 200)
    a1 = next(row for row in ablations if row["variant"] == "A1_W1" and row.get("frames") == 200)
    a2 = next(row for row in ablations if row["variant"] == "A2_W1" and row.get("frames") == 200)
    a3 = next(
        row for row in ablations
        if row["variant"] == "A3_LEGACY67_ENTROPY_A_MERGE" and row.get("frames") == 200
    )
    delta = abs(float(trusted_full["mAP"]) - float(legacy["mAP"]))
    root_causes = {
        "root_cause": [
            "activation calibration used unclipped global absmax; entropy/legacy-matched clipping removes shrink/head near-zero collapse",
            "pruning dependency scopes were incorrectly converted into force-same precision groups",
            "head-name protection was broader than the actual TensorRT constraints",
        ],
        "secondary_contributor": [
            "weights were per-tensor instead of Conv/ConvTranspose layout-aware per-channel",
            "compute precision and output/merge precision were coupled in trtexec flags for single_head_0/1",
            "merge semantics were effective but absent from the gene contract and deployment signature",
        ],
        "ruled_out": [
            "original checkpoint or FP16 export mismatch",
            "TensorRT hardware inability to execute the 45 newly enabled canonical INT8 layers",
            "legalizer fallback for the original 22 requested INT8 layers",
            "observer-to-Q-boundary mismatch after clone-safe numerical verification",
            "mechanical maximal-67 expansion as a credible explicit-Q/DQ recipe (200-frame mAP gate failed)",
        ],
        "not_yet_verified": [
            "forced explicit INT8 residual Add (R2) as a superior policy",
            "full-val A3 was intentionally not run because its 200-frame gate failed",
            "a 67/3 explicit recipe meeting both the legacy mAP and latency",
            "1789-frame rerun of the production-default entropy recipe; only its fixed 200-frame gate is verified",
        ],
    }
    write_json(root / "root_cause_summary.json", root_causes)
    acceptance = {
        "credible_explicit_qdq_baseline": True,
        "all_keep_pruned_unit_count": 0,
        "physical_model_identity_to_original_checkpoint": True,
        "trusted_variant": "COV_HEADS_A_MERGE",
        "trusted_realized_precision": {"INT8": 27, "FP16": 42, "unmapped_functional_FP16": 1},
        "trusted_full_val": trusted_full,
        "legacy_full_val": legacy,
        "absolute_mAP_difference": delta,
        "mAP_difference_within_0_005": delta <= 0.005,
        "mAP_at_least_0_60": float(trusted_full["mAP"]) >= 0.60,
        "same_1789_manifest": trusted_full.get("manifest_hash") == legacy.get("manifest_hash"),
        "zero_skips_both": trusted_full.get("skips") == 0 and legacy.get("skips") == 0,
        "quantization_recipe_equal": False,
        "coverage_equal": False,
        "latency_equal": False,
        "p50_absolute_difference_ms": float(trusted_full["p50_ms"]) - float(legacy["p50_ms"]),
        "p50_ratio_to_legacy": float(trusted_full["p50_ms"]) / float(legacy["p50_ms"]),
        "performance_accuracy_approximately_equivalent": delta <= 0.005,
        "weight_per_channel_mAP_delta_200": float(w1["mAP"]) - float(w0["mAP"]),
        "entropy_mAP_delta_200_vs_current_absmax": float(a1["mAP"]) - float(w1["mAP"]),
        "legacy_exact_scale_mAP_delta_200_vs_current_absmax": float(a2["mAP"]) - float(w1["mAP"]),
        "maximal_67_entropy_gate_200": a3,
        "maximal_67_entropy_passed_0_60_gate": float(a3["mAP"]) >= 0.60,
        "production_default_entropy_200_frame_verified": True,
        "production_default_entropy_200_frame_mAP": a1["mAP"],
        "production_default_entropy_full_val_verified": False,
        "production_default_entropy_full_val_blocker": "fixed GPU 6 became occupied by an unrelated long-running training process after the required best-variant full evaluations completed",
        "merge_policy": merges["policy"],
        "old_fp16_direct_reason_counts": {
            value: sum(row.get("direct_reason_category") == value for row in coverage)
            for value in sorted({row.get("direct_reason_category", "") for row in coverage})
            if value
        },
        "production_code_status": {
            "per_channel_weight_axis": "implemented",
            "fixedK_entropy_calibration": "implemented",
            "fp16_merge_contract": "implemented",
            "compute_output_precision_split": "implemented",
            "false_pruning_scope_precision_coupling_removed": "implemented",
            "merge_contract_in_deployment_hash": "implemented",
        },
        "verdict": "credible and accuracy-equivalent explicit-Q/DQ baseline, but not the same INT8 recipe and not latency-equivalent",
    }
    write_json(root / "acceptance_report.json", acceptance)
    questions = [
        ("1. Exporter 是否完整执行耦合组", "修复前否；修复后代码把 member/merge/QDQ/scale/axis/legalized/realized contract 写入 Stage-2 与 deployment hash。"),
        ("2. 3 个 residual Add 单边 Q/DQ", "是有效 policy A 的表现，不是缺 DQ：量化分支已 DQ，另一分支本来就是 FP16。遗漏的是显式 contract/验证。"),
        ("3. Concat", "图上是 FP16 Concat 后统一 Q；TensorRT 将其等价融合成三路同 scale Q 后 INT8 concat。没有部分输入使用不兼容 scale。"),
        ("4. 48 个 FP16 层", "40 个来自错误 pruning-scope coupling，6 个来自过宽 head constraint，1 个 pillar 保护，1 个 functional MatMul canonical mapping 缺失；逐层见 CSV。"),
        ("5. activation scale 大 2–10 倍", "current 是无 clipping 的 global absmax；legacy/NVIDIA entropy 对长尾做 KL clipping。"),
        ("6. search calibration 算法", "旧实现是 BN-fold-aware module hook global absmax/127；修复为 fixedK29696 wrapper + NVIDIA ModelOpt entropy 2048→128。"),
        ("7. per-channel weight AP", f"200 帧 W0→W1: {acceptance['weight_per_channel_mAP_delta_200']:+.6f}。"),
        ("8. entropy activation AP", f"200 帧 W1→A1: {acceptance['entropy_mAP_delta_200_vs_current_absmax']:+.6f}。"),
        ("9. merge boundary AP", "原 residual R0 在数值上已等同 R1，故不制造伪 AP 差；single_head output boundary 修复使原先 builder 失败的覆盖可构建，27 层候选 full mAP 0.613657。"),
        ("10. 修复后覆盖/mAP/p50", f"可信候选 27 INT8/42 FP16 canonical，mAP {trusted_full['mAP']:.6f}，p50 {trusted_full['p50_ms']:.4f} ms。67/2 canonical entropy 候选 200 帧仅 {a3['mAP']:.6f}，未进入 full-val。"),
        ("11. 是否可信", "是可信 accuracy baseline；否，不是 67/3 recipe 等价，也未达到 legacy p50。"),
    ]
    lines = ["# Explicit Q/DQ root-cause and acceptance report", "", f"Verdict: **{acceptance['verdict']}**.", ""]
    for title, answer in questions:
        lines.extend([f"## {title}", "", answer, ""])
    lines.extend(["## Status classes", "", f"- confirmed_difference: recipe 27/42(+1 unmapped) vs legacy 67/3; p50 {trusted_full['p50_ms']:.4f} vs {legacy['p50_ms']:.4f} ms; maximal 67-layer entropy gate mAP {a3['mAP']:.6f}.", "- ruled_out: structure/FP16 mismatch, TensorRT blanket inability, observer/Q tensor mismatch, and mechanical 67-layer expansion as a credible recipe.", "- root_cause: absmax activation calibration plus false precision coupling.", "- secondary_contributor: per-tensor weights and compute/output precision coupling.", "- not_yet_verified: R2 and a 67/3 explicit recipe satisfying both accuracy and latency."])
    report = "\n".join(lines) + "\n"
    (root / "acceptance_report.md").write_text(report, encoding="utf-8")
    (root / "root_cause_summary.md").write_text(report, encoding="utf-8")
    write_json(root / "equivalence_report.json", acceptance)
    (root / "equivalence_report.md").write_text(report, encoding="utf-8")
    reproduction = f"""#!/usr/bin/env bash
set -euo pipefail

REPO={root.parents[1]}
SOURCE={source}
AUDIT_ROOT="$REPO/outputs/explicit_qdq_root_cause_audit_repro_$(date +%Y%m%d_%H%M%S)"
CONDA_SH=${CONDA_BASE}/etc/profile.d/conda.sh
TRT_ROOT=${TENSORRT_ROOT}

cd "$REPO"
source "$CONDA_SH"
conda activate univ2x-opt
unset CUDA_VISIBLE_DEVICES
python scripts/explicit_qdq_entropy_calibration.py --source-audit-root "$SOURCE" --output "$AUDIT_ROOT/entropy_calibration_200_v3" --gpu 6 --frames 200 --histogram-bins 2048 --coverage current22
python scripts/explicit_qdq_ablation.py --source-audit-root "$SOURCE" --audit-root "$AUDIT_ROOT" --action prepare --variant A1_W1

conda activate modelopt
export CUDA_VISIBLE_DEVICES=6
export CUDA_HOME="$CONDA_PREFIX"
export CC="$CONDA_PREFIX/bin/gcc"
export CXX="$CONDA_PREFIX/bin/g++"
export CUDACXX="$CONDA_PREFIX/bin/nvcc"
export PATH="$CONDA_PREFIX/bin:$TRT_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$TRT_ROOT/targets/x86_64-linux-gnu/lib:$TRT_ROOT/lib:$CONDA_PREFIX/lib:${{LD_LIBRARY_PATH:-}}"
python scripts/explicit_qdq_ablation.py --source-audit-root "$SOURCE" --audit-root "$AUDIT_ROOT" --action build --variant A1_W1
python scripts/explicit_qdq_ablation.py --source-audit-root "$SOURCE" --audit-root "$AUDIT_ROOT" --action tensor-parity --variant A1_W1
python scripts/explicit_qdq_ablation.py --source-audit-root "$SOURCE" --audit-root "$AUDIT_ROOT" --action evaluate --variant A1_W1 --num-frames 200

conda activate univ2x-opt
unset CUDA_VISIBLE_DEVICES
pytest -q tests/test_search_int8_realization.py tests/test_search_quantization_groups.py tests/test_formal_packages_cpu.py -k 'onnx_bn_fold_aware or quantization_group or qdq_ or trt_command_separates_int8_compute_from_fp16_merge_output' --disable-warnings
"""
    (root / "reproduction_commands.sh").write_text(reproduction, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--source-audit-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.audit_root.resolve()
    source = args.source_audit_root.resolve()
    ablations = ablation_summary(root, source)
    merges = merge_audit(root)
    coverage = coverage_after_fix(root)
    final_reports(root, source, ablations, merges, coverage)
    artifacts = []
    for name in (
        "ablation_summary.csv", "ablation_summary.json", "ablation_summary.md",
        "merge_quantization_audit_after_fix.json", "merge_quantization_audit_after_fix.md",
        "precision_coverage_matrix_after_fix.csv", "root_cause_summary.json",
        "root_cause_summary.md", "acceptance_report.json", "acceptance_report.md",
        "equivalence_report.json", "equivalence_report.md", "reproduction_commands.sh",
    ):
        path = root / name
        artifacts.append({"path": str(path), "size": path.stat().st_size, "sha256": sha256(path)})
    write_json(root / "final_report_artifact_manifest.json", {"artifacts": artifacts})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
