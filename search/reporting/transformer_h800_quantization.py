"""Evidence-only reports for the dual-model H800 Transformer audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from search.model_families.transformer.fp8_profiles import FP8_PROFILE_ROLES
from search.model_families.transformer.precision_contract import precision_profiles
from search.model_families.transformer.search_space import (
    build_profile_library,
    build_search_space,
)
from search.model_families.transformer.smoothquant_profiles import smoothquant_profiles


MODELS = ("lidar_cobevt", "lidar_v2xvit")
SECTIONS = (
    "baselines", "precision_sensitivity", "smoothquant", "smoothquant_sm90",
    "bf16", "fp8", "accumulator",
)


def _read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in values for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in values:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def _evaluation(directory: Path, protocol: str) -> dict[str, Any]:
    return _read_json(directory / "evaluation" / protocol / "evaluation_acceptance.json", {}) or {}


def _precision_label(storage: str, output: str) -> str:
    return storage if storage == output else f"{storage}_OUTPUT_{output}"


def _matmul_label(prefix: str, operand: str, accumulator: str) -> str:
    return f"{prefix}_{operand}A{accumulator}"


def _contract_semantics(profile: str) -> dict[str, str] | None:
    contracts = precision_profiles().get(profile)
    if contracts is None:
        return None
    q = contracts["q_projection"]
    k = contracts["k_projection"]
    qk_projection = _precision_label(q.storage_precision, q.output_precision)
    if (q.storage_precision, q.output_precision) != (k.storage_precision, k.output_precision):
        qk_projection = f"Q={qk_projection};K={_precision_label(k.storage_precision, k.output_precision)}"
    qk = contracts["qk_matmul"]
    av = contracts["av_matmul"]
    return {
        "qk_projection_precision": qk_projection,
        "v_projection_precision": _precision_label(
            contracts["v_projection"].storage_precision,
            contracts["v_projection"].output_precision,
        ),
        "output_projection_precision": _precision_label(
            contracts["output_projection"].storage_precision,
            contracts["output_projection"].output_precision,
        ),
        "qk_operand_accumulator_profile": _matmul_label(
            "Q", qk.operand_precision, qk.accumulator_precision
        ),
        "softmax_precision": _precision_label(
            contracts["softmax"].storage_precision,
            contracts["softmax"].output_precision,
        ),
        "av_operand_accumulator_profile": _matmul_label(
            "A", av.operand_precision, av.accumulator_precision
        ),
        "layernorm_precision": _precision_label(
            contracts["layernorm"].storage_precision,
            contracts["layernorm"].output_precision,
        ),
        "ffn1_precision": _precision_label(
            contracts["ffn1"].storage_precision, contracts["ffn1"].output_precision
        ),
        "ffn2_precision": _precision_label(
            contracts["ffn2"].storage_precision, contracts["ffn2"].output_precision
        ),
        "residual_precision": _precision_label(
            contracts["residual_add"].storage_precision,
            contracts["residual_add"].output_precision,
        ),
    }


def _profile_semantics(profile: str) -> dict[str, str]:
    exact = _contract_semantics(profile)
    if exact is not None:
        return exact
    # SmoothQuant and FP8 profiles are F3-derived: LayerNorm/QK are protected
    # FP32, while the remaining core is FP16.  Only explicitly selected
    # projections are replaced by Q/DQ-backed low precision.
    semantics = _contract_semantics("B3_F3")
    assert semantics is not None
    sq = {row.profile_id: row for row in smoothquant_profiles()}.get(profile)
    selected_precision = "INT8" if sq is not None else "FP8_E4M3"
    selected_roles = sq.int8_roles if sq is not None else FP8_PROFILE_ROLES.get(profile, ())
    role_fields = {
        "q_projection": "qk_projection_precision",
        "k_projection": "qk_projection_precision",
        "v_projection": "v_projection_precision",
        "output_projection": "output_projection_precision",
        "ffn1": "ffn1_precision",
        "ffn2": "ffn2_precision",
    }
    for role in selected_roles:
        output = "FP32" if role in {"q_projection", "k_projection"} else "FP16"
        semantics[role_fields[role]] = f"{selected_precision}_DQ_{output}"
    return semantics


def _formal_latency(root: Path, model: str) -> dict[str, dict[str, Any]]:
    rows = _read_json(root / "latency" / model / "formal_latency.json", []) or []
    result = {}
    for row in rows:
        if not row.get("baseline_replay"):
            result[f"{row.get('section')}:{row.get('profile')}"] = dict(row)
    baselines = [row for row in rows if row.get("baseline_replay")]
    if baselines:
        result["baselines:B1_TRT_ATTN_FP32"] = dict(baselines[0])
    return result


def collect_profiles(root: Path, model: str) -> list[dict[str, Any]]:
    reference_dir = root / "baselines" / model / "B1_TRT_ATTN_FP32"
    reference500 = _evaluation(reference_dir, "fixed500")
    reference_map = float(reference500["mAP"]) if reference500.get("status") == "ok" else None
    latency = _formal_latency(root, model)
    latency_ref = latency.get("baselines:B1_TRT_ATTN_FP32", {})
    alpha_selection = _read_json(
        root / "smoothquant_alpha_sm90" / model / "alpha_selection.json", {}
    ) or {}
    selected_alpha = alpha_selection.get("selected", {}).get("alpha")
    rows = []
    for section in SECTIONS:
        section_dir = root / section / model
        if not section_dir.is_dir():
            continue
        for directory in sorted(path for path in section_dir.iterdir() if path.is_dir()):
            result = _read_json(directory / "baseline_result.json", {}) or {}
            if not result and directory.name == "B0_PYTORCH_FP32":
                result = {
                    "status": "ok",
                    "engine_exists": False,
                    "pytorch_checkpoint_baseline": True,
                }
            if not result:
                continue
            profile = directory.name
            invalid_legacy_bf16_contract = (
                section == "precision_sensitivity"
                and profile in {
                    "P13_QKV_BF16",
                    "P14_QK_BF16",
                    "P15_FFN_BF16",
                    "P16_FULL_ATTENTION_BF16",
                }
            )
            report_profile = profile
            if invalid_legacy_bf16_contract:
                report_profile = f"{profile}__INVALID_OLD_CONTRACT"
            elif section == "smoothquant_sm90":
                alpha = result.get("alpha")
                alpha_tag = str(alpha).replace(".", "p") if alpha is not None else "unknown"
                report_profile = f"{profile}__SM90_ALPHA_{alpha_tag}"
            alpha_contract_match = not (
                section == "smoothquant_sm90"
                and (
                    selected_alpha is None
                    or result.get("alpha") is None
                    or abs(float(result["alpha"]) - float(selected_alpha)) > 1.0e-12
                )
            )
            fixed50 = _evaluation(directory, "fixed50")
            fixed500 = _evaluation(directory, "fixed500")
            realized = _read_csv(directory / "requested_realized.csv")
            conflicts = [row for row in realized if str(row.get("conflict", ""))]
            int8 = sum(str(row.get("realized_precision", "")).upper() == "INT8" for row in realized)
            fp8 = sum(str(row.get("realized_precision", "")).upper() == "FP8" for row in realized)
            bf16 = sum(str(row.get("realized_precision", "")).upper() == "BF16" for row in realized)
            fp16 = sum(str(row.get("realized_precision", "")).upper() == "FP16" for row in realized)
            fp32 = sum(str(row.get("realized_precision", "")).upper() == "FP32" for row in realized)
            delta = (
                float(fixed500["mAP"]) - reference_map
                if reference_map is not None and fixed500.get("status") == "ok"
                else None
            )
            fixed500_safe = delta is not None and abs(delta) <= 0.003
            if delta is None:
                safety = "not_yet_verified"
            elif abs(delta) <= 0.003:
                safety = "SAFE_FIXED500"
            elif delta >= -0.010:
                safety = "BORDERLINE"
            else:
                safety = "UNSAFE"
            timing = latency.get(f"{section}:{profile}", {})
            latency_benefit = bool(
                timing and latency_ref and float(timing["p50_ms"]) < float(latency_ref["p50_ms"])
            )
            status = (
                "invalid_contract"
                if invalid_legacy_bf16_contract
                else str(result.get("status", "not_yet_verified"))
            )
            library_status = (
                "unsupported" if status in {
                    "precision_fallback", "precision_conflict",
                    "engine_build_failed", "invalid_contract"
                }
                else "rejected" if safety == "UNSAFE"
                else "experimental"
            )
            qdq = _read_csv(directory / "qdq_inventory.csv")
            accumulator_rows = [
                row for row in realized if row.get("role") in {"qk_matmul", "av_matmul"}
            ]
            accumulator_level = (
                "A" if accumulator_rows and all("accumulator_level_A" in str(row.get("evidence_source", "")) for row in accumulator_rows)
                else "C" if any(str(row.get("realized_accumulator", "")).lower() == "unknown" for row in accumulator_rows)
                else "B"
            )
            row = {
                "model": model,
                "section": section,
                "profile_id": report_profile,
                "source_profile_id": profile,
                "build_status": status,
                "selected_model_alpha": selected_alpha,
                "alpha_contract_match": alpha_contract_match,
                "invalid_legacy_bf16_contract": invalid_legacy_bf16_contract,
                "status": library_status,
                "physical_graph_legal": bool(
                    result.get(
                        "physical_structure_frozen",
                        section not in {"smoothquant", "smoothquant_sm90", "fp8"},
                    )
                ),
                "onnx_success": bool(result.get("typed_onnx_sha256") or result.get("final_qdq_sha256")),
                "engine_success": bool(result.get("engine_exists")),
                "requested_realized_match": not conflicts and bool(realized),
                "no_precision_conflict": not conflicts and bool(realized),
                "zero_skip": int(fixed500.get("skipped", -1)) == 0 if fixed500 else False,
                "fixed50_mAP": fixed50.get("mAP"),
                "fixed50_status": fixed50.get("status", "not_yet_verified"),
                "fixed500_mAP": fixed500.get("mAP"),
                "fixed500_evaluation_forward_p50_ms": fixed500.get("forward_p50_ms"),
                "fixed500_delta_mAP": delta,
                "fixed500_safety": safety,
                "fixed500_safe": fixed500_safe,
                "formal_latency_p50_ms": timing.get("p50_ms"),
                "formal_latency_p90_ms": timing.get("p90_ms"),
                "formal_latency_p95_ms": timing.get("p95_ms"),
                "formal_latency_p99_ms": timing.get("p99_ms"),
                "formal_latency_mean_ms": timing.get("mean_ms"),
                "formal_latency_std_ms": timing.get("std_ms"),
                "formal_latency_speedup_vs_b1": (
                    float(latency_ref["p50_ms"]) / float(timing["p50_ms"])
                    if timing and latency_ref else None
                ),
                "formal_latency_kernel_count": timing.get("kernel_count"),
                "formal_latency_cast_reformat_count": timing.get("cast_reformat_count"),
                "formal_latency_baseline_replay_drift_ratio": timing.get(
                    "baseline_replay_p50_drift_ratio"
                ),
                "latency_benefit": latency_benefit,
                "realized_int8_count": int8,
                "realized_fp8_count": fp8,
                "realized_bf16_count": bf16,
                "realized_fp16_count": fp16,
                "realized_fp32_count": fp32,
                "qdq_count": len(qdq),
                "precision_conflict_count": len(conflicts),
                "accumulator_evidence_level": accumulator_level,
                "accumulator_searchable": accumulator_level == "A" and not conflicts,
                **_profile_semantics(profile),
            }
            if alpha_contract_match and all(
                row[key]
                for key in (
                    "physical_graph_legal", "onnx_success", "engine_success",
                    "requested_realized_match", "zero_skip", "fixed500_safe",
                    "latency_benefit", "no_precision_conflict",
                )
            ) and status == "ok":
                row["status"] = "allowed"
            rows.append(row)
    return rows


def _consolidate_precision(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    realized_all = []
    qdq_all = []
    for model in MODELS:
        model_rows = []
        for section in SECTIONS:
            base = root / section / model
            if not base.is_dir():
                continue
            for directory in sorted(path for path in base.iterdir() if path.is_dir()):
                rows = _read_csv(directory / "requested_realized.csv")
                for row in rows:
                    row.update(section=section, profile=directory.name, model=model)
                model_rows.extend(rows)
                qrows = _read_csv(directory / "qdq_inventory.csv")
                for row in qrows:
                    row.update(section=section, profile=directory.name, model=model)
                qdq_all.extend(qrows)
        realized_all.extend(model_rows)
        _write_csv(root / "precision" / f"requested_realized_{model.removeprefix('lidar_')}.csv", model_rows)
    _write_csv(root / "precision" / "qdq_inventory.csv", qdq_all)
    _write_csv(
        root / "precision" / "precision_conflicts.csv",
        [row for row in realized_all if str(row.get("conflict", ""))],
    )
    _write_csv(
        root / "precision" / "fusion_tactic_inventory.csv",
        [row for row in realized_all if str(row.get("fusion_kind", "")) or str(row.get("tactic", ""))],
    )
    accumulator = []
    for row in realized_all:
        if row.get("role") not in {"qk_matmul", "av_matmul"}:
            continue
        source = str(row.get("evidence_source", ""))
        level = "A" if "accumulator_level_A" in source else "C" if str(row.get("realized_accumulator", "")).lower() == "unknown" else "B"
        accumulator.append(
            {
                **row,
                "operator": "QK" if row.get("role") == "qk_matmul" else "AV",
                "evidence_level": level,
                "searchable": level == "A" and not str(row.get("conflict", "")),
            }
        )
    _write_csv(root / "precision" / "accumulator_evidence.csv", accumulator)
    return realized_all, accumulator


def _consolidate_inventory(root: Path) -> dict[str, Any]:
    inventories = {
        model: _read_json(root / "inventory" / model / "inventory.json", {}) or {}
        for model in MODELS
    }
    schemas = [value.get("schema", {}) for value in inventories.values() if value]
    if schemas and any(schema != schemas[0] for schema in schemas[1:]):
        raise RuntimeError("cross_model_transformer_role_schema_mismatch")
    schema = schemas[0] if schemas else {
        "schema_version": "canonical-transformer-role-schema-v1",
        "canonical_roles": [],
    }
    _write_json(root / "inventory" / "canonical_transformer_role_schema.json", schema)
    unsupported = []
    for model, inventory in inventories.items():
        for row in inventory.get("missing_role_mapping", ()):
            unsupported.append({**dict(row), "model": model, "status": "missing_role_mapping"})
    _write_csv(root / "inventory" / "unsupported_role_mapping.csv", unsupported)
    return {
        "schema_version": schema.get("schema_version"),
        "unsupported_role_mapping_count": len(unsupported),
        "by_model": {
            model: len(value.get("missing_role_mapping", ()))
            for model, value in inventories.items()
        },
    }


def _conclusion(root: Path, profiles: Mapping[str, list[dict[str, Any]]], accumulator: list[dict[str, Any]]) -> str:
    def profile(model: str, name: str) -> dict[str, Any]:
        candidates = [
            row for row in profiles[model]
            if row["profile_id"] == name or row.get("source_profile_id") == name
        ]
        return next(
            (row for row in candidates if row.get("section") == "smoothquant_sm90"),
            candidates[0] if candidates else {},
        )

    def metric(model: str, name: str) -> str:
        row = profile(model, name)
        fixed500 = row.get("fixed500_mAP")
        latency = row.get("formal_latency_p50_ms")
        return (
            f"fixed50={row.get('fixed50_mAP', 'not_yet_verified')}, "
            f"fixed500={fixed500 if fixed500 is not None else 'not_yet_verified'}, "
            f"p50={latency if latency is not None else 'not_yet_verified'} ms"
        )

    def alpha(model: str) -> tuple[Any, Any]:
        payload = _read_json(
            root / "smoothquant_alpha_sm90" / model / "alpha_selection.json", {}
        ) or {}
        selected = payload.get("selected", {})
        return (
            selected.get("alpha", "not_yet_verified"),
            selected.get("scale_stability_delta", "not_yet_verified"),
        )

    def safe_names(model: str) -> list[str]:
        return [
            str(row["profile_id"])
            for row in profiles[model]
            if row.get("section") != "smoothquant"
            and row.get("fixed500_safety") == "SAFE_FIXED500"
            and row.get("requested_realized_match")
            and row.get("alpha_contract_match", True)
        ]

    def allowed_names(model: str) -> list[str]:
        return [
            str(row["profile_id"])
            for row in profiles[model]
            if row.get("status") == "allowed"
        ]

    def safe_low_precision(model: str) -> list[str]:
        return [
            str(row["profile_id"])
            for row in profiles[model]
            if row.get("section") in {"smoothquant_sm90", "bf16", "fp8"}
            and row.get("fixed500_safety") == "SAFE_FIXED500"
            and row.get("requested_realized_match")
            and row.get("alpha_contract_match", True)
        ]

    def latency_delta(model: str, name: str) -> str:
        row = profile(model, name)
        baseline = profile(model, "B1_TRT_ATTN_FP32")
        value = row.get("formal_latency_p50_ms")
        reference = baseline.get("formal_latency_p50_ms")
        if value is None or reference is None:
            return "not_yet_verified"
        return f"{float(value) - float(reference):+.6f} ms ({float(value) / float(reference):.4f}x)"

    cobevt_p12 = profile("lidar_cobevt", "P12_FULL_ATTENTION_FP16")
    v2_p12 = profile("lidar_v2xvit", "P12_FULL_ATTENTION_FP16")
    fp8 = [row for rows in profiles.values() for row in rows if "FP8" in row["profile_id"]]
    bf16 = [row for rows in profiles.values() for row in rows if "BF16" in row["profile_id"]]
    fused = sum(
        any(
            token in str(row.get("fusion_kind", "")).lower()
            for token in ("mha", "attention")
        )
        for row in accumulator
    )
    accumulator_matrix = _read_json(
        root / "accumulator" / "accumulator_matrix.json", []
    ) or []
    accumulator_searchable = [
        f"{row['model']}:{row['experiment']}"
        for row in accumulator_matrix
        if row.get("searchable")
    ]
    co_alpha, co_stability = alpha("lidar_cobevt")
    v2_alpha, v2_stability = alpha("lidar_v2xvit")
    lines = [
        "# H800 Transformer Quantization Root Conclusion",
        "",
        f"1. CoBEVT H800 baselines: B0 PyTorch {metric('lidar_cobevt', 'B0_PYTORCH_FP32')}; B1 Attention-FP32 TRT {metric('lidar_cobevt', 'B1_TRT_ATTN_FP32')}; B2 strict-FP16 {metric('lidar_cobevt', 'B2_TRT_STRICT_FP16')}; F3 {metric('lidar_cobevt', 'B3_F3')}.",
        f"2. V2XViT H800 baselines: B0 PyTorch {metric('lidar_v2xvit', 'B0_PYTORCH_FP32')}; B1 Attention-FP32 TRT {metric('lidar_v2xvit', 'B1_TRT_ATTN_FP32')}; B2 strict-FP16 {metric('lidar_v2xvit', 'B2_TRT_STRICT_FP16')}; F3 {metric('lidar_v2xvit', 'B3_F3')}.",
        f"3. strict FP16: CoBEVT is catastrophic at fixed50 (P12/full={cobevt_p12.get('fixed50_mAP', 'not_yet_verified')}); V2XViT is not catastrophic (P12/full={v2_p12.get('fixed50_mAP', 'not_yet_verified')}).",
        "4. Every isolated FP16 role remains non-catastrophic for CoBEVT, but the joint full-attention FP16 contract collapses; V2XViT is broadly FP16-safe. The fixed50 matrices, not profile names, are the evidence.",
        "5. F3 protects LayerNorm and QK at FP32 and recovers Q/K projection outputs to FP32. Softmax, AV and residual are FP16 in F3, so this audit does not claim they require FP32 protection.",
        f"6. SQ1 (INT8 Q/K projection then DQ FP32 QK): CoBEVT {metric('lidar_cobevt', 'SQ1_QK')}; V2XViT {metric('lidar_v2xvit', 'SQ1_QK')}.",
        f"7. SQ2/SQ3: CoBEVT SQ2 {metric('lidar_cobevt', 'SQ2_QKV')}, SQ3 {metric('lidar_cobevt', 'SQ3_QKVO')}; V2XViT SQ2 {metric('lidar_v2xvit', 'SQ2_QKV')}, SQ3 {metric('lidar_v2xvit', 'SQ3_QKVO')}. These alpha=0.7 screening rows remain observational and are not admitted when they differ from the selected model alpha; failed builds have no accuracy or latency claim.",
        "8. Joint FFN INT8 compiler failures are profile/model specific: CoBEVT SQ4/SQ5 and V2XViT SQ4 failed, while V2XViT SQ5 built but was BORDERLINE at fixed500. Separately built FFN1/FFN2 diagnostics may be promoted only after their own fixed500 and formal-latency gates; they are not evidence that the failed joint graph is safe.",
        f"9. SmoothQuant alpha: CoBEVT={co_alpha}; V2XViT={v2_alpha}. Selection is per-model and uses seven normalized calibration50/200 numerical metrics.",
        f"10. calibration50/200 selected-scale stability delta: CoBEVT={co_stability}; V2XViT={v2_stability}. Hashes and per-alpha rows are retained under smoothquant_alpha_sm90.",
        f"11. BF16 records={len(bf16)}. CoBEVT H4 {metric('lidar_cobevt', 'H4_FULL_TRANSFORMER_BF16_PROTECTED_QK')}; V2XViT H4 {metric('lidar_v2xvit', 'H4_FULL_TRANSFORMER_BF16_PROTECTED_QK')}. H2 BF16A32 is rejected when its accumulator remains unknown.",
        f"12. FP8 records={len(fp8)}. CoBEVT selected H7 {metric('lidar_cobevt', 'H7_FFN_FP8')}; V2XViT selected H9 {metric('lidar_v2xvit', 'H9_ALL_LINEAR_FP8_PROTECTED_CORE')}. Projection FP8 plus DQ FP32 is not native FP8 QK.",
        "13. H800 engines and timing caches were rebuilt. Exact 4090 tactic comparison is not_yet_verified because the 4090 EngineInspector artifacts are absent locally; its report-level AP/latency is context only.",
        f"14. Complete fused-MHA evidence rows={fused}. A zero count means primitive/fused subgraphs were inspected without a complete fused-MHA contract.",
        f"15. Level-A independently searchable accumulator profiles={accumulator_searchable}. FP16-default is observational; F16A32/BF16A32 remain unsupported or conflicted without exact metadata.",
        "16. Native INT8 QK is not implemented. INT8/FP8 projection Q/DQ around FP32 QK is projection quantization, never a native low-precision attention core.",
        "17. Lowest independently admissible QK core is FP32A32. FP16-default has no Level-A accumulator proof.",
        "18. Lowest independently admissible AV core is FP32A32; F3 may execute AV FP16-default only as a fixed jointly verified phenotype.",
        f"19. Fixed500-safe profiles: CoBEVT={safe_names('lidar_cobevt')}; V2XViT={safe_names('lidar_v2xvit')}. Safe low-precision subsets: CoBEVT={safe_low_precision('lidar_cobevt')}; V2XViT={safe_low_precision('lidar_v2xvit')}.",
        f"20. Fully admitted profiles: CoBEVT={allowed_names('lidar_cobevt')}; V2XViT={allowed_names('lidar_v2xvit')}. Portable/model-specific separation is in the profile library.",
        "21. Only complete profiles admitted by every gate enter verified_joint; no untested Cartesian product is opened.",
        f"22. Small Greedy is permitted only if the non-baseline admitted lists above are nonempty. SQ1 p50 deltas: CoBEVT {latency_delta('lidar_cobevt', 'SQ1_QK')}; V2XViT {latency_delta('lidar_v2xvit', 'SQ1_QK')}.",
        "23. Full1789 is still required for an eventual final search winner. This audit uses fixed500 as the profile-library accuracy gate by design.",
        "24. Remaining limits: TensorRT compiler rejection of some joint FFN/QK profiles, opaque default accumulators, non-separable fused-role latency, no native INT8/FP8 QK or AV contract, and missing exact 4090 inspector counterparts.",
        "",
    ]
    return "\n".join(lines)


def generate(root: Path) -> dict[str, Any]:
    inventory = _consolidate_inventory(root)
    realized, accumulator = _consolidate_precision(root)
    profiles = {model: collect_profiles(root, model) for model in MODELS}
    _write_csv(root / "cobevt_quantization_matrix.csv", profiles["lidar_cobevt"])
    _write_csv(root / "v2xvit_quantization_matrix.csv", profiles["lidar_v2xvit"])
    cross = [row for rows in profiles.values() for row in rows]
    _write_csv(root / "transformer_quantization_cross_model.csv", cross)
    keyed = {
        "cobevt": profiles["lidar_cobevt"],
        "v2xvit": profiles["lidar_v2xvit"],
    }
    library = build_profile_library(keyed)
    search_space = build_search_space(library)
    _write_json(root / "transformer_precision_profile_library.json", library)
    _write_json(root / "transformer_quantization_search_space.json", search_space)
    conclusion = _conclusion(root, profiles, accumulator)
    (root / "root_conclusion.md").write_text(conclusion, encoding="utf-8")
    result = {
        "models": {model: len(rows) for model, rows in profiles.items()},
        "requested_realized_rows": len(realized),
        "accumulator_evidence_rows": len(accumulator),
        "inventory": inventory,
        "allowed": {
            model: [row["profile_id"] for row in rows if row["status"] == "allowed"]
            for model, rows in profiles.items()
        },
    }
    _write_json(root / "reports" / "report_generation.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    result = generate(Path(args.output_root).resolve())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
