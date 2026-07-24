"""Pure decision helpers for staged accuracy-collapse attribution."""

from __future__ import annotations

from typing import Any


def classify_root_cause(*, strict_map: float, structural_fp32_map: float, structural_fp16_map: float | None, joint_fresh_map: float | None, stale_calibration_detected: bool, ort_structural_fp32_map: float | None, trt_structural_fp32_map: float | None, catastrophic_drop: float = 0.15, backend_drop: float = 0.05) -> dict[str, Any]:
    """Classify only when a stage's prerequisite evidence is present."""
    structural_drop = float(strict_map) - float(structural_fp32_map)
    if structural_drop >= catastrophic_drop:
        return {"root_cause_class": "structural_collapse", "decisive_stage": "pytorch_physical_fp32", "structural_map_drop": structural_drop}
    if stale_calibration_detected and joint_fresh_map is not None:
        return {"root_cause_class": "stale_calibration_or_scale", "decisive_stage": "fresh_calibration"}
    if ort_structural_fp32_map is not None and structural_fp32_map - ort_structural_fp32_map >= backend_drop:
        return {"root_cause_class": "export_semantic_mismatch", "decisive_stage": "onnxruntime"}
    upstream_map = ort_structural_fp32_map if ort_structural_fp32_map is not None else structural_fp32_map
    if trt_structural_fp32_map is not None and upstream_map - trt_structural_fp32_map >= backend_drop:
        return {"root_cause_class": "tensorrt_numeric_mismatch", "decisive_stage": "tensorrt"}
    if structural_fp16_map is not None and joint_fresh_map is not None and structural_fp16_map - joint_fresh_map >= catastrophic_drop:
        return {"root_cause_class": "quantization_collapse", "decisive_stage": "fresh_mixed_ptq"}
    return {"root_cause_class": "mixed_or_unresolved", "decisive_stage": "insufficient_unique_evidence"}


def calibration_key_audit(fields: dict[str, Any], *, expected_structure_hash: str) -> dict[str, Any]:
    required = ("physical_structure_hash", "precision_map_hash", "calibration_manifest_hash", "checkpoint_hash")
    missing = [name for name in required if not fields.get(name)]
    structure_matches = fields.get("physical_structure_hash") == expected_structure_hash
    return {"complete": not missing and structure_matches, "missing_fields": missing, "structure_matches": structure_matches, "stale_cross_structure_risk": not structure_matches}
