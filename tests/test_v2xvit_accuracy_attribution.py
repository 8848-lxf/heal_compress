from search.model_family.accuracy_attribution import calibration_key_audit, classify_root_cause


def test_structural_collapse_precedes_missing_later_backend_evidence():
    result = classify_root_cause(strict_map=0.5742, structural_fp32_map=0.2432, structural_fp16_map=None, joint_fresh_map=None, stale_calibration_detected=False, ort_structural_fp32_map=None, trt_structural_fp32_map=None)
    assert result["root_cause_class"] == "structural_collapse"
    assert result["decisive_stage"] == "pytorch_physical_fp32"


def test_export_mismatch_requires_normal_structural_control_then_ort_drop():
    result = classify_root_cause(strict_map=0.57, structural_fp32_map=0.56, structural_fp16_map=0.55, joint_fresh_map=0.54, stale_calibration_detected=False, ort_structural_fp32_map=0.20, trt_structural_fp32_map=0.19)
    assert result["root_cause_class"] == "export_semantic_mismatch"


def test_missing_precision_hash_is_incomplete_not_cross_structure_stale():
    result = calibration_key_audit({"physical_structure_hash": "physical-a", "precision_map_hash": None, "calibration_manifest_hash": "manifest-a", "checkpoint_hash": "checkpoint-a"}, expected_structure_hash="physical-a")
    assert result["complete"] is False
    assert result["missing_fields"] == ["precision_map_hash"]
    assert result["stale_cross_structure_risk"] is False


def test_calibration_key_rejects_cross_structure_reuse():
    result = calibration_key_audit({"physical_structure_hash": "physical-old", "precision_map_hash": "precision-a", "calibration_manifest_hash": "manifest-a", "checkpoint_hash": "checkpoint-a"}, expected_structure_hash="physical-new")
    assert result["complete"] is False
    assert result["stale_cross_structure_risk"] is True
