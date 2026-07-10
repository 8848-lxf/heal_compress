from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
UNIAD = ROOT.parent
for path in (UNIAD, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class SnapshotToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(8, 12, 3, padding=1, bias=True)
        self.grouped = nn.Conv2d(256, 256, 3, padding=1, groups=32, bias=False)
        self.deblock = nn.ConvTranspose2d(16, 32, 2, stride=2, bias=False)
        self.bn = nn.BatchNorm2d(12)
        self.fc = nn.Linear(12, 4)


class LinearOnlyToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 2, bias=False)


def test_full_physical_snapshot_contains_unchanged_weighted_modules_once() -> None:
    from tools.latency_lut.physical_structure_v2 import build_physical_structure_snapshot_v2

    model = SnapshotToy().eval()
    snapshot = build_physical_structure_snapshot_v2(model, state_dict=model.state_dict())
    rows = snapshot["modules"]

    assert snapshot["snapshot_schema_version"] == "physical-structure-snapshot-v2"
    assert [row["canonical_order"] for row in rows] == list(range(len(rows)))
    assert len({row["canonical_module_name"] for row in rows}) == len(rows)
    assert {row["canonical_module_name"] for row in rows} == {"conv", "grouped", "deblock", "bn", "fc"}
    assert next(row for row in rows if row["canonical_module_name"] == "conv")["weight_shape"] == [12, 8, 3, 3]
    assert next(row for row in rows if row["canonical_module_name"] == "bn")["num_features"] == 12


def test_physical_hash_v2_is_deterministic_and_ignores_nonstructural_metadata() -> None:
    from tools.latency_lut.physical_structure_v2 import build_physical_structure_snapshot_v2, compute_physical_hash_v2

    snapshot = build_physical_structure_snapshot_v2(SnapshotToy().eval())
    shuffled = {"non_structural_note": "different", **snapshot, "modules": [dict(reversed(list(row.items()))) for row in snapshot["modules"]]}

    first = compute_physical_hash_v2(snapshot, legacy_structure_hash="legacy-a", legacy_shape_hash="legacy-b")
    second = compute_physical_hash_v2(shuffled, legacy_structure_hash="changed-only-in-output", legacy_shape_hash="also-output-only")
    assert first["hash_schema_version"] == "physical-structure-v2"
    assert first["structure_hash_v2"] == second["structure_hash_v2"]
    assert first["shape_hash_v2"] == second["shape_hash_v2"]

    changed = json.loads(json.dumps(snapshot))
    next(row for row in changed["modules"] if row["canonical_module_name"] == "conv")["weight_shape"][0] = 16
    assert compute_physical_hash_v2(changed)["shape_hash_v2"] != first["shape_hash_v2"]


def test_application_ledger_has_terminal_status_and_skip_reason() -> None:
    from tools.latency_lut.physical_structure_v2 import build_physical_application_ledger, validate_physical_application_ledger

    requests = {
        "requests": [
            {"request_id": "r0", "dependency_domain_id": "d0", "module_name": "conv", "affected_modules": ["conv"], "requested_before": {"out_channels": 12}, "requested_after": {"out_channels": 8}},
            {"request_id": "r1", "dependency_domain_id": "d1", "module_name": "fc", "affected_modules": ["fc"], "requested_before": {"out_features": 4}, "requested_after": {"out_features": 2}},
        ]
    }
    snapshot = {
        "modules": [
            {"canonical_module_name": "conv", "out_channels": 8},
            {"canonical_module_name": "fc", "out_features": 4},
        ]
    }
    ledger = build_physical_application_ledger(requests, snapshot, physical_plan={"requests": []}, legacy_migration=True)

    assert [row["status"] for row in ledger["entries"]] == ["applied", "skipped"]
    assert ledger["entries"][1]["skip_reason"] == "unknown_legacy_provenance"
    assert validate_physical_application_ledger(requests, ledger)["passed"] is True


def test_ledger_validation_rejects_request_without_terminal_entry() -> None:
    from tools.latency_lut.physical_structure_v2 import validate_physical_application_ledger

    result = validate_physical_application_ledger(
        {"requests": [{"request_id": "r0"}, {"request_id": "r1"}]},
        {"entries": [{"request_id": "r0", "status": "applied"}]},
    )
    assert result["passed"] is False
    assert result["missing_request_ids"] == ["r1"]


def test_weight_layout_interpretation_handles_grouped_and_convtranspose() -> None:
    from tools.latency_lut.physical_structure_v2 import interpret_weight_shape

    grouped = interpret_weight_shape("Conv2d", [512, 16, 3, 3], groups=32)
    deblock = interpret_weight_shape("ConvTranspose2d", [256, 128, 4, 4], groups=1)

    assert grouped["logical_in_channels"] == 512
    assert grouped["logical_out_channels"] == 512
    assert grouped["layout"] == "[C_out,C_in/groups,kH,kW]"
    assert deblock["logical_in_channels"] == 256
    assert deblock["logical_out_channels"] == 128
    assert deblock["layout"] == "[C_in,C_out/groups,kH,kW]"


def _write_qdq_conv(path: Path) -> None:
    weight = np.ones((12, 8, 3, 3), dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("QuantizeLinear", ["weight", "scale", "zero"], ["weight_q"], name="weight_q_node"),
            helper.make_node("DequantizeLinear", ["weight_q", "scale", "zero"], ["weight_dq"], name="weight_dq_node"),
            helper.make_node("Conv", ["input", "weight_dq"], ["output"], name="canonical_conv", kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
        ],
        "qdq",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 8, 4, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 12, 4, 4])],
        [
            numpy_helper.from_array(weight, "weight"),
            numpy_helper.from_array(np.asarray([0.1], dtype=np.float32), "scale"),
            numpy_helper.from_array(np.asarray([0], dtype=np.int8), "zero"),
        ],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)]), str(path))


def test_qdq_weight_trace_reaches_original_initializer(tmp_path: Path) -> None:
    from tools.latency_lut.physical_structure_v2 import trace_onnx_weight_to_initializer

    path = tmp_path / "qdq.onnx"
    _write_qdq_conv(path)
    trace = trace_onnx_weight_to_initializer(path, "canonical_conv")

    assert trace["consumed_weight_tensor"] == "weight_dq"
    assert trace["root_initializer"] == "weight"
    assert trace["root_initializer_shape"] == [12, 8, 3, 3]
    assert [row["op_type"] for row in trace["trace_chain"]] == ["DequantizeLinear", "QuantizeLinear"]


def test_deployment_profile_hash_changes_with_precision_not_metadata() -> None:
    from tools.latency_lut.physical_structure_v2 import compute_deployment_profile_hash_v2

    base = compute_deployment_profile_hash_v2(
        shape_hash_v2="shape",
        profile={"layer_precision_assignment": {"conv": "int8"}, "random_seed": 1},
        requested_int8_modules=["conv"],
        qdq_policy_version="qdq-v1",
        canonical_mapping_version="canonical-v2",
        trt_build_policy_version="trt-v12",
    )
    metadata_changed = compute_deployment_profile_hash_v2(
        shape_hash_v2="shape",
        profile={"layer_precision_assignment": {"conv": "int8"}, "random_seed": 999},
        requested_int8_modules=["conv"],
        qdq_policy_version="qdq-v1",
        canonical_mapping_version="canonical-v2",
        trt_build_policy_version="trt-v12",
    )
    precision_changed = compute_deployment_profile_hash_v2(
        shape_hash_v2="shape",
        profile={"layer_precision_assignment": {"conv": "fp16"}},
        requested_int8_modules=[],
        qdq_policy_version="qdq-v1",
        canonical_mapping_version="canonical-v2",
        trt_build_policy_version="trt-v12",
    )
    assert base == metadata_changed
    assert base != precision_changed


def test_physical_preflight_accepts_matmul_export_weight_transpose(tmp_path: Path) -> None:
    from tools.latency_lut.physical_structure_v2 import (
        build_physical_structure_snapshot_v2,
        compute_physical_hash_v2,
        run_physical_structure_preflight,
    )

    subnet_dir = tmp_path / "subnet_000"
    profile_dir = subnet_dir / "profile_000"
    (subnet_dir / "onnx").mkdir(parents=True)
    (profile_dir / "onnx").mkdir(parents=True)
    model = LinearOnlyToy().eval()
    torch.save({"model_object": model}, subnet_dir / "pruned_model_object.pth")
    torch.save({"state_dict": model.state_dict()}, subnet_dir / "pruned_state_dict_with_manifest.pth")
    snapshot = build_physical_structure_snapshot_v2(model)
    (subnet_dir / "physical_structure_snapshot_v2.json").write_text(json.dumps(snapshot), encoding="utf-8")
    (subnet_dir / "physical_hash_v2.json").write_text(json.dumps(compute_physical_hash_v2(snapshot)), encoding="utf-8")

    weight = numpy_helper.from_array(np.ones((3, 2), dtype=np.float32), "fc_weight_exported")
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["input", "fc_weight_exported"], ["output"], name="__canonical__fc__MatMul__call00000")],
        "linear",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])],
        [weight],
    )
    onnx_model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)])
    onnx.save(onnx_model, str(subnet_dir / "onnx/model_signal_maxk.onnx"))
    onnx.save(onnx_model, str(profile_dir / "onnx/model_mixed_qdq.onnx"))
    mapping = {
        "entries": [
            {
                "canonical_module_name": "fc",
                "onnx_node_name_unique": "__canonical__fc__MatMul__call00000",
                "onnx_node_name_original": "/fc/MatMul",
                "onnx_weight_initializer": "fc_weight_exported",
            }
        ]
    }
    (profile_dir / "canonical_precision_mapping.json").write_text(json.dumps(mapping), encoding="utf-8")

    report = run_physical_structure_preflight(subnet_dir=subnet_dir, profile_dir=profile_dir)

    assert report["preflight_passed"] is True
    assert report["checks"][0]["base_onnx_weight_shape"] == [3, 2]
    assert report["checks"][0]["weight_layout"]["layout"] == "matrix"


def test_successful_label_v2_metadata_merge_preserves_measurements_and_legacy_hashes() -> None:
    from tools.latency_lut.migrate_v12_physical_structure_v2 import merge_successful_label_v2_metadata

    original = {
        "label_available": True,
        "structure_hash": "legacy-structure",
        "shape_hash": "legacy-shape",
        "evaluated_frames": 300,
        "mAP": 0.314,
        "forward_latency_p50_ms": 2.7,
    }
    merged = merge_successful_label_v2_metadata(
        original,
        physical_hash={
            "hash_schema_version": "physical-structure-v2",
            "structure_hash_v2": "physical-structure",
            "shape_hash_v2": "physical-shape",
        },
        deployment_profile_hash_v2="deployment-profile",
        valid=True,
    )

    assert merged["structure_hash"] == "legacy-structure"
    assert merged["shape_hash"] == "legacy-shape"
    assert merged["mAP"] == 0.314
    assert merged["forward_latency_p50_ms"] == 2.7
    assert merged["evaluated_frames"] == 300
    assert merged["structure_hash_v2"] == "physical-structure"
    assert merged["physical_metadata_v2_valid"] is True
