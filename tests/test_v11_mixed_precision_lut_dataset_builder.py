from __future__ import annotations

import argparse
import json
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
UNIAD = ROOT.parent
for path in (UNIAD, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class ResidualConcatToy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 8, 1)
        self.branch_a = nn.Conv2d(8, 8, 3, padding=1)
        self.branch_b = nn.Conv2d(8, 8, 3, padding=1)
        self.fuse = nn.Conv2d(16, 8, 1)
        self.head = nn.Conv2d(8, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        a = self.branch_a(x)
        b = self.branch_b(x)
        merged = a + b
        cat = torch.cat([merged, x], dim=1)
        return self.head(self.fuse(cat))


def test_precision_coupling_tracer_records_add_and_concat_boundaries_without_forcing_same_precision() -> None:
    from heal_compress.tracer.dependency_tracer import build_dependency_graph
    from heal_compress.tracer.precision_coupling_tracer import build_precision_coupling_groups

    model = ResidualConcatToy().eval()
    sample = torch.randn(1, 3, 4, 4)
    graph = build_dependency_graph(model, sample)

    groups = build_precision_coupling_groups(model, graph, sample)

    reasons = {group.reason for group in groups}
    assert "residual" in reasons or "elementwise" in reasons
    assert "concat" in reasons
    assert all(not group.force_same_precision for group in groups if group.reason in {"residual", "elementwise", "concat"})


def test_mixed_precision_profile_respects_groups_and_records_fallback() -> None:
    from heal_compress.tracer.precision_coupling_tracer import PrecisionGroup
    from tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder import sample_mixed_precision_profile

    groups = [
        PrecisionGroup("pg0", ["branch_a", "branch_b"], "residual", ["fp32", "fp16", "int8"], "fp16", True),
        PrecisionGroup("pg1", ["scatter"], "unsupported_int8", ["fp32", "fp16"], "fp16", True),
        PrecisionGroup("pg2", ["head"], "head_constraint", ["fp32", "fp16"], "fp16", True),
    ]

    profile = sample_mixed_precision_profile(
        subnet_id="subnet_000",
        structure_hash="abc",
        groups=groups,
        random_seed=1,
        precision_modes=["int8"],
    )

    assert profile["precision_group_assignments"]["pg0"]["final_precision"] == "int8"
    assert profile["layer_precision_assignment"]["branch_a"] == profile["layer_precision_assignment"]["branch_b"]
    assert profile["precision_group_assignments"]["pg1"]["requested_precision"] == "int8"
    assert profile["precision_group_assignments"]["pg1"]["final_precision"] == "fp16"
    assert profile["fallback_layer_count"] >= 1
    assert profile["int8_layer_count"] == 2


def test_dense_conv_shape_gate_does_not_forbid_non_8_aligned_int8_channels() -> None:
    from tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder import module_int8_shape_eligibility

    for channels in (4, 12, 20, 24, 28):
        result = module_int8_shape_eligibility(
            {
                "module_name": f"dense_{channels}",
                "groups": 1,
                "after_in": channels,
                "after_out": channels,
            }
        )
        assert result["int8_shape_supported"] is True
        assert result["reason"] == "ordinary_conv_no_extra_alignment_required"


def test_grouped_conv_shape_gate_uses_safe_per_group_set() -> None:
    from tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder import module_int8_shape_eligibility

    for per_group in (4, 8, 16, 32):
        result = module_int8_shape_eligibility(
            {
                "module_name": f"grouped_{per_group}",
                "groups": 32,
                "after_in": 32 * per_group,
                "after_out": 32 * per_group,
            }
        )
        assert result["int8_shape_supported"] is True

    for per_group in (12, 18, 20, 24, 28):
        result = module_int8_shape_eligibility(
            {
                "module_name": f"grouped_{per_group}",
                "groups": 32,
                "after_in": 32 * per_group,
                "after_out": 32 * per_group,
            }
        )
        assert result["int8_shape_supported"] is False
        assert result["reason"] == "grouped_conv_int8_per_group_shape_not_supported"


def test_profile_shape_gate_fallbacks_unsupported_grouped_int8_but_keeps_dense_int8() -> None:
    from tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder import apply_deployment_aware_precision_legality

    profile = {
        "precision_group_assignments": {
            "pg_dense": {
                "member_modules": ["dense_c12"],
                "reason": "user_constraint",
                "allowed_precisions": ["fp32", "fp16", "int8"],
                "requested_precision": "int8",
                "final_precision": "int8",
            },
            "pg_grouped": {
                "member_modules": ["grouped_c24"],
                "reason": "user_constraint",
                "allowed_precisions": ["fp32", "fp16", "int8"],
                "requested_precision": "int8",
                "final_precision": "int8",
            },
        },
        "layer_precision_assignment": {"dense_c12": "int8", "grouped_c24": "int8"},
    }
    shapes = {
        "dense_c12": {"groups": 1, "after_in": 12, "after_out": 12},
        "grouped_c24": {"groups": 32, "after_in": 32 * 24, "after_out": 32 * 24},
    }

    gated = apply_deployment_aware_precision_legality(profile, shapes)

    assert gated["precision_group_assignments"]["pg_dense"]["final_precision"] == "int8"
    assert gated["precision_group_assignments"]["pg_grouped"]["final_precision"] == "fp16"
    assert gated["precision_group_assignments"]["pg_grouped"]["fallback_reason"] == "grouped_conv_int8_per_group_shape_not_supported"
    assert gated["layer_precision_assignment"]["grouped_c24"] == "fp16"


def test_qdq_insert_report_schema() -> None:
    from tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder import make_qdq_insert_report

    report = make_qdq_insert_report(
        input_onnx="model.onnx",
        output_onnx="model_mixed_qdq.onnx",
        profile={"precision_group_assignments": {"pg": {"final_precision": "int8"}}},
        calibration_frame_ids=[1, 2],
        scale_table={"pg": {"scale": 0.1}},
        success=True,
    )

    assert report["success"] is True
    assert report["uses_mixed_precision_qdq"] is True
    assert report["calibration_frame_count"] == 2
    assert report["int8_precision_group_count"] == 1


def _write_tiny_conv_onnx(path: Path) -> None:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    graph = helper.make_graph(
        [
            helper.make_node(
                "Conv",
                ["input", "stem.weight", "stem.bias"],
                ["stem_out"],
                name="/stem/Conv",
            ),
            helper.make_node(
                "Conv",
                ["stem_out", "head.weight", "head.bias"],
                ["output"],
                name="/head/Conv",
            ),
        ],
        "tiny_qdq",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 4, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 4, 4])],
        [
            numpy_helper.from_array(np.ones((4, 3, 1, 1), dtype=np.float32), "stem.weight"),
            numpy_helper.from_array(np.zeros((4,), dtype=np.float32), "stem.bias"),
            numpy_helper.from_array(np.ones((2, 4, 1, 1), dtype=np.float32), "head.weight"),
            numpy_helper.from_array(np.zeros((2,), dtype=np.float32), "head.bias"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))


def _write_signal_maxk_like_onnx(path: Path) -> None:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["voxel_features", "stem.weight"], ["stem_out"], name="/encoder_m1/stem/MatMul"),
            helper.make_node("MatMul", ["stem_out", "cls_head.weight"], ["cls_preds"], name="/cls_head/MatMul"),
            helper.make_node("MatMul", ["stem_out", "reg_head.weight"], ["reg_preds"], name="/reg_head/MatMul"),
            helper.make_node("MatMul", ["stem_out", "dir_head.weight"], ["dir_preds"], name="/dir_head/MatMul"),
        ],
        "signal_maxk_like",
        [
            helper.make_tensor_value_info("voxel_features", TensorProto.FLOAT, [29696, 128]),
            helper.make_tensor_value_info("voxel_coords", TensorProto.INT32, [29696, 4]),
            helper.make_tensor_value_info("voxel_num_points", TensorProto.INT32, [29696]),
            helper.make_tensor_value_info("record_len", TensorProto.INT32, [1]),
            helper.make_tensor_value_info("pairwise_t_matrix", TensorProto.FLOAT, [1, 2, 2, 4, 4]),
            helper.make_tensor_value_info("valid_voxel_mask", TensorProto.FLOAT, [29696]),
        ],
        [
            helper.make_tensor_value_info("cls_preds", TensorProto.FLOAT, [29696, 2]),
            helper.make_tensor_value_info("reg_preds", TensorProto.FLOAT, [29696, 14]),
            helper.make_tensor_value_info("dir_preds", TensorProto.FLOAT, [29696, 4]),
        ],
        [
            numpy_helper.from_array(np.ones((128, 64), dtype=np.float32), "stem.weight"),
            numpy_helper.from_array(np.ones((64, 2), dtype=np.float32), "cls_head.weight"),
            numpy_helper.from_array(np.ones((64, 14), dtype=np.float32), "reg_head.weight"),
            numpy_helper.from_array(np.ones((64, 4), dtype=np.float32), "dir_head.weight"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))


def _write_signal_maxk_like_onnx_with_trt_plugin(path: Path) -> None:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["voxel_features", "stem.weight"], ["stem_out"], name="/encoder_m1/stem/MatMul"),
            helper.make_node(
                "PointPillarScatterTRT",
                ["stem_out", "voxel_coords", "valid_voxel_mask"],
                ["scatter_out"],
                name="/PointPillarScatterTRT",
            ),
            helper.make_node("MatMul", ["scatter_out", "cls_head.weight"], ["cls_preds"], name="/cls_head/MatMul"),
            helper.make_node("MatMul", ["scatter_out", "reg_head.weight"], ["reg_preds"], name="/reg_head/MatMul"),
            helper.make_node("MatMul", ["scatter_out", "dir_head.weight"], ["dir_preds"], name="/dir_head/MatMul"),
        ],
        "signal_maxk_like_with_trt_plugin",
        [
            helper.make_tensor_value_info("voxel_features", TensorProto.FLOAT, [29696, 128]),
            helper.make_tensor_value_info("voxel_coords", TensorProto.INT32, [29696, 4]),
            helper.make_tensor_value_info("voxel_num_points", TensorProto.INT32, [29696]),
            helper.make_tensor_value_info("pairwise_t_matrix", TensorProto.FLOAT, [1, 2, 2, 4, 4]),
            helper.make_tensor_value_info("valid_voxel_mask", TensorProto.FLOAT, [29696]),
        ],
        [
            helper.make_tensor_value_info("cls_preds", TensorProto.FLOAT, [29696, 2]),
            helper.make_tensor_value_info("reg_preds", TensorProto.FLOAT, [29696, 14]),
            helper.make_tensor_value_info("dir_preds", TensorProto.FLOAT, [29696, 4]),
        ],
        [
            numpy_helper.from_array(np.ones((128, 64), dtype=np.float32), "stem.weight"),
            numpy_helper.from_array(np.ones((64, 2), dtype=np.float32), "cls_head.weight"),
            numpy_helper.from_array(np.ones((64, 14), dtype=np.float32), "reg_head.weight"),
            numpy_helper.from_array(np.ones((64, 4), dtype=np.float32), "dir_head.weight"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)])
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))


def _qdq_test_ctx(tmp_path: Path, profile: dict) -> dict:
    import argparse

    subnet_dir = tmp_path / "subnets" / "subnet_000"
    source_onnx = subnet_dir / "onnx" / "model_signal_maxk.onnx"
    _write_signal_maxk_like_onnx(source_onnx)
    (subnet_dir / "onnx" / "real_onnx_export_report.json").write_text(
        json.dumps(
            {
                "export_success": True,
                "onnx_path": str(source_onnx),
                "input_names": [
                    "voxel_features",
                    "voxel_coords",
                    "voxel_num_points",
                    "record_len",
                    "pairwise_t_matrix",
                    "valid_voxel_mask",
                ],
                "output_names": ["cls_preds", "reg_preds", "dir_preds"],
                "source_exporter": "test_signal_maxk_like",
            }
        ),
        encoding="utf-8",
    )
    profile_dir = subnet_dir / "profile_001"
    profile_dir.mkdir(parents=True, exist_ok=True)
    return {
        "args": argparse.Namespace(calib_train_frames=2, fixed_k=29696),
        "subnet_dir": subnet_dir,
        "profile_dir": profile_dir,
        "profile": profile,
    }


def test_onnx_qdq_stage_allows_signal_maxk_trt_plugin_checker_skip(tmp_path: Path) -> None:
    import json
    import tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder as builder

    subnet_dir = tmp_path / "subnets/subnet_000"
    real_onnx = subnet_dir / "onnx/model_signal_maxk.onnx"
    _write_signal_maxk_like_onnx_with_trt_plugin(real_onnx)
    builder.write_json(
        subnet_dir / "onnx/real_onnx_export_report.json",
        {
            "export_success": True,
            "onnx_path": str(real_onnx),
            "input_names": ["voxel_features", "voxel_coords", "voxel_num_points", "pairwise_t_matrix", "valid_voxel_mask"],
            "output_names": ["cls_preds", "reg_preds", "dir_preds"],
        },
    )
    profile = {
        "precision_group_assignments": {
            "pg_encoder": {"member_modules": ["encoder_m1"], "final_precision": "int8", "allowed_precisions": ["fp32", "fp16", "int8"]},
        },
        "layer_precision_assignment": {"encoder_m1": "int8"},
        "fallback_layers": [],
    }

    report = builder.run_onnx_qdq_stage(
        {
            "args": argparse.Namespace(calib_train_frames=2),
            "subnet_dir": subnet_dir,
            "profile_dir": subnet_dir / "profile_000",
            "profile": profile,
        }
    )
    qdq_report = json.loads((subnet_dir / "profile_000/qdq_insert_report.json").read_text(encoding="utf-8"))
    check_report = json.loads((subnet_dir / "profile_000/onnx_check_report.json").read_text(encoding="utf-8"))

    assert report["success"] is True
    assert qdq_report["inserted_qdq_nodes"]
    assert check_report["checker"] == "onnx.checker_skipped_for_tensorrt_custom_plugin"
    assert check_report["custom_plugin_op_types"] == ["PointPillarScatterTRT"]


def test_real_signal_maxk_onnx_export_has_heal_bindings(tmp_path: Path, monkeypatch) -> None:
    import quant_deploy.pruned_signal_maxk_exporter as exporter

    def fake_formal_export(**kwargs):
        _write_signal_maxk_like_onnx(kwargs["output_onnx_path"])
        return {"success": True, "source_exporter": "fake_signal_maxk"}

    monkeypatch.setattr(exporter, "_run_formal_signal_maxk_export", fake_formal_export)
    report = exporter.export_pruned_lidar_pyramid_signal_maxk_onnx(
        pruned_model_path=tmp_path / "real_heal_pruned.pth",
        pruning_manifest_path=tmp_path / "manifest.json",
        output_onnx_path=tmp_path / "model_signal_maxk.onnx",
        model_config=tmp_path / "config.yaml",
        checkpoint=tmp_path / "checkpoint.pth",
        heal_root=tmp_path,
        fixed_k=29696,
        calibration_or_dummy_batch_source="train",
        dynamic_axes=True,
    )

    assert report["export_success"] is True
    assert {"voxel_features", "voxel_coords", "voxel_num_points", "record_len", "pairwise_t_matrix"} <= set(report["input_names"])
    assert report["input_names"] != ["input.1"]
    assert {"cls_preds", "reg_preds", "dir_preds"} <= set(report["output_names"])


def test_v11_builder_rejects_toy_onnx_for_real_eval(tmp_path: Path, monkeypatch) -> None:
    import tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder as builder

    _make_sampled_subnets(tmp_path, subnets=1)
    stale_qdq = tmp_path / "subnets/subnet_000/profile_000/onnx/model_mixed_qdq.onnx"
    stale_qdq.parent.mkdir(parents=True, exist_ok=True)
    stale_qdq.write_bytes(b"stale toy qdq")
    monkeypatch.setattr(builder, "run_engine_build_stage", lambda ctx: (_ for _ in ()).throw(AssertionError("toy ONNX must not reach build")))

    rc = builder.main(
        [
            "--mode",
            "expand-profiles-per-engine",
            "--source-dir",
            str(tmp_path),
            "--max-subnets",
            "1",
            "--precision-profiles-per-subnet",
            "1",
            "--eval-engines",
            "true",
        ]
    )

    assert rc == 0
    row = next(csv.DictReader((tmp_path / "mixed_precision_profile_index.csv").open()))
    assert row["status"] == "real_onnx_export_failed"
    assert row["eval_success"] == "False"
    report = json.loads((tmp_path / "subnets/subnet_000/onnx/real_onnx_export_report.json").read_text(encoding="utf-8"))
    assert report["export_success"] is False
    assert "ToyMixedPrecisionSubnet" in report["failure_reason"] or "not a real HEAL pruned" in report["failure_reason"]
    assert not stale_qdq.exists()


def test_subnet_artifact_audit_marks_toy_dataset_unusable(tmp_path: Path) -> None:
    import csv
    import tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder as builder

    _make_sampled_subnets(tmp_path, subnets=2)

    summary = builder.audit_subnet_artifacts(tmp_path)
    rows = list(csv.DictReader((tmp_path / "subnet_artifact_audit.csv").open()))
    report = json.loads((tmp_path / "subnet_artifact_audit_summary.json").read_text(encoding="utf-8"))

    assert summary["toy_count"] == 2
    assert summary["real_heal_pruned_count"] == 0
    assert summary["dataset_usable_for_real_lut"] is False
    assert len(rows) == 2
    assert {row["artifact_type"] for row in rows} == {"toy"}
    assert all(row["can_export_signal_maxk_onnx"] == "false" for row in rows)
    assert report["dataset_usable_for_real_lut"] is False


def test_sample_real_heal_mode_uses_real_subnet_generator(tmp_path: Path, monkeypatch) -> None:
    import tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder as builder

    called: dict[str, object] = {}

    def fake_real_generator(args, output_dir):
        called["mode"] = args.mode
        called["output_dir"] = str(output_dir)
        builder.write_json(output_dir / "dataset_manifest.json", {"successful_subnet_count": 1, "real_heal_pruned_subnets": True})
        return [{"subnet_id": "subnet_000", "structure_hash": "realhash", "shape_invariant_passed": True}]

    monkeypatch.setattr(builder, "generate_real_heal_subnets", fake_real_generator)
    monkeypatch.setattr(builder, "_write_subnet_artifacts", lambda *a, **k: (_ for _ in ()).throw(AssertionError("toy writer must not run")))

    rc = builder.main(
        [
            "--mode",
            "sample-real-heal",
            "--num-subnets",
            "1",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert rc == 0
    assert called == {"mode": "sample-real-heal", "output_dir": str(tmp_path)}


def test_qdq_insert_on_real_signal_maxk_onnx(tmp_path: Path) -> None:
    import onnx
    import tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder as builder

    subnet_dir = tmp_path / "subnets/subnet_000"
    real_onnx = subnet_dir / "onnx/model_signal_maxk.onnx"
    _write_signal_maxk_like_onnx(real_onnx)
    builder.write_json(
        subnet_dir / "onnx/real_onnx_export_report.json",
        {"export_success": True, "onnx_path": str(real_onnx), "input_names": ["voxel_features", "voxel_coords", "voxel_num_points", "record_len", "pairwise_t_matrix"], "output_names": ["cls_preds", "reg_preds", "dir_preds"]},
    )
    profile = {
        "precision_group_assignments": {
            "pg_encoder": {"member_modules": ["encoder_m1"], "final_precision": "int8", "allowed_precisions": ["fp32", "fp16", "int8"]},
            "pg_head": {"member_modules": ["cls_head"], "final_precision": "fp16", "allowed_precisions": ["fp32", "fp16"]},
        },
        "layer_precision_assignment": {"encoder_m1": "int8", "cls_head": "fp16"},
        "fallback_layers": [],
    }
    ctx = {"args": argparse.Namespace(calib_train_frames=2), "subnet_dir": subnet_dir, "profile_dir": subnet_dir / "profile_000", "profile": profile}

    report = builder.run_onnx_qdq_stage(ctx)
    qdq_report = json.loads((subnet_dir / "profile_000/qdq_insert_report.json").read_text(encoding="utf-8"))
    model = onnx.load(str(subnet_dir / "profile_000/onnx/model_mixed_qdq.onnx"))

    assert report["success"] is True
    assert qdq_report["success"] is True
    assert qdq_report["inserted_qdq_nodes"]
    assert any(node.op_type == "QuantizeLinear" for node in model.graph.node)
    assert any(node.op_type == "DequantizeLinear" for node in model.graph.node)


def test_precision_constraints_use_real_onnx_node_names(tmp_path: Path) -> None:
    import tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder as builder

    onnx_path = tmp_path / "model_signal_maxk.onnx"
    _write_signal_maxk_like_onnx(onnx_path)
    profile = {
        "precision_group_assignments": {
            "pg_encoder": {"member_modules": ["encoder_m1"], "final_precision": "int8"},
            "pg_cls": {"member_modules": ["cls_head"], "final_precision": "fp16"},
        },
        "layer_precision_assignment": {"encoder_m1": "int8", "cls_head": "fp16"},
    }

    specs = builder._onnx_precision_constraint_specs(onnx_path, profile)

    assert "/encoder_m1/stem/MatMul:int8" in specs
    assert "/cls_head/MatMul:fp16" in specs
    assert "encoder_m1:int8" not in specs


def test_no_synthetic_eval_success() -> None:
    from tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder import full_engine_training_sample

    eval_row = {"eval_success": True, "synthetic_used": True, "validation_dataloader_used": False}
    sample = full_engine_training_sample("subnet_000", "profile_000", "hash", {}, [], eval_row)

    assert sample["label_available"] is False


def test_onnx_qdq_stage_inserts_real_qdq_nodes_for_int8_profile(tmp_path: Path) -> None:
    import onnx
    import tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder as builder

    profile = {
        "precision_group_assignments": {
            "pg_stem": {
                "member_modules": ["encoder_m1"],
                "final_precision": "int8",
                "allowed_precisions": ["fp32", "fp16", "int8"],
            },
            "pg_head": {
                "member_modules": ["cls_head"],
                "final_precision": "fp16",
                "allowed_precisions": ["fp32", "fp16"],
            },
        },
        "layer_precision_assignment": {"encoder_m1": "int8", "cls_head": "fp16"},
        "fallback_layers": [],
    }

    report = builder.run_onnx_qdq_stage(_qdq_test_ctx(tmp_path, profile))
    qdq_report = json.loads((tmp_path / "subnets/subnet_000/profile_001/qdq_insert_report.json").read_text(encoding="utf-8"))
    model = onnx.load(str(tmp_path / "subnets/subnet_000/profile_001/onnx/model_mixed_qdq.onnx"))
    qdq_nodes = [node for node in model.graph.node if node.op_type in {"QuantizeLinear", "DequantizeLinear"}]

    assert report["success"] is True
    assert qdq_report["success"] is True
    assert qdq_report["inserted_qdq_nodes"]
    assert len(qdq_report["inserted_qdq_nodes"]) == len(qdq_nodes)
    assert {row["module_name"] for row in qdq_report["inserted_qdq_nodes"]} == {"encoder_m1"}
    assert qdq_report["skipped_non_int8_layers"] == [{"module_name": "cls_head", "final_precision": "fp16"}]


def test_qdq_insert_matches_signal_maxk_exported_backbone_aliases(tmp_path: Path) -> None:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    import tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder as builder

    input_onnx = tmp_path / "model_signal_maxk.onnx"
    graph = helper.make_graph(
        [
            helper.make_node(
                "Conv",
                ["voxel_features", "layer0.0.conv1.weight"],
                ["layer0_out"],
                name="/layer0/layer0.0/conv1/Conv",
            )
        ],
        "signal_maxk_exported_backbone_alias",
        [helper.make_tensor_value_info("voxel_features", TensorProto.FLOAT, [1, 8, 4, 4])],
        [helper.make_tensor_value_info("layer0_out", TensorProto.FLOAT, [1, 8, 4, 4])],
        [numpy_helper.from_array(np.ones((8, 8, 1, 1), dtype=np.float32), "layer0.0.conv1.weight")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    onnx.checker.check_model(model)
    onnx.save(model, str(input_onnx))
    output_onnx = tmp_path / "model_mixed_qdq.onnx"
    profile = {
        "precision_group_assignments": {
            "pg_backbone": {
                "member_modules": ["backbone_m1.resnet.layer0.0.conv1"],
                "requested_precision": "int8",
                "final_precision": "int8",
                "allowed_precisions": ["fp32", "fp16", "int8"],
            }
        },
        "layer_precision_assignment": {"backbone_m1.resnet.layer0.0.conv1": "int8"},
        "fallback_layers": [],
    }

    qdq = builder.insert_mixed_precision_qdq(
        input_onnx=input_onnx,
        output_onnx=output_onnx,
        profile=profile,
        scale_table={"pg_backbone": {"scale": 0.1}},
    )

    assert qdq["inserted_qdq_nodes"]
    assert qdq["unmatched_int8_precision_groups"] == []
    assert {row["precision_group_id"] for row in qdq["inserted_qdq_nodes"]} == {"pg_backbone"}


def test_qdq_insert_does_not_overmatch_signal_maxk_block_prefix(tmp_path: Path) -> None:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    import tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder as builder

    input_onnx = tmp_path / "model_signal_maxk.onnx"
    graph = helper.make_graph(
        [
            helper.make_node(
                "Conv",
                ["voxel_features", "layer0.1.conv3.weight"],
                ["layer0_out"],
                name="/layer0/layer0.1/conv3/Conv",
            )
        ],
        "signal_maxk_block_prefix",
        [helper.make_tensor_value_info("voxel_features", TensorProto.FLOAT, [1, 8, 4, 4])],
        [helper.make_tensor_value_info("layer0_out", TensorProto.FLOAT, [1, 8, 4, 4])],
        [numpy_helper.from_array(np.ones((8, 8, 1, 1), dtype=np.float32), "layer0.1.conv3.weight")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    onnx.checker.check_model(model)
    onnx.save(model, str(input_onnx))
    profile = {
        "precision_group_assignments": {
            "pg_conv1": {
                "member_modules": ["backbone_m1.resnet.layer0.1.conv1"],
                "requested_precision": "int8",
                "final_precision": "int8",
                "allowed_precisions": ["fp32", "fp16", "int8"],
            }
        },
        "layer_precision_assignment": {"backbone_m1.resnet.layer0.1.conv1": "int8"},
        "fallback_layers": [],
    }

    qdq = builder.insert_mixed_precision_qdq(
        input_onnx=input_onnx,
        output_onnx=tmp_path / "model_mixed_qdq.onnx",
        profile=profile,
        scale_table={"pg_conv1": {"scale": 0.1}},
    )

    assert qdq["inserted_qdq_nodes"] == []
    assert qdq["unmatched_int8_precision_groups"] == ["pg_conv1"]


def test_qdq_insert_does_not_match_pyramid_layer0_conv1_to_unsuffixed_backbone_node(tmp_path: Path) -> None:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    import tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder as builder

    input_onnx = tmp_path / "model_signal_maxk.onnx"
    graph = helper.make_graph(
        [
            helper.make_node(
                "Conv",
                ["voxel_features", "layer0.2.conv1.weight"],
                ["layer0_out"],
                name="/layer0/layer0.2/conv1/Conv",
            )
        ],
        "signal_maxk_unsuffixed_backbone_node",
        [helper.make_tensor_value_info("voxel_features", TensorProto.FLOAT, [1, 8, 4, 4])],
        [helper.make_tensor_value_info("layer0_out", TensorProto.FLOAT, [1, 8, 4, 4])],
        [numpy_helper.from_array(np.ones((8, 8, 1, 1), dtype=np.float32), "layer0.2.conv1.weight")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    onnx.checker.check_model(model)
    onnx.save(model, str(input_onnx))
    profile = {
        "precision_group_assignments": {
            "pg_pyramid": {
                "member_modules": ["pyramid_backbone.resnet.layer0.2.conv1"],
                "requested_precision": "int8",
                "final_precision": "int8",
                "allowed_precisions": ["fp32", "fp16", "int8"],
            }
        },
        "layer_precision_assignment": {"pyramid_backbone.resnet.layer0.2.conv1": "int8"},
        "fallback_layers": [],
    }

    qdq = builder.insert_mixed_precision_qdq(
        input_onnx=input_onnx,
        output_onnx=tmp_path / "model_mixed_qdq.onnx",
        profile=profile,
        scale_table={"pg_pyramid": {"scale": 0.1}},
    )

    assert qdq["inserted_qdq_nodes"] == []
    assert qdq["unmatched_int8_precision_groups"] == ["pg_pyramid"]


def test_high_int8_profile_keeps_signal_maxk_deblocks_out_of_int8_sampling() -> None:
    from heal_compress.tracer.precision_coupling_tracer import PrecisionGroup
    from tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder import sample_stratified_mixed_precision_profile

    groups = [
        PrecisionGroup("pg_backbone", ["pyramid_backbone.resnet.layer1.0.conv1"], "user_constraint", ["fp32", "fp16", "int8"], "fp16", True),
        PrecisionGroup("pg_deblock", ["pyramid_backbone.deblocks.0.0"], "user_constraint", ["fp32", "fp16", "int8"], "fp16", True),
        PrecisionGroup("pg_head", ["pyramid_backbone.single_head_0"], "head_constraint", ["fp32", "fp16"], "fp16", True),
    ]

    profile = sample_stratified_mixed_precision_profile(
        subnet_id="subnet_000",
        structure_hash="hash",
        groups=groups,
        profile_index=3,
        profile_seed=20260708,
        subnet_index=0,
    )

    assert profile["precision_group_assignments"]["pg_backbone"]["final_precision"] == "int8"
    assert profile["precision_group_assignments"]["pg_deblock"]["final_precision"] != "int8"


def test_onnx_qdq_stage_fails_int8_profile_when_no_onnx_node_matched(tmp_path: Path) -> None:
    import tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder as builder

    profile = {
        "precision_group_assignments": {
            "pg_missing": {
                "member_modules": ["missing"],
                "final_precision": "int8",
                "allowed_precisions": ["fp32", "fp16", "int8"],
            }
        },
        "layer_precision_assignment": {"missing": "int8"},
        "fallback_layers": [],
    }

    report = builder.run_onnx_qdq_stage(_qdq_test_ctx(tmp_path, profile))
    qdq_report = json.loads((tmp_path / "subnets/subnet_000/profile_001/qdq_insert_report.json").read_text(encoding="utf-8"))

    assert report["success"] is False
    assert report["status"] == "onnx_or_qdq_failed"
    assert qdq_report["success"] is False
    assert qdq_report["failure_reason"] == "int8_profile_has_no_inserted_qdq_nodes"
    assert qdq_report["inserted_qdq_nodes"] == []


def test_onnx_qdq_stage_counts_overlapping_int8_groups_as_matched(tmp_path: Path) -> None:
    import tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder as builder

    profile = {
        "precision_group_assignments": {
            "pg_residual": {
                "member_modules": ["encoder_m1"],
                "final_precision": "int8",
                "allowed_precisions": ["fp32", "fp16", "int8"],
            },
            "pg_concat": {
                "member_modules": ["encoder_m1", "cls_head"],
                "final_precision": "int8",
                "allowed_precisions": ["fp32", "fp16", "int8"],
            },
        },
        "layer_precision_assignment": {"encoder_m1": "int8", "cls_head": "int8"},
        "fallback_layers": [],
    }

    report = builder.run_onnx_qdq_stage(_qdq_test_ctx(tmp_path, profile))
    qdq_report = json.loads((tmp_path / "subnets/subnet_000/profile_001/qdq_insert_report.json").read_text(encoding="utf-8"))

    assert report["success"] is True
    assert qdq_report["success"] is True
    assert set(qdq_report["matched_int8_precision_groups"]) == {"pg_residual", "pg_concat"}
    assert qdq_report["unmatched_int8_precision_groups"] == []


def test_onnx_qdq_stage_allows_fp16_only_profile_without_qdq(tmp_path: Path) -> None:
    import onnx
    import tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder as builder

    profile = {
        "precision_group_assignments": {
            "pg_stem": {
                "member_modules": ["encoder_m1"],
                "final_precision": "fp16",
                "allowed_precisions": ["fp32", "fp16", "int8"],
            }
        },
        "layer_precision_assignment": {"encoder_m1": "fp16"},
        "fallback_layers": [],
    }

    report = builder.run_onnx_qdq_stage(_qdq_test_ctx(tmp_path, profile))
    qdq_report = json.loads((tmp_path / "subnets/subnet_000/profile_001/qdq_insert_report.json").read_text(encoding="utf-8"))
    model = onnx.load(str(tmp_path / "subnets/subnet_000/profile_001/onnx/model_mixed_qdq.onnx"))

    assert report["success"] is True
    assert qdq_report["success"] is True
    assert qdq_report["inserted_qdq_nodes"] == []
    assert not [node for node in model.graph.node if node.op_type in {"QuantizeLinear", "DequantizeLinear"}]


def test_trtexec_build_uses_profile_layer_constraints_not_unconditional_fp16(tmp_path: Path, monkeypatch) -> None:
    import argparse
    import subprocess
    import tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder as builder

    trtexec = tmp_path / "trtexec"
    trtexec.write_text("#!/bin/sh\n", encoding="utf-8")
    trtexec.chmod(0o755)
    onnx_path = tmp_path / "model.onnx"
    _write_tiny_conv_onnx(onnx_path)
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        for part in cmd:
            if str(part).startswith("--saveEngine="):
                Path(str(part).split("=", 1)[1]).write_bytes(b"engine")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok")

    monkeypatch.setattr(builder.subprocess, "run", fake_run)
    args = argparse.Namespace(
        trtexec=str(trtexec),
        trt_root=str(tmp_path),
        trt_build_timeout_seconds=5,
    )
    profile = {
        "precision_group_assignments": {
            "pg_stem": {"member_modules": ["stem"], "final_precision": "int8"},
            "pg_head": {"member_modules": ["head"], "final_precision": "fp16"},
        },
        "layer_precision_assignment": {"stem": "int8", "head": "fp16"},
    }

    report = builder.build_engine_with_trtexec(
        args=args,
        onnx_path=onnx_path,
        engine_path=tmp_path / "engine.plan",
        profile=profile,
        build_log_path=tmp_path / "build.log",
        layer_info_path=tmp_path / "layer_info.json",
    )

    cmd = captured["cmd"]
    assert report["build_success"] is True
    assert "--precisionConstraints=obey" in cmd
    assert any(part.startswith("--layerPrecisions=/stem/Conv:int8,/head/Conv:fp16") for part in cmd)
    assert any(part.startswith("--layerOutputTypes=/stem/Conv:int8,/head/Conv:fp16") for part in cmd)

    fp32_profile = {
        "precision_group_assignments": {"pg_stem": {"member_modules": ["stem"], "final_precision": "fp32"}},
        "layer_precision_assignment": {"stem": "fp32"},
    }
    builder.build_engine_with_trtexec(
        args=args,
        onnx_path=onnx_path,
        engine_path=tmp_path / "engine_fp32.plan",
        profile=fp32_profile,
        build_log_path=tmp_path / "build_fp32.log",
        layer_info_path=tmp_path / "layer_info_fp32.json",
    )
    assert "--fp16" not in captured["cmd"]
    assert "--int8" not in captured["cmd"]


def test_signal_maxk_engine_build_requires_pointpillar_plugin(tmp_path: Path, monkeypatch) -> None:
    import argparse
    import subprocess
    import tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder as builder

    trtexec = tmp_path / "trtexec"
    trtexec.write_text("#!/bin/sh\n", encoding="utf-8")
    trtexec.chmod(0o755)
    onnx_path = tmp_path / "model_signal_maxk.onnx"
    _write_signal_maxk_like_onnx_with_trt_plugin(onnx_path)
    called = {"trtexec": False}

    def fake_run(cmd, **kwargs):
        called["trtexec"] = True
        return subprocess.CompletedProcess(cmd, 0, stdout="should not run")

    monkeypatch.setattr(builder.subprocess, "run", fake_run)
    args = argparse.Namespace(
        trtexec=str(trtexec),
        trt_root=str(tmp_path),
        trt_build_timeout_seconds=5,
        plugin=str(tmp_path / "missing_plugin.so"),
        fixed_k=29696,
    )
    profile = {
        "precision_group_assignments": {"pg_encoder": {"member_modules": ["encoder_m1"], "final_precision": "fp16"}},
        "layer_precision_assignment": {"encoder_m1": "fp16"},
    }

    report = builder.build_engine_with_trtexec(
        args=args,
        onnx_path=onnx_path,
        engine_path=tmp_path / "engine.plan",
        profile=profile,
        build_log_path=tmp_path / "build.log",
        layer_info_path=tmp_path / "layer_info.json",
    )

    assert report["build_success"] is False
    assert report["failure_reason"].startswith("pointpillar_scatter_plugin_not_found:")
    assert called["trtexec"] is False


def test_signal_maxk_engine_build_passes_static_plugin_to_trtexec(tmp_path: Path, monkeypatch) -> None:
    import argparse
    import subprocess
    import tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder as builder

    trtexec = tmp_path / "trtexec"
    trtexec.write_text("#!/bin/sh\n", encoding="utf-8")
    trtexec.chmod(0o755)
    plugin = tmp_path / "libpointpillar_scatter_trt.so"
    plugin.write_bytes(b"plugin")
    onnx_path = tmp_path / "model_signal_maxk.onnx"
    _write_signal_maxk_like_onnx_with_trt_plugin(onnx_path)
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        for part in cmd:
            if str(part).startswith("--saveEngine="):
                Path(str(part).split("=", 1)[1]).write_bytes(b"engine")
        return subprocess.CompletedProcess(cmd, 0, stdout="ok")

    monkeypatch.setattr(builder.subprocess, "run", fake_run)
    args = argparse.Namespace(
        trtexec=str(trtexec),
        trt_root=str(tmp_path),
        trt_build_timeout_seconds=5,
        plugin=str(plugin),
        fixed_k=29696,
    )
    profile = {
        "precision_group_assignments": {"pg_encoder": {"member_modules": ["encoder_m1"], "final_precision": "fp16"}},
        "layer_precision_assignment": {"encoder_m1": "fp16"},
    }

    report = builder.build_engine_with_trtexec(
        args=args,
        onnx_path=onnx_path,
        engine_path=tmp_path / "engine.plan",
        profile=profile,
        build_log_path=tmp_path / "build.log",
        layer_info_path=tmp_path / "layer_info.json",
    )

    assert report["build_success"] is True
    assert f"--staticPlugins={plugin}" in captured["cmd"]
    assert report["static_plugin"] == str(plugin)


def test_engine_build_can_defer_layer_info_export_until_after_save_engine(tmp_path: Path, monkeypatch) -> None:
    import argparse
    import subprocess
    import tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder as builder

    trtexec = tmp_path / "trtexec"
    trtexec.write_text("#!/bin/sh\n", encoding="utf-8")
    trtexec.chmod(0o755)
    onnx_path = tmp_path / "model.onnx"
    _write_tiny_conv_onnx(onnx_path)
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if any(str(part).startswith("--saveEngine=") for part in cmd):
            for part in cmd:
                if str(part).startswith("--saveEngine="):
                    Path(str(part).split("=", 1)[1]).write_bytes(b"engine")
            return subprocess.CompletedProcess(cmd, 0, stdout="build ok")
        if any(str(part).startswith("--loadEngine=") for part in cmd):
            for part in cmd:
                if str(part).startswith("--exportLayerInfo="):
                    Path(str(part).split("=", 1)[1]).write_text('{"Layers":[{"Name":"x"}]}', encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, stdout="layer info ok")
        return subprocess.CompletedProcess(cmd, 1, stdout="unexpected")

    monkeypatch.setattr(builder.subprocess, "run", fake_run)
    args = argparse.Namespace(
        trtexec=str(trtexec),
        trt_root=str(tmp_path),
        trt_build_timeout_seconds=5,
        export_layer_info_during_build=False,
    )
    profile = {
        "precision_group_assignments": {"pg_stem": {"member_modules": ["stem"], "final_precision": "fp16"}},
        "layer_precision_assignment": {"stem": "fp16"},
    }

    report = builder.build_engine_with_trtexec(
        args=args,
        onnx_path=onnx_path,
        engine_path=tmp_path / "engine.plan",
        profile=profile,
        build_log_path=tmp_path / "build.log",
        layer_info_path=tmp_path / "layer_info.json",
    )

    assert report["build_success"] is True
    assert len(calls) == 2
    assert any(str(part).startswith("--saveEngine=") for part in calls[0])
    assert not any(str(part).startswith("--exportLayerInfo=") for part in calls[0])
    assert any(str(part).startswith("--loadEngine=") for part in calls[1])
    assert any(str(part).startswith("--exportLayerInfo=") for part in calls[1])
    assert report["layer_info_export_report"]["success"] is True


def test_tensorrt_engine_runner_loads_plugin_before_deserialize(tmp_path: Path, monkeypatch) -> None:
    import sys
    import types
    from heal_compress.trt_runtime.engine_runner import TensorRTEngineRunner
    import heal_compress.trt_runtime.plugins as trt_plugins

    engine_path = tmp_path / "engine.plan"
    engine_path.write_bytes(b"engine")
    plugin_path = tmp_path / "libpointpillar_scatter_trt.so"
    plugin_path.write_bytes(b"plugin")
    events: list[str] = []

    class FakeLogger:
        ERROR = 0

        def __init__(self, level: int) -> None:
            self.level = level

    class FakeRuntime:
        def __init__(self, logger: object) -> None:
            self.logger = logger

        def deserialize_cuda_engine(self, payload: bytes) -> object:
            events.append("deserialize")

            class FakeEngine:
                def create_execution_context(self) -> object:
                    return object()

            return FakeEngine()

    fake_trt = types.SimpleNamespace(Logger=FakeLogger, Runtime=FakeRuntime)
    monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)
    monkeypatch.setattr(trt_plugins.ctypes, "CDLL", lambda path, mode: events.append(f"load:{Path(path).name}"))
    monkeypatch.setattr("heal_compress.trt_runtime.engine_runner.torch.cuda.Stream", lambda device: object())

    TensorRTEngineRunner(engine_path, types.SimpleNamespace(type="cuda"), plugin_path=plugin_path)

    assert events == ["load:libpointpillar_scatter_trt.so", "deserialize"]


def test_schema_rows_for_engine_component_and_training_sample() -> None:
    from tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder import (
        component_lut_sample_row,
        engine_eval_summary_row,
        full_engine_training_sample,
    )

    eval_row = engine_eval_summary_row("subnet_000", "profile_000", "hash", {}, {}, {})
    comp_row = component_lut_sample_row("subnet_000", "profile_000", "hash", {})
    sample = full_engine_training_sample("subnet_000", "profile_000", "hash", {}, [comp_row], eval_row)

    assert {"total_p50_ms", "forward_mean_ms", "AP@0.03", "fallback_layer_count"} <= set(eval_row)
    assert {"trt_layer_name", "precision_group_id", "latency_ms", "final_precision"} <= set(comp_row)
    assert sample["subnet_id"] == "subnet_000"
    assert "component_features" in sample


def test_pipeline_gates_block_later_stages() -> None:
    from tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder import gate_engine_build, gate_eval, gate_onnx_export

    assert gate_onnx_export({"passed": False})["allowed"] is False
    assert gate_engine_build({"success": False})["allowed"] is False
    assert gate_eval({"build_success": False})["allowed"] is False


def test_sample_only_generates_required_index_files(tmp_path: Path) -> None:
    from tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder import main

    rc = main(
        [
            "--mode",
            "sample-only",
            "--num-subnets",
            "2",
            "--precision-profiles-per-subnet",
            "1",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert rc == 0
    assert (tmp_path / "subnet_index.csv").exists()
    assert (tmp_path / "mixed_precision_profile_index.csv").exists()
    assert (tmp_path / "dataset_manifest.json").exists()
    manifest = json.loads((tmp_path / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["successful_subnet_count"] == 2


def test_stratified_profiles_have_templates_and_unique_hashes() -> None:
    from heal_compress.tracer.precision_coupling_tracer import PrecisionGroup
    from tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder import (
        precision_assignment_hash,
        sample_stratified_mixed_precision_profile,
    )

    groups = [
        PrecisionGroup("pg0", ["branch_a", "branch_b"], "residual", ["fp32", "fp16", "int8"], "fp16", False),
        PrecisionGroup("pg1", ["fuse"], "user_constraint", ["fp32", "fp16", "int8"], "fp16", True),
        PrecisionGroup("pg2", ["head"], "head_constraint", ["fp32", "fp16"], "fp16", True),
    ]
    profiles = [
        sample_stratified_mixed_precision_profile(
            subnet_id="subnet_000",
            structure_hash="hash",
            groups=groups,
            profile_index=i,
            profile_seed=20260708,
            subnet_index=0,
        )
        for i in range(1, 5)
    ]

    templates = {profile["profile_template_id"] for profile in profiles}
    hashes = {precision_assignment_hash(profile) for profile in profiles}
    assert [profile["profile_template_id"] for profile in profiles[:3]] == ["low_int8", "medium_int8", "high_int8"]
    assert {"low_int8", "medium_int8", "high_int8"} <= templates
    assert all(profile["int8_group_count"] > 0 for profile in profiles[:3])
    assert len(hashes) == len(profiles)
    assert any(
        profile["layer_precision_assignment"]["branch_a"] != profile["layer_precision_assignment"]["branch_b"]
        for profile in profiles
    )
    for profile in profiles:
        assert profile["precision_assignment_hash"] == precision_assignment_hash(profile)


def test_high_int8_profile_does_not_duplicate_existing_all_int8_profile() -> None:
    from heal_compress.tracer.precision_coupling_tracer import PrecisionGroup
    from tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder import (
        precision_assignment_hash,
        sample_stratified_mixed_precision_profile,
    )

    groups = [
        PrecisionGroup("pg_0000", ["branch_a", "branch_b"], "residual", ["fp32", "fp16", "int8"], "fp16", True),
        PrecisionGroup("pg_0001", ["stem"], "user_constraint", ["fp32", "fp16", "int8"], "fp16", True),
        PrecisionGroup("pg_0002", ["fuse"], "user_constraint", ["fp32", "fp16", "int8"], "fp16", True),
        PrecisionGroup("pg_0003", ["head"], "head_constraint", ["fp32", "fp16"], "fp16", True),
    ]
    existing_all_int8_profile = {
        "precision_group_assignments": {
            "pg_0000": {"member_modules": ["branch_a", "branch_b"], "requested_precision": "int8", "final_precision": "int8"},
            "pg_0001": {"member_modules": ["stem"], "requested_precision": "int8", "final_precision": "int8"},
            "pg_0002": {"member_modules": ["fuse"], "requested_precision": "int8", "final_precision": "int8"},
            "pg_0003": {"member_modules": ["head"], "requested_precision": "fp16", "final_precision": "fp16"},
        }
    }

    high = sample_stratified_mixed_precision_profile(
        subnet_id="subnet_017",
        structure_hash="hash",
        groups=groups,
        profile_index=3,
        profile_seed=20260708,
        subnet_index=17,
    )

    assert high["profile_template_id"] == "high_int8"
    assert high["int8_group_count"] == 2
    assert precision_assignment_hash(high) != precision_assignment_hash(existing_all_int8_profile)


def test_stratified_profile_records_concat_fp16_boundary_counts() -> None:
    from heal_compress.tracer.precision_coupling_tracer import PrecisionGroup
    from tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder import sample_stratified_mixed_precision_profile

    groups = [
        PrecisionGroup("pg_concat", ["branch_a", "branch_b"], "concat", ["fp32", "fp16", "int8"], "fp16", False),
        PrecisionGroup("pg_fuse", ["fuse"], "user_constraint", ["fp32", "fp16", "int8"], "fp16", True),
    ]

    profile = sample_stratified_mixed_precision_profile(
        subnet_id="subnet_000",
        structure_hash="hash",
        groups=groups,
        profile_index=3,
        profile_seed=20260708,
        subnet_index=0,
    )

    assert profile["concat_fp16_boundary_count"] == 1
    assert profile["concat_requantize_after_count"] >= 0


def test_overlapping_precision_groups_are_assigned_consistently() -> None:
    from heal_compress.tracer.precision_coupling_tracer import PrecisionGroup
    from tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder import sample_stratified_mixed_precision_profile

    groups = [
        PrecisionGroup("pg0", ["branch_a", "branch_b"], "residual", ["fp32", "fp16", "int8"], "fp16", True),
        PrecisionGroup("pg1", ["branch_a", "stem"], "user_constraint", ["fp32", "fp16", "int8"], "fp16", True),
        PrecisionGroup("pg2", ["head"], "head_constraint", ["fp32", "fp16"], "fp16", True),
    ]

    profile = sample_stratified_mixed_precision_profile(
        subnet_id="subnet_000",
        structure_hash="hash",
        groups=groups,
        profile_index=3,
        profile_seed=20260708,
        subnet_index=0,
    )

    for group_id, row in profile["precision_group_assignments"].items():
        for module in row["member_modules"]:
            assert profile["layer_precision_assignment"][module] == row["final_precision"], group_id


def test_metadata_only_training_sample_has_no_labels() -> None:
    from tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder import (
        engine_eval_summary_row,
        full_engine_training_sample,
    )

    row = engine_eval_summary_row("subnet_000", "profile_001", "hash", {}, {}, {}, eval_success=False)
    sample = full_engine_training_sample("subnet_000", "profile_001", "hash", {}, [], row)

    assert sample["label_available"] is False
    assert sample["measured_forward_latency"] is None
    assert sample["AP/mAP"] is None


def test_training_sample_requires_all_formal_label_gates() -> None:
    from tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder import (
        engine_eval_summary_row,
        full_engine_training_sample,
    )

    eval_row = engine_eval_summary_row(
        "subnet_000",
        "profile_001",
        "hash",
        {"forward_mean_ms": 1.0, "total_mean_ms": 2.0, "evaluated_frames": 10},
        {"AP@0.03": 0.1, "AP@0.30": 0.1, "AP@0.50": 0.1, "AP@0.70": 0.1, "mAP": 0.1},
        {"int8_group_count": 1},
        build_success=True,
        eval_success=True,
        precision_realization_passed=True,
        smoke_success=True,
        synthetic_used=False,
        validation_dataloader_used=True,
    )

    sample = full_engine_training_sample("subnet_000", "profile_001", "hash", {"int8_group_count": 1}, [], eval_row)

    assert sample["label_available"] is False
    assert sample["measured_forward_latency"] is None


def test_expand_profiles_reuses_existing_subnets_and_build_eval_are_separate(tmp_path: Path) -> None:
    from tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder import main

    rc = main(
        [
            "--mode",
            "sample-only",
            "--num-subnets",
            "2",
            "--precision-profiles-per-subnet",
            "1",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    before = {
        path.parent.name: json.loads(path.read_text(encoding="utf-8"))["structure_hash"]
        for path in sorted((tmp_path / "subnets").glob("subnet_*/pruning_manifest.json"))
    }

    rc = main(
        [
            "--mode",
            "expand-profiles",
            "--source-dir",
            str(tmp_path),
            "--max-subnets",
            "2",
            "--precision-profiles-per-subnet",
            "3",
            "--profile-sampling-policy",
            "stratified",
            "--profile-seed",
            "20260708",
            "--build-engines",
            "false",
            "--eval-engines",
            "false",
        ]
    )

    assert rc == 0
    after = {
        path.parent.name: json.loads(path.read_text(encoding="utf-8"))["structure_hash"]
        for path in sorted((tmp_path / "subnets").glob("subnet_*/pruning_manifest.json"))
    }
    assert after == before
    for subnet in ("subnet_000", "subnet_001"):
        profile_paths = sorted((tmp_path / "subnets" / subnet).glob("profile_*/mixed_precision_profile.json"))
        assert len(profile_paths) == 3
        hashes = [json.loads(path.read_text(encoding="utf-8"))["precision_assignment_hash"] for path in profile_paths]
        assert len(hashes) == len(set(hashes))
    rows = list(csv.DictReader((tmp_path / "mixed_precision_profile_index.csv").open()))
    assert len(rows) == 6
    assert {row["profile_template_id"] for row in rows} >= {"existing_or_baseline_profile", "low_int8", "medium_int8"}
    assert all(row["eval_success"] == "False" for row in rows)
    manifest = json.loads((tmp_path / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["new_pruning_performed"] is False
    assert manifest["profiles_per_subnet_requested"] == 3


def _make_sampled_subnets(tmp_path: Path, *, subnets: int = 1) -> None:
    from tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder import main

    rc = main(
        [
            "--mode",
            "sample-only",
            "--num-subnets",
            str(subnets),
            "--precision-profiles-per-subnet",
            "1",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0


def test_per_engine_pipeline_runs_stages_in_order_and_writes_immediately(tmp_path: Path, monkeypatch) -> None:
    import tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder as builder

    _make_sampled_subnets(tmp_path, subnets=1)
    calls: list[str] = []

    def stage(name: str):
        def _run(ctx):
            calls.append(name)
            if name == "profile":
                return {"profile_legality_passed": True, "status": "profile_legality_passed"}
            if name == "onnx":
                return {"success": True, "status": "onnx_qdq_passed", "output_onnx": str(ctx["profile_dir"] / "onnx" / "model_mixed_qdq.onnx")}
            if name == "build":
                return {"build_success": True, "success": True, "status": "engine_build_passed", "engine_path": str(ctx["profile_dir"] / "engine.plan"), "uses_int8_flag": False}
            if name == "structure":
                return {"structure_check_passed": True, "status": "engine_structure_passed", "matched_compute_layer_count": 1}
            if name == "precision":
                return {"precision_realization_passed": True, "status": "engine_precision_passed", "precision_realization_mismatch_count": 0}
            if name == "smoke":
                return {"success": True, "status": "trt_smoke_passed", "synthetic_used": False, "validation_dataloader_used": True}
            return {
                "eval_success": True,
                "status": "eval_success",
                "synthetic_used": False,
                "validation_dataloader_used": True,
                "latency_summary": {"forward_mean_ms": 1.0, "total_mean_ms": 2.0},
                "ap": {"AP@0.03": 0.1, "AP@0.30": 0.1, "AP@0.50": 0.1, "AP@0.70": 0.1, "mAP": 0.1},
                "per_frame_rows": [{"frame_id": 0, "success": True}],
            }
        return _run

    monkeypatch.setattr(builder, "run_profile_legality_stage", stage("profile"))
    monkeypatch.setattr(builder, "run_onnx_qdq_stage", stage("onnx"))
    monkeypatch.setattr(builder, "run_engine_build_stage", stage("build"))
    monkeypatch.setattr(builder, "run_engine_structure_stage", stage("structure"))
    monkeypatch.setattr(builder, "run_engine_precision_stage", stage("precision"))
    monkeypatch.setattr(builder, "run_trt_smoke_stage", stage("smoke"))
    monkeypatch.setattr(builder, "run_real_eval_stage", stage("eval"))

    rc = builder.main(
        [
            "--mode",
            "expand-profiles-per-engine",
            "--source-dir",
            str(tmp_path),
            "--max-subnets",
            "1",
            "--precision-profiles-per-subnet",
            "1",
            "--build-engines",
            "true",
            "--eval-engines",
            "true",
            "--resume",
        ]
    )

    assert rc == 0
    assert calls == ["profile", "onnx", "build", "structure", "precision", "smoke", "eval"]
    rows = list(csv.DictReader((tmp_path / "mixed_precision_profile_index.csv").open()))
    assert len(rows) == 1
    assert rows[0]["eval_success"] == "True"
    progress = json.loads((tmp_path / "progress_state.json").read_text(encoding="utf-8"))
    assert progress["profiles_completed"] == 1
    assert progress["engines_eval_success"] == 1


def test_per_engine_structure_failure_blocks_precision_smoke_eval(tmp_path: Path, monkeypatch) -> None:
    import tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder as builder

    _make_sampled_subnets(tmp_path, subnets=1)
    calls: list[str] = []

    monkeypatch.setattr(builder, "run_profile_legality_stage", lambda ctx: calls.append("profile") or {"profile_legality_passed": True, "status": "ok"})
    monkeypatch.setattr(builder, "run_onnx_qdq_stage", lambda ctx: calls.append("onnx") or {"success": True, "status": "ok"})
    monkeypatch.setattr(builder, "run_engine_build_stage", lambda ctx: calls.append("build") or {"build_success": True, "success": True, "status": "ok", "engine_path": str(ctx["profile_dir"] / "engine.plan"), "uses_int8_flag": False})
    monkeypatch.setattr(builder, "run_engine_structure_stage", lambda ctx: calls.append("structure") or {"structure_check_passed": False, "status": "engine_structure_mismatch", "failure_reason": "toy_input_1_detected"})
    monkeypatch.setattr(builder, "run_engine_precision_stage", lambda ctx: (_ for _ in ()).throw(AssertionError("precision blocked")))
    monkeypatch.setattr(builder, "run_trt_smoke_stage", lambda ctx: (_ for _ in ()).throw(AssertionError("smoke blocked")))
    monkeypatch.setattr(builder, "run_real_eval_stage", lambda ctx: (_ for _ in ()).throw(AssertionError("eval blocked")))

    rc = builder.main(
        [
            "--mode",
            "expand-profiles-per-engine",
            "--source-dir",
            str(tmp_path),
            "--max-subnets",
            "1",
            "--precision-profiles-per-subnet",
            "1",
            "--eval-engines",
            "true",
        ]
    )

    assert rc == 0
    assert calls == ["profile", "onnx", "build", "structure"]
    failure = json.loads((tmp_path / "subnets/subnet_000/profile_000/profile_failure_report.json").read_text(encoding="utf-8"))
    assert failure["stage_failed"] == "engine_structure_mismatch"
    rows = list(csv.DictReader((tmp_path / "mixed_precision_profile_index.csv").open()))
    assert rows[0]["status"] == "engine_structure_mismatch"


def test_per_engine_profile_legality_failure_blocks_later_stages(tmp_path: Path, monkeypatch) -> None:
    import tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder as builder

    _make_sampled_subnets(tmp_path, subnets=1)
    calls: list[str] = []

    def fail_profile(ctx):
        calls.append("profile")
        return {"profile_legality_passed": False, "status": "profile_legality_failed", "failure_reason": "duplicate_precision_assignment_hash"}

    def forbidden(ctx):
        raise AssertionError("later stage must not run after profile legality failure")

    monkeypatch.setattr(builder, "run_profile_legality_stage", fail_profile)
    monkeypatch.setattr(builder, "run_onnx_qdq_stage", forbidden)
    monkeypatch.setattr(builder, "run_engine_build_stage", forbidden)
    monkeypatch.setattr(builder, "run_engine_precision_stage", forbidden)
    monkeypatch.setattr(builder, "run_trt_smoke_stage", forbidden)
    monkeypatch.setattr(builder, "run_real_eval_stage", forbidden)

    rc = builder.main(
        [
            "--mode",
            "expand-profiles-per-engine",
            "--source-dir",
            str(tmp_path),
            "--max-subnets",
            "1",
            "--precision-profiles-per-subnet",
            "1",
            "--eval-engines",
            "true",
        ]
    )

    assert rc == 0
    assert calls == ["profile"]
    failure = json.loads((tmp_path / "subnets/subnet_000/profile_000/profile_failure_report.json").read_text(encoding="utf-8"))
    assert failure["stage_failed"] == "profile_legality_failed"
    assert not (tmp_path / "subnets/subnet_000/profile_000/qdq_insert_report.json").exists()


def test_per_engine_onnx_failure_overwrites_stale_engine_artifacts(tmp_path: Path, monkeypatch) -> None:
    import tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder as builder

    _make_sampled_subnets(tmp_path, subnets=1)
    profile_dir = tmp_path / "subnets/subnet_000/profile_000"
    (profile_dir / "engine.plan").write_bytes(b"stale")
    builder.write_json(profile_dir / "build_report.json", {"build_success": True, "status": "engine_build_passed"})

    monkeypatch.setattr(builder, "run_profile_legality_stage", lambda ctx: {"profile_legality_passed": True, "status": "ok"})
    monkeypatch.setattr(builder, "run_onnx_qdq_stage", lambda ctx: {"success": False, "status": "onnx_or_qdq_failed", "failure_reason": "int8_profile_has_no_inserted_qdq_nodes"})
    monkeypatch.setattr(builder, "run_engine_build_stage", lambda ctx: (_ for _ in ()).throw(AssertionError("build blocked")))
    monkeypatch.setattr(builder, "run_engine_precision_stage", lambda ctx: (_ for _ in ()).throw(AssertionError("precision blocked")))
    monkeypatch.setattr(builder, "run_trt_smoke_stage", lambda ctx: (_ for _ in ()).throw(AssertionError("smoke blocked")))
    monkeypatch.setattr(builder, "run_real_eval_stage", lambda ctx: (_ for _ in ()).throw(AssertionError("eval blocked")))

    rc = builder.main(
        [
            "--mode",
            "expand-profiles-per-engine",
            "--source-dir",
            str(tmp_path),
            "--max-subnets",
            "1",
            "--precision-profiles-per-subnet",
            "1",
            "--eval-engines",
            "true",
        ]
    )

    assert rc == 0
    assert not (profile_dir / "engine.plan").exists()
    build_report = json.loads((profile_dir / "build_report.json").read_text(encoding="utf-8"))
    assert build_report["build_success"] is False
    assert build_report["status"] == "engine_build_blocked"
    precision_report = json.loads((profile_dir / "engine_precision_realization_report.json").read_text(encoding="utf-8"))
    assert precision_report["status"] == "engine_precision_not_run"
    rows = list(csv.DictReader((tmp_path / "mixed_precision_profile_index.csv").open()))
    assert rows[0]["build_success"] == "False"
    assert rows[0]["status"] == "onnx_or_qdq_failed"


def test_per_engine_precision_mismatch_blocks_eval_without_allow_flag(tmp_path: Path, monkeypatch) -> None:
    import tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder as builder

    _make_sampled_subnets(tmp_path, subnets=1)
    calls: list[str] = []

    monkeypatch.setattr(builder, "run_profile_legality_stage", lambda ctx: calls.append("profile") or {"profile_legality_passed": True, "status": "ok"})
    monkeypatch.setattr(builder, "run_onnx_qdq_stage", lambda ctx: calls.append("onnx") or {"success": True, "status": "ok"})
    monkeypatch.setattr(builder, "run_engine_build_stage", lambda ctx: calls.append("build") or {"build_success": True, "success": True, "status": "ok", "engine_path": str(ctx["profile_dir"] / "engine.plan"), "uses_int8_flag": False})
    monkeypatch.setattr(builder, "run_engine_structure_stage", lambda ctx: calls.append("structure") or {"structure_check_passed": True, "status": "ok"})
    monkeypatch.setattr(builder, "run_engine_precision_stage", lambda ctx: calls.append("precision") or {"precision_realization_passed": False, "status": "engine_precision_mismatch", "precision_realization_mismatch_count": 1})
    monkeypatch.setattr(builder, "run_trt_smoke_stage", lambda ctx: (_ for _ in ()).throw(AssertionError("smoke blocked")))
    monkeypatch.setattr(builder, "run_real_eval_stage", lambda ctx: (_ for _ in ()).throw(AssertionError("eval blocked")))

    rc = builder.main(
        [
            "--mode",
            "expand-profiles-per-engine",
            "--source-dir",
            str(tmp_path),
            "--max-subnets",
            "1",
            "--precision-profiles-per-subnet",
            "1",
            "--eval-engines",
            "true",
        ]
    )

    assert rc == 0
    assert calls == ["profile", "onnx", "build", "structure", "precision"]
    rows = list(csv.DictReader((tmp_path / "mixed_precision_profile_index.csv").open()))
    assert rows[0]["eval_success"] == "False"
    assert rows[0]["status"] == "engine_precision_mismatch"
    smoke_report = json.loads((tmp_path / "subnets/subnet_000/profile_000/trt_smoke_report.json").read_text(encoding="utf-8"))
    eval_report = json.loads((tmp_path / "subnets/subnet_000/profile_000/eval_report.json").read_text(encoding="utf-8"))
    assert smoke_report["status"] == "trt_smoke_not_run"
    assert smoke_report["blocked_by"] == "engine_precision_mismatch"
    assert eval_report["status"] == "eval_not_run"
    assert eval_report["blocked_by"] == "engine_precision_mismatch"


def test_engine_structure_check_rejects_toy_input_and_accepts_signal_maxk_bindings(tmp_path: Path) -> None:
    import tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder as builder

    subnet_dir = tmp_path / "subnets/subnet_000"
    profile_dir = subnet_dir / "profile_000"
    profile_dir.mkdir(parents=True, exist_ok=True)
    builder.write_json(
        subnet_dir / "pruning_manifest.json",
        {
            "subnet_id": "subnet_000",
            "structure_hash": "hash",
            "actual_param_prune_ratio": 0.2,
            "actual_channel_prune_ratio": 0.1,
            "round_to": 4,
            "module_channel_before_after": {
                "encoder_m1": {"C_in_before": 128, "C_out_before": 64, "C_in_after": 128, "C_out_after": 64},
                "cls_head": {"C_in_before": 64, "C_out_before": 2, "C_in_after": 64, "C_out_after": 2},
            },
        },
    )
    real_onnx = subnet_dir / "onnx/model_signal_maxk.onnx"
    _write_signal_maxk_like_onnx(real_onnx)
    builder.write_json(
        subnet_dir / "onnx/real_onnx_export_report.json",
        {
            "export_success": True,
            "onnx_path": str(real_onnx),
            "input_names": ["voxel_features", "voxel_coords", "voxel_num_points", "record_len", "pairwise_t_matrix", "valid_voxel_mask"],
            "output_names": ["cls_preds", "reg_preds", "dir_preds"],
        },
    )
    builder.write_json(
        profile_dir / "trt_layer_info.json",
        {
            "Layers": [
                {"Name": "/encoder_m1/stem/MatMul", "LayerType": "MatrixMultiply", "Metadata": "[ONNX Layer: /encoder_m1/stem/MatMul]", "Outputs": [{"Format/Datatype": "Half"}]},
                {"Name": "/cls_head/MatMul", "LayerType": "MatrixMultiply", "Metadata": "[ONNX Layer: /cls_head/MatMul]", "Outputs": [{"Format/Datatype": "Half"}]},
            ]
        },
    )

    passed = builder.run_engine_structure_stage(
        {
            "subnet_dir": subnet_dir,
            "profile_dir": profile_dir,
            "structure_hash": "hash",
        }
    )

    assert passed["structure_check_passed"] is True
    assert passed["onnx_input_names"] != ["input.1"]
    assert passed["matched_compute_layer_count"] >= 2
    assert {"structure_check_passed", "structure_hash", "onnx_input_names", "onnx_output_names", "matched_compute_layer_count", "unmatched_compute_layers", "channel_alignment_passed", "round_to", "failure_reason"} <= set(passed)

    toy_profile_dir = subnet_dir / "profile_001"
    toy_profile_dir.mkdir(parents=True, exist_ok=True)
    toy_onnx = subnet_dir / "onnx/model_signal_maxk.onnx"
    _write_tiny_conv_onnx(toy_onnx)
    builder.write_json(
        subnet_dir / "onnx/real_onnx_export_report.json",
        {"export_success": True, "onnx_path": str(toy_onnx), "input_names": ["input.1"], "output_names": ["17"]},
    )
    builder.write_json(toy_profile_dir / "trt_layer_info.json", {"Layers": []})

    failed = builder.run_engine_structure_stage(
        {
            "subnet_dir": subnet_dir,
            "profile_dir": toy_profile_dir,
            "structure_hash": "hash",
        }
    )

    assert failed["structure_check_passed"] is False
    assert failed["status"] == "engine_structure_mismatch"
    assert "toy_input_1" in failed["failure_reason"]


def test_canonical_shape_check_prefers_physical_module_channel_shapes(tmp_path: Path) -> None:
    import onnx
    from onnx import TensorProto, helper
    import tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder as builder

    onnx_path = tmp_path / "model.onnx"
    graph = helper.make_graph(
        [
            helper.make_node(
                "Conv",
                ["x", "conv.weight"],
                ["y"],
                name="__canonical__stem__Conv__call00001",
                kernel_shape=[3, 3],
                pads=[1, 1, 1, 1],
            )
        ],
        "shape_check",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 64, 8, 8])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 36, 8, 8])],
        [helper.make_tensor("conv.weight", TensorProto.FLOAT, [36, 64, 3, 3], [0.0] * (36 * 64 * 3 * 3))],
    )
    onnx.save(helper.make_model(graph), onnx_path)

    manifest = {
        "before_after_shapes": [
            {
                "module_name": "stem",
                "after": {"attrs": {"in_channels": 36, "out_channels": 36, "groups": 1}},
            }
        ],
        "module_channel_before_after": [
            {
                "module_name": "stem",
                "after": {"in_channels": 64, "out_channels": 36, "groups": 1},
            }
        ],
    }
    mapping = {
        "entries": [
            {
                "canonical_module_name": "stem",
                "onnx_node_name_unique": "__canonical__stem__Conv__call00001",
                "onnx_node_name_original": "/stem/Conv",
                "onnx_op_type": "Conv",
                "onnx_weight_initializer": "conv.weight",
            }
        ]
    }

    passed, checks = builder._canonical_shape_consistency_report(
        onnx_path=onnx_path,
        manifest=manifest,
        canonical_mapping=mapping,
    )

    assert passed is True
    assert checks[0]["manifest_after_attrs"] == {"in_channels": 64, "out_channels": 36, "groups": 1}


def test_channel_alignment_report_does_not_hard_fail_fp16_deblock_convtranspose() -> None:
    import tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder as builder

    passed, violations = builder._channel_alignment_report(
        {
            "round_to": 4,
            "module_channel_before_after": [
                {
                    "module_name": "pyramid_backbone.deblocks.1.0",
                    "module_type": "ConvTranspose2d",
                    "after": {"in_channels": 72, "out_channels": 66, "groups": 1},
                },
                {
                    "module_name": "pyramid_backbone.resnet.layer1.0.conv1",
                    "module_type": "Conv2d",
                    "after": {"in_channels": 72, "out_channels": 66, "groups": 1},
                },
            ],
        }
    )

    assert passed is False
    assert violations == [
        {
            "module_name": "pyramid_backbone.resnet.layer1.0.conv1",
            "field": "out_channels",
            "value": 66,
            "round_to": 4,
        }
    ]


def test_manifest_shape_fallback_uses_sampling_before_when_no_physical_delta() -> None:
    import tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder as builder

    manifest = {
        "materialized_from_random_dependency_domains": True,
        "before_after_shapes": [
            {
                "module_name": "conv",
                "module_type": "Conv2d",
                "before": {"attrs": {"in_channels": 64, "out_channels": 64, "groups": 1}},
                "after": {"attrs": {"in_channels": 56, "out_channels": 56, "groups": 1}},
            }
        ],
        "module_channel_before_after": [],
    }

    rows = builder._manifest_shapes_by_module(manifest)

    assert builder._after_attrs_from_manifest_shape(rows["conv"]) == {"in_channels": 64, "out_channels": 64, "groups": 1}


def test_manifest_shape_fallback_uses_physical_delta_over_sampling_request() -> None:
    import tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder as builder

    manifest = {
        "materialized_from_random_dependency_domains": True,
        "before_after_shapes": [
            {"module_name": "conv", "before": {"attrs": {"in_channels": 64, "out_channels": 64, "groups": 1}}, "after": {"attrs": {"in_channels": 56, "out_channels": 56, "groups": 1}}}
        ],
        "module_channel_before_after": [
            {"module_name": "conv", "after": {"in_channels": 60, "out_channels": 60, "groups": 1}}
        ],
    }

    rows = builder._manifest_shapes_by_module(manifest)

    assert builder._after_attrs_from_manifest_shape(rows["conv"]) == {"in_channels": 60, "out_channels": 60, "groups": 1}


def test_manifest_shape_snapshot_has_priority_over_sampling_and_delta() -> None:
    import tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder as builder

    manifest = {
        "materialized_from_random_dependency_domains": True,
        "before_after_shapes": [{"module_name": "conv", "before": {"attrs": {"in_channels": 64, "out_channels": 64}}, "after": {"attrs": {"in_channels": 56, "out_channels": 56}}}],
        "module_channel_before_after": [{"module_name": "conv", "after": {"in_channels": 60, "out_channels": 60}}],
        "physical_structure_snapshot_v2": {
            "modules": [{"canonical_module_name": "conv", "module_type": "Conv2d", "in_channels": 64, "out_channels": 64, "groups": 1, "weight_shape": [64, 64, 3, 3]}]
        },
    }

    rows = builder._manifest_shapes_by_module(manifest)

    assert builder._after_attrs_from_manifest_shape(rows["conv"]) == {"in_channels": 64, "out_channels": 64, "groups": 1}


def test_physical_preflight_failure_blocks_trtexec(tmp_path: Path, monkeypatch) -> None:
    import tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder as builder

    profile_dir = tmp_path / "subnets/subnet_000/profile_000"
    profile_dir.mkdir(parents=True)
    calls: list[str] = []
    monkeypatch.setattr(builder, "run_profile_legality_stage", lambda ctx: {"profile_legality_passed": True})
    monkeypatch.setattr(builder, "run_onnx_qdq_stage", lambda ctx: {"success": True, "output_onnx": str(profile_dir / "onnx/model_mixed_qdq.onnx")})
    monkeypatch.setattr(builder, "run_physical_structure_preflight_stage", lambda ctx: {"preflight_passed": False, "failure_reason": "shape_mismatch"})
    monkeypatch.setattr(builder, "run_engine_build_stage", lambda ctx: calls.append("trtexec") or {"build_success": True})

    result = builder.run_one_profile_pipeline(
        {
            "args": argparse.Namespace(allow_precision_mismatch_eval=False),
            "subnet_dir": profile_dir.parents[1],
            "subnet_id": "subnet_000",
            "subnet_index": 0,
            "structure_hash": "hash",
            "groups": [],
            "profile": {"layer_precision_assignment": {}},
            "profile_id": "profile_000",
            "profile_index": 0,
            "profile_dir": profile_dir,
            "existing_hashes": set(),
        }
    )

    assert result["status"] == "physical_structure_preflight_failed"
    assert calls == []



def test_pilot_latency_report_reads_flat_latency_summary() -> None:
    import tools.latency_lut.run_v11_random_deployment_aware_pilot_engine_eval as pilot

    report = {"latency_summary": {"forward_mean_ms": 3.8, "forward_p50_ms": 2.1, "forward_p90_ms": 2.4}}

    assert pilot._latency_value(report, "forward_latency_ms", "mean") == 3.8
    assert pilot._latency_value(report, "forward_latency_ms", "p50") == 2.1
    assert pilot._latency_value(report, "forward_latency_ms", "p90") == 2.4


def test_precision_realization_exempts_reformat_copy_but_keeps_real_conv_mismatch(tmp_path: Path) -> None:
    import tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder as builder

    profile_dir = tmp_path / "profile_000"
    profile_dir.mkdir()
    builder.write_json(
        profile_dir / "trt_layer_info.json",
        {
            "Layers": [
                {
                    "Name": "/stem/Conv",
                    "LayerType": "CaskConvolution",
                    "Metadata": "[ONNX Layer: stem_Conv_stem_weight_QuantizeLinear]\u001e[ONNX Layer: /stem/Conv]",
                    "Inputs": [{"Format/Datatype": "Float"}],
                    "Outputs": [{"Format/Datatype": "Float"}],
                    "Weights": {"Type": "Float"},
                    "TacticName": "sm80_fp32_tactic",
                },
                {
                    "Name": "/branch_a/Conv",
                    "LayerType": "CaskConvolution",
                    "Metadata": "[ONNX Layer: /branch_a/Conv]",
                    "Inputs": [{"Format/Datatype": "Float"}],
                    "Outputs": [{"Format/Datatype": "Float"}],
                    "Weights": {"Type": "Float"},
                    "TacticName": "sm80_fp32_tactic",
                },
                {
                    "Name": "/stem/Conv_output_0 copy",
                    "LayerType": "Reformat",
                    "Metadata": "[ONNX Layer: /Concat]",
                    "Inputs": [{"Format/Datatype": "Float"}],
                    "Outputs": [{"Format/Datatype": "Float"}],
                    "TacticName": "0x00000000000003e8",
                },
                {
                    "Name": "/fuse/Conv",
                    "LayerType": "CaskConvolution",
                    "Metadata": "[ONNX Layer: /fuse/Conv]",
                    "Inputs": [{"Format/Datatype": "Float"}],
                    "Outputs": [{"Format/Datatype": "Float"}],
                    "Weights": {"Type": "Float"},
                    "TacticName": "sm50_xmma_conv_fprop_fp32",
                },
            ],
            "Bindings": ["input.1", "17"],
        },
    )

    report = builder.run_engine_precision_stage(
        {
            "profile_dir": profile_dir,
            "profile": {
                "layer_precision_assignment": {
                    "stem": "int8",
                    "branch_a": "int8",
                    "fuse": "fp16",
                }
            },
        }
    )

    assert report["hidden_reformat_count"] == 1
    assert report["unresolved_layer_mapping_count"] == 0
    assert report["precision_realization_mismatch_count"] == 3
    assert [
        {k: row[k] for k in ("trt_layer_name", "original_module_name", "expected", "engine_actual_precision")}
        for row in report["mismatch_layers"]
    ] == [
        {
            "trt_layer_name": "/stem/Conv",
            "original_module_name": "stem",
            "expected": "int8",
            "engine_actual_precision": "fp32",
        },
        {
            "trt_layer_name": "/branch_a/Conv",
            "original_module_name": "branch_a",
            "expected": "int8",
            "engine_actual_precision": "fp32",
        },
        {
            "trt_layer_name": "/fuse/Conv",
            "original_module_name": "fuse",
            "expected": "fp16",
            "engine_actual_precision": "fp32",
        },
    ]
    assert report["mismatch_layers"][0]["failure_reason"] == "requested_int8_compute_not_realized"
    assert report["mismatch_layers"][1]["failure_reason"] == "requested_int8_compute_not_realized"


def test_precision_realization_fails_int8_profile_without_qdq_or_mapped_int8_layer(tmp_path: Path) -> None:
    import tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder as builder

    profile_dir = tmp_path / "profile_001"
    profile_dir.mkdir()
    builder.write_json(profile_dir / "trt_layer_info.json", {"Layers": []})
    builder.write_json(
        profile_dir / "qdq_insert_report.json",
        {"success": True, "inserted_qdq_nodes": [], "matched_int8_precision_groups": [], "unmatched_int8_precision_groups": []},
    )
    profile = {
        "precision_group_assignments": {
            "pg_encoder": {"member_modules": ["encoder_m1"], "final_precision": "int8", "requested_precision": "int8"},
        },
        "layer_precision_assignment": {"encoder_m1": "int8"},
    }

    report = builder.run_engine_precision_stage({"profile_dir": profile_dir, "profile": profile})

    assert report["precision_realization_passed"] is False
    assert report["status"] == "engine_precision_mismatch"
    assert report["precision_realization_mismatch_count"] >= 1
    assert any(row["original_module_name"] == "encoder_m1" for row in report["mismatch_layers"])


def test_precision_realization_checks_fused_int8_conv_even_when_name_contains_qdq(tmp_path: Path) -> None:
    import tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder as builder

    profile_dir = tmp_path / "profile_001"
    profile_dir.mkdir()
    builder.write_json(
        profile_dir / "trt_layer_info.json",
        {
            "Layers": [
                {
                    "Name": "onnx::Conv_1059 + layer0_layer0_0_conv1_Conv_backbone_m1_resnet_layer0_0_conv1_weight_QuantizeLinear + /layer0/layer0.0/conv1/Conv",
                    "LayerType": "CaskConvolution",
                    "Metadata": "[ONNX Layer: layer0_layer0_0_conv1_Conv_backbone_m1_resnet_layer0_0_conv1_weight_QuantizeLinear]\u001e[ONNX Layer: /layer0/layer0.0/conv1/Conv]",
                    "Inputs": [{"Format/Datatype": "Int8"}],
                    "Outputs": [{"Format/Datatype": "Int8"}],
                    "Weights": {"Type": "Int8"},
                    "TacticName": "i8i8_i8i32",
                }
            ]
        },
    )
    builder.write_json(
        profile_dir / "qdq_insert_report.json",
        {"success": True, "inserted_qdq_nodes": [{"module_name": "backbone_m1.resnet.layer0.0.conv1"}]},
    )
    profile = {
        "precision_group_assignments": {
            "pg_backbone": {"member_modules": ["backbone_m1.resnet.layer0.0.conv1"], "final_precision": "int8", "requested_precision": "int8"},
        },
        "layer_precision_assignment": {"backbone_m1.resnet.layer0.0.conv1": "int8"},
    }

    report = builder.run_engine_precision_stage({"profile_dir": profile_dir, "profile": profile})

    assert report["precision_realization_passed"] is True
    assert report["precision_realization_mismatch_count"] == 0
    assert report["qdq_folded_count"] == 1


def test_precision_realization_accepts_cask_convolution_int8_compute(tmp_path: Path) -> None:
    import tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder as builder

    profile_dir = tmp_path / "profile_001"
    profile_dir.mkdir()
    builder.write_json(
        profile_dir / "trt_layer_info.json",
        {
            "Layers": [
                {
                    "Name": "/dense/Conv",
                    "LayerType": "CaskConvolution",
                    "Metadata": "[ONNX Layer: /dense/Conv]",
                    "Inputs": [{"Format/Datatype": "Int8"}],
                    "Outputs": [{"Format/Datatype": "Int8"}],
                    "Weights": {"Type": "Int8"},
                    "TacticName": "sm75_i8i8_i8i32",
                }
            ]
        },
    )
    builder.write_json(
        profile_dir / "qdq_insert_report.json",
        {"success": True, "inserted_qdq_nodes": [{"module_name": "dense"}]},
    )

    report = builder.run_engine_precision_stage(
        {
            "profile_dir": profile_dir,
            "profile": {
                "precision_group_assignments": {
                    "pg_dense": {"member_modules": ["dense"], "final_precision": "int8"},
                },
                "layer_precision_assignment": {"dense": "int8"},
            },
        }
    )

    assert report["precision_realization_passed"] is True
    assert report["precision_realization_mismatch_count"] == 0
    assert report["int8_realized_layer_count"] == 1
    assert report["int8_compute_fp16_boundary_count"] == 0
    assert report["int8_realized_layers"][0]["realization_status"] == "int8_realized"


def test_precision_realization_accepts_cask_jit_conv_int8_compute_with_fp16_boundary(tmp_path: Path) -> None:
    import tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder as builder

    profile_dir = tmp_path / "profile_001"
    profile_dir.mkdir()
    builder.write_json(
        profile_dir / "trt_layer_info.json",
        {
            "Layers": [
                {
                    "Name": "/residual/Conv + PWN(/residual/Add + /residual/relu/Relu)",
                    "LayerType": "CaskJitConv",
                    "Metadata": "[ONNX Layer: /residual/Conv]\u001e[ONNX Layer: /residual/output_DequantizeLinear]\u001e[ONNX Layer: /residual/Add]\u001e[ONNX Layer: /residual/relu/Relu]",
                    "Inputs": [{"Format/Datatype": "Int8"}],
                    "Outputs": [{"Format/Datatype": "Half"}],
                    "Weights": {"Type": "Int8"},
                    "TacticName": "sm80_int8int8_fused_residual",
                }
            ]
        },
    )
    builder.write_json(
        profile_dir / "qdq_insert_report.json",
        {"success": True, "inserted_qdq_nodes": [{"module_name": "residual"}]},
    )

    report = builder.run_engine_precision_stage(
        {
            "profile_dir": profile_dir,
            "profile": {
                "precision_group_assignments": {
                    "pg_residual": {"member_modules": ["residual"], "final_precision": "int8"},
                },
                "layer_precision_assignment": {"residual": "int8"},
            },
        }
    )

    assert report["precision_realization_passed"] is True
    assert report["precision_realization_mismatch_count"] == 0
    assert report["int8_realized_layer_count"] == 1
    assert report["int8_compute_fp16_boundary_count"] == 1
    assert report["int8_realized_layers"][0]["realization_status"] == "int8_compute_fp16_boundary"
    assert report["int8_realized_layers"][0]["boundary_dtype"] == "fp16"


def test_precision_realization_keeps_true_mismatch_when_int8_compute_absent(tmp_path: Path) -> None:
    import tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder as builder

    profile_dir = tmp_path / "profile_001"
    profile_dir.mkdir()
    builder.write_json(
        profile_dir / "trt_layer_info.json",
        {
            "Layers": [
                {
                    "Name": "/dense/Conv",
                    "LayerType": "CaskConvolution",
                    "Metadata": "[ONNX Layer: /dense/Conv]",
                    "Inputs": [{"Format/Datatype": "Half"}],
                    "Outputs": [{"Format/Datatype": "Half"}],
                    "Weights": {"Type": "Half"},
                    "TacticName": "sm80_f16f16",
                }
            ]
        },
    )
    builder.write_json(
        profile_dir / "qdq_insert_report.json",
        {"success": True, "inserted_qdq_nodes": [{"module_name": "dense"}]},
    )

    report = builder.run_engine_precision_stage(
        {
            "profile_dir": profile_dir,
            "profile": {
                "precision_group_assignments": {
                    "pg_dense": {"member_modules": ["dense"], "final_precision": "int8"},
                },
                "layer_precision_assignment": {"dense": "int8"},
            },
        }
    )

    assert report["precision_realization_passed"] is False
    assert report["precision_realization_mismatch_count"] == 1
    assert report["mismatch_layers"][0]["failure_reason"] == "requested_int8_compute_not_realized"


def test_stale_profile_failure_report_older_than_success_is_resolved_without_blocking_completion(tmp_path: Path) -> None:
    import os
    import tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder as builder

    profile_dir = tmp_path / "profile_001"
    profile_dir.mkdir()
    builder.write_json(profile_dir / "mixed_precision_profile.json", {})
    builder.write_json(profile_dir / "qdq_insert_report.json", {"success": True})
    (profile_dir / "engine.plan").write_bytes(b"engine")
    builder.write_json(profile_dir / "engine_structure_check_report.json", {"structure_check_passed": True})
    builder.write_json(profile_dir / "engine_precision_realization_report.json", {"precision_realization_passed": True})
    builder.write_json(profile_dir / "trt_smoke_report.json", {"success": True})
    builder.write_json(
        profile_dir / "eval_report.json",
        {
            "eval_success": True,
            "synthetic_used": False,
            "validation_dataloader_used": True,
            "evaluated_frames": 1000,
        },
    )
    builder.write_json(
        profile_dir / "profile_failure_report.json",
        {"stage_failed": "engine_precision_mismatch", "failure_reason": "mismatch_count=29"},
    )
    old = (profile_dir / "eval_report.json").stat().st_mtime - 60.0
    os.utime(profile_dir / "profile_failure_report.json", (old, old))

    assert builder._profile_success_complete(profile_dir) is True
    status = builder.resolve_stale_profile_failure_report(profile_dir, resolved_by="unit_test")

    assert status["status"] == "stale_failure_report"
    resolved = json.loads((profile_dir / "profile_failure_report.json").read_text(encoding="utf-8"))
    assert resolved["resolved"] is True
    assert resolved["resolution_status"] == "stale_failure_report"


def test_per_engine_resume_skips_success_and_reevaluates_missing_eval(tmp_path: Path, monkeypatch) -> None:
    import tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder as builder

    _make_sampled_subnets(tmp_path, subnets=1)
    calls: list[str] = []

    def ok_eval(ctx):
        calls.append(str(ctx["profile_id"]))
        return {
            "eval_success": True,
            "status": "eval_success",
            "synthetic_used": False,
            "validation_dataloader_used": True,
            "evaluated_frames": 1000,
            "latency_summary": {},
            "ap": {},
            "per_frame_rows": [],
        }

    def ok_profile(ctx):
        return {"profile_legality_passed": True, "status": "ok"}

    def ok_onnx(ctx):
        builder.write_json(ctx["profile_dir"] / "qdq_insert_report.json", {"success": True})
        return {"success": True, "status": "ok"}

    def ok_build(ctx):
        (ctx["profile_dir"] / "engine.plan").write_bytes(b"engine")
        return {"build_success": True, "success": True, "status": "ok", "engine_path": str(ctx["profile_dir"] / "engine.plan"), "uses_int8_flag": False}

    def ok_structure(ctx):
        report = {"structure_check_passed": True, "status": "ok"}
        builder.write_json(ctx["profile_dir"] / "engine_structure_check_report.json", report)
        return report

    def ok_precision(ctx):
        report = {"precision_realization_passed": True, "status": "ok", "precision_realization_mismatch_count": 0}
        builder.write_json(ctx["profile_dir"] / "engine_precision_realization_report.json", report)
        return report

    def ok_smoke(ctx):
        report = {"success": True, "status": "ok", "synthetic_used": False, "validation_dataloader_used": True}
        builder.write_json(ctx["profile_dir"] / "trt_smoke_report.json", report)
        return report

    monkeypatch.setattr(builder, "run_profile_legality_stage", ok_profile)
    monkeypatch.setattr(builder, "run_onnx_qdq_stage", ok_onnx)
    monkeypatch.setattr(builder, "run_engine_build_stage", ok_build)
    monkeypatch.setattr(builder, "run_engine_structure_stage", ok_structure)
    monkeypatch.setattr(builder, "run_engine_precision_stage", ok_precision)
    monkeypatch.setattr(builder, "run_trt_smoke_stage", ok_smoke)
    monkeypatch.setattr(builder, "run_real_eval_stage", ok_eval)

    args = [
        "--mode",
        "expand-profiles-per-engine",
        "--source-dir",
        str(tmp_path),
        "--max-subnets",
        "1",
        "--precision-profiles-per-subnet",
        "2",
        "--eval-engines",
        "true",
        "--resume",
    ]
    assert builder.main(args) == 0
    calls.clear()
    eval_report = tmp_path / "subnets/subnet_000/profile_001/eval_report.json"
    payload = json.loads(eval_report.read_text(encoding="utf-8"))
    payload["eval_success"] = False
    payload["failure_reason"] = "real_trt_validation_eval_not_wired"
    eval_report.write_text(json.dumps(payload), encoding="utf-8")
    builder.write_csv(
        tmp_path / "failure_summary.csv",
        [
            {
                "subnet_id": "subnet_000",
                "profile_id": "profile_001",
                "stage_failed": "trt_smoke_failed",
                "failure_reason": "stale_failure",
            }
        ],
    )

    assert builder.main(args) == 0
    assert calls == ["profile_001"]
    stale_rows = list(csv.DictReader((tmp_path / "failure_summary.csv").open()))
    assert stale_rows == []


def test_dual_gpu_worker_queue_skips_complete_profiles(tmp_path: Path) -> None:
    import tools.latency_lut.run_v11_dual_gpu_profile_workers as dual
    import tools.latency_lut.run_v11_mixed_precision_lut_dataset_builder as builder

    _make_sampled_subnets(tmp_path, subnets=2)
    complete = tmp_path / "subnets/subnet_000/profile_000"
    complete.mkdir(parents=True, exist_ok=True)
    builder.write_json(complete / "mixed_precision_profile.json", {"profile_id": "profile_000"})
    builder.write_json(complete / "qdq_insert_report.json", {"success": True})
    (complete / "engine.plan").write_bytes(b"engine")
    builder.write_json(complete / "engine_structure_check_report.json", {"structure_check_passed": True})
    builder.write_json(complete / "engine_precision_realization_report.json", {"precision_realization_passed": True})
    builder.write_json(complete / "trt_smoke_report.json", {"success": True})
    builder.write_json(
        complete / "eval_report.json",
        {
            "eval_success": True,
            "synthetic_used": False,
            "validation_dataloader_used": True,
            "evaluated_frames": 1000,
        },
    )

    args = dual.parse_args(
        [
            "--source-dir",
            str(tmp_path),
            "--max-subnets",
            "2",
            "--precision-profiles-per-subnet",
            "2",
            "--resume",
        ]
    )

    jobs, subnet_rows = dual.build_pending_jobs(args)

    assert len(subnet_rows) == 2
    assert [(job.subnet_id, job.profile_id) for job in jobs] == [
        ("subnet_000", "profile_001"),
        ("subnet_001", "profile_000"),
        ("subnet_001", "profile_001"),
    ]


def test_dual_gpu_worker_result_updates_indexes_serially(tmp_path: Path) -> None:
    import tools.latency_lut.run_v11_dual_gpu_profile_workers as dual

    result = {
        "status": "eval_success",
        "index_row": {
            "subnet_id": "subnet_000",
            "profile_id": "profile_001",
            "build_success": True,
            "eval_success": True,
            "engine_structure_check_passed": True,
            "precision_realization_passed": True,
            "smoke_success": True,
        },
        "eval_row": {
            "subnet_id": "subnet_000",
            "profile_id": "profile_001",
            "eval_success": True,
            "synthetic_used": False,
            "validation_dataloader_used": True,
        },
        "component_rows": [{"subnet_id": "subnet_000", "profile_id": "profile_001", "trt_layer_name": "layer"}],
        "training_row": {"subnet_id": "subnet_000", "profile_id": "profile_001", "label_available": True},
    }
    state = dual.create_empty_index_state(
        tmp_path,
        subnet_rows=[{"subnet_id": "subnet_000", "structure_hash": "hash"}],
        args=dual.parse_args(["--source-dir", str(tmp_path), "--precision-profiles-per-subnet", "4"]),
    )

    dual.apply_worker_result(state, result, total_subnets=1, profiles_per_subnet=4)

    assert list(csv.DictReader((tmp_path / "mixed_precision_profile_index.csv").open()))[0]["profile_id"] == "profile_001"
    progress = json.loads((tmp_path / "progress_state.json").read_text(encoding="utf-8"))
    assert progress["profiles_completed"] == 1
    assert progress["engines_eval_success"] == 1
    assert progress["current_profile"] == ""
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["engine_eval_success_count"] == 1


def test_dual_gpu_failure_threshold_marks_stopped_early(tmp_path: Path) -> None:
    import tools.latency_lut.run_v11_dual_gpu_profile_workers as dual

    args = dual.parse_args(
        [
            "--source-dir",
            str(tmp_path),
            "--max-consecutive-failures",
            "2",
            "--max-total-gate-failures",
            "3",
        ]
    )
    state = dual.create_empty_index_state(tmp_path, subnet_rows=[], args=args)
    failure = {
        "status": "engine_precision_mismatch",
        "index_row": {"subnet_id": "subnet_000", "profile_id": "profile_000", "build_success": True},
        "eval_row": {"subnet_id": "subnet_000", "profile_id": "profile_000", "eval_success": False},
        "component_rows": [],
        "training_row": {"subnet_id": "subnet_000", "profile_id": "profile_000", "label_available": False},
        "failure": {
            "subnet_id": "subnet_000",
            "profile_id": "profile_000",
            "stage_failed": "engine_precision_mismatch",
            "failure_reason": "mismatch_count=1",
            "traceback": "",
            "recovery_action": "fix_precision_constraints",
        },
    }

    should_stop = dual.apply_worker_result(state, failure, total_subnets=1, profiles_per_subnet=4)
    assert should_stop is False
    should_stop = dual.apply_worker_result(
        state,
        {
            **failure,
            "index_row": {**failure["index_row"], "profile_id": "profile_001"},
            "eval_row": {**failure["eval_row"], "profile_id": "profile_001"},
            "training_row": {**failure["training_row"], "profile_id": "profile_001"},
            "failure": {**failure["failure"], "profile_id": "profile_001"},
        },
        total_subnets=1,
        profiles_per_subnet=4,
    )

    assert should_stop is True
    progress = json.loads((tmp_path / "progress_state.json").read_text(encoding="utf-8"))
    assert progress["stopped_early"] is True
    assert progress["stop_reason"].startswith("failure_threshold")
    assert progress["failure_stage_counts"] == {"engine_precision_mismatch": 2}


def test_dual_gpu_worker_command_uses_single_gpu_and_worker_mode(tmp_path: Path) -> None:
    import tools.latency_lut.run_v11_dual_gpu_profile_workers as dual

    job = dual.ProfileJob(
        subnet_dir=tmp_path / "subnets/subnet_000",
        subnet_id="subnet_000",
        subnet_index=0,
        structure_hash="hash",
        profile_id="profile_001",
        profile_index=1,
        profile_dir=tmp_path / "subnets/subnet_000/profile_001",
    )
    args = dual.parse_args(
        [
            "--source-dir",
            str(tmp_path),
            "--gpus",
            "6,7",
            "--precision-profiles-per-subnet",
            "4",
            "--eval-engines",
            "true",
        ]
    )

    cmd, env, log_path = dual.build_worker_command(args, job, gpu="7", result_path=job.profile_dir / "worker_result.json")

    assert "--worker" in cmd
    assert "--worker-profile-index" in cmd
    assert "1" in cmd
    assert env["CUDA_VISIBLE_DEVICES"] == "7"
    assert log_path.name.endswith(".log")
