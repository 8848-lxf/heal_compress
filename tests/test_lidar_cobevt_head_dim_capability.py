from __future__ import annotations

from dataclasses import replace

import torch


def test_candidate_matrix_covers_requested_shape_families_and_profiles():
    from search.model_families.lidar_cobevt.head_dim_capability import (
        build_synthetic_candidate_matrix,
    )

    rows = build_synthetic_candidate_matrix(
        tensorrt_version="10.9.0.34",
        gpu_architecture="sm89",
    )

    assert len(rows) == 282
    assert {row.structure_family for row in rows} == {
        "uniform",
        "qk_only",
        "v_only",
    }
    uniform_widths = {
        row.d_qk
        for row in rows
        if row.structure_family == "uniform"
    }
    assert uniform_widths == {
        4,
        6,
        8,
        10,
        12,
        14,
        16,
        20,
        24,
        28,
        32,
        40,
        48,
        56,
        64,
        80,
        96,
        128,
    }
    assert {
        row.precision_profile
        for row in rows
        if row.structure_family == "uniform"
        and row.graph_variant == "projection_attention"
    } == {
        "P0_strict_fp32",
        "P1_strict_fp16_native",
        "P2_f3_mixed",
        "P3_int8_projections_qk_fp32",
        "P4_int8_native_attention",
    }


def test_candidate_shape_semantics_and_same_shape_reference_are_explicit():
    from search.model_families.lidar_cobevt.head_dim_capability import (
        HeadDimCandidate,
    )

    row = HeadDimCandidate(
        graph_variant="projection_attention",
        structure_family="qk_only",
        d_qk=24,
        d_v=32,
        precision_profile="P2_f3_mixed",
        tensorrt_version="10.9.0.34",
        gpu_architecture="sm89",
    )

    assert row.num_heads == 8
    assert row.embed_dim == 256
    assert row.q_projection_out == 192
    assert row.k_projection_out == 192
    assert row.v_projection_out == 256
    assert row.out_projection_in == 256
    assert row.out_projection_out == 256
    assert row.same_shape_fp32_reference_id.endswith("__P0_strict_fp32")
    assert len(row.candidate_hash) == 64


def test_candidate_hash_owns_shape_precision_trt_and_gpu_architecture():
    from search.model_families.lidar_cobevt.head_dim_capability import (
        HeadDimCandidate,
    )

    row = HeadDimCandidate(
        graph_variant="core_attention",
        structure_family="uniform",
        d_qk=24,
        d_v=24,
        precision_profile="P1_strict_fp16_native",
        tensorrt_version="10.9.0.34",
        gpu_architecture="sm89",
    )
    changes = (
        {"d_qk": 28, "d_v": 28},
        {"precision_profile": "P2_f3_mixed"},
        {"tensorrt_version": "10.9.1"},
        {"gpu_architecture": "sm90"},
        {"graph_variant": "projection_attention"},
    )

    for values in changes:
        assert replace(row, **values).candidate_hash != row.candidate_hash


def _small_candidate(*, d_qk: int, d_v: int, family: str, profile: str):
    from search.model_families.lidar_cobevt.head_dim_capability import (
        HeadDimCandidate,
    )

    return HeadDimCandidate(
        graph_variant="projection_attention",
        structure_family=family,
        d_qk=d_qk,
        d_v=d_v,
        precision_profile=profile,
        tensorrt_version="10.9.0.34",
        gpu_architecture="sm89",
        num_heads=2,
        embed_dim=16,
        token_length=4,
        window_shape=(2, 2),
        window_groups=3,
    )


def test_projection_graph_shapes_follow_independent_qk_and_v_dimensions():
    from search.model_families.lidar_cobevt.head_dim_synthetic import (
        build_synthetic_graph,
    )

    qk_only = _small_candidate(
        d_qk=12, d_v=32, family="qk_only", profile="P0_strict_fp32"
    )
    graph, inputs, _manifest = build_synthetic_graph(qk_only)

    assert graph.q_proj.weight.shape == (24, 16)
    assert graph.k_proj.weight.shape == (24, 16)
    assert graph.v_proj.weight.shape == (64, 16)
    assert graph.out_proj.weight.shape == (16, 64)
    output = graph(**inputs)
    assert output.shape == (3, 4, 16)
    assert torch.isfinite(output).all()

    v_only = _small_candidate(
        d_qk=32, d_v=12, family="v_only", profile="P0_strict_fp32"
    )
    graph, inputs, _manifest = build_synthetic_graph(v_only)
    assert graph.q_proj.weight.shape == (64, 16)
    assert graph.v_proj.weight.shape == (24, 16)
    assert graph.out_proj.weight.shape == (16, 24)
    assert graph(**inputs).shape == (3, 4, 16)


def test_f3_manifest_keeps_only_qk_scale_and_matmul_fp32():
    from search.model_families.lidar_cobevt.head_dim_synthetic import (
        build_synthetic_graph,
    )

    candidate = _small_candidate(
        d_qk=16, d_v=16, family="uniform", profile="P2_f3_mixed"
    )
    graph, inputs, manifest = build_synthetic_graph(candidate)
    output = graph(**inputs)

    assert torch.isfinite(output).all()
    assert manifest["q_projection"] == "FP16"
    assert manifest["k_projection"] == "FP16"
    assert manifest["v_projection"] == "FP16"
    assert manifest["qk_scale"] == "FP32"
    assert manifest["qk_matmul"] == "FP32"
    assert manifest["softmax"] == "FP16"
    assert manifest["av_matmul"] == "FP16"
    assert manifest["out_projection"] == "FP16"
    assert manifest["qk_input_cast"] == "FP16_TO_FP32"


def test_explicit_int8_profiles_export_real_qdq(tmp_path):
    import onnx

    from search.model_families.lidar_cobevt.head_dim_synthetic import (
        export_synthetic_onnx,
    )

    counts = {}
    for profile in (
        "P3_int8_projections_qk_fp32",
        "P4_int8_native_attention",
    ):
        candidate = _small_candidate(
            d_qk=16, d_v=16, family="uniform", profile=profile
        )
        destination = tmp_path / f"{profile}.onnx"
        report = export_synthetic_onnx(candidate, destination)
        model = onnx.load(str(destination))
        node_types = [node.op_type for node in model.graph.node]
        counts[profile] = (
            node_types.count("QuantizeLinear"),
            node_types.count("DequantizeLinear"),
        )
        assert report["onnx_export_success"] is True
        assert report["onnx_sha256"]
        assert counts[profile][0] > 0
        assert counts[profile][1] > 0

    assert counts["P4_int8_native_attention"][0] > counts[
        "P3_int8_projections_qk_fp32"
    ][0]


def _layer(name, metadata, dtype, *, tactic=""):
    return {
        "Name": name,
        "LayerType": "kgen" if "mha" in name.lower() else "gemm",
        "TacticName": tactic,
        "Metadata": metadata,
        "Inputs": [{"Name": "input", "Format/Datatype": dtype}],
        "Outputs": [{"Name": "output", "Format/Datatype": dtype}],
    }


def test_inspector_parser_distinguishes_primitive_from_complete_fused_mha():
    from search.reporting.cobevt_head_dim_capability import (
        inspect_attention_layers,
    )

    primitive = inspect_attention_layers(
        [
            _layer("qk_matmul", "[ONNX Layer: /qk_matmul/MatMul]", "Half"),
            _layer("softmax", "[ONNX Layer: /softmax/Softmax]", "Half"),
            _layer("av_matmul", "[ONNX Layer: /av_matmul/MatMul]", "Half"),
        ]
    )
    assert primitive["fused_mha_detected"] is False
    assert primitive["fusion_kind"] == "primitive"
    assert primitive["realized_qk_precision"] == "FP16"
    assert primitive["realized_softmax_precision"] == "FP16"
    assert primitive["realized_av_precision"] == "FP16"

    fused = inspect_attention_layers(
        [
            _layer(
                "_gemm_mha_v2_attention",
                "[ONNX Layer: /qk_matmul/MatMul]"
                "[ONNX Layer: /softmax/Softmax]"
                "[ONNX Layer: /av_matmul/MatMul]",
                "Half",
                tactic="_gemm_mha_v2_sm89_fp16",
            )
        ]
    )
    assert fused["fused_mha_detected"] is True
    assert fused["fusion_kind"] == "complete_fused_mha"
    assert fused["attention_execution_layer_count"] == 1


def test_support_class_requires_runtime_precision_identity_and_real_fusion():
    from search.reporting.cobevt_head_dim_capability import (
        classify_support,
    )

    base = {
        "onnx_export_success": True,
        "trt_build_success": True,
        "runtime_success": True,
        "precision_identity": True,
        "fused_mha_detected": False,
    }
    assert classify_support(base) == "supported_primitive"
    assert classify_support({**base, "fused_mha_detected": True}) == (
        "supported_fused_mha"
    )
    assert classify_support({**base, "precision_identity": False}) == (
        "supported_with_fallback"
    )
    assert classify_support({**base, "runtime_success": False}) == (
        "unsupported_build"
    )
    assert classify_support({**base, "trt_build_success": False}) == (
        "unsupported_build"
    )
    assert classify_support({**base, "onnx_export_success": False}) == (
        "unsupported_export"
    )


def test_requested_realized_audit_reports_fallback_and_unknown_accumulators():
    from search.reporting.cobevt_head_dim_capability import (
        audit_requested_realized_precision,
    )

    requested = {
        "q_projection": "INT8",
        "k_projection": "INT8",
        "v_projection": "INT8",
        "qk_matmul": "INT8",
        "softmax": "FP32",
        "av_matmul": "INT8",
        "out_projection": "INT8",
    }
    realized = {
        "q_projection": "INT8",
        "k_projection": "INT8",
        "v_projection": "INT8",
        "qk_matmul": "FP16",
        "softmax": "FP32",
        "av_matmul": "FP16",
        "out_projection": "INT8",
    }

    audit = audit_requested_realized_precision(requested, realized)

    assert audit["precision_identity"] is False
    assert audit["fallback_count"] == 2
    assert audit["fallback_roles"] == ["av_matmul", "qk_matmul"]
    assert audit["qk_accumulator_precision"] == "unknown"
    assert audit["av_accumulator_precision"] == "unknown"


def test_tensorrt_evidence_rejects_root_or_version_mismatch(tmp_path):
    import json

    import pytest

    from search.orchestration.lidar_cobevt_head_dim_capability import (
        resolve_tensorrt_evidence,
    )

    trt_root = tmp_path / "TensorRT-10.9"
    trtexec = trt_root / "targets/x86_64-linux-gnu/bin/trtexec"
    trtexec.parent.mkdir(parents=True)
    trtexec.write_bytes(b"trtexec")
    previous = tmp_path / "previous"
    previous.mkdir()
    (previous / "run_manifest.json").write_text(
        json.dumps(
            {
                "environment": {
                    "resolved_tensorrt_root": str(trt_root),
                    "tensorrt_version": "10.9",
                    "cuda_version": "11.8",
                }
            }
        )
    )
    report = previous / "candidate/build_report.json"
    report.parent.mkdir()
    report.write_text(
        json.dumps(
            {
                "builder_command": {
                    "command": [str(trtexec), "--stronglyTyped"]
                }
            }
        )
    )

    resolved = resolve_tensorrt_evidence(
        previous,
        python_tensorrt_version="10.9.0.34",
        trtexec_version="10.9.0",
    )
    assert resolved["tensorrt_root"] == str(trt_root.resolve())
    assert resolved["trtexec_sha256"]

    with pytest.raises(RuntimeError, match="tensorrt_version_mismatch"):
        resolve_tensorrt_evidence(
            previous,
            python_tensorrt_version="10.10.0",
            trtexec_version="10.9.0",
        )

    other = tmp_path / "other/targets/x86_64-linux-gnu/bin/trtexec"
    other.parent.mkdir(parents=True)
    other.write_bytes(b"other")
    report.write_text(
        json.dumps(
            {"builder_command": {"command": [str(other), "--stronglyTyped"]}}
        )
    )
    with pytest.raises(RuntimeError, match="tensorrt_root_mismatch"):
        resolve_tensorrt_evidence(
            previous,
            python_tensorrt_version="10.9.0.34",
            trtexec_version="10.9.0",
        )


def test_build_signature_owns_candidate_onnx_qdq_trtexec_and_architecture():
    from search.orchestration.lidar_cobevt_head_dim_capability import (
        capability_build_signature,
    )

    fields = {
        "candidate_hash": "candidate",
        "onnx_sha256": "onnx",
        "qdq_scale_hash": "scale",
        "trtexec_sha256": "trtexec",
        "tensorrt_version": "10.9.0.34",
        "gpu_architecture": "sm89",
    }
    baseline = capability_build_signature(**fields)
    assert len(baseline) == 64
    for key, value in fields.items():
        changed = dict(fields)
        changed[key] = value + "-other"
        assert capability_build_signature(**changed) != baseline


def test_trtexec_shapes_use_real_cobevt_window_and_token_layout():
    from search.model_families.lidar_cobevt.head_dim_capability import (
        HeadDimCandidate,
    )
    from search.orchestration.lidar_cobevt_head_dim_capability import (
        candidate_input_shapes,
    )

    projection = HeadDimCandidate(
        graph_variant="projection_attention",
        structure_family="qk_only",
        d_qk=24,
        d_v=32,
        precision_profile="P2_f3_mixed",
        tensorrt_version="10.9.0.34",
        gpu_architecture="sm89",
    )
    assert candidate_input_shapes(projection) == {
        "x": (512, 32, 256),
        "attention_mask": (512, 32, 32),
        "relative_position_bias": (1, 8, 32, 32),
    }

    core = replace(
        projection,
        graph_variant="core_attention",
        structure_family="uniform",
        d_v=24,
    )
    assert candidate_input_shapes(core) == {
        "q": (512, 8, 32, 24),
        "k": (512, 8, 32, 24),
        "v": (512, 8, 32, 24),
    }
