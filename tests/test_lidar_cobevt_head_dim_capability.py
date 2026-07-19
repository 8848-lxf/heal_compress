from __future__ import annotations

from dataclasses import replace
from pathlib import Path

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


def test_exported_attention_roles_have_stable_onnx_names(tmp_path):
    import onnx

    from search.model_families.lidar_cobevt.head_dim_synthetic import (
        export_synthetic_onnx,
    )

    candidate = _small_candidate(
        d_qk=16, d_v=16, family="uniform", profile="P0_strict_fp32"
    )
    destination = tmp_path / "named_roles.onnx"
    export_synthetic_onnx(candidate, destination)
    names = {node.name for node in onnx.load(str(destination)).graph.node}

    assert "/qk_matmul/MatMul" in names
    assert "/softmax/Softmax" in names
    assert "/av_matmul/MatMul" in names


def test_export_traces_one_group_but_records_real_512_group_build_shape(tmp_path):
    import onnx

    from search.model_families.lidar_cobevt.head_dim_capability import (
        HeadDimCandidate,
    )
    from search.model_families.lidar_cobevt.head_dim_synthetic import (
        export_synthetic_onnx,
    )

    candidate = HeadDimCandidate(
        graph_variant="projection_attention",
        structure_family="uniform",
        d_qk=24,
        d_v=24,
        precision_profile="P0_strict_fp32",
        tensorrt_version="10.9.0.34",
        gpu_architecture="sm89",
    )
    destination = tmp_path / "dynamic_groups.onnx"
    report = export_synthetic_onnx(candidate, destination, trace_window_groups=1)
    model = onnx.load(str(destination))
    dimensions = model.graph.input[0].type.tensor_type.shape.dim

    assert report["trace_input_shapes"]["x"] == [1, 32, 256]
    assert report["target_input_shapes"]["x"] == [512, 32, 256]
    assert dimensions[0].dim_param == "window_groups"


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


def test_parity_metrics_cover_general_qk_softmax_and_av_signals():
    from search.reporting.cobevt_head_dim_capability import (
        attention_tensor_parity,
    )

    reference = torch.tensor(
        [[[[2.0, 1.0, -1.0, -2.0], [0.0, 1.0, 2.0, 3.0]]]]
    )
    candidate = reference.clone()
    qk = attention_tensor_parity(reference, candidate, role="qk_score")
    assert qk["cosine_similarity"] == 1.0
    assert qk["top1_agreement"] == 1.0
    assert qk["top4_overlap"] == 1.0
    assert qk["sign_flip_ratio"] == 0.0

    probability = torch.softmax(reference, dim=-1)
    softmax = attention_tensor_parity(
        probability, probability.clone(), role="softmax"
    )
    assert softmax["row_sum_max_error"] < 1.0e-6
    assert softmax["kl_divergence"] == 0.0
    assert softmax["js_divergence"] == 0.0

    av = attention_tensor_parity(reference, reference * 0.5, role="av_output")
    assert av["relative_l2_error"] > 0.0
    assert av["channel_energy_relative_error"] > 0.0


def test_search_contract_uses_only_runtime_precision_identity_evidence():
    from search.reporting.cobevt_head_dim_capability import (
        derive_head_dim_search_contract,
    )

    common = {
        "structure_family": "uniform",
        "runtime_success": True,
        "precision_identity": True,
        "numerical_safe": True,
        "accuracy_safe": True,
        "support_class": "supported_primitive",
    }
    rows = [
        {
            **common,
            "d_qk": 24,
            "d_v": 24,
            "precision_profile": "P1_strict_fp16_native",
            "fused_mha_detected": False,
        },
        {
            **common,
            "d_qk": 16,
            "d_v": 16,
            "precision_profile": "P2_f3_mixed",
            "fused_mha_detected": False,
        },
        {
            **common,
            "d_qk": 32,
            "d_v": 32,
            "precision_profile": "P4_int8_native_attention",
            "fused_mha_detected": True,
            "support_class": "supported_fused_mha",
        },
        {
            **common,
            "d_qk": 12,
            "d_v": 12,
            "precision_profile": "P4_int8_native_attention",
            "precision_identity": False,
            "support_class": "supported_with_fallback",
        },
    ]

    contract = derive_head_dim_search_contract(
        rows,
        hardware_scope={
            "gpu_model": "RTX 4090",
            "compute_capability": "8.9",
            "tensorrt_version": "10.9.0.34",
            "cuda_version": "11.8",
        },
    )

    assert contract["uniform_attention"][
        "fp16_primitive_supported_head_dims"
    ] == [24]
    assert contract["uniform_attention"]["f3_supported_head_dims"] == [16]
    assert contract["uniform_attention"]["int8_fused_supported_head_dims"] == [
        32
    ]
    assert 12 not in contract["search_space_recommendation"][
        "legal_structural_widths"
    ]
    assert {
        "d_qk": 12,
        "d_v": 12,
        "precision_profile": "P4_int8_native_attention",
        "reason": "precision_fallback",
    } in contract["search_space_recommendation"][
        "forbidden_precision_shape_pairs"
    ]


def test_capability_matrix_writer_preserves_failures_and_machine_readable_rows(
    tmp_path,
):
    import json

    from search.reporting.cobevt_head_dim_capability import (
        write_capability_matrix,
    )

    rows = [
        {
            "candidate_id": "ok",
            "structure_family": "uniform",
            "d_qk": 24,
            "d_v": 24,
            "precision_profile": "P1_strict_fp16_native",
            "support_class": "supported_primitive",
            "failure_reason": "",
        },
        {
            "candidate_id": "failed",
            "structure_family": "uniform",
            "d_qk": 6,
            "d_v": 6,
            "precision_profile": "P4_int8_native_attention",
            "support_class": "unsupported_build",
            "failure_reason": "no tactic",
        },
    ]
    paths = write_capability_matrix(tmp_path, rows)

    assert json.loads(paths["json"].read_text()) == rows
    assert "no tactic" in paths["csv"].read_text()
    assert "supported_primitive" in paths["markdown"].read_text()
    assert "unsupported_build" in paths["markdown"].read_text()


def test_trtexec_command_is_strongly_typed_detailed_and_fresh(tmp_path):
    from search.orchestration.lidar_cobevt_head_dim_capability import (
        build_trtexec_command,
    )

    candidate = _small_candidate(
        d_qk=16, d_v=16, family="uniform", profile="P4_int8_native_attention"
    )
    command = build_trtexec_command(
        candidate,
        trtexec=tmp_path / "trtexec",
        onnx_path=tmp_path / "attention.onnx",
        engine_path=tmp_path / "engine.plan",
        layer_info_path=tmp_path / "engine_layer_info.json",
        profile_path=tmp_path / "engine_profile.json",
    )

    assert "--stronglyTyped" in command
    assert "--noTF32" in command
    assert "--noBuilderCache" in command
    assert "--profilingVerbosity=detailed" in command
    assert "--dumpLayerInfo" in command
    assert "--dumpProfile" in command
    assert any(value.startswith("--exportLayerInfo=") for value in command)
    assert any(value.startswith("--exportProfile=") for value in command)
    assert "--fp16" not in command
    assert "--int8" not in command
    assert not any("precisionConstraints" in value for value in command)
    shapes = next(value for value in command if value.startswith("--optShapes="))
    assert "x:3x4x16" in shapes
    assert "attention_mask:3x4x4" in shapes
    assert "relative_position_bias" not in shapes


def test_prepare_run_writes_all_candidates_and_rejects_nonempty_output(tmp_path):
    import json

    import pytest

    from search.orchestration.lidar_cobevt_head_dim_capability import (
        prepare_capability_run,
    )

    output = tmp_path / "run"
    result = prepare_capability_run(
        output,
        environment={
            "tensorrt_version": "10.9.0.34",
            "gpu_architecture": "sm89",
            "gpu_model": "NVIDIA GeForce RTX 4090",
            "compute_capability": "8.9",
            "cuda_version": "11.8",
            "driver_version": "580.105.08",
        },
        code_commit="a9c5151",
    )

    candidates = json.loads((output / "candidate_matrix.json").read_text())
    assert result["candidate_count"] == 282
    assert len(candidates) == 282
    assert (output / "run_manifest.json").is_file()
    assert (output / "resolved_config.yaml").is_file()
    assert (output / "synthetic").is_dir()
    assert (output / "real_cobevt").is_dir()
    assert (output / "failures").is_dir()

    with pytest.raises(RuntimeError, match="capability_output_not_empty"):
        prepare_capability_run(
            output,
            environment={
                "tensorrt_version": "10.9.0.34",
                "gpu_architecture": "sm89",
            },
            code_commit="a9c5151",
        )


def test_export_phase_writes_complete_candidate_provenance_without_engine(tmp_path):
    import json

    from search.orchestration.lidar_cobevt_head_dim_capability import (
        export_capability_candidates,
        prepare_capability_run,
    )

    output = tmp_path / "run"
    prepare_capability_run(
        output,
        environment={
            "tensorrt_version": "10.9.0.34",
            "gpu_architecture": "sm89",
        },
        code_commit="a9c5151",
    )
    candidate_id = "projection_attention__uniform__qk24_v24__P2_f3_mixed"
    result = export_capability_candidates(
        output,
        only_candidate_ids={candidate_id},
        trace_window_groups=1,
    )

    assert result == {"attempted": 1, "failed": 0, "succeeded": 1}
    candidate_dirs = list((output / "synthetic").rglob("candidate.json"))
    assert len(candidate_dirs) == 1
    candidate_dir = candidate_dirs[0].parent
    candidate = json.loads((candidate_dir / "candidate.json").read_text())
    requested = json.loads(
        (candidate_dir / "requested_precision.json").read_text()
    )
    export = json.loads((candidate_dir / "export_report.json").read_text())
    assert candidate["candidate_id"] == candidate_id
    assert requested["qk_matmul"] == "FP32"
    assert requested["q_projection"] == "FP16"
    assert export["onnx_export_success"] is True
    assert export["onnx_sha256"]
    assert (candidate_dir / "attention.onnx").is_file()
    assert not list(output.rglob("*.plan"))


def test_diagnostic_onnx_exposes_qk_softmax_and_av_without_changing_production(
    tmp_path,
):
    import onnx

    from search.model_families.lidar_cobevt.head_dim_synthetic import (
        export_synthetic_diagnostic_onnx,
        export_synthetic_onnx,
    )

    candidate = _small_candidate(
        d_qk=16, d_v=16, family="uniform", profile="P2_f3_mixed"
    )
    production = tmp_path / "production.onnx"
    diagnostic = tmp_path / "diagnostic.onnx"
    export_synthetic_onnx(candidate, production, trace_window_groups=1)
    report = export_synthetic_diagnostic_onnx(
        candidate, diagnostic, trace_window_groups=1
    )

    production_outputs = [row.name for row in onnx.load(str(production)).graph.output]
    diagnostic_outputs = [row.name for row in onnx.load(str(diagnostic)).graph.output]
    assert production_outputs == ["output"]
    assert diagnostic_outputs == [
        "output",
        "qk_score",
        "softmax_output",
        "av_output",
    ]
    assert report["diagnostic_only"] is True
    assert report["latency_eligible"] is False


def test_runtime_input_cases_are_deterministic_and_cover_numeric_boundaries():
    from search.model_families.lidar_cobevt.head_dim_capability import (
        HeadDimCandidate,
    )
    from search.model_families.lidar_cobevt.head_dim_synthetic import (
        SYNTHETIC_INPUT_CASES,
        build_synthetic_inputs,
    )

    candidate = HeadDimCandidate(
        graph_variant="core_attention",
        structure_family="uniform",
        d_qk=8,
        d_v=8,
        precision_profile="P0_strict_fp32",
        tensorrt_version="10.9.0.34",
        gpu_architecture="sm89",
        num_heads=2,
        embed_dim=16,
        token_length=4,
        window_shape=(2, 2),
        window_groups=2,
    )
    assert SYNTHETIC_INPUT_CASES == (
        "deterministic",
        "random_normal",
        "large_range",
        "small_margin_qk",
        "cancellation_heavy",
    )
    first = build_synthetic_inputs(candidate, input_case="deterministic")
    second = build_synthetic_inputs(candidate, input_case="deterministic")
    assert all(torch.equal(first[name], second[name]) for name in first)

    normal = build_synthetic_inputs(candidate, input_case="random_normal")
    large = build_synthetic_inputs(candidate, input_case="large_range")
    assert large["q"].abs().max() > normal["q"].abs().max() * 8

    small_margin = build_synthetic_inputs(candidate, input_case="small_margin_qk")
    assert torch.max(torch.abs(small_margin["q"] - small_margin["k"])) < 1.0e-3

    cancellation = build_synthetic_inputs(
        candidate, input_case="cancellation_heavy"
    )
    assert torch.allclose(cancellation["q"][..., 0::2], -cancellation["q"][..., 1::2])


def test_build_phase_records_engine_inspector_and_never_loads_plugin(tmp_path):
    import json
    from types import SimpleNamespace

    from search.orchestration.lidar_cobevt_head_dim_capability import (
        build_capability_candidates,
        export_capability_candidates,
        prepare_capability_run,
    )

    trt_root = tmp_path / "TensorRT-10.9"
    trtexec = trt_root / "targets/x86_64-linux-gnu/bin/trtexec"
    trtexec.parent.mkdir(parents=True)
    trtexec.write_bytes(b"trtexec")
    output = tmp_path / "run"
    environment = {
        "tensorrt_version": "10.9.0.34",
        "gpu_architecture": "sm89",
        "tensorrt_root": str(trt_root),
        "trtexec_path": str(trtexec),
        "trtexec_sha256": "trtexec-sha",
    }
    prepare_capability_run(output, environment=environment, code_commit="commit")
    candidate_id = "projection_attention__uniform__qk24_v24__P1_strict_fp16_native"
    export_capability_candidates(output, only_candidate_ids={candidate_id})

    def fake_runner(command, **_kwargs):
        assert not any("Plugins" in value or "plugin" in value for value in command)
        engine = next(value.split("=", 1)[1] for value in command if value.startswith("--saveEngine="))
        layers = next(value.split("=", 1)[1] for value in command if value.startswith("--exportLayerInfo="))
        profile = next(value.split("=", 1)[1] for value in command if value.startswith("--exportProfile="))
        Path(engine).write_bytes(b"engine")
        Path(layers).write_text(
            json.dumps(
                {
                    "Layers": [
                        _layer("qk_matmul", "[ONNX Layer: /qk_matmul/MatMul]", "Half"),
                        _layer("softmax", "[ONNX Layer: /softmax/Softmax]", "Half"),
                        _layer("av_matmul", "[ONNX Layer: /av_matmul/MatMul]", "Half"),
                    ]
                }
            )
        )
        Path(profile).write_text("[]")
        return SimpleNamespace(returncode=0, stdout="TensorRT build ok")

    result = build_capability_candidates(
        output,
        environment=environment,
        physical_gpu=2,
        only_candidate_ids={candidate_id},
        command_runner=fake_runner,
    )

    assert result == {"attempted": 1, "failed": 0, "succeeded": 1}
    build_path = next((output / "synthetic").rglob("build_report.json"))
    build = json.loads(build_path.read_text())
    assert build["trt_build_success"] is True
    assert build["fusion_kind"] == "primitive"
    assert build["fused_mha_detected"] is False
    assert build["engine_sha256"]
    assert build["build_signature"]
    assert "--stronglyTyped" in build["builder_command"]

    diagnostic_result = build_capability_candidates(
        output,
        environment=environment,
        physical_gpu=2,
        only_candidate_ids={candidate_id},
        command_runner=fake_runner,
        diagnostic=True,
    )
    assert diagnostic_result == {"attempted": 1, "failed": 0, "succeeded": 1}
    diagnostic_path = next(
        (output / "synthetic").rglob("diagnostic_build_report.json")
    )
    diagnostic = json.loads(diagnostic_path.read_text())
    assert diagnostic["diagnostic_only"] is True
    assert diagnostic["latency_eligible"] is False
    assert Path(diagnostic["engine_path"]).name == "diagnostic_engine.plan"
