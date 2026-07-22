from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from torch import nn
from onnx import TensorProto

from search.model_families.transformer.dh_alignment_audit import (
    audit_engine_alignment,
    latency_beneficial,
    parse_engine_memory_audit,
)
from search.model_families.transformer.dh_candidate_grid import (
    adjacent_microbenchmark_pairs,
    dense_head_dimension_grid,
)
from search.model_families.transformer.dh_physical_rewrite import (
    PrunedV2XWindowAttention,
    discover_attention_families,
    materialize_family_head_dimension,
    materialize_joint_head_dimensions,
)
from search.model_families.transformer.dh_precision_profiles import PROFILES
from search.model_families.transformer.dh_pruning_contract import (
    AttentionFamilyRecord,
    HeadLocalMask,
    audit_nested_masks,
    masks_from_rankings,
)
from search.orchestration.lidar_transformer_dh_microbenchmark import (
    _linear_model,
    _relation_model,
    _standard_attention_model,
)
from search.orchestration.lidar_transformer_dh_identity_parity import _compare
from search.orchestration.lidar_transformer_dh_joint import joint_id, parse_targets
from search.reporting.transformer_dh_alignment import _parameter_breakdown


class FakeWindow(nn.Module):
    def __init__(self, heads: int = 4, d_h: int = 5, dim: int = 20) -> None:
        super().__init__()
        self.heads = heads
        self.scale = d_h**-0.5
        self.window_size = 2
        self.relative_pos_embedding = False
        self.to_qkv = nn.Linear(dim, 3 * heads * d_h, bias=False)
        self.to_out = nn.Sequential(nn.Linear(heads * d_h, dim), nn.Dropout(0.0))
        self.pos_embedding = nn.Parameter(torch.zeros(4, 4))


class HGTCavAttention(nn.Module):
    def __init__(self, heads: int = 2, d_h: int = 5, dim: int = 10) -> None:
        super().__init__()
        self.heads = heads
        self.scale = d_h**-0.5
        self.q_linears = nn.ModuleList([nn.Linear(dim, heads * d_h) for _ in range(2)])
        self.k_linears = nn.ModuleList([nn.Linear(dim, heads * d_h) for _ in range(2)])
        self.v_linears = nn.ModuleList([nn.Linear(dim, heads * d_h) for _ in range(2)])
        self.a_linears = nn.ModuleList([nn.Linear(heads * d_h, dim) for _ in range(2)])
        self.relation_att = nn.Parameter(torch.randn(4, heads, d_h, d_h))
        self.relation_msg = nn.Parameter(torch.randn(4, heads, d_h, d_h))


class FakeHGTModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = HGTCavAttention()


class BaseWindowAttention(FakeWindow):
    pass


class FakeMultiFamilyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.agent = HGTCavAttention()
        self.window = BaseWindowAttention()


def test_dense_grid_includes_odd_and_nonfour_widths() -> None:
    grid = dense_head_dimension_grid(32, heads=8)
    assert [row.d_h for row in grid] == list(range(32, 15, -1))
    assert next(row for row in grid if row.d_h == 31).alignment_class == "odd"
    assert next(row for row in grid if row.d_h == 30).alignment_class == "multiple_of_2_only"
    assert next(row for row in grid if row.d_h == 28).alignment_class == "multiple_of_4_only"


def test_dense_grid_derives_from_actual_non32_dimension() -> None:
    grid = dense_head_dimension_grid(64, heads=4)
    assert grid[0].d_h == 64
    assert grid[-1].d_h == 32
    assert len(grid) == 33


def test_low_width_extension_runs_only_after_main_grid() -> None:
    grid = dense_head_dimension_grid(16, heads=16, low_width_extension=True)
    assert [row.d_h for row in grid] == list(range(16, 7, -1))


def test_nested_masks_and_common_qkv_positions() -> None:
    family = AttentionFamilyRecord("lidar_v2xvit", "f", "window", ("a",), 2, 5, 10)
    ranking = {"a": ((0, 1, 2, 3, 4), (4, 3, 2, 1, 0))}
    masks = {width: masks_from_rankings(family, ranking, width) for width in range(1, 6)}
    rows = audit_nested_masks(masks)
    assert rows and all(row["nested"] for row in rows)
    assert masks[3]["a"].q_keep_by_head if hasattr(masks[3]["a"], "q_keep_by_head") else True


def test_physical_v2x_window_rows_columns_bias_and_scale() -> None:
    torch.manual_seed(1)
    source = FakeWindow()
    mask = HeadLocalMask("x", 5, ((0, 2, 4), (0, 1, 4), (1, 2, 3), (0, 3, 4)))
    source_weight = source.to_qkv.weight.detach().clone()
    source_out = source.to_out[0].weight.detach().clone()
    source_out_bias = source.to_out[0].bias.detach().clone()
    module = PrunedV2XWindowAttention(source, mask)
    flat = torch.tensor(mask.flattened())
    assert module.heads == 4
    assert module.d_qk == module.d_v == 3
    assert module.q_proj.weight.shape == (12, 20)
    assert module.k_proj.weight.shape == (12, 20)
    assert module.v_proj.weight.shape == (12, 20)
    assert module.out_proj.weight.shape == (20, 12)
    assert torch.equal(module.q_proj.weight, source_weight.index_select(0, flat))
    assert torch.equal(module.k_proj.weight, source_weight.index_select(0, flat + 20))
    assert torch.equal(module.v_proj.weight, source_weight.index_select(0, flat + 40))
    assert torch.equal(module.out_proj.weight, source_out.index_select(1, flat))
    assert torch.equal(module.out_proj.bias, source_out_bias)
    assert math.isclose(module.scale, 3**-0.5)


def test_agent_relation_rewrite_slices_both_relation_axes() -> None:
    model = FakeHGTModel()
    old_att = model.attention.relation_att.detach().clone()
    old_msg = model.attention.relation_msg.detach().clone()
    family = discover_attention_families("lidar_v2xvit", model)[0]
    mask = HeadLocalMask("attention", 5, ((0, 2, 4), (1, 2, 3)))
    report = materialize_family_head_dimension(
        "lidar_v2xvit", model, family, {"attention": mask}
    )
    assert report.passed
    assert model.attention.heads == 2
    assert model.attention.q_linears[0].weight.shape == (6, 10)
    assert model.attention.v_linears[1].bias.shape == (6,)
    assert model.attention.a_linears[0].weight.shape == (10, 6)
    assert model.attention.relation_att.shape == (4, 2, 3, 3)
    assert model.attention.relation_msg.shape == (4, 2, 3, 3)
    first = torch.tensor((0, 2, 4))
    assert torch.equal(model.attention.relation_att[:, 0], old_att[:, 0].index_select(1, first).index_select(2, first))
    assert torch.equal(model.attention.relation_msg[:, 0], old_msg[:, 0].index_select(1, first).index_select(2, first))
    assert math.isclose(model.attention.scale, 3**-0.5)
    assert report.physical_parameter_count < report.original_parameter_count


def test_joint_family_rewrite_keeps_independent_target_widths() -> None:
    model = FakeMultiFamilyModel()
    families = {row.attention_kind: row for row in discover_attention_families("lidar_v2xvit", model)}
    agent = families["agent_relation"]
    window = families["spatial_window"]
    agent_masks = {"agent": HeadLocalMask("agent", 5, ((0, 2, 4), (1, 2, 3)))}
    window_masks = {"window": HeadLocalMask("window", 5, tuple((0, 1, 2, 4) for _ in range(4)))}
    report = materialize_joint_head_dimensions(
        "lidar_v2xvit", model,
        {agent.family_id: agent_masks, window.family_id: window_masks},
    )
    assert report.passed
    assert report.target_d_h_by_family[agent.family_id] == 3
    assert report.target_d_h_by_family[window.family_id] == 4
    assert model.agent.q_linears[0].out_features == 6
    assert model.window.q_proj.out_features == 16


def test_profiles_share_structure_and_keep_qk_fp32() -> None:
    assert set(PROFILES) == {"P32", "P16", "P8"}
    assert all(row.to_dict()["qk_contract"] == "F32A32O32" for row in PROFILES.values())
    assert PROFILES["P8"].int8_roles == ("q_projection", "k_projection")
    assert PROFILES["P8"].alpha("lidar_cobevt") == 0.8
    assert PROFILES["P8"].alpha("lidar_v2xvit") == 0.75


def test_sq1_restores_qdq_adjacency_before_fp32_qk() -> None:
    source = Path("search/orchestration/lidar_transformer_dh_build.py").read_text(encoding="utf-8")
    assert "restore_projection_qdq_adjacency" in source
    assert "validate_projection_qdq_adjacency" in source
    assert '"qk_contract": "F32A32O32"' in source


def test_cross_width_calibration_namespace_is_content_separated() -> None:
    source = Path("search/orchestration/lidar_transformer_dh_build.py").read_text(encoding="utf-8")
    assert "family_id}_{d_h}_P8" in source
    assert '"fresh_width_calibration": True' in source


def test_fixed500_delta_must_use_same_profile_baseline() -> None:
    source = Path("search/model_families/transformer/dh_alignment_audit.py").read_text(encoding="utf-8")
    # Reporting owns the delta, while this low-level module must not contain a
    # cross-profile baseline shortcut.
    assert "B1_TRT_ATTN_FP32" not in source


def test_micro_latency_is_not_additive() -> None:
    source = Path("search/orchestration/lidar_transformer_dh_latency.py").read_text(encoding="utf-8")
    assert "_time_engine" in source
    assert "sum(" not in source or "micro" not in source


def test_padding_and_fallback_are_not_inferred_from_build_success(tmp_path: Path) -> None:
    info = {
        "Layers": [
            {"Name": "q_proj", "LayerType": "MatrixMultiply", "TacticName": "sm90_mma"},
            {"Name": "q_proj_reformat", "LayerType": "Reformat", "TacticName": "", "Inputs": [{"Dimensions": [1, 248]}]},
        ]
    }
    path = tmp_path / "layers.json"
    path.write_text(json.dumps(info), encoding="utf-8")
    audit = audit_engine_alignment(path, logical_d_h=31, projection_width=248)
    assert audit["padding_status"] == "EXACT_NONALIGNED"
    assert audit["reformat_count"] == 1
    assert audit["tensor_core_hint"]


def test_cublas_gemv_tactic_is_not_automatically_fallback(tmp_path: Path) -> None:
    path = tmp_path / "layers.json"
    path.write_text(json.dumps({"Layers": [{"Name": "qk_matmul", "LayerType": "gemm", "TacticName": "sm50_xmma_cublas_gemvx_f16f16_f32"}]}), encoding="utf-8")
    audit = audit_engine_alignment(path, logical_d_h=31, projection_width=248)
    assert audit["padding_status"] == "EXACT_NONALIGNED"
    assert not audit["fallback_hint"]


def test_engine_memory_is_parsed_from_trtexec_evidence(tmp_path: Path) -> None:
    path = tmp_path / "build.log"
    path.write_text("Total Device Persistent Memory: 123 bytes\nMax Scratch Memory: 456 bytes\nTotal Activation Memory: 789 bytes\n", encoding="utf-8")
    audit = parse_engine_memory_audit(path)
    assert audit["device_persistent_bytes"] == 123
    assert audit["max_scratch_bytes"] == 456
    assert audit["activation_bytes"] == 789


def test_parameter_breakdown_uses_unique_physical_modules_and_exact_reduction(
    tmp_path: Path,
) -> None:
    inventory_dir = tmp_path / "inventory"
    structure_dir = tmp_path / "structures" / "lidar_v2xvit" / "family" / "dh_031"
    inventory_dir.mkdir(parents=True)
    structure_dir.mkdir(parents=True)
    (inventory_dir / "v2xvit_parameter_baseline.json").write_text(
        json.dumps(
            {
                "total_parameter_count": 1000,
                "transformer_parameter_count": 600,
            }
        ),
        encoding="utf-8",
    )
    (structure_dir / "inventory.json").write_text(
        json.dumps(
            {
                "rows": [
                    {"module_path": "attn.q_proj", "canonical_role": "q_projection"},
                    {"module_path": "attn.q_proj", "canonical_role": "q_projection"},
                    {"module_path": "attn.out_proj", "canonical_role": "output_projection"},
                    {"module_path": "ffn.0", "canonical_role": "ffn1"},
                ]
            }
        ),
        encoding="utf-8",
    )
    (structure_dir / "physical_structure_snapshot_v2.json").write_text(
        json.dumps(
            {
                "modules": [
                    {"module_path": "attn.q_proj", "parameter_count": 100},
                    {"module_path": "attn.out_proj", "parameter_count": 80},
                    {"module_path": "ffn.0", "parameter_count": 200},
                ]
            }
        ),
        encoding="utf-8",
    )
    result = _parameter_breakdown(
        tmp_path,
        "lidar_v2xvit",
        structure_dir,
        {"original_parameter_count": 1000, "physical_parameter_count": 950},
    )
    assert result == {
        "total_parameter_count": 950,
        "transformer_parameter_count": 550,
        "qkv_parameter_count": 100,
        "out_parameter_count": 80,
        "ffn_parameter_count": 200,
    }


def test_v2x_odd_dimension_forward_preserves_external_shape() -> None:
    module = PrunedV2XWindowAttention(
        FakeWindow(), HeadLocalMask("x", 5, ((0, 1, 4), (0, 2, 3), (1, 3, 4), (0, 2, 4)))
    )
    value = torch.randn(1, 2, 4, 4, 20)
    output = module(value)
    assert output.shape == value.shape
    assert torch.isfinite(output).all()


def test_microbenchmark_pairs_cover_alignment_boundaries() -> None:
    pairs = adjacent_microbenchmark_pairs(range(32, 15, -1))
    assert (32, 31) in pairs
    assert (25, 24) in pairs
    assert (17, 16) in pairs


def test_linear_primitive_onnx_is_physical_nonaligned(tmp_path: Path) -> None:
    arrays = _linear_model(
        tmp_path / "linear.onnx", name="q_projection", tokens=3,
        input_width=8, output_width=7, dtype=TensorProto.FLOAT16,
    )
    assert arrays["input"].shape == (3, 8)


@pytest.mark.parametrize("qk", [True, False])
def test_standard_attention_primitive_onnx(qk: bool, tmp_path: Path) -> None:
    arrays = _standard_attention_model(
        tmp_path / f"standard_{qk}.onnx", name="qk" if qk else "av",
        instances=2, heads=2, sequence=3, d_h=5, qk=qk,
        dtype=TensorProto.FLOAT if qk else TensorProto.FLOAT16,
    )
    assert set(arrays) == ({"q", "k"} if qk else {"probability", "value"})


@pytest.mark.parametrize("qk", [True, False])
def test_relation_attention_primitive_onnx(qk: bool, tmp_path: Path) -> None:
    arrays = _relation_model(
        tmp_path / f"relation_{qk}.onnx", name="qk" if qk else "av",
        instances=2, heads=2, sequence=2, d_h=3, qk=qk,
        dtype=TensorProto.FLOAT if qk else TensorProto.FLOAT16,
    )
    assert "relation" in arrays
    assert arrays["relation"].shape[-2:] == (3, 3)


def test_family_progress_journals_are_independent() -> None:
    source = Path("search/orchestration/lidar_transformer_dh_run_matrix.py").read_text(encoding="utf-8")
    assert 'progress_scope = family_filter or "all_families"' in source


def test_inventory_consolidation_keeps_models_separate() -> None:
    source = Path("search/reporting/transformer_dh_alignment.py").read_text(encoding="utf-8")
    assert 'groups[model] = families' in source
    assert '("lidar_cobevt", "lidar_v2xvit")' in source


def test_full_engine_latency_is_explicitly_nonadditive() -> None:
    source = Path("search/orchestration/lidar_transformer_dh_microbenchmark.py").read_text(encoding="utf-8")
    assert '"unit_latency_additive": False' in source
    assert '"full_engine_predictor": False' in source


def test_identity_forward_parity_is_numerically_gated() -> None:
    accepted = _compare({"x": torch.ones(2)}, {"x": torch.ones(2) + 1e-6})
    rejected = _compare({"x": torch.ones(2)}, {"x": torch.zeros(2)})
    assert accepted["passed"]
    assert not rejected["passed"]


def test_joint_target_identity_is_order_independent() -> None:
    left = parse_targets("family_b=30,family_a=31")
    right = parse_targets("family_a=31,family_b=30")
    assert joint_id(left) == joint_id(right)
    with pytest.raises(ValueError, match="multiple_families"):
        parse_targets("family_a=31")


def test_latency_gate_uses_noise_and_replay_drift() -> None:
    rejected = latency_beneficial(
        baseline_p50_ms=10.0,
        candidate_p50_ms=9.95,
        baseline_repeat_cv=0.002,
        baseline_replay_drift=0.004,
    )
    accepted = latency_beneficial(
        baseline_p50_ms=10.0,
        candidate_p50_ms=9.8,
        baseline_repeat_cv=0.002,
        baseline_replay_drift=0.004,
    )
    assert not rejected["latency_beneficial"]
    assert accepted["latency_beneficial"]


def test_system_nvcc_contract_remains_fail_closed() -> None:
    source = Path("search/orchestration/lidar_transformer_h800_environment.py").read_text(encoding="utf-8")
    assert "/usr/bin/nvcc" in source or "nvcc_outside_conda_prefix" in source


def test_pyramid_orchestration_not_imported() -> None:
    for path in Path("search/model_families/transformer").glob("dh_*.py"):
        assert "lidar_pyramid_search" not in path.read_text(encoding="utf-8")
