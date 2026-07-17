from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Dropout(), nn.Linear(256, 256))


class BaseWindowAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.heads = 8
        self.to_qkv = nn.Linear(256, 768, bias=False)
        self.to_out = nn.Sequential(nn.Linear(256, 256), nn.Dropout())


class SplitAttn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(256, 256, bias=False)
        self.bn1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 768, bias=False)


class HGTCavAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.heads = 8
        self.q_linears = nn.ModuleList([nn.Linear(256, 256), nn.Linear(256, 256)])
        self.k_linears = nn.ModuleList([nn.Linear(256, 256), nn.Linear(256, 256)])
        self.v_linears = nn.ModuleList([nn.Linear(256, 256), nn.Linear(256, 256)])
        self.a_linears = nn.ModuleList([nn.Linear(256, 256), nn.Linear(256, 256)])
        self.relation_att = nn.Parameter(torch.randn(4, 8, 32, 32))
        self.relation_msg = nn.Parameter(torch.randn(4, 8, 32, 32))


class V2XFusionBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = HGTCavAttention()


class FakeV2XViT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone_m1 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1))
        self.deblock = nn.ConvTranspose2d(64, 128, 2, stride=2)
        self.fusion_net = nn.Module()
        self.fusion_net.block = V2XFusionBlock()
        self.fusion_net.window = BaseWindowAttention()
        self.fusion_net.split_attn = SplitAttn()
        self.fusion_net.ff = FeedForward()
        self.cls_head = nn.Conv2d(256, 2, 1)


def _config():
    return {
        "train_params": {"max_cav": 2},
        "cav_lidar_range": [-102.4, -51.2, -3.5, 102.4, 51.2, 1.5],
        "heter": {
            "modality_setting": {
                "m1": {
                    "preprocess": {
                        "args": {"voxel_size": [0.4, 0.4, 4], "max_voxel_test": 70000}
                    }
                }
            }
        },
        "model": {
            "core_method": "heter_model_baseline",
            "args": {
                "ego_modality": "m1",
                "fusion_method": "v2xvit",
                "lidar_range": [-102.4, -51.2, -3.5, 102.4, 51.2, 1.5],
                "m1": {"core_method": "point_pillar"},
            },
        },
    }


def test_v2xvit_registry_detects_without_touching_lidar_pyramid() -> None:
    from search.model_family import detect_model_family, registered_model_families

    assert "heal_lidar_v2xvit" in registered_model_families()
    assert detect_model_family(_config()).family_id == "heal_lidar_v2xvit"


def test_v2xvit_audit_tracks_module_and_functional_weighted_ops() -> None:
    from search.model_family import get_model_family

    audit = get_model_family("heal_lidar_v2xvit").audit(FakeV2XViT(), _config())
    by_id = {row.canonical_id: row for row in audit.weighted_ops}
    assert "functional::fusion_net.block.attention.relation_att" in by_id
    assert "functional::fusion_net.block.attention.relation_msg" in by_id
    assert by_id["module::backbone_m1.0"].weight_axis == 0
    assert by_id["module::deblock"].weight_axis == 1
    assert by_id["module::fusion_net.window.to_qkv"].allowed_precisions == ("FP32", "FP16")
    assert by_id["module::fusion_net.window.to_qkv"].potential_precisions[-1] == "INT8"
    assert by_id["module::cls_head"].production_enabled is False


def test_v2xvit_audit_exposes_head_ffn_merge_and_plugin_gates() -> None:
    from search.model_family import get_model_family

    audit = get_model_family("heal_lidar_v2xvit").audit(FakeV2XViT(), _config())
    kinds = {row.domain_kind for row in audit.pruning_domains}
    assert "transformer_ffn_hidden_width" in kinds
    assert "whole_attention_head_bundle" in kinds
    assert "whole_heterogeneous_attention_head_bundle" in kinds
    assert all(
        row.production_enabled
        for row in audit.pruning_domains
        if row.domain_kind == "transformer_ffn_hidden_width"
    )
    assert all(
        not row.production_enabled
        for row in audit.pruning_domains
        if row.domain_kind != "transformer_ffn_hidden_width"
    )
    assert any(row.merge_kind == "transformer_residual_add" for row in audit.merge_boundaries)
    plugin = audit.plugin_requirements[0]
    assert plugin.plugin_key == "pointpillar_scatter_trt"
    assert plugin.required is True
    assert audit.input_contract["grid_size_xyz"] == [512, 256, 1]
    assert "canonical_onnx_export_not_yet_smoked" in audit.blockers


def test_v2xvit_export_policy_requires_explicit_fixed_k() -> None:
    from search.model_family.export import HealV2XViTExportPolicy

    policy = HealV2XViTExportPolicy(fixed_k=128, max_agents=2)
    assert policy.fixed_k == 128
    assert policy.output_names == ("cls_preds", "reg_preds", "dir_preds")
    try:
        HealV2XViTExportPolicy(fixed_k=0)
    except ValueError as error:
        assert str(error) == "v2xvit_fixed_k_must_be_positive"
    else:
        raise AssertionError("zero fixed_k must be rejected")


def test_v2xvit_export_policy_loads_only_valid_frozen_manifest(tmp_path: Path) -> None:
    import json

    from search.model_family.calibration_manifest import (
        V2XVIT_TRAIN200_SCHEMA,
        finalize_v2xvit_train_manifest,
    )
    from search.model_family.export import HealV2XViTExportPolicy

    manifest = finalize_v2xvit_train_manifest(
        {
            "schema_version": V2XVIT_TRAIN200_SCHEMA,
            "family_id": "heal_lidar_v2xvit",
            "split": "train",
            "fixed_k_contract": {"alignment": 256},
            "input_contract": {"max_agents": 2, "modality": "m1"},
            "samples": [
                {
                    "dataset_index": index,
                    "vehicle_frame_id": f"frame-{index}",
                    "record_len": 2,
                    "voxel_count": 1000 + index,
                }
                for index in range(200)
            ],
        }
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    policy = HealV2XViTExportPolicy.from_frozen_train_manifest(path)
    assert policy.fixed_k == 1280
    assert policy.max_agents == 2
    assert policy.modality == "m1"
    assert policy.calibration_manifest_hash == manifest["manifest_hash"]


def test_v2xvit_train_manifest_selection_and_fixed_k_are_deterministic() -> None:
    from search.model_family.calibration_manifest import (
        V2XVIT_TRAIN200_SCHEMA,
        evenly_spaced_indices,
        finalize_v2xvit_train_manifest,
        sample_seed,
        validate_v2xvit_train_manifest,
    )

    assert evenly_spaced_indices(4811, 200)[:5] == [0, 24, 48, 73, 97]
    assert evenly_spaced_indices(4811, 200)[-5:] == [4713, 4737, 4762, 4786, 4810]
    assert sample_seed(20260717, 4496) == 20265213
    manifest = finalize_v2xvit_train_manifest(
        {
            "schema_version": V2XVIT_TRAIN200_SCHEMA,
            "family_id": "heal_lidar_v2xvit",
            "split": "train",
            "fixed_k_contract": {"alignment": 256},
            "input_contract": {"max_points_per_voxel": 32, "max_agents": 2},
            "samples": [
                {
                    "dataset_index": 0,
                    "vehicle_frame_id": "000010",
                    "record_len": 1,
                    "voxel_count": 5793,
                },
                {
                    "dataset_index": 4496,
                    "vehicle_frame_id": "015547",
                    "record_len": 2,
                    "voxel_count": 27666,
                },
            ],
        },
        expected_sample_count=2,
    )
    assert manifest["fixed_k_contract"]["value"] == 27904
    assert manifest["fixed_k_contract"]["alignment_margin_voxels"] == 238
    assert manifest["fixed_k_contract"]["truncated_sample_count"] == 0
    assert manifest["fixed_k_contract"]["full_train_split_upper_bound_claimed"] is False
    validate_v2xvit_train_manifest(manifest, expected_sample_count=2)


def test_v2xvit_train_manifest_hash_rejects_sample_tampering() -> None:
    import copy

    from search.model_family.calibration_manifest import (
        V2XVIT_TRAIN200_SCHEMA,
        finalize_v2xvit_train_manifest,
        validate_v2xvit_train_manifest,
    )

    manifest = finalize_v2xvit_train_manifest(
        {
            "schema_version": V2XVIT_TRAIN200_SCHEMA,
            "family_id": "heal_lidar_v2xvit",
            "split": "train",
            "fixed_k_contract": {"alignment": 256},
            "samples": [
                {
                    "dataset_index": 3,
                    "vehicle_frame_id": "frame-3",
                    "record_len": 2,
                    "voxel_count": 1000,
                }
            ],
        },
        expected_sample_count=1,
    )
    tampered = copy.deepcopy(manifest)
    tampered["samples"][0]["voxel_count"] = 999
    try:
        validate_v2xvit_train_manifest(tampered, expected_sample_count=1)
    except ValueError as error:
        assert str(error) == "v2xvit_train_manifest_hash_mismatch"
    else:
        raise AssertionError("tampered calibration manifest must be rejected")


def test_v2xvit_ffn_domain_width_materializes_exact_ranked_mask() -> None:
    from search.candidate import CandidateGenotype
    from search.canonicalization import SearchSpaceSpec, canonicalize_candidate
    from search.model_family import get_model_family
    from search.model_family.pruning import materialize_v2xvit_ffn_pruning
    from search.model_family.search_space import (
        build_ranked_v2xvit_ffn_domains,
        build_v2xvit_ffn_atomic_units,
    )

    model = FakeV2XViT()
    audit = get_model_family("heal_lidar_v2xvit").audit(model, _config())
    units, _capabilities = build_v2xvit_ffn_atomic_units(model, audit)
    assert len(units) == 256
    scores = {row.stable_id: float(row.root_indices[0]) for row in units}
    domains = build_ranked_v2xvit_ffn_domains(units, scores)
    assert len(domains) == 1
    domain = domains[0]
    assert domain.legal_widths == (64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256)
    space = SearchSpaceSpec(
        pruning_unit_ids=[row.stable_id for row in units],
        precision_layer_ids=[],
        pruning_domains=tuple(domains),
    )
    phenotype = canonicalize_candidate(
        CandidateGenotype(pruning_width_genes={domain.domain_id: 240}), space
    )
    result = materialize_v2xvit_ffn_pruning(model, phenotype, domains)
    first = result.model.get_submodule("fusion_net.ff.net.0")
    second = result.model.get_submodule("fusion_net.ff.net.3")
    assert first.out_features == 240
    assert second.in_features == 240
    assert len(phenotype.pruned_unit_ids) == 16
    assert result.parameter_count_after < result.parameter_count_before
    replay = materialize_v2xvit_ffn_pruning(model, phenotype, domains)
    replay.model.load_state_dict(result.model.state_dict(), strict=True)


def test_v2xvit_precision_groups_come_from_active_canonical_capabilities() -> None:
    from search.model_family import get_model_family
    from search.model_family.search_space import build_v2xvit_quantization_groups

    model = FakeV2XViT()
    audit = get_model_family("heal_lidar_v2xvit").audit(model, _config())
    active = ["backbone_m1.0", "fusion_net.ff.net.0", "cls_head"]
    groups = build_v2xvit_quantization_groups(
        model, audit, active_module_paths=active
    )
    assert len(groups) == 3
    by_module = {row.module_paths[0]: row for row in groups}
    assert by_module["backbone_m1.0"].allowed_precisions == (
        "FP32",
        "FP16",
        "INT8",
    )
    assert by_module["backbone_m1.0"].metadata["production_int8_admitted"] is True
    assert by_module["fusion_net.ff.net.0"].allowed_precisions == ("FP32", "FP16")
    assert by_module["cls_head"].metadata["production_int8_admitted"] is False
    assert all(row.group_id.startswith("v2xvit_qg::module::") for row in groups)


def test_v2xvit_identity_sttf_preserves_grid_sample_and_masks_agents() -> None:
    from search.model_family.export.heal_v2xvit import _identity_sttf_and_roi

    value = torch.randn(1, 2, 8, 16, 4)
    mask = torch.tensor([[1.0, 0.0]])
    observed, communication = _identity_sttf_and_roi(value, mask, use_roi_mask=True)
    assert observed.shape == value.shape
    assert torch.allclose(observed, value, atol=1.0e-5, rtol=1.0e-5)
    assert communication.shape == (1, 8, 16, 1, 2)
    assert torch.count_nonzero(communication[..., 0]).item() == 8 * 16
    assert torch.count_nonzero(communication[..., 1]).item() == 0


def test_v2xvit_search_readiness_blocks_unverified_genes() -> None:
    from search.model_family import build_model_family_search_readiness, get_model_family
    from search.model_family.readiness import (
        PRUNING_REQUIRED_EVIDENCE,
        QUANTIZATION_REQUIRED_EVIDENCE,
    )

    audit = get_model_family("heal_lidar_v2xvit").audit(FakeV2XViT(), _config())
    smoke = build_model_family_search_readiness(
        audit,
        {
            "strict_state_dict_load": True,
            "onnx_export": True,
            "onnx_checker": True,
            "tensor_parity": True,
        },
    )
    assert smoke.quantization_search_ready is False
    assert smoke.pruning_search_ready is False
    assert smoke.ready_precision_gene_ids == ()
    assert smoke.ready_pruning_domain_ids == ()
    assert "strongly_typed_parser" in smoke.missing_quantization_evidence
    assert "physical_materializer" in smoke.missing_pruning_evidence

    all_evidence = {key: True for key in (*QUANTIZATION_REQUIRED_EVIDENCE, *PRUNING_REQUIRED_EVIDENCE)}
    gated = build_model_family_search_readiness(audit, all_evidence)
    assert "module::backbone_m1.0" in gated.ready_precision_gene_ids
    assert "module::fusion_net.window.to_qkv" not in gated.ready_precision_gene_ids
    assert gated.ready_pruning_domain_ids == ("ffn_hidden::fusion_net.ff",)
    assert gated.joint_search_ready is True


def test_v2xvit_onnx_mapping_resolves_functional_weight_and_inactive_type_branch(
    tmp_path: Path,
) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    import numpy as np

    from search.model_family.contracts import ModelFamilyAudit, WeightedOpCapability
    from search.model_family.onnx_mapping import build_v2xvit_onnx_mapping

    def capability(
        canonical_id: str,
        module_path: str,
        source_kind: str,
        *,
        parameter_name: str = "",
    ) -> WeightedOpCapability:
        return WeightedOpCapability(
            canonical_id=canonical_id,
            module_path=module_path,
            op_type="Linear" if source_kind == "module" else "FunctionalEinsumWeight",
            source_kind=source_kind,
            weight_shape=(4, 4),
            allowed_precisions=("FP32", "FP16"),
            potential_precisions=("FP32", "FP16", "INT8"),
            default_precision="FP16",
            weight_granularity="per_output_channel",
            weight_axis=0,
            input_scale_owner="input",
            output_boundary="output",
            production_enabled=False,
            metadata={"parameter_name": parameter_name} if parameter_name else {},
        )

    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["x", "linear.weight"], ["linear_y"], name="linear/MatMul"),
            helper.make_node(
                "Gather", ["model.block.relation_att", "relation_index"], ["selected_relation"], name="relation/Gather", axis=0
            ),
            helper.make_node(
                "Einsum", ["linear_y", "selected_relation"], ["z"], name="attention/Einsum", equation="ij,jk->ik"
            ),
        ],
        "v2xvit_mapping",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
        [helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 4])],
        initializer=[
            numpy_helper.from_array(np.ones((4, 4), dtype=np.float32), name="linear.weight"),
            numpy_helper.from_array(
                np.ones((1, 4, 4), dtype=np.float32), name="model.block.relation_att"
            ),
            numpy_helper.from_array(np.asarray([0], dtype=np.int64), name="relation_index"),
        ],
    )
    path = tmp_path / "model.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)
    audit = ModelFamilyAudit(
        schema_version="test",
        family_id="heal_lidar_v2xvit",
        model_type="Fake",
        parameter_count=1,
        weighted_ops=(
            capability("module::linear", "linear", "module"),
            capability(
                "functional::block.relation_att",
                "block",
                "functional_parameter",
                parameter_name="block.relation_att",
            ),
            capability("module::block.q_linears.1", "block.q_linears.1", "module"),
        ),
        pruning_domains=(),
        merge_boundaries=(),
        deployment_operators=(),
        plugin_requirements=(),
        input_contract={},
        blockers=(),
    )
    mapping = build_v2xvit_onnx_mapping(
        path,
        audit,
        [
            {
                "module_path": "linear",
                "module_type": "Linear",
                "call_index": 0,
                "mapped_onnx_op_type": "MatMul",
                "weight_shape": [4, 4],
                "groups": 1,
            }
        ],
    )
    by_id = {row.canonical_id: row for row in mapping.weighted_entries}
    assert by_id["module::linear"].mapping_status == "active_mapped"
    assert by_id["functional::block.relation_att"].mapping_status == "active_functional_weight_mapped"
    assert by_id["module::block.q_linears.1"].mapping_status == "inactive_by_export_specialization"
    assert mapping.unresolved == ()
    assert mapping.metadata["active_weighted_capability_count"] == 2
    assert mapping.metadata["inactive_weighted_capability_count"] == 1
    assert mapping.metadata["realized_graph_mapping_complete"] is True
