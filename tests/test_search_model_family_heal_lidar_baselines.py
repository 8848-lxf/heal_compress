from __future__ import annotations

import pytest
import torch.nn as nn


class PillarVFE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pfn_layers = nn.ModuleList([nn.Module()])
        self.pfn_layers[0].linear = nn.Linear(10, 64, bias=False)


class PointPillarScatter(nn.Module):
    pass


class BaseBEVBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            self._block(64, 64, 4),
            self._block(64, 128, 6),
            self._block(128, 256, 9),
        ])
        self.deblocks = nn.ModuleList([
            self._deblock(64, 1),
            self._deblock(128, 2),
            self._deblock(256, 4),
        ])

    @staticmethod
    def _block(in_channels: int, out_channels: int, count: int) -> nn.Sequential:
        rows: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        for _ in range(count - 1):
            rows.extend((
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ))
        return nn.Sequential(*rows)

    @staticmethod
    def _deblock(in_channels: int, stride: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, 128, stride, stride=stride, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )


class DoubleConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
        )


class DownsampleConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([DoubleConv()])


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pillar_vfe = PillarVFE()
        self.scatter = PointPillarScatter()


class MaxFusion(nn.Module):
    pass


class CoBEVT(nn.Module):
    pass


class PixelWeightLayer(nn.Module):
    def __init__(self, last_channels: int = 1) -> None:
        super().__init__()
        self.conv1_1 = nn.Conv2d(512, 128, 1)
        self.bn1_1 = nn.BatchNorm2d(128)
        self.conv1_2 = nn.Conv2d(128, 32, 1)
        self.bn1_2 = nn.BatchNorm2d(32)
        self.conv1_3 = nn.Conv2d(32, 8, 1)
        self.bn1_3 = nn.BatchNorm2d(8)
        self.conv1_4 = nn.Conv2d(8, last_channels, 1)


class DiscoFusion(nn.Module):
    def __init__(self, last_channels: int = 1) -> None:
        super().__init__()
        self.pixel_weight_layer = PixelWeightLayer(last_channels)


class HeterModelBaseline(nn.Module):
    def __init__(self, fusion: nn.Module) -> None:
        super().__init__()
        self.encoder_m1 = Encoder()
        self.backbone_m1 = BaseBEVBackbone()
        self.shrinker_m1 = DownsampleConv()
        self.fusion_net = fusion
        self.cls_head = nn.Conv2d(256, 2, 1)
        self.reg_head = nn.Conv2d(256, 14, 1)
        self.dir_head = nn.Conv2d(256, 4, 1)


def _config(fusion_method: str) -> dict:
    return {
        "train_params": {"max_cav": 2},
        "heter": {"modality_setting": {"m1": {"preprocess": {"args": {"max_voxel_test": 70000}}}}},
        "model": {
            "core_method": "heter_model_baseline",
            "args": {
                "ego_modality": "m1",
                "fusion_method": fusion_method,
                "anchor_number": 2,
                "dir_args": {"num_bins": 2},
                "m1": {"core_method": "point_pillar"},
            },
        },
    }


def test_baseline_providers_detect_only_their_own_config() -> None:
    from search.model_family import detect_model_family, registered_model_families

    assert {
        "heal_lidar_fcooper", "heal_lidar_disco", "heal_lidar_cobevt"
    } <= set(registered_model_families())
    assert detect_model_family(_config("max")).family_id == "heal_lidar_fcooper"
    assert detect_model_family(_config("disconet")).family_id == "heal_lidar_disco"
    assert detect_model_family(_config("cobevt")).family_id == "heal_lidar_cobevt"
    bad = _config("max")
    bad["model"]["args"]["m1"]["core_method"] = "second"
    with pytest.raises(RuntimeError, match="no_model_family_provider_matches_config"):
        detect_model_family(bad)


def test_baseline_audit_rejects_config_and_real_structure_mismatches() -> None:
    from search.model_family import get_model_family

    fcooper = get_model_family("heal_lidar_fcooper")
    with pytest.raises(RuntimeError, match="config_mismatch"):
        fcooper.audit(HeterModelBaseline(MaxFusion()), _config("disconet"))
    with pytest.raises(RuntimeError, match="fusion_net"):
        fcooper.audit(HeterModelBaseline(DiscoFusion()), _config("max"))


def test_fcooper_audit_lists_every_weighted_op_and_protects_pfn_heads() -> None:
    from search.model_family import get_model_family

    model = HeterModelBaseline(MaxFusion())
    audit = get_model_family("heal_lidar_fcooper").audit(model, _config("max"))
    actual = {f"module::{name}" for name, module in model.named_modules() if name and isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d, nn.Linear))}
    by_id = {row.canonical_id: row for row in audit.weighted_ops}
    assert set(by_id) == actual
    assert by_id["module::encoder_m1.pillar_vfe.pfn_layers.0.linear"].allowed_precisions == ("FP32", "FP16")
    assert by_id["module::cls_head"].output_boundary == "fixed_detection_output_semantics"
    assert by_id["module::cls_head"].production_enabled is False
    assert by_id["module::backbone_m1.blocks.0.0"].production_enabled is True
    assert by_id["module::backbone_m1.deblocks.0.0"].weight_axis == 1
    assert by_id["module::backbone_m1.deblocks.0.0"].production_enabled is True
    production = [row for row in audit.pruning_domains if row.production_enabled]
    assert len(production) == 21
    assert all(len(row.legal_widths) > 1 for row in production)
    protected = {row.domain_id: row for row in audit.pruning_domains if not row.production_enabled}
    assert "protected_output::cls_head" in protected
    assert protected["protected_output::cls_head"].legal_widths == (2,)


def test_fcooper_max_merge_and_fixed_input_plugin_contract() -> None:
    from search.model_family import get_model_family

    audit = get_model_family("heal_lidar_fcooper").audit(HeterModelBaseline(MaxFusion()), _config("max"))
    merge = audit.merge_boundaries[0]
    assert merge.merge_kind == "agentwise_max"
    assert merge.policy == "FP16_merge"
    assert audit.input_contract["fixed_k"] == 29696
    assert audit.input_contract["max_agents"] == 2
    assert audit.plugin_requirements[0].required is True
    assert audit.plugin_requirements[0].compatibility_status == "verified_fixedk29696_strict_fp32_and_strongly_typed_qdq_h800"
    assert all(row.production_enabled for row in audit.deployment_operators)
    assert merge.production_enabled is True
    assert audit.blockers == ("baseline_train200_entropy_calibration_and_int8_accuracy_evidence_not_available",)


def test_cobevt_provider_records_transformer_merge_and_unified_strategy() -> None:
    from search.model_family import get_model_family

    audit = get_model_family("heal_lidar_cobevt").audit(
        HeterModelBaseline(CoBEVT()), _config("cobevt")
    )
    assert audit.merge_boundaries[0].merge_kind == "transformer_residual_window_merge"
    assert audit.merge_boundaries[0].policy == (
        "explicit_qk_softmax_av_fp32_with_weighted_qdq"
    )
    assert audit.metadata["fusion_pruning_strategy"] == (
        "unified_tracer_cnn_attention_dh_ffn_domains"
    )


def test_provider_physical_audit_accepts_consistently_pruned_width_only_when_explicit() -> None:
    from search.model_family import get_model_family

    model = HeterModelBaseline(MaxFusion())
    block = model.backbone_m1.blocks[2]
    block[0] = nn.Conv2d(128, 252, 3, padding=1, bias=False)
    block[1] = nn.BatchNorm2d(252)
    block[3] = nn.Conv2d(252, 256, 3, padding=1, bias=False)
    provider = get_model_family("heal_lidar_fcooper")

    with pytest.raises(RuntimeError, match="backbone_original_width"):
        provider.audit(model, _config("max"))
    physical = provider.audit(
        model,
        _config("max"),
        require_original_widths=False,
    )

    domain = next(
        row
        for row in physical.pruning_domains
        if row.domain_id == "heal_lidar::backbone_m1.blocks.2.0::out"
    )
    assert domain.original_width == 252
    assert physical.metadata["physical_width_audit"] is True


def test_disconet_audit_protects_hidden_and_single_channel_logits() -> None:
    from search.model_family import get_model_family

    audit = get_model_family("heal_lidar_disco").audit(HeterModelBaseline(DiscoFusion()), _config("disconet"))
    by_id = {row.canonical_id: row for row in audit.weighted_ops}
    assert by_id["module::fusion_net.pixel_weight_layer.conv1_1"].allowed_precisions == ("FP32", "FP16")
    assert by_id["module::fusion_net.pixel_weight_layer.conv1_1"].production_enabled is False
    assert by_id["module::fusion_net.pixel_weight_layer.conv1_4"].output_boundary == "single_channel_pixel_weight_logits_before_agent_softmax_fp16"
    assert all(row.policy == "FP16_merge" for row in audit.merge_boundaries)
    assert all(row.production_enabled for row in audit.merge_boundaries)
    assert {row.merge_kind for row in audit.merge_boundaries} == {"channel_concat", "agent_softmax_weighted_sum"}
    operators = {row.capability_id: row for row in audit.deployment_operators}
    assert {"Concat", "Softmax", "ReduceSum"} <= set(operators["agent_feature_fusion"].op_kinds)
    pruning = {row.domain_id: row for row in audit.pruning_domains}
    assert pruning["heal_lidar::fusion_net.pixel_weight_layer.conv1_1::out"].production_enabled is True
    assert pruning["heal_lidar::fusion_net.pixel_weight_layer.conv1_2::out"].production_enabled is True
    assert pruning["heal_lidar::fusion_net.pixel_weight_layer.conv1_3::out"].production_enabled is False
    assert pruning["protected_output::fusion_net.pixel_weight_layer.conv1_4"].original_width == 1


def test_disconet_audit_fails_closed_for_non_scalar_pixel_logits() -> None:
    from search.model_family import get_model_family

    with pytest.raises(RuntimeError, match="pixel_weight_logit_channels"):
        get_model_family("heal_lidar_disco").audit(
            HeterModelBaseline(DiscoFusion(last_channels=2)), _config("disconet")
        )
