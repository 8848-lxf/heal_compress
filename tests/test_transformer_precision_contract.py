"""Transformer precision, Softmax and SmoothQuant contracts."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from heal_compress.search.pruning_space.transformer_domains import build_transformer_pruning_domains
from heal_compress.search.quantization_space.smoothquant import (
    SMOOTHQUANT_ALPHA_GRID,
    SmoothQuantRegistry,
    make_smoothquant_record,
    select_smoothquant_alpha,
    smoothquant_scale,
    smoothquant_transform,
)
from heal_compress.search.quantization_space.transformer_precision import (
    RealizedTransformerPrecision,
    SEARCH_PRECISION_STATES,
    assert_transformer_precision_realized,
    build_transformer_precision_units,
    expected_softmax_realization,
    validate_external_precision_profile,
)


class Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.heads = 4
        self.to_qkv = nn.Linear(32, 3 * 4 * 8, bias=False)
        self.to_out = nn.Sequential(nn.Linear(4 * 8, 32, bias=False), nn.Dropout(0.0))
        self.attend = nn.Softmax(dim=-1)


class FFN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, 32)


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = Attention()
        self.ffn = FFN()
        self.norm = nn.LayerNorm(32)


def _precision_units():
    model = Model()
    _, attention, ffn = build_transformer_pruning_domains(
        model,
        model_name="toy",
        allow_identity_ranking=True,
    )
    return build_transformer_precision_units(
        attention,
        ffn,
        model=model,
        layernorm_paths=("norm",),
    )


def test_search_contract_has_only_three_coupled_weight_activation_states() -> None:
    assert SEARCH_PRECISION_STATES == ("W32A32", "W16A16", "W8A8")
    units = _precision_units()
    weighted = [unit for unit in units if not unit.activation_only]
    assert weighted
    assert all(unit.allowed_states == SEARCH_PRECISION_STATES for unit in weighted)
    assert any(unit.role == "fused_qkv_projection" and "W8A8" in unit.allowed_states for unit in weighted)


def test_qk_and_layernorm_are_protected_fp32_and_illegal_requests_fail() -> None:
    units = _precision_units()
    qk = next(unit for unit in units if unit.role == "qk_matmul")
    norm = next(unit for unit in units if unit.role == "layernorm")
    assert qk.allowed_states == norm.allowed_states == ("A32",)
    assert qk.protected and norm.protected
    with pytest.raises(ValueError, match="request_illegal"):
        validate_external_precision_profile({qk.unit_id: "A8"}, units)


def test_softmax_a8_means_float_compute_with_quantized_output_not_native_exp() -> None:
    contract = expected_softmax_realization("A8")
    assert contract == {
        "softmax_compute": "FP32",
        "softmax_input": "FP32",
        "softmax_output": "INT8",
        "qdq": True,
        "semantics": "floating_softmax_quantized_output",
    }
    plugin = expected_softmax_realization("A8", native_int8_plugin=True)
    assert plugin["softmax_compute"] == "INT8_PLUGIN"
    assert plugin["semantics"] == "native_int8_plugin"


def test_requested_realized_qk_or_softmax_conflict_fails_closed() -> None:
    units = _precision_units()
    requested = {unit.unit_id: unit.default_state for unit in units}
    evidence = []
    for unit in units:
        state = requested[unit.unit_id]
        internal = "FP32" if state.endswith("32") else "FP16" if state.endswith("16") else "INT8"
        evidence.append(RealizedTransformerPrecision(
            unit_id=unit.unit_id,
            role=unit.role,
            requested_state=state,
            realized_weight=internal,
            realized_activation=internal,
            compute_precision=internal,
            accumulator_precision="FP32" if unit.role == "qk_matmul" else "unknown",
            output_precision=internal,
            qdq=False,
            cast=False,
            reformat=False,
        ))
    assert_transformer_precision_realized(requested, units, evidence)
    qk_index = next(index for index, unit in enumerate(units) if unit.role == "qk_matmul")
    broken = list(evidence)
    row = broken[qk_index]
    broken[qk_index] = RealizedTransformerPrecision(
        **{**row.to_dict(), "compute_precision": "FP16"}
    )
    with pytest.raises(RuntimeError, match="qk_fp32_contract_conflict"):
        assert_transformer_precision_realized(requested, units, broken)


def test_smoothquant_is_equivalent_before_quantization_and_hashes_are_stable() -> None:
    torch.manual_seed(11)
    linear = nn.Linear(6, 4, bias=True).eval()
    activation = torch.randn(3, 5, 6)
    scale = smoothquant_scale(activation, linear.weight, alpha=0.75)
    transformed_activation, transformed_weight, transformed_bias = smoothquant_transform(
        activation, linear, scale
    )
    expected = linear(activation)
    actual = torch.nn.functional.linear(transformed_activation, transformed_weight, transformed_bias)
    torch.testing.assert_close(actual, expected, rtol=1.0e-5, atol=1.0e-6)
    first = make_smoothquant_record(
        model="toy",
        family="toy_attention",
        unit_id="qkv",
        module_paths=("attn.to_qkv",),
        alpha=0.75,
        structure_hash="structure-a",
        calibration_manifest_hash="calibration-a",
        scale=scale,
        fused_qkv_shared_input=True,
        selection_evidence={"proxy": 1.0},
    )
    second = make_smoothquant_record(
        model="toy",
        family="toy_attention",
        unit_id="qkv",
        module_paths=("attn.to_qkv",),
        alpha=0.75,
        structure_hash="structure-a",
        calibration_manifest_hash="calibration-a",
        scale=scale.clone(),
        fused_qkv_shared_input=True,
        selection_evidence={"proxy": 1.0},
    )
    assert first == second
    assert first.activation_scale_hash != first.weight_scale_hash
    assert first.fused_qkv_shared_input


def test_smoothquant_registry_rejects_silent_cross_structure_reuse() -> None:
    scale = torch.ones(8)
    record = make_smoothquant_record(
        model="toy",
        family="attention",
        unit_id="qk",
        module_paths=("q", "k"),
        alpha=0.7,
        structure_hash="width-8",
        calibration_manifest_hash="manifest",
        scale=scale,
        fused_qkv_shared_input=False,
        selection_evidence={"anchor": 0.1},
    )
    registry = SmoothQuantRegistry()
    registry.freeze(record)
    assert registry.require(
        model="toy", family="attention", unit_id="qk",
        structure_hash="width-8", calibration_manifest_hash="manifest",
    ) == record
    with pytest.raises(RuntimeError, match="calibration_incompatible"):
        registry.require(
            model="toy", family="attention", unit_id="qk",
            structure_hash="width-4", calibration_manifest_hash="manifest",
        )
    assert registry.manifest()["alpha_is_search_gene"] is False


def test_alpha_is_selected_offline_from_frozen_grid_and_anchor() -> None:
    rows = [
        {
            "unit_id": "qkv", "structure_hash": "s", "calibration_manifest_hash": "m",
            "alpha": alpha, "calibration_proxy": abs(alpha - 0.75), "anchor_loss": abs(alpha - 0.8),
        }
        for alpha in SMOOTHQUANT_ALPHA_GRID
    ]
    selected = select_smoothquant_alpha(rows, anchor_weight=0.5)
    assert selected["alpha"] in SMOOTHQUANT_ALPHA_GRID
    assert selected["alpha"] == 0.75
