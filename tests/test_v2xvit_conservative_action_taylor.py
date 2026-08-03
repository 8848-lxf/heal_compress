from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from search.candidate import CandidatePhenotype


def _phenotype(pruned: list[str], precision: dict[str, str]) -> CandidatePhenotype:
    return CandidatePhenotype.from_dict(
        {"pruned_unit_ids": pruned, "precision_profile": precision}
    )


def test_structural_gate_action_sums_only_newly_removed_units() -> None:
    from search.proxy.conservative_action_taylor import StructuralGateTaylorProxy

    proxy = StructuralGateTaylorProxy(
        unit_terms={
            "u0": {"first": 1.0, "second": 0.25},
            "u1": {"first": 2.0, "second": 0.50},
        },
        domain_units={"domain": ("u0", "u1")},
    )
    row = proxy.pruning_action_breakdown(
        _phenotype(["u0"], {}), _phenotype(["u0", "u1"], {})
    )

    assert row["delta_J_struct"] == pytest.approx(2.5)
    assert row["first_order_abs_sum"] == pytest.approx(2.0)
    assert row["second_order_abs_sum"] == pytest.approx(0.5)
    assert row["newly_removed_unit_ids"] == ["u1"]
    assert row["risk_refund"] == 0.0


def test_structural_gate_missing_mapping_fails_closed() -> None:
    from search.proxy.conservative_action_taylor import StructuralGateTaylorProxy

    with pytest.raises(RuntimeError, match="structural_gate_mapping_incomplete"):
        StructuralGateTaylorProxy(
            unit_terms={"u0": {"first": 1.0, "second": 0.0}},
            domain_units={"domain": ("u0", "u1")},
        )


def test_activation_taylor_uses_elementwise_abs_before_reduction() -> None:
    from search.proxy.conservative_action_taylor import ActivationActionTaylorProxy
    from search.proxy.joint_weight_activation_taylor import (
        JointOutputTaylorStatistics,
        TaylorDeploymentUnit,
    )

    unit = TaylorDeploymentUnit(
        unit_id="activation::layer",
        module_path="layer",
        unit_type="weighted_input",
        boundary="module_input",
        precision_owner="layer",
        quantizer_id="q::layer",
        metadata={"precision_gene_id": "gene"},
    )
    stats = JointOutputTaylorStatistics(
        baseline_outputs={"activation::layer": (torch.tensor([1.0, -1.0]),)},
        gradients={"activation::layer": (torch.tensor([1.0, 1.0]),)},
        calibration_manifest_hash="calibration",
        unit_manifest_hash="units",
        sample_count=1,
    )
    proxy = ActivationActionTaylorProxy(
        statistics=stats,
        units=(unit,),
        gene_to_unit_ids={"gene": (unit.unit_id,)},
    )
    row = proxy.quantization_action_breakdown(
        _phenotype([], {"layer": "FP32"}),
        _phenotype([], {"layer": "FP16"}),
        changed_gene_id="gene",
    )

    delta = torch.tensor([1.0, -1.0]).half().float() - torch.tensor([1.0, -1.0])
    expected_first = (torch.tensor([1.0, 1.0]) * delta).abs().sum().item()
    assert row["delta_J_AQ"] == pytest.approx(expected_first)
    assert row["elementwise_abs_before_reduction"] is True
    assert row["cross_sample_signed_cancellation"] is False


def test_activation_taylor_detects_cross_element_cancellation() -> None:
    from search.proxy.conservative_action_taylor import activation_taylor_terms

    value = torch.tensor([1.0, -1.0])
    gradient = torch.tensor([1.0, 1.0])
    delta = torch.tensor([0.25, -0.25])
    first, second = activation_taylor_terms(value, gradient, delta)

    assert first.sum().item() == pytest.approx(0.5)
    assert (gradient * delta).sum().abs().item() == 0.0
    assert second.sum().item() > 0.0


def test_streaming_activation_taylor_precomputes_sample_mean_without_tensors() -> None:
    from search.proxy.conservative_action_taylor import (
        collect_streaming_activation_action_statistics,
    )
    from search.proxy.joint_weight_activation_taylor import TaylorDeploymentUnit

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = torch.nn.Linear(2, 2, bias=False)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.layer(value)

    model = Tiny()
    unit = TaylorDeploymentUnit(
        unit_id="activation::layer",
        module_path="layer",
        unit_type="weighted_input",
        boundary="module_input",
        precision_owner="layer",
        quantizer_id="q::layer",
    )
    proxy, manifest = collect_streaming_activation_action_statistics(
        model,
        (
            torch.tensor([[1.001, -0.499]], requires_grad=True),
            torch.tensor([[0.251, 2.003]], requires_grad=True),
        ),
        forward_fn=lambda module, value: module(value),
        loss_fn=lambda output, _batch: output.square().sum(),
        units=(unit,),
        gene_to_unit_ids={"gene": (unit.unit_id,)},
        precision_ladders={"gene": ("FP32", "FP16", "INT8")},
        calibration_manifest_hash="fixed-manifest",
    )

    first = proxy.quantization_action_breakdown(
        _phenotype([], {"layer": "FP32"}),
        _phenotype([], {"layer": "FP16"}),
        changed_gene_id="gene",
    )
    second = proxy.quantization_action_breakdown(
        _phenotype([], {"layer": "FP16"}),
        _phenotype([], {"layer": "INT8"}),
        changed_gene_id="gene",
    )
    assert manifest["sample_count"] == 2
    assert manifest["statistics_tensors_persisted"] is False
    assert manifest["sample_reduction"].startswith("mean_of_per_sample")
    assert first["delta_J_AQ"] >= 0.0
    assert second["delta_J_AQ"] >= 0.0


def test_structural_gate_statistics_average_multiple_batches() -> None:
    from search.proxy.conservative_action_taylor import (
        collect_structural_gate_statistics,
    )
    from search.pruning_space.local_domains import LocalPruningDomain

    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = torch.nn.Linear(2, 2, bias=False)
            torch.nn.init.eye_(self.layer.weight)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.layer(value)

    domain = LocalPruningDomain(
        domain_id="domain",
        root_module_path="layer",
        root_axis="out",
        scope_id="scope",
        kind="dense",
        original_width=2,
        total_original_width=2,
        ordered_unit_ids=("u0", "u1"),
        legal_widths=(1, 2),
        width_to_pruned_unit_ids={1: ("u0",), 2: ()},
        unit_root_indices={"u0": (0,), "u1": (1,)},
    )
    proxy, manifest = collect_structural_gate_statistics(
        Tiny(),
        (domain,),
        (torch.tensor([[1.0, 2.0]]), torch.tensor([[2.0, 1.0]])),
        forward_fn=lambda module, value: module(value),
        loss_fn=lambda output, _batch: output.square().sum(),
    )

    assert manifest["sample_count"] == 2
    assert len(manifest["task_losses"]) == 2
    assert proxy.unit_terms["u0"]["first"] > 0.0
    assert proxy.unit_terms["u1"]["first"] > 0.0


def test_winner_selector_does_not_reward_pruning_or_mixed_size() -> None:
    from search.greedy.conservative_joint import select_budget_winner

    rows = [
        {
            "candidate_hash": "wide",
            "cumulative_total_taylor": 10.0,
            "current_retention": 0.301,
            "R_parameter_retention": 0.80,
            "mixed_weight_size_bytes": 1000.0,
        },
        {
            "candidate_hash": "aggressive",
            "cumulative_total_taylor": 10.0 * (1.0 + 5.0e-9),
            "current_retention": 0.301,
            "R_parameter_retention": 0.50,
            "mixed_weight_size_bytes": 100.0,
        },
    ]

    winner = select_budget_winner(rows, target=0.30)
    assert winner["candidate_hash"] == "wide"


def test_stage1_runtime_audit_has_no_model_or_deployment_calls() -> None:
    from search.greedy.conservative_joint import empty_search_loop_runtime_audit

    assert empty_search_loop_runtime_audit() == {
        "search_loop_forward_calls": 0,
        "search_loop_backward_calls": 0,
        "search_loop_physical_exports": 0,
        "search_loop_onnx_exports": 0,
        "search_loop_trt_builds": 0,
    }


def test_precision_action_combines_weight_and_activation_without_interaction() -> None:
    from search.greedy.conservative_joint import combine_precision_action_risk

    row = combine_precision_action_risk(
        {"delta_J_WQ": 2.0, "first_order_abs_sum": 1.5, "second_order_abs_sum": 0.5},
        {"delta_J_AQ": 3.0, "first_order_abs_sum": 2.5, "second_order_abs_sum": 0.5},
    )
    assert row["delta_J_precision"] == 5.0
    assert row["joint_taylor_used_for_fitness"] is False
    assert row["cross_residual_used_for_fitness"] is False
    assert row["risk_refund"] == 0.0


def test_stage2_selection_is_physical_and_phenotype_deduplicated() -> None:
    from search.greedy.conservative_joint import select_stage2_budget_pool

    rows = [
        {
            "candidate_hash": "a",
            "physical_hash": "p0",
            "precision_hash": "q0",
            "phenotype_hash": "x0",
            "cumulative_total_taylor": 1.0,
            "current_retention": 0.30,
            "R_parameter_retention": 0.70,
            "pruned_unit_count": 10,
            "int8_count": 20,
            "attention_ffn_width_sum": 100,
        },
        {
            "candidate_hash": "a-precision-variant",
            "physical_hash": "p0",
            "precision_hash": "q1",
            "phenotype_hash": "x1",
            "cumulative_total_taylor": 0.9,
            "current_retention": 0.30,
            "R_parameter_retention": 0.70,
            "pruned_unit_count": 10,
            "int8_count": 30,
            "attention_ffn_width_sum": 100,
        },
        {
            "candidate_hash": "b",
            "physical_hash": "p1",
            "precision_hash": "q1",
            "phenotype_hash": "x1",
            "cumulative_total_taylor": 1.2,
            "current_retention": 0.301,
            "R_parameter_retention": 0.90,
            "pruned_unit_count": 3,
            "int8_count": 25,
            "attention_ffn_width_sum": 120,
        },
    ]

    selected = select_stage2_budget_pool(rows, target=0.30, maximum=5)
    assert [row["candidate_hash"] for row in selected] == [
        "a-precision-variant",
        "b",
    ]
    assert "lowest_total_taylor" in selected[0]["selection_reasons"]
    assert "highest_parameter_retention" in selected[1]["selection_reasons"]


def test_budget_retention_uses_original_fp32_not_legal_start_normalization() -> None:
    from search.greedy.conservative_joint import absolute_fp32_retention

    bops = {
        "bops_total": 36.0,
        "R_bops_vs_fp32": 0.296,
        "R_bops_vs_fp16_deploy": 1.07,
    }
    assert absolute_fp32_retention(bops) == pytest.approx(0.296)
