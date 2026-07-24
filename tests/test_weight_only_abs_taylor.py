import pytest
import torch


def test_fisher_collector_preserves_elementwise_abs_before_sample_reduction():
    from search.proxy.fisher_proxy import collect_task_loss_fisher_statistics

    model = torch.nn.Linear(1, 1, bias=False)
    batches = [torch.tensor([[1.0]]), torch.tensor([[-1.0]])]
    stats, report = collect_task_loss_fisher_statistics(
        model,
        batches,
        forward_fn=lambda module, batch: module(batch),
        loss_fn=lambda output, _batch: output.sum(),
        calibration_manifest_hash="abs-test",
    )
    assert stats.gradients["weight"].item() == pytest.approx(0.0)
    assert stats.absolute_gradients["weight"].item() == pytest.approx(1.0)
    assert report["absolute_value_before_sample_reduction"] is True


def test_elementwise_abs_prevents_cross_parameter_cancellation():
    from search.proxy.joint_weight_taylor import JointWeightTaylorProxy

    first, second, score = JointWeightTaylorProxy._cost_terms(
        torch.tensor([1.0, 1.0]),
        torch.tensor([1.0, -1.0]),
        torch.tensor([1.0, 1.0]),
    )
    assert first.sum().item() == pytest.approx(2.0)
    assert second.sum().item() == pytest.approx(1.0)
    assert score.sum().item() == pytest.approx(3.0)
    assert bool((score >= 0).all())


def test_pruning_action_counts_only_newly_removed_elements_without_refund():
    from search.candidate import CandidatePhenotype, PrecisionDecision
    from search.proxy.fisher_proxy import FisherStatistics
    from search.proxy.joint_weight_taylor import JointWeightTaylorProxy
    from search.proxy.parameter_slice_resolver import ParameterSlice

    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    stats = FisherStatistics(
        gradients={"weight": torch.tensor([[1.0, -1.0], [1.0, -1.0]])},
        fisher_diag={"weight": torch.ones(2, 2)},
        absolute_gradients={"weight": torch.ones(2, 2)},
        manifest_hash="action-test",
    )
    proxy = JointWeightTaylorProxy(
        model,
        statistics=stats,
        unit_to_parameter_slices={
            "row0": [ParameterSlice("weight", "", 0, (0,), "prune_weight_slice")]
        },
    )
    current = CandidatePhenotype(precision_profile={"": PrecisionDecision("FP32", "FP32")})
    successor = CandidatePhenotype(
        pruned_unit_ids=["row0"],
        precision_profile={"": PrecisionDecision("FP32", "FP32")},
    )
    breakdown = proxy.pruning_action_breakdown(current, successor)
    assert breakdown["newly_pruned_parameter_count"] == 2
    assert breakdown["delta_J_prune"] == pytest.approx(1.0 + 2.0 + 0.5 * (1.0 + 4.0))
    assert breakdown["risk_refund"] == 0.0


def test_weight_quantization_action_is_adjacent_and_nonnegative():
    from search.candidate import CandidatePhenotype, PrecisionDecision
    from search.proxy.fisher_proxy import FisherStatistics
    from search.proxy.joint_weight_taylor import JointWeightTaylorProxy

    model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False))
    stats = FisherStatistics(
        gradients={"0.weight": torch.ones(2, 2)},
        fisher_diag={"0.weight": torch.ones(2, 2)},
        absolute_gradients={"0.weight": torch.ones(2, 2)},
        manifest_hash="quant-test",
    )
    proxy = JointWeightTaylorProxy(model, statistics=stats, unit_to_parameter_slices={})
    fp32 = CandidatePhenotype(precision_profile={"0": PrecisionDecision("FP32", "FP32")})
    fp16 = CandidatePhenotype(precision_profile={"0": PrecisionDecision("FP16", "FP16")})
    int8 = CandidatePhenotype(precision_profile={"0": PrecisionDecision("INT8", "INT8")})
    q16 = proxy.weight_quantization_action_breakdown(fp32, fp16)
    q8 = proxy.weight_quantization_action_breakdown(fp16, int8)
    assert q16["delta_J_WQ"] >= 0.0
    assert q8["delta_J_WQ"] >= 0.0
