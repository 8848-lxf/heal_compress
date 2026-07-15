from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@dataclass
class DummyContext:
    model: object
    trace_result: object
    atomic_prune_units: list[object]
    search_space: object
    pruning_action_catalog: object | None = None


def test_global_anchor_context_exposes_all_legal_trace_units_without_plugin_gene() -> None:
    import torch.nn as nn

    from pruning.types import AtomicPruneUnit
    from search.anchors.joint_taylor_runner import apply_global_anchor_pruning_context
    from search.canonicalization import SearchSpaceSpec

    model = nn.Sequential()
    model.add_module("conv_a", nn.Conv2d(4, 32, 1))
    model.add_module("conv_b", nn.Conv2d(32, 32, 1))
    legal_a = AtomicPruneUnit("scope_a", "conv_a", "out", [0], ["c0"], 0.0)
    legal_b = AtomicPruneUnit("scope_b", "conv_b", "out", [0], ["c1"], 0.0)
    protected = AtomicPruneUnit(
        "scope_head", "conv_b", "out", [1], ["c2"], 0.0, protected=True
    )
    context = DummyContext(
        model=model,
        trace_result=SimpleNamespace(atomic_prune_units=[legal_a, legal_b, protected]),
        atomic_prune_units=[legal_a],
        search_space=SearchSpaceSpec(
            pruning_unit_ids=[legal_a.stable_id],
            precision_layer_ids=["conv_a", "conv_b"],
        ),
    )

    expanded, audit = apply_global_anchor_pruning_context(
        context,
        grouped_conv_mode="independent_group_topk",
        grouped_conv_align=4,
        grouped_allowed_channels_per_group=(4, 8, 16, 32),
    )

    assert set(expanded.search_space.pruning_unit_ids) == {
        legal_a.stable_id,
        legal_b.stable_id,
    }
    assert protected.stable_id not in expanded.search_space.pruning_unit_ids
    assert all("scatter" not in value.lower() for value in expanded.search_space.pruning_unit_ids)
    assert audit["source"] == "existing_formal_trace_atomic_prune_units"
    assert audit["tracer_modified"] is False


def test_anchor_precision_variants_keep_same_pruned_units() -> None:
    from search.anchors.joint_taylor_runner import build_anchor_precision_phenotype
    from search.canonicalization import SearchSpaceSpec
    from search.quantization_space.types import QuantizationSearchGroup

    groups = (
        QuantizationSearchGroup(
            "pg0", ("conv0",), (), ("FP32", "FP16", "INT8"), False, "", 0, 10, 10.0
        ),
        QuantizationSearchGroup(
            "pg1", ("head",), (), ("FP32", "FP16"), True, "head", 1, 10, 10.0
        ),
    )
    space = SearchSpaceSpec(
        pruning_unit_ids=["u"],
        precision_layer_ids=["conv0", "head"],
        quantization_groups=groups,
    )

    fp32 = build_anchor_precision_phenotype(space, ["u"], "strict_fp32")
    fp16 = build_anchor_precision_phenotype(space, ["u"], "strict_fp16")
    int8 = build_anchor_precision_phenotype(space, ["u"], "maximal_legal_int8")

    assert fp32.pruned_unit_ids == fp16.pruned_unit_ids == int8.pruned_unit_ids == ["u"]
    assert set(fp32.realized_precision_profile.values()) == {"FP32"}
    assert set(fp16.realized_precision_profile.values()) == {"FP16"}
    assert int8.realized_precision_profile == {"conv0": "INT8", "head": "FP16"}


def test_tau_row_requires_full_manifest_and_all_deployment_audits() -> None:
    from search.anchors.joint_taylor_runner import stage2_result_to_tau_row

    task = {
        "anchor_id": "prune_0100",
        "precision_variant": "strict_fp16",
        "candidate_hash": "candidate",
        "requested_prune_rate": 0.1,
        "realized_prune_rate": 0.11,
    }
    result = {
        "status": "ok",
        "mAP": 0.7,
        "evaluated": 1789,
        "skipped": 0,
        "physical_hash": "physical",
        "engine_hash": "engine",
        "requested_precision_profile_hash": "precision",
        "realized_precision_profile_hash": "precision",
        "precision_identity_passed": True,
        "deployment_audits_passed": True,
        "calibration_manifest_hash": "calibration",
        "validation_manifest_hash": "validation",
        "BOPS_retention": 0.2,
    }
    proxy = {"total_importance": 2.0, "first_order_sum": 1.0, "second_order_fisher_sum": 1.0}

    accepted = stage2_result_to_tau_row(task, result, proxy, required_frames=1789)
    incomplete = stage2_result_to_tau_row(
        task, {**result, "evaluated": 1788}, proxy, required_frames=1789
    )
    changed = stage2_result_to_tau_row(
        task,
        {**result, "realized_precision_profile_hash": "changed"},
        proxy,
        required_frames=1789,
    )

    assert accepted["valid_for_tau"] is True
    assert incomplete["valid_for_tau"] is False
    assert "full_validation_incomplete" in incomplete["tau_rejection_reasons"]
    assert changed["valid_for_tau"] is False
    assert "precision_identity_failed" in changed["tau_rejection_reasons"]
