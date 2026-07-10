from __future__ import annotations

from types import SimpleNamespace

import torch.nn as nn


def _item(name, module, direction):
    return SimpleNamespace(name=name, module=module, direction=direction, reason="")


def _group(group_id, items, channels=16):
    return SimpleNamespace(group_id=group_id, items=items, num_channels=channels, protected=False, protected_reason="", meta={})


def test_full_model_surface_keeps_non_grouped_conv_prunable():
    from heal_compress.pruning.full_model_surface import apply_full_model_prunable_surface

    conv = nn.Conv2d(16, 32, 3)
    group = _group("ordinary", [_item("neck.conv", conv, "out")], channels=32)

    report = apply_full_model_prunable_surface([group], group_conv_policy="A")

    assert group.protected is False
    assert report["num_prunable_coupled_units"] == 1
    assert report["num_protected_coupled_units"] == 0


def test_full_model_surface_allows_grouped_conv_input_for_v97_resolver():
    from heal_compress.pruning.full_model_surface import apply_full_model_prunable_surface

    producer = nn.Conv2d(16, 32, 1)
    grouped = nn.Conv2d(32, 32, 3, groups=8)
    group = _group(
        "producer_to_grouped_input",
        [_item("block.conv1", producer, "out"), _item("block.conv2", grouped, "in")],
        channels=32,
    )

    report = apply_full_model_prunable_surface([group], group_conv_policy="A")

    assert group.protected is False
    assert group.protected_reason == ""
    assert report["num_prunable_coupled_units"] == 1


def test_full_model_surface_allows_grouped_conv_output_resolver_A():
    from heal_compress.pruning.full_model_surface import apply_full_model_prunable_surface

    grouped = nn.Conv2d(32, 32, 3, groups=8)
    next_conv = nn.Conv2d(32, 64, 1)
    group = _group(
        "grouped_output",
        [_item("block.conv2", grouped, "out"), _item("block.conv3", next_conv, "in")],
        channels=32,
    )

    apply_full_model_prunable_surface([group], group_conv_policy="A")

    assert group.protected is False


def test_full_model_surface_protects_convtranspose_deblock_contract():
    from heal_compress.pruning.full_model_surface import apply_full_model_prunable_surface

    deblock = nn.ConvTranspose2d(64, 64, 2, stride=2)
    group = _group("deblock", [_item("pyramid_backbone.deblocks.0.0", deblock, "out")], channels=64)

    apply_full_model_prunable_surface([group], group_conv_policy="A")

    assert group.protected is True
    assert group.protected_reason == "protected_convtranspose_deblock_or_fpn_output_contract"


def test_full_model_surface_param_budget_uses_unique_modules():
    from heal_compress.pruning.full_model_surface import surface_inventory

    conv = nn.Conv2d(4, 8, 1, bias=False)
    prunable = _group("conv_out", [_item("shared.conv", conv, "out")], channels=8)
    protected = _group("conv_in_contract", [_item("shared.conv", conv, "in")], channels=4)
    protected.protected = True
    protected.protected_reason = "fixed_width_shape_contract"

    report = surface_inventory([prunable, protected], total_model_params=conv.weight.numel(), group_conv_policy="A")

    assert report["total_prunable_params"] == conv.weight.numel()
    assert report["total_protected_params"] == 0
    assert report["total_prunable_params"] + report["total_protected_params"] == report["total_model_params"]
