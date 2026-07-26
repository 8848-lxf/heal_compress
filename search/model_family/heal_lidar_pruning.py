"""Graph-safe structured pruning for HEAL LiDAR F-Cooper and DiscoNet.

The two baselines share the same PointPillar/BEV backbone.  Their final feature
width differs only at fusion: F-Cooper carries one shared channel mask through
agent-wise max, while DiscoNet maps each feature channel to both halves of the
neighbor/ego concatenation consumed by ``PixelWeightLayer.conv1_1``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any, Sequence

import torch.nn as nn

from pruning.api import replay_pruning
from pruning.types import SamplingPruningRequest
from tracer.types import AtomicPruneUnit, DependencyMember

from ..adapters.pruning_adapter import FormalPruningAdapter
from ..pruning_space.action_catalog import PruningSearchAction
from ..pruning_space.grouped_bundle_adapter import request_from_pruning_actions
from .contracts import ModelFamilyAudit


SUPPORTED_HEAL_LIDAR_BASELINE_FAMILIES = (
    "heal_lidar_fcooper",
    "heal_lidar_disco",
)


@dataclass(frozen=True)
class HealLidarPruningDomainSpec:
    domain_id: str
    root_module_path: str
    root_axis: str
    original_width: int
    members: tuple[tuple[str, str, str], ...]
    domain_kind: str = "dense_coupled_output_channels"
    protected: bool = False
    protection_reason: str = ""


@dataclass(frozen=True)
class HealLidarBaselinePruningTopology:
    family_id: str
    backbone_conv_paths: tuple[tuple[str, ...], ...]
    backbone_bn_paths: tuple[tuple[str, ...], ...]
    deblock_conv_paths: tuple[str, ...]
    shrinker_conv_paths: tuple[str, str]
    feature_width: int
    concat_width: int
    domain_specs: tuple[HealLidarPruningDomainSpec, ...]
    fixed_output_contracts: dict[str, int]
    original_contract_verified: bool

    @property
    def production_domain_specs(self) -> tuple[HealLidarPruningDomainSpec, ...]:
        return tuple(row for row in self.domain_specs if not row.protected)


def _family_id(value: str | ModelFamilyAudit) -> str:
    family_id = str(value.family_id if isinstance(value, ModelFamilyAudit) else value)
    if family_id not in SUPPORTED_HEAL_LIDAR_BASELINE_FAMILIES:
        raise RuntimeError(f"unsupported_heal_lidar_pruning_family:{family_id}")
    return family_id


def _require_module(
    model: nn.Module,
    path: str,
    expected: type[nn.Module] | tuple[type[nn.Module], ...],
) -> nn.Module:
    try:
        module = model.get_submodule(path)
    except AttributeError as exc:
        raise RuntimeError(f"heal_lidar_pruning_topology_missing:{path}") from exc
    if not isinstance(module, expected):
        expected_name = (
            "|".join(value.__name__ for value in expected)
            if isinstance(expected, tuple)
            else expected.__name__
        )
        raise RuntimeError(
            f"heal_lidar_pruning_topology_type:{path}:{type(module).__name__}:"
            f"expected={expected_name}"
        )
    return module


def _attfusion_scale_audit(model: nn.Module, feature_width: int) -> dict[str, Any]:
    """Require AttFusion's parameter-free attention scale to match its width.

    ``ScaledDotProductAttention.sqrt_dim`` is plain Python/NumPy metadata and
    therefore is not rewritten by channel pruning or ``state_dict`` replay.
    Leaving it at ``sqrt(original_width)`` changes the physical model after the
    shrinker output is pruned even though every tensor shape remains legal.
    """

    attention = _require_module(model, "fusion_net.att", nn.Module)
    if not hasattr(attention, "sqrt_dim"):
        raise RuntimeError("heal_lidar_attfusion_sqrt_dim_missing")
    observed = float(getattr(attention, "sqrt_dim"))
    expected = math.sqrt(float(feature_width))
    return {
        "feature_width": int(feature_width),
        "observed_sqrt_dim": observed,
        "expected_sqrt_dim": expected,
        "passed": bool(math.isfinite(observed) and math.isclose(observed, expected, rel_tol=0.0, abs_tol=1.0e-12)),
    }


def _synchronize_attfusion_scale(model: nn.Module, feature_width: int) -> dict[str, Any]:
    attention = _require_module(model, "fusion_net.att", nn.Module)
    if not hasattr(attention, "sqrt_dim"):
        raise RuntimeError("heal_lidar_attfusion_sqrt_dim_missing")
    before = float(getattr(attention, "sqrt_dim"))
    setattr(attention, "sqrt_dim", math.sqrt(float(feature_width)))
    audit = _attfusion_scale_audit(model, feature_width)
    if not audit["passed"]:
        raise RuntimeError(f"heal_lidar_attfusion_sqrt_dim_update_failed:{audit}")
    return {**audit, "before_sqrt_dim": before, "updated": not math.isclose(before, audit["expected_sqrt_dim"], rel_tol=0.0, abs_tol=1.0e-12)}


def _conv_bn_paths(model: nn.Module, prefix: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    sequence = _require_module(model, prefix, nn.Sequential)
    children = list(sequence.named_children())
    conv_paths: list[str] = []
    bn_paths: list[str] = []
    for position, (name, module) in enumerate(children):
        if not isinstance(module, nn.Conv2d):
            continue
        following = children[position + 1] if position + 1 < len(children) else None
        if following is None or not isinstance(following[1], nn.BatchNorm2d):
            raise RuntimeError(f"heal_lidar_pruning_conv_without_batchnorm:{prefix}.{name}")
        conv_paths.append(f"{prefix}.{name}")
        bn_paths.append(f"{prefix}.{following[0]}")
    if not conv_paths:
        raise RuntimeError(f"heal_lidar_pruning_empty_backbone_block:{prefix}")
    return tuple(conv_paths), tuple(bn_paths)


def _identity_member(path: str, axis: str, dependency_type: str, channel: int) -> DependencyMember:
    return DependencyMember(
        module_path=path,
        axis=axis,
        indices=[channel],
        dependency_type=dependency_type,
        index_map={channel: [channel]},
    )


def _domain(
    *,
    root: str,
    width: int,
    members: Sequence[tuple[str, str, str]],
    domain_kind: str = "dense_coupled_output_channels",
    protected: bool = False,
    protection_reason: str = "",
) -> HealLidarPruningDomainSpec:
    return HealLidarPruningDomainSpec(
        domain_id=f"heal_lidar::{root}::out",
        root_module_path=root,
        root_axis="out",
        original_width=int(width),
        members=tuple(members),
        domain_kind=domain_kind,
        protected=protected,
        protection_reason=protection_reason,
    )


def validate_heal_lidar_baseline_pruning_topology(
    model: nn.Module,
    family: str | ModelFamilyAudit,
    *,
    require_original_widths: bool = True,
) -> HealLidarBaselinePruningTopology:
    """Validate the exact BEV/shrinker/fusion dependency contract.

    ``require_original_widths=False`` validates a physically pruned replay: all
    coupled widths must still agree, but production roots may be narrower.
    """

    family_id = _family_id(family)
    if type(model).__name__ != "HeterModelBaseline":
        raise RuntimeError("heal_lidar_pruning_topology_model_type")
    backbone = _require_module(model, "backbone_m1", nn.Module)
    blocks = getattr(backbone, "blocks", None)
    deblocks = getattr(backbone, "deblocks", None)
    if not isinstance(blocks, nn.ModuleList) or len(blocks) != 3:
        raise RuntimeError("heal_lidar_pruning_requires_three_backbone_levels")
    if not isinstance(deblocks, nn.ModuleList) or len(deblocks) != 3:
        raise RuntimeError("heal_lidar_pruning_requires_three_deblocks")

    all_conv_paths: list[tuple[str, ...]] = []
    all_bn_paths: list[tuple[str, ...]] = []
    canonical_widths = (64, 128, 256)
    previous_output = 64
    for level in range(3):
        conv_paths, bn_paths = _conv_bn_paths(model, f"backbone_m1.blocks.{level}")
        all_conv_paths.append(conv_paths)
        all_bn_paths.append(bn_paths)
        for position, (conv_path, bn_path) in enumerate(zip(conv_paths, bn_paths)):
            conv = _require_module(model, conv_path, nn.Conv2d)
            bn = _require_module(model, bn_path, nn.BatchNorm2d)
            expected_input = previous_output if position == 0 else int(
                _require_module(model, conv_paths[position - 1], nn.Conv2d).out_channels
            )
            if conv.in_channels != expected_input or bn.num_features != conv.out_channels:
                raise RuntimeError(f"heal_lidar_pruning_backbone_chain_mismatch:{conv_path}")
            if require_original_widths and conv.out_channels != canonical_widths[level]:
                raise RuntimeError(f"heal_lidar_pruning_backbone_original_width:{conv_path}")
        previous_output = int(_require_module(model, conv_paths[-1], nn.Conv2d).out_channels)

    deblock_paths: list[str] = []
    fixed_contracts: dict[str, int] = {}
    for level in range(3):
        deblock_prefix = f"backbone_m1.deblocks.{level}"
        deblock = _require_module(model, deblock_prefix, nn.Sequential)
        direct = list(deblock.named_children())
        if len(direct) < 2 or not isinstance(direct[0][1], (nn.Conv2d, nn.ConvTranspose2d)):
            raise RuntimeError(f"heal_lidar_pruning_invalid_deblock:{deblock_prefix}")
        conv_path = f"{deblock_prefix}.{direct[0][0]}"
        conv = direct[0][1]
        bn = direct[1][1]
        block_last = _require_module(model, all_conv_paths[level][-1], nn.Conv2d)
        if not isinstance(bn, nn.BatchNorm2d):
            raise RuntimeError(f"heal_lidar_pruning_deblock_batchnorm_missing:{deblock_prefix}")
        if conv.in_channels != block_last.out_channels or conv.out_channels != 128 or bn.num_features != 128:
            raise RuntimeError(f"heal_lidar_pruning_deblock_contract:{deblock_prefix}")
        deblock_paths.append(conv_path)
        fixed_contracts[conv_path] = 128
    concat_width = sum(
        int(_require_module(model, path, (nn.Conv2d, nn.ConvTranspose2d)).out_channels)
        for path in deblock_paths
    )
    if concat_width != 384:
        raise RuntimeError(f"heal_lidar_pruning_concat_width:{concat_width}")

    shrinker = _require_module(model, "shrinker_m1", nn.Module)
    layers = getattr(shrinker, "layers", None)
    if not isinstance(layers, nn.ModuleList) or len(layers) != 1:
        raise RuntimeError("heal_lidar_pruning_requires_one_shrinker_doubleconv")
    shrink_sequence = _require_module(model, "shrinker_m1.layers.0.double_conv", nn.Sequential)
    shrink_convs = [
        (name, module)
        for name, module in shrink_sequence.named_children()
        if isinstance(module, nn.Conv2d)
    ]
    if len(shrink_convs) != 2:
        raise RuntimeError("heal_lidar_pruning_requires_two_shrinker_convs")
    shrink_paths = tuple(f"shrinker_m1.layers.0.double_conv.{name}" for name, _ in shrink_convs)
    shrink_first, shrink_final = (module for _, module in shrink_convs)
    if shrink_first.in_channels != concat_width or shrink_final.in_channels != shrink_first.out_channels:
        raise RuntimeError("heal_lidar_pruning_shrinker_chain_mismatch")
    if require_original_widths and (shrink_first.out_channels != 256 or shrink_final.out_channels != 256):
        raise RuntimeError("heal_lidar_pruning_shrinker_original_width")
    feature_width = int(shrink_final.out_channels)

    for head_path in ("cls_head", "reg_head", "dir_head"):
        head = _require_module(model, head_path, nn.Conv2d)
        if head.in_channels != feature_width:
            raise RuntimeError(f"heal_lidar_pruning_head_input_mismatch:{head_path}")
        fixed_contracts[head_path] = int(head.out_channels)

    expected_fusion_class = "MaxFusion" if family_id == "heal_lidar_fcooper" else "DiscoFusion"
    fusion = _require_module(model, "fusion_net", nn.Module)
    if type(fusion).__name__ != expected_fusion_class:
        raise RuntimeError(f"heal_lidar_pruning_fusion_type:{type(fusion).__name__}")
    if family_id == "heal_lidar_attfusion":
        scale_audit = _attfusion_scale_audit(model, feature_width)
        if not scale_audit["passed"]:
            raise RuntimeError(
                f"heal_lidar_attfusion_sqrt_dim_mismatch:{scale_audit}"
            )

    domains: list[HealLidarPruningDomainSpec] = []
    for level, (conv_paths, bn_paths) in enumerate(zip(all_conv_paths, all_bn_paths)):
        for position, (conv_path, bn_path) in enumerate(zip(conv_paths, bn_paths)):
            conv = _require_module(model, conv_path, nn.Conv2d)
            members: list[tuple[str, str, str]] = [
                (conv_path, "out", "root_output"),
                (bn_path, "channel", "batchnorm_output_coupling"),
            ]
            if position + 1 < len(conv_paths):
                members.append((conv_paths[position + 1], "in", "sequential_conv_input"))
            else:
                members.append((deblock_paths[level], "in", "fixed_deblock_input"))
                if level + 1 < len(all_conv_paths):
                    members.append((all_conv_paths[level + 1][0], "in", "next_backbone_level_input"))
            domains.append(_domain(root=conv_path, width=conv.out_channels, members=members))

    domains.append(_domain(
        root=shrink_paths[0],
        width=shrink_first.out_channels,
        members=(
            (shrink_paths[0], "out", "root_output"),
            (shrink_paths[1], "in", "shrinker_second_conv_input"),
        ),
        domain_kind="shrinker_hidden_width",
    ))
    feature_members: list[tuple[str, str, str]] = [
        (shrink_paths[1], "out", "root_output"),
        ("cls_head", "in", "fixed_detection_head_input"),
        ("reg_head", "in", "fixed_detection_head_input"),
        ("dir_head", "in", "fixed_detection_head_input"),
    ]

    if family_id == "heal_lidar_disco":
        pixel = "fusion_net.pixel_weight_layer"
        conv1_1 = _require_module(model, f"{pixel}.conv1_1", nn.Conv2d)
        bn1_1 = _require_module(model, f"{pixel}.bn1_1", nn.BatchNorm2d)
        conv1_2 = _require_module(model, f"{pixel}.conv1_2", nn.Conv2d)
        bn1_2 = _require_module(model, f"{pixel}.bn1_2", nn.BatchNorm2d)
        conv1_3 = _require_module(model, f"{pixel}.conv1_3", nn.Conv2d)
        bn1_3 = _require_module(model, f"{pixel}.bn1_3", nn.BatchNorm2d)
        conv1_4 = _require_module(model, f"{pixel}.conv1_4", nn.Conv2d)
        if conv1_1.in_channels != 2 * feature_width:
            raise RuntimeError("heal_lidar_pruning_disco_double_half_input_mismatch")
        if not (
            bn1_1.num_features == conv1_1.out_channels == conv1_2.in_channels
            and bn1_2.num_features == conv1_2.out_channels == conv1_3.in_channels
            and bn1_3.num_features == conv1_3.out_channels == conv1_4.in_channels
        ):
            raise RuntimeError("heal_lidar_pruning_disco_hidden_chain_mismatch")
        if require_original_widths and (
            conv1_1.out_channels != 128 or conv1_2.out_channels != 32 or conv1_3.out_channels != 8
        ):
            raise RuntimeError("heal_lidar_pruning_disco_original_hidden_width")
        if conv1_3.out_channels != 8 or conv1_4.out_channels != 1:
            raise RuntimeError("heal_lidar_pruning_disco_protected_logit_contract")
        feature_members.append(
            (f"{pixel}.conv1_1", "in", "disconet_neighbor_ego_concat_double_half")
        )
        domains.extend((
            _domain(
                root=f"{pixel}.conv1_1",
                width=conv1_1.out_channels,
                members=(
                    (f"{pixel}.conv1_1", "out", "root_output"),
                    (f"{pixel}.bn1_1", "channel", "batchnorm_output_coupling"),
                    (f"{pixel}.conv1_2", "in", "pixel_weight_next_conv_input"),
                ),
                domain_kind="disconet_pixel_weight_hidden_width",
            ),
            _domain(
                root=f"{pixel}.conv1_2",
                width=conv1_2.out_channels,
                members=(
                    (f"{pixel}.conv1_2", "out", "root_output"),
                    (f"{pixel}.bn1_2", "channel", "batchnorm_output_coupling"),
                    (f"{pixel}.conv1_3", "in", "pixel_weight_next_conv_input"),
                ),
                domain_kind="disconet_pixel_weight_hidden_width",
            ),
            _domain(
                root=f"{pixel}.conv1_3",
                width=conv1_3.out_channels,
                members=(
                    (f"{pixel}.conv1_3", "out", "root_output"),
                    (f"{pixel}.bn1_3", "channel", "batchnorm_output_coupling"),
                    (f"{pixel}.conv1_4", "in", "single_logit_conv_input"),
                ),
                domain_kind="disconet_protected_logit_hidden_width",
                protected=True,
                protection_reason="eight_channel_pre_logit_domain_protected_until_alignment_and_ap_evidence",
            ),
        ))
        fixed_contracts[f"{pixel}.conv1_4"] = 1

    domains.append(_domain(
        root=shrink_paths[1],
        width=feature_width,
        members=feature_members,
        domain_kind=(
            "disconet_shared_fusion_feature_width"
            if family_id == "heal_lidar_disco"
            else "fcooper_shared_fusion_feature_width"
        ),
    ))

    return HealLidarBaselinePruningTopology(
        family_id=family_id,
        backbone_conv_paths=tuple(all_conv_paths),
        backbone_bn_paths=tuple(all_bn_paths),
        deblock_conv_paths=tuple(deblock_paths),
        shrinker_conv_paths=(shrink_paths[0], shrink_paths[1]),
        feature_width=feature_width,
        concat_width=concat_width,
        domain_specs=tuple(domains),
        fixed_output_contracts=fixed_contracts,
        original_contract_verified=bool(require_original_widths),
    )


def build_heal_lidar_baseline_atomic_units(
    model: nn.Module,
    family: str | ModelFamilyAudit,
) -> list[AtomicPruneUnit]:
    """Build one immutable dependency-closed unit per production root channel."""

    topology = validate_heal_lidar_baseline_pruning_topology(model, family)
    units: list[AtomicPruneUnit] = []
    for domain in topology.domain_specs:
        if domain.protected:
            continue
        for channel in range(domain.original_width):
            members: list[DependencyMember] = []
            for module_path, axis, dependency_type in domain.members:
                if dependency_type == "disconet_neighbor_ego_concat_double_half":
                    local = [channel, topology.feature_width + channel]
                    complete_map = {
                        root: [root, topology.feature_width + root]
                        for root in range(topology.feature_width)
                    }
                    members.append(DependencyMember(
                        module_path=module_path,
                        axis=axis,
                        indices=local,
                        dependency_type=dependency_type,
                        index_map=complete_map,
                    ))
                else:
                    members.append(_identity_member(module_path, axis, dependency_type, channel))
            coupled_id = f"heal_lidar::{topology.family_id}::{domain.domain_id}::channel::{channel:04d}"
            units.append(AtomicPruneUnit(
                scope_id=domain.domain_id,
                root_module_path=domain.root_module_path,
                root_axis=domain.root_axis,
                root_indices=[channel],
                source_coupled_unit_ids=[coupled_id],
                members=members,
                constraints={
                    "original_channel_count": domain.original_width,
                    "family_id": topology.family_id,
                    "domain_kind": domain.domain_kind,
                    "physical_materializer": "heal_lidar_baseline_plan_first_v1",
                    "dense_alignment": 4,
                    "minimum_retained_ratio": 0.10,
                    "fixed_deblock_outputs": True,
                    "fixed_concat_width": topology.concat_width,
                },
            ))
    if not units:
        raise RuntimeError("heal_lidar_pruning_space_empty")
    return units


def _request_from_selection(
    selection: SamplingPruningRequest | Sequence[PruningSearchAction],
) -> SamplingPruningRequest:
    if isinstance(selection, SamplingPruningRequest):
        return selection
    actions = list(selection)
    if not all(isinstance(action, PruningSearchAction) for action in actions):
        raise TypeError("heal_lidar_materializer_requires_request_or_pruning_actions")
    return request_from_pruning_actions(actions)


def materialize_heal_lidar_baseline(
    model: nn.Module,
    selection: SamplingPruningRequest | Sequence[PruningSearchAction],
    *,
    family: str | ModelFamilyAudit,
    example_inputs: Any | None = None,
    require_zero_alignment_repair: bool = True,
) -> dict[str, Any]:
    """Run the formal plan-first pipeline and family-specific closure checks."""

    family_id = _family_id(family)
    original = validate_heal_lidar_baseline_pruning_topology(model, family_id)
    request = _request_from_selection(selection)
    adapter = FormalPruningAdapter()
    forward_inputs = (example_inputs,) if isinstance(example_inputs, Mapping) else example_inputs
    result = adapter.materialize_from_request(
        model,
        request,
        example_inputs=forward_inputs,
        fixed_output_contracts=original.fixed_output_contracts,
    )
    repaired = [
        f"{row.module_path}:{row.axis}"
        for row in result["plan"].entries
        if bool(row.repaired)
    ]
    if repaired and require_zero_alignment_repair:
        raise RuntimeError(f"heal_lidar_unexpected_alignment_repair:{repaired}")
    ledger = result["ledger"].to_dict()
    if ledger["status_counts"].get("skipped", 0):
        raise RuntimeError("heal_lidar_materialization_skipped_request")
    validation = result["validation"]
    if not validation.passed:
        raise RuntimeError(f"heal_lidar_physical_validation_failed:{validation.issues}")
    attfusion_scale_audit: dict[str, Any] = {}
    if family_id == "heal_lidar_attfusion":
        realized_width = int(_require_module(result["model"], "cls_head", nn.Conv2d).in_channels)
        attfusion_scale_audit = _synchronize_attfusion_scale(
            result["model"], realized_width
        )
    physical_topology = validate_heal_lidar_baseline_pruning_topology(
        result["model"], family_id, require_original_widths=False
    )

    replayed = replay_pruning(model, result["plan"], in_place=False).model
    if family_id == "heal_lidar_attfusion":
        _synchronize_attfusion_scale(replayed, physical_topology.feature_width)
    incompatible = replayed.load_state_dict(result["model"].state_dict(), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("heal_lidar_strict_replay_reload_failed")
    replay_topology = validate_heal_lidar_baseline_pruning_topology(
        replayed, family_id, require_original_widths=False
    )
    if replay_topology.feature_width != physical_topology.feature_width:
        raise RuntimeError("heal_lidar_replay_feature_width_mismatch")

    return {
        **result,
        "request": request,
        "family_id": family_id,
        "original_topology": original,
        "physical_topology": physical_topology,
        "strict_replay_reload_verified": True,
        "attfusion_scale_audit": attfusion_scale_audit,
        "ledger_summary": ledger,
    }


__all__ = [
    "HealLidarBaselinePruningTopology",
    "HealLidarPruningDomainSpec",
    "SUPPORTED_HEAL_LIDAR_BASELINE_FAMILIES",
    "build_heal_lidar_baseline_atomic_units",
    "materialize_heal_lidar_baseline",
    "validate_heal_lidar_baseline_pruning_topology",
]
