"""CoBEVT-specific legal fusion widths and atomic physical materialization."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.nn as nn


def _parameter(value: torch.Tensor, original: nn.Parameter) -> nn.Parameter:
    return nn.Parameter(value.detach().clone(), requires_grad=original.requires_grad)


def _indices(values: Iterable[int], device: torch.device) -> torch.Tensor:
    return torch.as_tensor(tuple(values), dtype=torch.long, device=device)


def _parameter_count(model: nn.Module) -> int:
    return sum(int(parameter.numel()) for parameter in model.parameters())


def legal_fusion_widths(
    *,
    original_width: int,
    dim_head: int,
    minimum_retained_ratio: float = 0.2,
    per_domain_max_prune_rate: float = 0.8,
    minimum_heads: int = 1,
) -> tuple[int, ...]:
    if original_width <= 0 or dim_head <= 0 or original_width % dim_head:
        raise ValueError("invalid_cobevt_attention_dimensions")
    if not 0.0 < minimum_retained_ratio <= 1.0:
        raise ValueError("invalid_minimum_retained_ratio")
    if not 0.0 <= per_domain_max_prune_rate < 1.0:
        raise ValueError("invalid_per_domain_max_prune_rate")
    minimum = max(
        int(minimum_heads) * dim_head,
        int(math.ceil(original_width * minimum_retained_ratio)),
        int(math.ceil(original_width * (1.0 - per_domain_max_prune_rate))),
    )
    aligned_minimum = int(math.ceil(minimum / dim_head) * dim_head)
    return tuple(range(aligned_minimum, original_width + 1, dim_head))


@dataclass(frozen=True)
class DecodedFusionWidth:
    original_width: int
    keep_width: int
    dim_head: int
    original_heads: int
    new_heads: int
    ranked_head_ids: tuple[int, ...]
    pruned_head_ids: tuple[int, ...]
    keep_head_ids: tuple[int, ...]
    pruned_channel_indices: tuple[int, ...]
    keep_channel_indices: tuple[int, ...]
    structure_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "dim_head": self.dim_head,
            "keep_channel_indices": list(self.keep_channel_indices),
            "keep_head_ids": list(self.keep_head_ids),
            "keep_width": self.keep_width,
            "new_heads": self.new_heads,
            "original_heads": self.original_heads,
            "original_width": self.original_width,
            "pruned_channel_indices": list(self.pruned_channel_indices),
            "pruned_head_ids": list(self.pruned_head_ids),
            "ranked_head_ids": list(self.ranked_head_ids),
            "structure_hash": self.structure_hash,
        }


def decode_fusion_width(
    original_width: int,
    keep_width: int,
    *,
    dim_head: int,
    ranked_head_ids: Iterable[int],
) -> DecodedFusionWidth:
    if original_width <= 0 or dim_head <= 0 or original_width % dim_head:
        raise ValueError("invalid_cobevt_attention_dimensions")
    if keep_width <= 0 or keep_width > original_width or keep_width % dim_head:
        raise ValueError(f"illegal_cobevt_fusion_keep_width:{keep_width}")
    original_heads = original_width // dim_head
    ranking = tuple(int(value) for value in ranked_head_ids)
    if len(ranking) != original_heads or set(ranking) != set(range(original_heads)):
        raise ValueError("cobevt_head_ranking_must_be_a_permutation")
    new_heads = keep_width // dim_head
    prune_count = original_heads - new_heads
    pruned_heads = tuple(ranking[:prune_count])
    keep_heads = tuple(sorted(set(range(original_heads)) - set(pruned_heads)))
    pruned_channels = tuple(
        channel
        for head in sorted(pruned_heads)
        for channel in range(head * dim_head, (head + 1) * dim_head)
    )
    keep_channels = tuple(
        channel
        for head in keep_heads
        for channel in range(head * dim_head, (head + 1) * dim_head)
    )
    payload = {
        "dim_head": dim_head,
        "keep_channels": keep_channels,
        "keep_heads": keep_heads,
        "keep_width": keep_width,
        "original_width": original_width,
        "ranking": ranking,
        "recipe": "lidar_cobevt_fusion_width_v1",
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return DecodedFusionWidth(
        original_width=original_width,
        keep_width=keep_width,
        dim_head=dim_head,
        original_heads=original_heads,
        new_heads=new_heads,
        ranked_head_ids=ranking,
        pruned_head_ids=pruned_heads,
        keep_head_ids=keep_heads,
        pruned_channel_indices=pruned_channels,
        keep_channel_indices=keep_channels,
        structure_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


@dataclass(frozen=True)
class CobevtPhysicalPruneReport:
    passed: bool
    original_embed_dim: int
    new_embed_dim: int
    original_heads: int
    new_heads: int
    original_parameter_count: int
    predicted_parameter_count: int
    physical_parameter_count: int
    structure_hash: str
    operations: tuple[dict[str, Any], ...]
    issues: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class CobevtFusionWidthDomain:
    domain_id: str
    root_module: str
    domain_kind: str
    original_width: int
    dim_head: int
    original_heads: int
    legal_keep_widths: tuple[int, ...]
    physical_replay_supported: bool
    protected: bool
    dependent_module_count: int


class CobevtPruningRecipe:
    """Family surgery for the CoBEVT fusion embedding closure."""

    def fusion_width_domain(self, model: nn.Module) -> CobevtFusionWidthDomain:
        shrink = model.shrinker_m1.layers[0].double_conv[2]
        if not isinstance(shrink, nn.Conv2d):
            raise RuntimeError("cobevt_fusion_input_conv_missing")
        attention_rows = [
            (name, module)
            for name, module in model.named_modules()
            if name.startswith("fusion_net") and hasattr(module, "heads")
        ]
        if not attention_rows:
            raise RuntimeError("cobevt_attention_metadata_missing")
        head_counts = {int(module.heads) for _, module in attention_rows}
        if len(head_counts) != 1:
            raise RuntimeError("cobevt_attention_head_counts_disagree")
        original_heads = next(iter(head_counts))
        original_width = int(shrink.out_channels)
        if original_heads <= 0 or original_width % original_heads:
            raise RuntimeError("cobevt_embed_dim_not_divisible_by_heads")
        dim_head = original_width // original_heads
        dependent = sum(
            1
            for name, module in model.named_modules()
            if name.startswith("fusion_net")
            and isinstance(module, (nn.Linear, nn.LayerNorm, nn.Embedding))
        )
        return CobevtFusionWidthDomain(
            domain_id="cobevt::fusion_embed",
            root_module="shrinker_m1.layers.0.double_conv.2",
            domain_kind="cobevt_attention_embedding",
            original_width=original_width,
            dim_head=dim_head,
            original_heads=original_heads,
            legal_keep_widths=legal_fusion_widths(
                original_width=original_width,
                dim_head=dim_head,
            ),
            physical_replay_supported=True,
            protected=False,
            dependent_module_count=dependent,
        )

    def decode_fusion_width_indices(
        self,
        *,
        original_width: int,
        keep_width: int,
        dim_head: int,
        ranked_head_ids: Iterable[int],
    ) -> DecodedFusionWidth:
        return decode_fusion_width(
            original_width,
            keep_width,
            dim_head=dim_head,
            ranked_head_ids=ranked_head_ids,
        )

    def decode_fusion_width(
        self,
        model: nn.Module,
        *,
        keep_width: int,
        ranked_head_ids: Iterable[int],
    ) -> DecodedFusionWidth:
        domain = self.fusion_width_domain(model)
        return self.decode_fusion_width_indices(
            original_width=domain.original_width,
            keep_width=keep_width,
            dim_head=domain.dim_head,
            ranked_head_ids=ranked_head_ids,
        )

    @staticmethod
    def _predicted_parameter_count(
        model: nn.Module, decoded: DecodedFusionWidth
    ) -> int:
        original = decoded.original_width
        new = decoded.keep_width
        new_heads = decoded.new_heads
        total = 0
        for name, parameter in model.named_parameters():
            shape = list(parameter.shape)
            if name == "shrinker_m1.layers.0.double_conv.2.weight":
                shape[0] = new
            elif name == "shrinker_m1.layers.0.double_conv.2.bias":
                shape[0] = new
            elif name.startswith("fusion_net"):
                if name.endswith("relative_position_bias_table.weight"):
                    shape[1] = new_heads
                elif len(shape) == 1 and shape[0] == original:
                    shape[0] = new
                elif len(shape) == 2:
                    if shape[1] == original:
                        shape[1] = new
                    if shape[0] == original * 3:
                        shape[0] = new * 3
                    elif shape[0] == original:
                        shape[0] = new
            elif name in {
                "cls_head.weight",
                "reg_head.weight",
                "dir_head.weight",
            }:
                shape[1] = new
            total += math.prod(shape)
        return int(total)

    def materialize_fusion_width(
        self, model: nn.Module, decoded: DecodedFusionWidth
    ) -> CobevtPhysicalPruneReport:
        original = decoded.original_width
        new = decoded.keep_width
        keep = decoded.keep_channel_indices
        qkv_keep = tuple(keep) + tuple(original + value for value in keep) + tuple(
            2 * original + value for value in keep
        )
        operations: list[dict[str, Any]] = []
        issues: list[dict[str, Any]] = []
        original_parameter_count = _parameter_count(model)
        predicted_parameter_count = self._predicted_parameter_count(model, decoded)

        shrink = model.shrinker_m1.layers[0].double_conv[2]
        if not isinstance(shrink, nn.Conv2d) or shrink.out_channels != original:
            raise RuntimeError("cobevt_fusion_input_conv_shape_mismatch")
        index = _indices(keep, shrink.weight.device)
        shrink.weight = _parameter(shrink.weight.data.index_select(0, index), shrink.weight)
        if shrink.bias is not None:
            shrink.bias = _parameter(shrink.bias.data.index_select(0, index), shrink.bias)
        shrink.out_channels = new
        operations.append({"module": "shrinker_m1.layers.0.double_conv.2", "axis": "out", "before": original, "after": new})

        for name, module in model.named_modules():
            if not name.startswith("fusion_net"):
                continue
            if isinstance(module, nn.LayerNorm):
                normalized = tuple(int(value) for value in module.normalized_shape)
                if normalized != (original,):
                    issues.append({"module": name, "reason": "unexpected_layernorm_shape", "shape": list(normalized)})
                    continue
                norm_index = _indices(keep, module.weight.device)
                if module.elementwise_affine:
                    module.weight = _parameter(module.weight.data.index_select(0, norm_index), module.weight)
                    module.bias = _parameter(module.bias.data.index_select(0, norm_index), module.bias)
                module.normalized_shape = (new,)
                operations.append({"module": name, "axis": "layernorm", "before": original, "after": new})
            elif isinstance(module, nn.Linear):
                before_in = int(module.in_features)
                before_out = int(module.out_features)
                weight = module.weight.data
                if before_in == original:
                    weight = weight.index_select(1, _indices(keep, weight.device))
                    module.in_features = new
                if before_out == original * 3:
                    out_keep = qkv_keep
                    module.out_features = new * 3
                elif before_out == original:
                    out_keep = keep
                    module.out_features = new
                else:
                    issues.append({"module": name, "reason": "unsupported_fusion_linear_output", "out_features": before_out})
                    continue
                out_index = _indices(out_keep, weight.device)
                weight = weight.index_select(0, out_index)
                module.weight = _parameter(weight, module.weight)
                if module.bias is not None:
                    module.bias = _parameter(module.bias.data.index_select(0, out_index), module.bias)
                operations.append({"module": name, "axis": "linear_in_out", "before": [before_in, before_out], "after": [module.in_features, module.out_features]})
            elif isinstance(module, nn.Embedding) and name.endswith(
                "relative_position_bias_table"
            ):
                if module.embedding_dim != decoded.original_heads:
                    issues.append({"module": name, "reason": "unexpected_bias_head_count", "heads": module.embedding_dim})
                    continue
                head_index = _indices(decoded.keep_head_ids, module.weight.device)
                module.weight = _parameter(module.weight.data.index_select(1, head_index), module.weight)
                module.embedding_dim = decoded.new_heads
                operations.append({"module": name, "axis": "embedding_dim", "before": decoded.original_heads, "after": decoded.new_heads})
            if hasattr(module, "heads"):
                before_heads = int(module.heads)
                if before_heads != decoded.original_heads:
                    issues.append({"module": name, "reason": "unexpected_attention_heads", "heads": before_heads})
                else:
                    module.heads = decoded.new_heads
                    operations.append({"module": name, "axis": "heads", "before": before_heads, "after": decoded.new_heads})

        for head_name in ("cls_head", "reg_head", "dir_head"):
            head = getattr(model, head_name)
            if not isinstance(head, nn.Conv2d) or head.in_channels != original:
                issues.append({"module": head_name, "reason": "unexpected_prediction_head_input"})
                continue
            head_index = _indices(keep, head.weight.device)
            head.weight = _parameter(head.weight.data.index_select(1, head_index), head.weight)
            head.in_channels = new
            operations.append({"module": head_name, "axis": "in", "before": original, "after": new})

        issues.extend(self._validate_materialized(model, decoded))
        physical_parameter_count = _parameter_count(model)
        if physical_parameter_count != predicted_parameter_count:
            issues.append(
                {
                    "reason": "predicted_physical_parameter_mismatch",
                    "predicted": predicted_parameter_count,
                    "physical": physical_parameter_count,
                }
            )
        return CobevtPhysicalPruneReport(
            passed=not issues,
            original_embed_dim=original,
            new_embed_dim=new,
            original_heads=decoded.original_heads,
            new_heads=decoded.new_heads,
            original_parameter_count=original_parameter_count,
            predicted_parameter_count=predicted_parameter_count,
            physical_parameter_count=physical_parameter_count,
            structure_hash=decoded.structure_hash,
            operations=tuple(operations),
            issues=tuple(issues),
        )

    @staticmethod
    def _validate_materialized(
        model: nn.Module, decoded: DecodedFusionWidth
    ) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        new = decoded.keep_width
        shrink = model.shrinker_m1.layers[0].double_conv[2]
        if shrink.out_channels != new:
            issues.append({"module": "shrinker_m1.layers.0.double_conv.2", "reason": "fusion_width_mismatch"})
        for name, module in model.named_modules():
            if not name.startswith("fusion_net"):
                continue
            if hasattr(module, "heads") and int(module.heads) != decoded.new_heads:
                issues.append({"module": name, "reason": "attention_heads_mismatch"})
            if isinstance(module, nn.LayerNorm) and tuple(module.normalized_shape) != (new,):
                issues.append({"module": name, "reason": "layernorm_width_mismatch"})
            if isinstance(module, nn.Linear) and (
                module.in_features != new or module.out_features not in {new, new * 3}
            ):
                issues.append({"module": name, "reason": "linear_width_mismatch", "shape": [module.in_features, module.out_features]})
            if isinstance(module, nn.Embedding) and name.endswith(
                "relative_position_bias_table"
            ) and module.embedding_dim != decoded.new_heads:
                issues.append({"module": name, "reason": "bias_head_width_mismatch"})
        for head_name in ("cls_head", "reg_head", "dir_head"):
            if getattr(model, head_name).in_channels != new:
                issues.append({"module": head_name, "reason": "prediction_head_input_mismatch"})
        return issues
