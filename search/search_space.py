"""Candidate subnet encoding and effective weight computation.

Encodes pruning decisions (per coupled channel group) and quantization
decisions (per layer weight bit-width) into a single candidate vector.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Supported weight bit-width labels
WEIGHT_BITS = ("FP16", "INT8", "INT4")

# Deployment status categories
DEPLOY_NATIVE_FP16 = "native_trt_fp16"
DEPLOY_NATIVE_INT8_QDQ = "native_trt_int8_qdq"
DEPLOY_INT4_WOQ_GEMM = "int4_woq_gemm"
DEPLOY_PLUGIN_REQUIRED = "plugin_required"
DEPLOY_FALLBACK_INT8 = "fallback_to_int8"
DEPLOY_FALLBACK_FP16 = "fallback_to_fp16"
DEPLOY_PROTECTED = "protected"


@dataclass
class CandidateEncoding:
    """Encoded candidate subnet configuration.

    Attributes:
        prune_vars: Map of group_id -> {0, 1} (0=pruned, 1=kept).
        bitwidth_vars: Map of layer_name -> bit-width label ('FP16'/'INT8'/'INT4').
        meta: Additional metadata (created_by, generation, etc.).
    """
    prune_vars: dict[str, int]
    bitwidth_vars: dict[str, str]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        return {
            "prune_vars": self.prune_vars,
            "bitwidth_vars": self.bitwidth_vars,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidateEncoding":
        """Deserialize from dict."""
        return cls(
            prune_vars={str(k): int(v) for k, v in data.get("prune_vars", {}).items()},
            bitwidth_vars={str(k): str(v) for k, v in data.get("bitwidth_vars", {}).items()},
            meta=dict(data.get("meta", {})),
        )

    @classmethod
    def full(
        cls,
        group_ids: list[str],
        layer_names: list[str],
        default_bit: str = "FP16",
    ) -> "CandidateEncoding":
        """Create a full-precision candidate (all groups kept, FP16 weights).

        Args:
            group_ids: All coupled channel group IDs.
            layer_names: All quantizable layer names.
            default_bit: Default bit-width label.

        Returns:
            Full-precision candidate encoding.
        """
        return cls(
            {gid: 1 for gid in group_ids},
            {name: default_bit for name in layer_names},
            {"created_by": "full"},
        )

    @classmethod
    def random(
        cls,
        group_ids: list[str],
        layer_names: list[str],
        weight_bits: list[str] | None = None,
        keep_probability: float = 0.75,
        protected_group_ids: set[str] | None = None,
    ) -> "CandidateEncoding":
        """Generate a random candidate encoding.

        Args:
            group_ids: All coupled channel group IDs.
            layer_names: All quantizable layer names.
            weight_bits: Candidate bit-width labels (default: FP16/INT8/INT4).
            keep_probability: Probability of keeping each group.
            protected_group_ids: Groups that must be kept.

        Returns:
            Random candidate encoding.
        """
        bits = weight_bits or list(WEIGHT_BITS)
        protected = protected_group_ids or set()
        prune_vars = {
            gid: 1 if gid in protected or random.random() < keep_probability else 0
            for gid in group_ids
        }
        bitwidth_vars = {name: random.choice(bits) for name in layer_names}
        return cls(prune_vars, bitwidth_vars, {"created_by": "random"})

    def clone(self) -> "CandidateEncoding":
        """Deep copy this encoding."""
        return CandidateEncoding(
            dict(self.prune_vars),
            dict(self.bitwidth_vars),
            dict(self.meta),
        )


class SearchSpaceEncoder:
    """Manages the candidate subnet search space.

    For each candidate encoding C, generates:
    - Channel masks M(z): zero out pruned groups
    - Weight pseudo-quantization Q_{b^w}(W): simulate quantization error
    - Effective weights: W_eff(C) = M(z) * Q_{b^w}(W)
    - Layer deployment status classification

    Args:
        model: The HEAL model.
        groups: List of coupled channel groups.
        quantizable_layers: List of layer names eligible for quantization.
        weight_bits: Candidate bit-width labels.
    """

    def __init__(
        self,
        model: nn.Module,
        groups: list[Any],
        quantizable_layers: list[str],
        weight_bits: list[str] | None = None,
    ):
        self.model = model
        self.groups = groups
        self.quantizable_layers = quantizable_layers
        self.weight_bits = weight_bits or list(WEIGHT_BITS)
        self.group_ids = [g.group_id for g in groups]
        self.protected_group_ids = {g.group_id for g in groups if g.is_protected}
        self._group_map = {g.group_id: g for g in groups}

    def classify_deploy_status(self, layer_name: str, bitwidth: str) -> str:
        """Determine the TRT deployment status for a layer at a given bit-width.

        Rules:
        - FP16: all layers -> native_trt_fp16
        - INT8: Conv2d and Linear -> native_trt_int8_qdq
        - INT4: Linear/GEMM -> int4_woq_gemm; Conv2d -> plugin_required

        Args:
            layer_name: Fully qualified layer name.
            bitwidth: Bit-width label ('FP16', 'INT8', 'INT4').

        Returns:
            Deployment status string.
        """
        modules = dict(self.model.named_modules())
        module = modules.get(layer_name)

        if bitwidth == "FP16":
            return DEPLOY_NATIVE_FP16
        if bitwidth == "INT8":
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                return DEPLOY_NATIVE_INT8_QDQ
            return DEPLOY_FALLBACK_FP16
        if bitwidth == "INT4":
            if isinstance(module, nn.Linear):
                return DEPLOY_INT4_WOQ_GEMM
            if isinstance(module, nn.Conv2d):
                return DEPLOY_PLUGIN_REQUIRED
            return DEPLOY_FALLBACK_INT8

        return DEPLOY_FALLBACK_FP16

    def build_channel_masks(
        self, candidate: CandidateEncoding,
    ) -> dict[str, torch.Tensor]:
        """Build output channel masks from pruning variables.

        Args:
            candidate: Candidate encoding with prune_vars.

        Returns:
            Map of layer_name -> binary mask tensor.
        """
        masks: dict[str, torch.Tensor] = {}
        modules = dict(self.model.named_modules())

        for group in self.groups:
            keep = candidate.prune_vars.get(group.group_id, 1)
            for layer_name in group.source_modules:
                module = modules.get(layer_name)
                if module is None or not hasattr(module, "weight"):
                    continue
                out_ch = int(module.weight.shape[0])
                mask = torch.ones(out_ch, dtype=module.weight.dtype,
                                  device=module.weight.device)
                if keep == 0:
                    mask.zero_()
                masks[layer_name] = mask
        return masks

    def random_candidate(self) -> CandidateEncoding:
        """Generate a random candidate encoding.

        Returns:
            Random CandidateEncoding.
        """
        return CandidateEncoding.random(
            self.group_ids,
            self.quantizable_layers,
            self.weight_bits,
            protected_group_ids=self.protected_group_ids,
        )

    def full_candidate(self) -> CandidateEncoding:
        """Generate a full-precision candidate (baseline).

        Returns:
            Full-precision CandidateEncoding.
        """
        return CandidateEncoding.full(
            self.group_ids,
            self.quantizable_layers,
        )
