"""Joint proxy objective function for candidate evaluation.

F(C) = l1*L_Taylor + l2*L_SQNR + l3*P_BOPS + l4*P_latency + l5*P_size + l6*P_deploy
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn

from .search_space import CandidateEncoding, SearchSpaceEncoder

logger = logging.getLogger(__name__)


class TaylorProxy:
    """Task accuracy loss proxy based on second-order Taylor expansion
    with empirical Fisher diagonal approximation.

    L_Taylor = sum_i [g_i * dw_i + 0.5 * h_i * dw_i^2]
    where dw_i = W_eff,i(C) - W_i

    Args:
        model: The HEAL model.
        group_to_layers: Map of group_id -> list of layer names.
        gradients: Map of param_name -> gradient tensor.
        fisher_diag: Map of param_name -> Fisher diagonal tensor.
    """

    def __init__(
        self,
        model: nn.Module,
        group_to_layers: dict[str, list[str]],
        gradients: dict[str, torch.Tensor] | None = None,
        fisher_diag: dict[str, torch.Tensor] | None = None,
    ):
        self.model = model
        self.group_to_layers = group_to_layers
        self.gradients = gradients or {}
        self.fisher_diag = fisher_diag or {}

    def evaluate(self, candidate: CandidateEncoding) -> float:
        """Compute Taylor proxy loss for a candidate.

        Args:
            candidate: Candidate encoding.

        Returns:
            Taylor proxy loss value.
        """
        modules = dict(self.model.named_modules())
        total = 0.0
        for gid, keep in candidate.prune_vars.items():
            if keep:
                continue
            for layer_name in self.group_to_layers.get(gid, []):
                module = modules.get(layer_name)
                if module is None or not hasattr(module, "weight"):
                    continue
                w = module.weight.detach()
                g = self.gradients.get(f"{layer_name}.weight")
                h = self.fisher_diag.get(f"{layer_name}.weight")
                # dw = -w (pruned to zero)
                if g is not None:
                    total += float((g * (-w)).sum().abs().cpu())
                if h is not None:
                    total += 0.5 * float((h * w.pow(2)).sum().cpu())
                else:
                    total += float(w.abs().mean().cpu())
        return total


class SQNRProxy:
    """Weight quantization error proxy based on Signal-to-Quantization-Noise Ratio.

    L_SQNR = sum_l rho_l * ||M_l*Q(W_l) - W_l||^2 / (||M_l*W_l||^2 + eps)

    Args:
        model: The HEAL model.
        search_space: Search space encoder for pseudo-quantization.
    """

    def __init__(self, model: nn.Module, search_space: SearchSpaceEncoder):
        self.model = model
        self.search_space = search_space
        self._cache: dict[str, dict[str, float]] = {}
        self._precompute()

    def _precompute(self) -> None:
        """Precompute SQNR for all layers and bit-widths."""
        from ..quantization.pseudo_quant import pseudo_quantize_weight

        for layer_name in self.search_space.quantizable_layers:
            module = dict(self.model.named_modules()).get(layer_name)
            if module is None or not hasattr(module, "weight"):
                continue
            self._cache[layer_name] = {}
            w = module.weight.detach()
            signal = float((w ** 2).mean().cpu()) + 1e-12
            for bit_label in self.search_space.weight_bits:
                qw = pseudo_quantize_weight(w, bit_label)
                noise = float(((w - qw) ** 2).mean().cpu()) + 1e-12
                self._cache[layer_name][bit_label] = noise / signal

    def evaluate(self, candidate: CandidateEncoding) -> float:
        """Compute SQNR proxy loss.

        Args:
            candidate: Candidate encoding.

        Returns:
            SQNR proxy loss value.
        """
        total = 0.0
        for layer_name, bit_label in candidate.bitwidth_vars.items():
            cached = self._cache.get(layer_name, {}).get(bit_label, 0.0)
            total += cached
        return total


class BOPSProxy:
    """Bit-Operations (BOPS) proxy for computational cost.

    BOPS(C) = sum_l [H*W*Kh*Kw*C_in*C_out/G * b_w * p_a]
    Penalty: P_BOPS = max(0, R_BOPS/T_BOPS - 1)^2

    Args:
        model: The HEAL model.
        group_to_layers: Map of group_id -> layer names.
        activation_bits: Activation precision in bits (default 16 for FP16).
        target_ratio: Target BOPS retention ratio.
    """

    BIT_MAP = {"FP16": 16, "INT8": 8, "INT4": 4}

    def __init__(
        self,
        model: nn.Module,
        group_to_layers: dict[str, list[str]],
        activation_bits: int = 16,
        target_ratio: float = 0.5,
    ):
        self.model = model
        self.group_to_layers = group_to_layers
        self.activation_bits = activation_bits
        self.target_ratio = target_ratio
        self.layer_params = self._collect_layer_params()
        self.baseline = sum(
            p * 16 * activation_bits for p in self.layer_params.values()
        ) or 1

    def _collect_layer_params(self) -> dict[str, int]:
        """Collect parameter counts for BOPS estimation."""
        params = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                params[name] = int(module.weight.numel())
        return params

    def evaluate(self, candidate: CandidateEncoding) -> float:
        """Compute BOPS retention ratio.

        Args:
            candidate: Candidate encoding.

        Returns:
            BOPS ratio (0 to 1+).
        """
        pruned_layers = set()
        for gid, keep in candidate.prune_vars.items():
            if keep == 0:
                pruned_layers.update(self.group_to_layers.get(gid, []))

        bops = 0.0
        for layer_name, params in self.layer_params.items():
            if layer_name in pruned_layers:
                continue
            bit_label = candidate.bitwidth_vars.get(layer_name, "FP16")
            w_bits = self.BIT_MAP.get(bit_label, 16)
            bops += params * w_bits * self.activation_bits
        return float(bops / self.baseline)

    def penalty(self, candidate: CandidateEncoding) -> float:
        """Compute BOPS constraint penalty.

        Args:
            candidate: Candidate encoding.

        Returns:
            Quadratic penalty value.
        """
        ratio = self.evaluate(candidate)
        excess = ratio / self.target_ratio - 1.0
        return max(0.0, excess) ** 2


class LatencyLUTProxy:
    """Latency proxy based on a lookup table.

    Offline-profiled latencies for (op_type, C_in, C_out, H, W, bit_width)
    combinations. Uses linear interpolation for missing entries.

    Args:
        lut_path: Path to the latency LUT YAML file.
        target_ms: Target latency in milliseconds.
    """

    def __init__(self, lut_path: str | None = None, target_ms: float = 50.0):
        self.lut_path = lut_path
        self.target_ms = target_ms
        self.layer_latency: dict[str, dict[str, float]] = {}
        self.default_latency = 1.0
        if lut_path:
            self._load_lut(lut_path)

    def _load_lut(self, path: str) -> None:
        """Load latency LUT from YAML file."""
        from ..utils.io_utils import load_yaml
        try:
            data = load_yaml(path)
            self.default_latency = float(data.get("default_latency_ms", 1.0))
            for layer, vals in data.get("layers", {}).items():
                self.layer_latency[layer] = {str(k): float(v) for k, v in vals.items()}
        except Exception as exc:
            logger.warning(f"Failed to load latency LUT from {path}: {exc}")

    def evaluate(self, candidate: CandidateEncoding) -> float:
        """Estimate total latency for a candidate.

        Args:
            candidate: Candidate encoding.

        Returns:
            Estimated latency in ms.
        """
        total = 0.0
        for layer, bit_label in candidate.bitwidth_vars.items():
            layer_lut = self.layer_latency.get(layer, {})
            lat = layer_lut.get(bit_label, self.default_latency)
            total += lat

        # Scale by pruning ratio
        if candidate.prune_vars:
            kept = sum(candidate.prune_vars.values())
            ratio = kept / max(len(candidate.prune_vars), 1)
            total *= ratio
        return total

    def penalty(self, candidate: CandidateEncoding) -> float:
        """Compute latency constraint penalty.

        Args:
            candidate: Candidate encoding.

        Returns:
            Quadratic penalty value.
        """
        lat = self.evaluate(candidate)
        excess = lat / self.target_ms - 1.0
        return max(0.0, excess) ** 2


class SizeProxy:
    """Model size proxy in bytes.

    Size(C) = sum_l Params_l(C) * bits_l(C) / 8

    Args:
        model: The HEAL model.
        group_to_layers: Map of group_id -> layer names.
        target_ratio: Target size retention ratio.
    """

    BIT_MAP = {"FP16": 16, "INT8": 8, "INT4": 4}

    def __init__(
        self,
        model: nn.Module,
        group_to_layers: dict[str, list[str]],
        target_ratio: float = 0.4,
    ):
        self.model = model
        self.group_to_layers = group_to_layers
        self.target_ratio = target_ratio
        self.layer_params = {
            name: int(module.weight.numel())
            for name, module in model.named_modules()
            if isinstance(module, (nn.Conv2d, nn.Linear))
        }
        self.baseline_bytes = sum(p * 2 for p in self.layer_params.values()) or 1  # FP16 baseline

    def evaluate(self, candidate: CandidateEncoding) -> float:
        """Compute model size ratio.

        Args:
            candidate: Candidate encoding.

        Returns:
            Size ratio relative to FP16 baseline.
        """
        pruned_layers = set()
        for gid, keep in candidate.prune_vars.items():
            if keep == 0:
                pruned_layers.update(self.group_to_layers.get(gid, []))

        total_bytes = 0.0
        for layer_name, params in self.layer_params.items():
            if layer_name in pruned_layers:
                continue
            bit_label = candidate.bitwidth_vars.get(layer_name, "FP16")
            bits = self.BIT_MAP.get(bit_label, 16)
            total_bytes += params * bits / 8
        return total_bytes / self.baseline_bytes

    def penalty(self, candidate: CandidateEncoding) -> float:
        """Compute size constraint penalty.

        Args:
            candidate: Candidate encoding.

        Returns:
            Quadratic penalty value.
        """
        ratio = self.evaluate(candidate)
        excess = ratio / self.target_ratio - 1.0
        return max(0.0, excess) ** 2


class DeployPenalty:
    """Deployment feasibility penalty.

    Penalizes layers with plugin_required status when the target hardware
    does not support the required plugin.

    Args:
        search_space: Search space encoder for deploy status classification.
        penalty_value: Penalty for infeasible deployment.
    """

    def __init__(self, search_space: SearchSpaceEncoder, penalty_value: float = 1e6):
        self.search_space = search_space
        self.penalty_value = penalty_value

    def evaluate(self, candidate: CandidateEncoding) -> float:
        """Compute deployment penalty.

        Args:
            candidate: Candidate encoding.

        Returns:
            Penalty value (0 if all layers are deployable).
        """
        penalty = 0.0
        for layer_name, bit_label in candidate.bitwidth_vars.items():
            status = self.search_space.classify_deploy_status(layer_name, bit_label)
            if status == "plugin_required":
                penalty += self.penalty_value
        return penalty


class ProxyObjectiveEvaluator:
    """Joint proxy objective function evaluator.

    F(C) = l1*L_Taylor + l2*L_SQNR + l3*P_BOPS + l4*P_latency
           + l5*P_size + l6*P_deploy

    All terms are normalized before weighting.

    Args:
        taylor: TaylorProxy instance (or None).
        sqnr: SQNRProxy instance (or None).
        bops: BOPSProxy instance (or None).
        latency: LatencyLUTProxy instance (or None).
        size: SizeProxy instance (or None).
        deploy: DeployPenalty instance (or None).
        lambdas: Dict of term name -> weight coefficient.
    """

    def __init__(
        self,
        taylor: TaylorProxy | None = None,
        sqnr: SQNRProxy | None = None,
        bops: BOPSProxy | None = None,
        latency: LatencyLUTProxy | None = None,
        size: SizeProxy | None = None,
        deploy: DeployPenalty | None = None,
        lambdas: dict[str, float] | None = None,
    ):
        self.taylor = taylor
        self.sqnr = sqnr
        self.bops = bops
        self.latency = latency
        self.size = size
        self.deploy = deploy
        self.lambdas = {
            "taylor": 1.0,
            "sqnr": 0.5,
            "bops": 2.0,
            "latency": 2.0,
            "size": 1.0,
            "deploy": 10.0,
        }
        if lambdas:
            self.lambdas.update(lambdas)

    def evaluate(self, candidate: CandidateEncoding) -> dict[str, float | bool]:
        """Evaluate the joint objective for a candidate.

        Args:
            candidate: Candidate encoding.

        Returns:
            Dict with individual term values, total 'score', and 'feasible' flag.
        """
        parts = {
            "taylor": self.taylor.evaluate(candidate) if self.taylor else 0.0,
            "sqnr": self.sqnr.evaluate(candidate) if self.sqnr else 0.0,
            "bops": self.bops.penalty(candidate) if self.bops else 0.0,
            "latency": self.latency.penalty(candidate) if self.latency else 0.0,
            "size": self.size.penalty(candidate) if self.size else 0.0,
            "deploy": self.deploy.evaluate(candidate) if self.deploy else 0.0,
        }

        score = sum(self.lambdas.get(k, 1.0) * v for k, v in parts.items())

        feasible = (
            parts["bops"] == 0.0
            and parts["latency"] == 0.0
            and parts["size"] == 0.0
            and parts["deploy"] == 0.0
        )

        return {
            **{k: float(v) for k, v in parts.items()},
            "score": float(score),
            "feasible": feasible,
        }
