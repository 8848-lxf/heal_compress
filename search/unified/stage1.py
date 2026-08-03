"""Model-agnostic Stage-1 Taylor term composition."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping


Proxy = Callable[[Any], float | Mapping[str, Any]]


def _term(proxy: Proxy, phenotype: Any, names: tuple[str, ...]) -> float:
    raw = proxy(phenotype)
    if isinstance(raw, Mapping):
        value = next((raw[name] for name in names if name in raw), None)
        if value is None:
            raise KeyError(f"stage1_taylor_term_missing:{','.join(names)}")
    else:
        value = raw
    result = float(value)
    if result < 0.0 or not math.isfinite(result):
        raise ValueError(f"stage1_taylor_term_invalid:{result}")
    return result


@dataclass(frozen=True)
class Stage1TaylorPolicy:
    include_activation_taylor: bool = False


class Stage1TaylorEvaluator:
    """Compose structural, weight-quantization and optional activation terms."""

    def __init__(
        self,
        *,
        structural_proxy: Proxy,
        weight_quantization_proxy: Proxy,
        activation_quantization_proxy: Proxy | None,
        policy: Stage1TaylorPolicy | None = None,
    ) -> None:
        self.structural_proxy = structural_proxy
        self.weight_quantization_proxy = weight_quantization_proxy
        self.activation_quantization_proxy = activation_quantization_proxy
        self.policy = policy or Stage1TaylorPolicy()
        if self.policy.include_activation_taylor and activation_quantization_proxy is None:
            raise ValueError("activation_taylor_enabled_but_proxy_missing")

    def __call__(self, phenotype: Any) -> dict[str, Any]:
        structural = _term(self.structural_proxy, phenotype, ("J_struct", "value"))
        weight = _term(self.weight_quantization_proxy, phenotype, ("J_WQ", "value"))
        activation = (
            _term(
                self.activation_quantization_proxy,  # type: ignore[arg-type]
                phenotype,
                ("J_AQ", "L_joint_weight_activation_taylor", "value"),
            )
            if self.policy.include_activation_taylor
            else 0.0
        )
        return {
            "J_struct": structural,
            "J_WQ": weight,
            "J_AQ": activation,
            "J_total": structural + weight + activation,
            "activation_taylor_included": self.policy.include_activation_taylor,
        }


__all__ = ["Stage1TaylorEvaluator", "Stage1TaylorPolicy"]
