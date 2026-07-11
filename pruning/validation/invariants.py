"""Live module attribute and weight-layout invariants."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from ..config import GroupedConvConfig
from ..policies.grouped_conv import validate_grouped_conv_shape


def validate_module_invariants(
    module_path: str,
    module: nn.Module,
    *,
    grouped_config: GroupedConvConfig | None = None,
) -> list[dict[str, Any]]:
    """Return structured issues for one physically materialized module."""

    issues: list[dict[str, Any]] = []
    cfg = grouped_config or GroupedConvConfig()

    def issue(code: str, detail: str) -> None:
        issues.append({"module_path": module_path, "code": code, "detail": detail})

    if isinstance(module, nn.Conv2d):
        expected = (module.out_channels, module.in_channels // module.groups)
        if tuple(module.weight.shape[:2]) != expected:
            issue("conv2d_weight_shape", f"expected prefix {expected}, got {tuple(module.weight.shape[:2])}")
    elif isinstance(module, nn.ConvTranspose2d):
        expected = (module.in_channels, module.out_channels // module.groups)
        if tuple(module.weight.shape[:2]) != expected:
            issue("convtranspose2d_weight_shape", f"expected prefix {expected}, got {tuple(module.weight.shape[:2])}")
    elif isinstance(module, nn.Linear):
        expected = (module.out_features, module.in_features)
        if tuple(module.weight.shape) != expected:
            issue("linear_weight_shape", f"expected {expected}, got {tuple(module.weight.shape)}")
    elif isinstance(module, nn.modules.batchnorm._BatchNorm):
        for name in ("weight", "bias", "running_mean", "running_var"):
            value = getattr(module, name, None)
            if value is not None and int(value.numel()) != module.num_features:
                issue("batchnorm_shape", f"{name} length {value.numel()} != {module.num_features}")
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        if module.in_channels % module.groups or module.out_channels % module.groups:
            issue("group_divisibility", "logical channels are not divisible by groups")
        if module.groups > 1:
            report = validate_grouped_conv_shape(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                groups=module.groups,
                allowed_channels_per_group=cfg.allowed_channels_per_group,
                depthwise_special_case=cfg.depthwise_special_case,
            )
            for violation in report.violations:
                issue("grouped_conv_legality", violation)
    return issues


__all__ = ["validate_module_invariants"]
