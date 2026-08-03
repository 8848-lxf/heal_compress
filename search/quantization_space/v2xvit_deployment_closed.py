"""Deployment-closed precision chromosome for V2X-ViT TensorRT 10.9."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence


_FIXED_ACTIVATION_ROLE = {
    "qk_matmul": "FP32",
    "layernorm": "FP32",
    "softmax": "FP16",
    "av_matmul": "FP16",
    "ffn_activation": "FP16",
    "residual_add": "FP16",
}


def deployment_close_v2xvit_quantization_groups(
    groups: Sequence[Any],
    *,
    buildable_int8_group_ids: set[str] | None = None,
) -> tuple[Any, ...]:
    """Remove unimplemented functional A8 loci and optionally gate weighted INT8."""

    result = []
    for group in groups:
        metadata = dict(group.metadata)
        role = str(metadata.get("transformer_role", ""))
        activation_only = bool(metadata.get("activation_only", False))
        if activation_only:
            if role not in _FIXED_ACTIVATION_ROLE:
                raise RuntimeError(
                    f"v2xvit_deployment_activation_role_unmapped:{group.group_id}:{role}"
                )
            precision = _FIXED_ACTIVATION_ROLE[role]
            result.append(
                replace(
                    group,
                    allowed_precisions=(precision,),
                    protected=True,
                    protection_reason=(
                        "deployment_closed_functional_boundary_no_exact_a8_qdq"
                    ),
                    metadata={
                        **metadata,
                        "deployment_closed": True,
                        "deployment_fixed_precision": precision,
                        "default_precision": precision,
                        "removed_states": sorted(
                            set(group.allowed_precisions) - {precision}
                        ),
                    },
                )
            )
            continue
        allowed = tuple(str(value).upper() for value in group.allowed_precisions)
        if buildable_int8_group_ids is not None and group.group_id not in buildable_int8_group_ids:
            allowed = tuple(value for value in allowed if value != "INT8")
        if not allowed:
            raise RuntimeError(f"v2xvit_deployment_weighted_group_empty:{group.group_id}")
        result.append(
            replace(
                group,
                allowed_precisions=allowed,
                metadata={
                    **metadata,
                    "deployment_closed": buildable_int8_group_ids is not None,
                    "int8_build_verified": (
                        None
                        if buildable_int8_group_ids is None
                        else group.group_id in buildable_int8_group_ids
                    ),
                },
            )
        )
    return tuple(result)


__all__ = ["deployment_close_v2xvit_quantization_groups"]
