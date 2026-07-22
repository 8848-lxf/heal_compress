"""The three fixed precision contracts for the d_h alignment sweep."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DhPrecisionProfile:
    profile_id: str
    base_profile: str
    int8_roles: tuple[str, ...] = ()
    alpha_by_model: tuple[tuple[str, float], ...] = ()

    def alpha(self, model: str) -> float | None:
        return dict(self.alpha_by_model).get(str(model))

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "base_profile": self.base_profile,
            "int8_roles": list(self.int8_roles),
            "alpha_by_model": dict(self.alpha_by_model),
            "qk_contract": "F32A32O32",
            "strongly_typed": True,
            "no_tf32": True,
        }


PROFILES = {
    "P32": DhPrecisionProfile("P32", "B1_TRT_ATTN_FP32"),
    "P16": DhPrecisionProfile("P16", "B3_F3"),
    "P8": DhPrecisionProfile(
        "P8",
        "B3_F3",
        int8_roles=("q_projection", "k_projection"),
        alpha_by_model=(("lidar_cobevt", 0.8), ("lidar_v2xvit", 0.75)),
    ),
}


def require_profile(profile_id: str) -> DhPrecisionProfile:
    try:
        profile = PROFILES[str(profile_id)]
    except KeyError as exc:
        raise ValueError(f"unsupported_dh_precision_profile:{profile_id}") from exc
    return profile


__all__ = ["DhPrecisionProfile", "PROFILES", "require_profile"]
