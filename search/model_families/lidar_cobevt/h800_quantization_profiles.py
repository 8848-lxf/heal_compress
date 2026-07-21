"""CoBEVT-specific H800 profile support layered on shared role contracts."""

from __future__ import annotations

from search.model_families.transformer.precision_contract import precision_profiles
from search.model_families.transformer.smoothquant_profiles import smoothquant_profiles


MODEL_FAMILY = "lidar_cobevt"
EXPECTED_ATTENTION_BLOCKS = 6
EXPECTED_HEADS = 8
EXPECTED_D_QK = 32
EXPECTED_D_V = 32


def requested_profiles() -> dict[str, dict]:
    return {
        **precision_profiles(),
        **{
            row.profile_id: {
                "profile_id": row.profile_id,
                "int8_roles": list(row.int8_roles),
                "protected_roles": list(row.protected_roles),
                "qk_contract": row.qk_contract,
            }
            for row in smoothquant_profiles()
        },
    }


__all__ = [
    "EXPECTED_ATTENTION_BLOCKS",
    "EXPECTED_D_QK",
    "EXPECTED_D_V",
    "EXPECTED_HEADS",
    "MODEL_FAMILY",
    "requested_profiles",
]
