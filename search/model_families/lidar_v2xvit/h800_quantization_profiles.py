"""V2X-ViT-specific H800 profile support; no CoBEVT name assumptions."""

from __future__ import annotations

from search.model_families.transformer.precision_contract import precision_profiles
from search.model_families.transformer.smoothquant_profiles import smoothquant_profiles


MODEL_FAMILY = "lidar_v2xvit"
EXPECTED_AGENT_ATTENTION_BLOCKS = 3
EXPECTED_WINDOW_ATTENTION_BLOCKS = 9
AGENT_HEADS = 8
AGENT_D_QK = 32
WINDOW_HEADS = (16, 8, 4)
WINDOW_D_QK = (16, 32, 64)


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
    "AGENT_D_QK",
    "AGENT_HEADS",
    "EXPECTED_AGENT_ATTENTION_BLOCKS",
    "EXPECTED_WINDOW_ATTENTION_BLOCKS",
    "MODEL_FAMILY",
    "WINDOW_D_QK",
    "WINDOW_HEADS",
    "requested_profiles",
]
