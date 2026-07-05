from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def default_fixed_width_boundaries() -> list[dict[str, Any]]:
    return [
        {
            "boundary_id": "pfn_encoder_to_pointpillar_scatter",
            "producer_patterns": [
                "encoder_m1.pillar_vfe.pfn_layers",
                "pillar_vfe.pfn_layers",
            ],
            "consumer_patterns": [
                "point_pillar_scatter",
                "scatter",
            ],
            "fixed_channel": 64,
            "protected_reason": "fixed_width_boundary:pfn_to_pointpillar_scatter",
            "can_prune_without_resolver": False,
        }
    ]


def protected_prefixes_for_fixed_width_boundaries(boundaries: list[dict[str, Any]] | None = None) -> list[str]:
    prefixes: list[str] = []
    for boundary in boundaries or default_fixed_width_boundaries():
        prefixes.extend(str(p) for p in boundary.get("producer_patterns", []) if p)
    return sorted(dict.fromkeys(prefixes))


def main() -> int:
    out = Path("outputs/latency_lut/fixed_width_boundary_registry_v84.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    data = {"fixed_width_boundaries": default_fixed_width_boundaries()}
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(data, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
