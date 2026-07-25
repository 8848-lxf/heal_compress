#!/usr/bin/env python3
"""Generate deterministic deployment-closed profiles for old005 bisection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable


def _is_cnn(group_id: str) -> bool:
    return group_id.startswith("cnn_precision::") and "shrinker_m1" not in group_id


def _is_shrinker(group_id: str) -> bool:
    return group_id.startswith("cnn_precision::") and "shrinker_m1" in group_id


def _is_ffn(group_id: str) -> bool:
    return group_id.startswith("transformer_precision::") and group_id.endswith(("::ffn1", "::ffn2"))


def _is_attention(group_id: str) -> bool:
    return group_id.startswith("transformer_precision::") and not _is_ffn(group_id)


def _is_agent_relation(group_id: str) -> bool:
    return _is_attention(group_id) and ".layers.0.0.fn::" in group_id


def _is_layer_window(group_id: str, layer: int) -> bool:
    return _is_attention(group_id) and f".encoder.layers.{layer}.0.layers.0.1.fn.pwmsa." in group_id


def build_profiles(
    old_winner: dict, precision_floor: dict
) -> dict[str, dict]:
    old_genotype = dict(old_winner["genotype"])
    template = dict(precision_floor["genotypes"]["P8-max-requested"])
    group_ids = tuple(sorted(template["precision_genes"]))
    widths = dict(old_genotype["pruning_width_genes"])

    selectors: dict[str, Callable[[str], bool]] = {
        "INT8-CNN-only": _is_cnn,
        "INT8-shrinker-only": _is_shrinker,
        "INT8-FFN-only": _is_ffn,
        "INT8-Attention-QKVO-only": _is_attention,
        "INT8-agent-relation-only": _is_agent_relation,
        "INT8-layer0-window-only": lambda value: _is_layer_window(value, 0),
        "INT8-layer1-window-only": lambda value: _is_layer_window(value, 1),
        "INT8-layer2-window-only": lambda value: _is_layer_window(value, 2),
        "repaired-maximal-mixed": lambda _value: True,
    }
    profiles: dict[str, dict] = {}
    for name, default in (("S32", "FP32"), ("S16", "FP16")):
        profiles[name] = {
            "pruning_width_genes": widths,
            "precision_genes": {group_id: default for group_id in group_ids},
            "meta": {"created_by": "old005_deployment_bisection", "profile": name},
        }
    for name, selector in selectors.items():
        precision = {
            group_id: "INT8" if selector(group_id) else "FP16"
            for group_id in group_ids
        }
        profiles[name] = {
            "pruning_width_genes": widths,
            "precision_genes": precision,
            "meta": {
                "created_by": "old005_deployment_bisection",
                "profile": name,
                "requested_int8_count": sum(value == "INT8" for value in precision.values()),
            },
        }
    return profiles


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-winner", type=Path, required=True)
    parser.add_argument("--precision-floor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old_winner = json.loads(args.old_winner.read_text(encoding="utf-8"))
    precision_floor = json.loads(args.precision_floor.read_text(encoding="utf-8"))
    payload = {
        "schema_version": "v2xvit-old005-build-bisection-v1",
        "source_candidate_hash": old_winner.get("candidate_hash"),
        "functional_contracts": {"S32": "P32", "default": "F3"},
        "genotypes": build_profiles(old_winner, precision_floor),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise RuntimeError(f"refusing_to_overwrite:{args.output}")
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
