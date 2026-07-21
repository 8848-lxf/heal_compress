"""Build fresh role-isolated Transformer precision sensitivity engines on H800."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from search.model_families.transformer.precision_contract import precision_profiles
from search.orchestration.lidar_transformer_h800_baselines import (
    _write_json,
    build_profile,
    reaudit_existing_profile,
)


FP16_PROFILES = tuple(f"P{index}_{name}" for index, name in (
    (1, "QKV_FP16"),
    (2, "QK_FP16"),
    (3, "SOFTMAX_FP16"),
    (4, "AV_FP16"),
    (5, "OUT_FP16"),
    (6, "LAYERNORM_FP16"),
    (7, "RESIDUAL_FP16"),
    (8, "FFN1_FP16"),
    (9, "FFN2_FP16"),
    (10, "ALL_FFN_FP16"),
    (11, "ALL_PROJECTION_FP16"),
    (12, "FULL_ATTENTION_FP16"),
))
BF16_PROFILES = (
    "P13_QKV_BF16",
    "P14_QK_BF16",
    "P15_FFN_BF16",
    "P16_FULL_ATTENTION_BF16",
)
H800_BF16_PROFILES = (
    "H1_QKV_BF16_QK_FP32",
    "H2_QKV_BF16_QK_BF16A32",
    "H3_FFN_BF16",
    "H4_FULL_TRANSFORMER_BF16_PROTECTED_QK",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model", choices=("lidar_cobevt", "lidar_v2xvit"), required=True)
    parser.add_argument("--profiles", default=",".join(FP16_PROFILES))
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--plugin", required=True)
    parser.add_argument(
        "--destination-section",
        choices=("precision_sensitivity", "bf16"),
        default="precision_sensitivity",
    )
    parser.add_argument("--reaudit-existing", action="store_true")
    args = parser.parse_args(argv)
    profiles = tuple(value.strip() for value in args.profiles.split(",") if value.strip())
    known = set(precision_profiles()) - {
        "B1_TRT_ATTN_FP32",
        "B2_TRT_STRICT_FP16",
        "B3_F3",
    }
    unknown = sorted(set(profiles) - known)
    if unknown:
        raise ValueError(f"unknown_transformer_sensitivity_profile:{unknown}")
    output_root = Path(args.output_root).resolve()
    if args.reaudit_existing:
        results = [
            reaudit_existing_profile(
                output_root=output_root,
                destination_section=args.destination_section,
                model_name=args.model,
                profile=profile,
            )
            for profile in profiles
        ]
    else:
        results = [
            build_profile(
                output_root=output_root,
                model_name=args.model,
                profile=profile,
                physical_gpu=args.physical_gpu,
                plugin_path=Path(args.plugin).resolve(),
                destination_section=args.destination_section,
            )
            for profile in profiles
        ]
    _write_json(
        output_root / args.destination_section / args.model / "sensitivity_build_matrix.json",
        results,
    )
    print(json.dumps(results, sort_keys=True))
    return 0 if all(row["status"] == "ok" for row in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
