from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _candidate(candidate_id: str, overrides: dict[str, str], pruning: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "deploy_mode": "single_engine_maxK",
        "fixed_K": 29696,
        "pruning": pruning or {"enabled": False},
        "precision_config": {"default": "FP16", "overrides": overrides},
    }


def _safe(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def _load_units(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    return list(data.get("units") or [])


def build(args: argparse.Namespace) -> dict[str, Any]:
    units = _load_units(args.inventory)
    conv = [u["unit_id"] for u in units if u.get("unit_type") in {"conv_bn_act", "gemm"}]
    int8 = [u["unit_id"] for u in units if u.get("unit_type") in {"conv_bn_act", "gemm"} and u.get("has_activation_scale") and u.get("int8_supported")]
    head = [u for u in conv if "head" in u]
    shrink = [u for u in conv if "shrink" in u] or ["shrink"]
    stage2 = [u for u in conv if "layer1" in u or "stage2" in u]
    stage3 = [u for u in conv if "layer2" in u or "stage3" in u]
    candidates = [
        _candidate("baseline_like_fp16_head_fp32_route2", {"detection_head": "FP32"}),
        _candidate("baseline_like_fp16_shrink_int8_head_fp16_route2", {"shrink": "INT8"}),
        _candidate("baseline_like_fp16_head_fp32_shrink_int8_route2", {"detection_head": "FP32", "shrink": "INT8"}),
        _candidate("baseline_like_fp16_stage2_fp32_route2", {"backbone.stage2": "FP32"}),
        _candidate("baseline_like_fp16_stage3_fp32_route2", {"backbone.stage3": "FP32"}),
    ]
    pool = (stage2[:12] + stage3[:12] + shrink[:4] + head[:6] + conv[:20])
    for uid in pool[:30]:
        candidates.append(_candidate(f"onehot_{_safe(uid)}_fp32_route2", {uid: "FP32"}))
        if uid in int8:
            candidates.append(_candidate(f"onehot_{_safe(uid)}_int8_route2", {uid: "INT8"}))
    for idx, uid in enumerate(pool[:20]):
        other = pool[(idx + 5) % len(pool)] if pool else uid
        overrides = {uid: "FP32", other: "INT8" if other in int8 else "FP32"}
        candidates.append(_candidate(f"pairwise_{idx:03d}_{_safe(uid)}_{_safe(other)}_route2", overrides))
    for idx in range(20):
        overrides = {}
        for off, uid in enumerate(pool[idx: idx + 4]):
            overrides[uid] = "INT8" if uid in int8 and (idx + off) % 3 == 0 else "FP32"
        candidates.append(_candidate(f"random_legal_route2_{idx:03d}", overrides))
    protected = ["encoder_m1", "pillar_vfe", "voxel_encoder", "backbone_m1.resnet.layer0", "cls_head", "reg_head", "dir_head"]
    for idx, importance in enumerate(["l1", "first_order_taylor", "second_order_fisher", "l1", "l1", "first_order_taylor", "second_order_fisher", "l1", "l1", "first_order_taylor"]):
        keep = [0.97, 0.95, 0.875, 0.75][idx % 4]
        overrides = {"detection_head": "FP32"} if idx % 3 == 0 else ({"shrink": "INT8"} if idx % 3 == 1 else {"detection_head": "FP32", "shrink": "INT8"})
        pruning = {"enabled": True, "source": "pruning_tool", "importance": importance, "scope": "global", "target_keep_ratio": keep, "min_keep_ratio": 0.875, "align": 8, "respect_group_conv_alignment": True, "extra_protected_prefixes": protected, "num_calib_batches": 1}
        candidates.append(_candidate(f"pruned_mixed_{idx:03d}_{importance}_keep{int(keep*1000)}_route2", overrides, pruning))
    by_id = {c["candidate_id"]: c for c in candidates}
    out = {"candidates": list(by_id.values()), "candidate_count": len(by_id)}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    structured = []
    width_choices = [8, 16, 32, 64, 128]
    selected_units = conv[:20]
    for idx in range(100):
        width = width_choices[idx % len(width_choices)]
        resolved = []
        for uid in selected_units:
            resolved.append({"unit_id": uid, "C_in": width, "C_out": width, "C_in_aligned8": width, "C_out_aligned8": width})
        structured.append(
            {
                "structure_candidate_id": f"structured_width_{idx:03d}",
                "group_widths": {f"group_{j:03d}": width_choices[(idx + j) % len(width_choices)] for j in range(min(8, len(selected_units)))},
                "resolved_units": resolved,
                "is_shape_legal": True,
                "illegal_reason": "",
                "note": "v4 legal width candidate scaffold; formal ChannelResolver coupling replay remains required before physical pruning export",
            }
        )
    Path(args.structured_width_output).write_text(json.dumps({"candidates": structured}, indent=2), encoding="utf-8")
    print(json.dumps({"candidate_count": len(by_id), "int8_units": len(int8), "conv_units": len(conv)}, indent=2))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", default="outputs/latency_lut/atomic_deployment_unit_inventory_v2.json")
    parser.add_argument("--output", default="outputs/latency_lut/full_engine_candidate_plan_v4.json")
    parser.add_argument("--structured-width-output", "--structured_width_output", dest="structured_width_output", default="outputs/latency_lut/structured_width_candidate_space_v4.json")
    return parser.parse_args()


def main() -> int:
    build(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
