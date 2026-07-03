from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


DEFAULT_PROTECTED = [
    "encoder_m1",
    "pillar_vfe",
    "voxel_encoder",
    "backbone_m1.resnet.layer0",
    "shrink_conv",
    "cls_head",
    "reg_head",
    "dir_head",
]


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.is_file():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _candidate(candidate_id: str, overrides: dict[str, str] | None = None, pruning: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "deploy_mode": "single_engine_maxK",
        "fixed_K": 29696,
        "pruning": pruning or {"enabled": False},
        "precision_config": {
            "default": "FP16",
            "overrides": dict(overrides or {}),
        },
    }


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")


def _select_units(inventory: dict[str, Any], measurements: list[dict[str, Any]]) -> dict[str, list[str]]:
    units = list(inventory.get("units") or [])
    measured_by_unit_precision: set[tuple[str, str]] = {
        (str(row.get("unit_id")), str(row.get("precision"))) for row in measurements if row.get("valid")
    }
    buckets: dict[str, list[str]] = {"early": [], "middle": [], "late": [], "shrink": [], "fusion": [], "head": [], "int8": []}
    for unit in units:
        uid = str(unit.get("unit_id"))
        stage = str(unit.get("parent_stage") or "")
        if unit.get("unit_type") not in {"conv_bn_act", "gemm"}:
            continue
        if (uid, "FP32") in measured_by_unit_precision or unit.get("has_lut_fp32"):
            if "stage1" in stage or "layer0" in uid:
                buckets["early"].append(uid)
            elif "stage2" in stage or "layer1" in uid:
                buckets["middle"].append(uid)
            elif "stage3" in stage or "layer2" in uid:
                buckets["late"].append(uid)
            elif "shrink" in stage or "shrink" in uid:
                buckets["shrink"].append(uid)
            elif "fusion" in stage or "fusion" in uid:
                buckets["fusion"].append(uid)
            elif "head" in stage or "head" in uid:
                buckets["head"].append(uid)
        if ((uid, "INT8_QDQ") in measured_by_unit_precision or unit.get("has_lut_int8")) and unit.get("has_activation_scale"):
            buckets["int8"].append(uid)
    for key in list(buckets):
        seen = []
        for uid in buckets[key]:
            if uid not in seen:
                seen.append(uid)
        buckets[key] = seen
    return buckets


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    inventory = _load_json(args.inventory)
    measurements = _read_jsonl(args.measurements)
    selected = _select_units(inventory, measurements)
    candidates: list[dict[str, Any]] = []
    block_specs = [
        ("baseline_like_fp16_shrink_fp32_route2", {"shrink": "FP32"}),
        ("baseline_like_fp16_fusion_fp32_route2", {"pyramid_fusion": "FP32"}),
        ("baseline_like_fp16_backbone_stage1_fp32_route2", {"backbone.stage1": "FP32"}),
        ("baseline_like_fp16_backbone_stage2_fp32_route2", {"backbone.stage2": "FP32"}),
        ("baseline_like_fp16_backbone_stage3_fp32_route2", {"backbone.stage3": "FP32"}),
        ("baseline_like_fp16_head_fp32_route2", {"detection_head": "FP32"}),
        ("baseline_like_fp16_head_fp32_shrink_int8_route2", {"detection_head": "FP32", "shrink": "INT8"}),
        ("baseline_like_fp16_shrink_int8_head_fp32_route2", {"shrink": "INT8", "detection_head": "FP32"}),
        ("baseline_like_fp16_backbone_stage2_int8_route2", {"backbone.stage2": "INT8"}),
        ("baseline_like_fp16_backbone_stage3_int8_route2", {"backbone.stage3": "INT8"}),
    ]
    candidates.extend(_candidate(cid, overrides) for cid, overrides in block_specs)
    onehot_units = (selected["early"][:3] + selected["middle"][:3] + selected["late"][:3] + selected["shrink"][:2] + selected["fusion"][:2] + selected["head"][:2])
    for uid in onehot_units[: max(0, int(args.num_onehot))]:
        candidates.append(_candidate(f"onehot_{_safe_name(uid)}_fp32_route2", {uid: "FP32"}))
        if uid in selected["int8"]:
            candidates.append(_candidate(f"onehot_{_safe_name(uid)}_int8_route2", {uid: "INT8"}))
    pair_units = onehot_units[: max(2, min(len(onehot_units), int(args.num_pairwise) + 1))]
    for idx in range(max(0, min(int(args.num_pairwise), len(pair_units) - 1))):
        a, b = pair_units[idx], pair_units[idx + 1]
        precision_b = "INT8" if b in selected["int8"] else "FP32"
        candidates.append(_candidate(f"pairwise_{_safe_name(a)}_{_safe_name(b)}_fp32_{precision_b.lower()}_route2", {a: "FP32", b: precision_b}))
    for idx in range(max(0, int(args.num_random))):
        overrides: dict[str, str] = {}
        pool = onehot_units or (selected["early"] + selected["middle"] + selected["late"] + selected["head"])
        for unit_index, uid in enumerate(pool[idx % len(pool): idx % len(pool) + 3] if pool else []):
            if uid in selected["int8"] and (idx + unit_index) % 3 == 0:
                overrides[uid] = "INT8"
            elif (idx + unit_index) % 2 == 0:
                overrides[uid] = "FP32"
        candidates.append(_candidate(f"random_legal_route2_{idx:03d}", overrides))
    pruned_specs = [
        ("global_taylor1_keep875_fp16_head_fp32_route2", "first_order_taylor", {"detection_head": "FP32"}),
        ("global_taylor1_keep875_fp16_shrink_int8_route2", "first_order_taylor", {"shrink": "INT8"}),
        ("global_taylor1_keep875_fp16_head_fp32_shrink_int8_route2", "first_order_taylor", {"detection_head": "FP32", "shrink": "INT8"}),
        ("global_fisher2_keep875_fp16_head_fp32_route2", "second_order_fisher", {"detection_head": "FP32"}),
        ("very_light_prune_97_protected_l1_fp16_shrink_int8_route2", "l1", {"shrink": "INT8"}),
    ]
    for cid, importance, overrides in pruned_specs:
        keep = 0.97 if cid.startswith("very_light") else 0.875
        pruning = {
            "enabled": True,
            "source": "pruning_tool",
            "importance": importance,
            "scope": "global",
            "target_keep_ratio": keep,
            "min_keep_ratio": 0.875,
            "align": 8,
            "respect_group_conv_alignment": True,
            "extra_protected_prefixes": DEFAULT_PROTECTED,
            "num_calib_batches": 1 if "taylor" in importance or "fisher" in importance else None,
        }
        candidates.append(_candidate(cid, overrides, pruning))
    by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    out = {"candidates": list(by_id.values()), "selected_units": selected}
    Path(args.plan).parent.mkdir(parents=True, exist_ok=True)
    Path(args.plan).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _candidate_has_int8(candidate: dict[str, Any]) -> bool:
    precision = candidate.get("precision_config") or {}
    values = [precision.get("default", "FP16")]
    values.extend((precision.get("overrides") or {}).values())
    return any(str(value).upper() == "INT8" for value in values)


def _execute_candidate(args: argparse.Namespace, candidate_path: Path, result_path: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "tools/latency_lut/run_full_engine_candidate_benchmark.py",
        "--candidate",
        str(candidate_path),
        "--output",
        str(result_path),
        "--val-subset-size",
        str(int(args.val_subset_size)),
        "--deploy-mode",
        "single_engine_maxK",
        "--fixed-k",
        "29696",
        "--trtexec",
        str(args.trtexec),
        "--device",
        str(int(args.device)),
        "--timeout",
        str(int(args.timeout)),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False, timeout=int(args.timeout) + 60)
    data: dict[str, Any]
    if result_path.is_file():
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    else:
        data = {}
    data.setdefault("candidate_id", candidate_path.stem)
    data["batch_command"] = cmd
    data["batch_returncode"] = proc.returncode
    data["batch_stdout_tail"] = proc.stdout[-4000:]
    if proc.returncode != 0 and data.get("success"):
        data["success"] = False
        data["status"] = "batch_command_failed"
    return data


def run(args: argparse.Namespace) -> dict[str, Any]:
    plan = build_plan(args)
    candidates = list(plan["candidates"])
    candidate_dir = Path(args.candidate_dir)
    result_dir = Path(args.result_dir)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    if args.limit is not None:
        candidates = candidates[: int(args.limit)]
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_path = candidate_dir / f"{candidate['candidate_id']}.json"
        result_path = result_dir / f"{candidate['candidate_id']}.result.json"
        candidate_path.write_text(json.dumps(candidate, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.dry_run:
            continue
        if args.resume and result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
        else:
            if _candidate_has_int8(candidate) and not any(candidate.get(name) for name in ("qdq_onnx_path", "route2_onnx_path", "mixed_precision_onnx")):
                result = {
                    "candidate_id": candidate["candidate_id"],
                    "success": False,
                    "status": "int8_mixed_qdq_onnx_not_prepared",
                    "failed_stage": "candidate_preflight",
                    "error": "INT8 mixed full-engine candidate requires explicit Route2 Q/DQ ONNX; refusing to fabricate T_real",
                }
                result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            else:
                result = _execute_candidate(args, candidate_path, result_path)
        if result.get("success"):
            results.append(result)
        else:
            failures.append(result)
    _write_jsonl(args.output, results)
    Path(args.failure_report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.failure_report).write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    stats = {
        "status": "dry_run" if args.dry_run else "success",
        "plan": str(args.plan),
        "num_candidates_in_plan": len(plan["candidates"]),
        "num_selected": len(candidates),
        "num_success_results": len(results),
        "num_failures": len(failures),
        "success_by_type": dict(Counter("int8" if _candidate_has_int8(row) else "fp" for row in results)),
        "failure_by_status": dict(Counter(row.get("status", "unknown") for row in failures)),
        "dry_run": bool(args.dry_run),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(
        "# Full Engine Calibration Batch v3 Report\n\n"
        f"- candidates in plan: {stats['num_candidates_in_plan']}\n"
        f"- selected: {stats['num_selected']}\n"
        f"- success result JSONs: {stats['num_success_results']}\n"
        f"- failures: {stats['num_failures']}\n"
        f"- failure by status: `{stats['failure_by_status']}`\n"
        f"- dry_run: `{stats['dry_run']}`\n\n"
        "Successful result JSONs still need decomposition and missing-key checks before entering `full_engine_calibration_dataset_v3.jsonl`.\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", default="outputs/latency_lut/atomic_deployment_unit_inventory_v2.json")
    parser.add_argument("--measurements", default="outputs/latency_lut/lut_measurements_v3.jsonl")
    parser.add_argument("--plan", default="outputs/latency_lut/full_engine_calibration_candidate_plan_v3.json")
    parser.add_argument("--candidate-dir", "--candidate_dir", dest="candidate_dir", default="outputs/latency_lut/full_engine_candidates_v3")
    parser.add_argument("--result-dir", "--result_dir", dest="result_dir", default="outputs/latency_lut/full_engine_candidate_results_v3")
    parser.add_argument("--output", default="outputs/latency_lut/full_engine_calibration_batch_v3_results.jsonl")
    parser.add_argument("--failure-report", "--failure_report", dest="failure_report", default="outputs/latency_lut/full_engine_calibration_batch_v3_failures.json")
    parser.add_argument("--report", default="outputs/latency_lut/full_engine_calibration_batch_v3_report.md")
    parser.add_argument("--num-onehot", "--num_onehot", dest="num_onehot", type=int, default=15)
    parser.add_argument("--num-pairwise", "--num_pairwise", dest="num_pairwise", type=int, default=10)
    parser.add_argument("--num-random", "--num_random", dest="num_random", type=int, default=10)
    parser.add_argument("--val-subset-size", "--val_subset_size", dest="val_subset_size", type=int, default=50)
    parser.add_argument("--trtexec", default="/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118/targets/x86_64-linux-gnu/bin/trtexec")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
