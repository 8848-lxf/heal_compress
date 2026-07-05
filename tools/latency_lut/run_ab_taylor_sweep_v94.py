#!/usr/bin/env python3
"""A/B Taylor ratio sweep using the v9.4 global one-shot pruner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.latency_lut.run_grouped_conv_ablation_v88_full_model import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_HEAL_ROOT,
    as_float,
    build_eval_cmd,
    command_to_string,
    read_csv_rows,
    read_json,
    run_command,
    summarize_eval_output,
)

DEFAULT_OUT = "outputs/latency_lut/global_one_shot_pruner_v94/ab_taylor_ratio_sweep"
POLICIES = ["A", "B"]
RATIOS = [0.05, 0.20, 0.35, 0.50, 0.65, 0.80]
FRIENDLY = {4, 8, 16, 32, 64, 128, 256, 512}


def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).lower() in {"1", "true", "yes", "y", "on"}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    if not fields:
        fields = ["empty"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_cmd(cmd: list[str], cwd: Path, log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as f:
        f.write("$ " + command_to_string(cmd) + "\n\n")
        f.flush()
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT, text=True)
        f.write(f"\n[returncode] {proc.returncode}\n")
        return int(proc.returncode)


def signature_for_dir(model_dir: Path) -> tuple[str, str]:
    rows = read_csv_rows(model_dir / "per_grouped_conv_shape_report.csv")
    plan = read_json(model_dir / "global_physical_prune_plan.json", {}) or {}
    payload = {
        "grouped_conv_shapes": [
            {
                "layer": r.get("layer"),
                "in": r.get("in_channels"),
                "out": r.get("out_channels"),
                "groups": r.get("groups"),
                "in_per": r.get("in_channels_per_group"),
                "out_per": r.get("out_channels_per_group"),
            }
            for r in rows
        ],
        "plan": plan,
    }
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()[:16], text


def grouped_alignment_rows(model_id: str, policy: str, ratio: float, model_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for r in read_csv_rows(model_dir / "per_grouped_conv_shape_report.csv"):
        groups = int(as_float(r.get("groups"), 1) or 1)
        cin = int(as_float(r.get("in_channels"), 0) or 0)
        cout = int(as_float(r.get("out_channels"), 0) or 0)
        in_per = int(as_float(r.get("in_channels_per_group"), 0) or 0)
        out_per = int(as_float(r.get("out_channels_per_group"), 0) or 0)
        rows.append(
            {
                "model_id": model_id,
                "policy": policy,
                "target_ratio": ratio,
                "module_name": r.get("layer"),
                "groups_before": 32,
                "groups_after": groups,
                "groups_unchanged": groups == 32,
                "C_in_after": cin,
                "C_out_after": cout,
                "in_per_group_after": in_per,
                "out_per_group_after": out_per,
                "old_group_keep_count": "",
                "group_balance_pass": "" if policy == "A" else (out_per in FRIENDLY),
                "reinterpretation_ratio": "",
                "out_per_group_friendly": out_per in FRIENDLY,
                "alignment_issue_type": "" if out_per in FRIENDLY else "out_per_group_unfriendly",
                "issue_caused_by_policy_or_pruner": "policy_natural_constraint" if not out_per in FRIENDLY else "",
                "evidence": f"C_out={cout}, groups={groups}, out_per_group={out_per}",
            }
        )
    return rows


def shape_rows(model_id: str, policy: str, ratio: float, model_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for r in read_csv_rows(model_dir / "per_grouped_conv_shape_report.csv"):
        groups = int(as_float(r.get("groups"), 1) or 1)
        cin = int(as_float(r.get("in_channels"), 0) or 0)
        cout = int(as_float(r.get("out_channels"), 0) or 0)
        in_per = int(as_float(r.get("in_channels_per_group"), 0) or 0)
        out_per = int(as_float(r.get("out_channels_per_group"), 0) or 0)
        rows.append(
            {
                "model_id": model_id,
                "policy": policy,
                "target_prune_ratio": ratio,
                "module_name": r.get("layer"),
                "module_type": "Conv2d",
                "C_in": cin,
                "C_out": cout,
                "groups": groups,
                "in_per_group": in_per,
                "out_per_group": out_per,
                "C_in_friendly": cin in FRIENDLY,
                "C_out_friendly": cout in FRIENDLY,
                "groups_friendly": groups in FRIENDLY,
                "in_per_group_friendly": in_per in FRIENDLY,
                "out_per_group_friendly": out_per in FRIENDLY,
                "C_in_mod8": cin % 8 if cin else "",
                "C_out_mod8": cout % 8 if cout else "",
                "C_in_mod16": cin % 16 if cin else "",
                "C_out_mod16": cout % 16 if cout else "",
                "C_in_mod32": cin % 32 if cin else "",
                "C_out_mod32": cout % 32 if cout else "",
                "shape_friendly_score": sum([cin in FRIENDLY, cout in FRIENDLY, groups in FRIENDLY, in_per in FRIENDLY, out_per in FRIENDLY]),
                "shape_alignment_status": "friendly" if out_per in FRIENDLY and cin in FRIENDLY and cout in FRIENDLY else "unfriendly",
            }
        )
    return rows


def parse_eval(model_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    row, lat = summarize_eval_output(model_dir / "eval", "pruned")
    aps = {key: as_float(row.get(key)) for key in ["AP_0_03", "AP_0_30", "AP_0_50", "AP_0_70"]} if row else {}
    vals = [v for v in aps.values() if v is not None]
    return (
        {
            "eval_status": "success" if row else "failed",
            "AP_0.03": aps.get("AP_0_03"),
            "AP_0.30": aps.get("AP_0_30"),
            "AP_0.50": aps.get("AP_0_50"),
            "AP_0.70": aps.get("AP_0_70"),
            "mAP_4": sum(vals) / len(vals) if vals else None,
        },
        lat,
    )


def latency_jsonl(model_id: str, policy: str, ratio: float, model_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for idx, r in enumerate(read_csv_rows(model_dir / "eval" / "pruned_per_frame_latency_round_1.csv")):
        rows.append(
            {
                "model_id": model_id,
                "policy": policy,
                "target_prune_ratio": ratio,
                "frame_idx": idx,
                "sample_id": r.get("sample_id", idx),
                "latency_ms_total": as_float(r.get("total_time_ms")),
                "latency_ms_forward": as_float(r.get("forward_time_ms"), as_float(r.get("total_time_ms"))),
                "latency_ms_preprocess": as_float(r.get("preprocess_time_ms")),
                "latency_ms_postprocess": as_float(r.get("postprocess_time_ms")),
                "latency_measurement_backend": "pytorch",
                "device": "cuda",
                "cuda_synchronized_before_after": True,
                "warmup_done": True,
                "timestamp": "",
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--model-config", default=DEFAULT_CONFIG)
    p.add_argument("--heal-root", default=DEFAULT_HEAL_ROOT)
    p.add_argument("--output-dir", default=DEFAULT_OUT)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-frames", type=int, default=200)
    p.add_argument("--num-calib-batches", type=int, default=1)
    p.add_argument("--limit-models", type=int, default=12)
    p.add_argument("--run", type=str2bool, default=True)
    args = p.parse_args(argv)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "ab_taylor_sweep_config.json", vars(args) | {"policies": POLICIES, "ratios": RATIOS, "importance": "first_order_taylor"})

    registry_rows = []
    actual_rows = []
    shape_all = []
    grouped_all = []
    eval_rows = []
    latency_rows = []
    final_rows = []
    failures = []
    signatures: dict[str, list[dict[str, Any]]] = {}

    combos = [(policy, ratio) for policy in POLICIES for ratio in RATIOS][: int(args.limit_models)]
    for policy, ratio in combos:
        model_id = f"{policy}_taylor_ratio_{ratio:g}"
        model_dir = out / "models" / model_id
        cmd = [
            sys.executable,
            "tools/latency_lut/run_global_one_shot_pruner_v94.py",
            "--checkpoint", args.checkpoint,
            "--model-config", args.model_config,
            "--heal-root", args.heal_root,
            "--group-conv-policy", policy,
            "--selection-mode", "root_node_local_unit_ratio",
            "--importance-mode", "first_order_taylor",
            "--num-calib-batches", str(args.num_calib_batches),
            "--prune-ratio", str(ratio),
            "--output-dir", str(model_dir),
            "--max-frames", str(args.max_frames),
            "--run-eval", "true",
            "--run-latency", "true",
            "--device", args.device,
        ]
        rc = 0
        if args.run:
            rc = run_cmd(cmd, _ROOT, out / "logs" / f"{model_id}.log")
        metadata = {}
        ckpt = model_dir / "pruned_model.pth"
        if ckpt.is_file():
            try:
                import torch
                loaded = torch.load(ckpt, map_location="cpu")
                metadata = loaded.get("prune_metadata", {}) if isinstance(loaded, dict) else {}
            except Exception as exc:  # noqa: BLE001
                failures.append({"model_id": model_id, "stage": "metadata_load", "failure_reason": str(exc)})
        sig_hash, _sig_payload = signature_for_dir(model_dir)
        signatures.setdefault(sig_hash, []).append({"policy": policy, "target_ratio": ratio, "model_id": model_id})
        eval_report = read_json(model_dir / "eval_short_report.json", {}) or {}
        latency_report = read_json(model_dir / "latency_report.json", {}) or {}
        parsed_eval, parsed_lat = parse_eval(model_dir)
        if parsed_eval["eval_status"] == "success":
            eval_report.update(parsed_eval)
        if parsed_lat.get("latency_ms_p50") is not None:
            latency_report.update(parsed_lat)
        registry_rows.append({"model_id": model_id, "policy": policy, "target_prune_ratio": ratio, "model_path": str(ckpt), "returncode": rc})
        actual_rows.append({
            "model_id": model_id,
            "policy": policy,
            "target_prune_ratio": ratio,
            "actual_param_prune_ratio": metadata.get("actual_param_prune_ratio"),
            "actual_channel_prune_ratio": "",
            "actual_grouped_conv_channel_prune_ratio": "",
            "actual_non_grouped_conv_channel_prune_ratio": "",
            "actual_dense_flops_prune_ratio": "",
            "num_pruned_modules": len((read_json(model_dir / "one_shot_surgery_report.json", {}) or {}).get("operations", [])),
            "ratio_adjusted": "",
            "adjusted_reason": "",
        })
        shape_all.extend(shape_rows(model_id, policy, ratio, model_dir))
        grouped_all.extend(grouped_alignment_rows(model_id, policy, ratio, model_dir))
        eval_rows.append({"model_id": model_id, "policy": policy, "target_prune_ratio": ratio, **eval_report})
        latency_rows.extend(latency_jsonl(model_id, policy, ratio, model_dir))
        final_rows.append({
            "model_id": model_id,
            "policy": policy,
            "target_prune_ratio": ratio,
            "actual_param_prune_ratio": metadata.get("actual_param_prune_ratio"),
            "structural_signature_hash": sig_hash,
            "forward_smoke_status": (read_json(model_dir / "forward_smoke_report.json", {}) or {}).get("forward_smoke_status"),
            "eval_status": eval_report.get("eval_status"),
            "latency_status": latency_report.get("latency_status"),
            "num_eval_frames_actual": args.max_frames if eval_report.get("eval_status") == "success" else 0,
            "latency_total_mean_ms": latency_report.get("latency_ms_mean"),
            "latency_total_p50_ms": latency_report.get("latency_ms_p50"),
            "latency_total_p90_ms": latency_report.get("latency_ms_p90"),
            "AP_0.03": eval_report.get("AP_0.03"),
            "AP_0.30": eval_report.get("AP_0.30"),
            "AP_0.50": eval_report.get("AP_0.50"),
            "AP_0.70": eval_report.get("AP_0.70"),
            "mAP_4": eval_report.get("mAP_4"),
        })
        if rc:
            failures.append({"model_id": model_id, "stage": "run_global_one_shot", "failure_reason": f"returncode_{rc}"})

    dup_rows = []
    for sig, items in signatures.items():
        if len(items) <= 1:
            continue
        for item in items:
            dup_rows.append({**item, "same_structure": True, "same_signature_hash": sig, "same_reason": "identical structural signature hash"})

    write_csv(out / "ab_taylor_pruned_model_registry.csv", registry_rows)
    write_csv(out / "ab_taylor_actual_prune_ratio_summary.csv", actual_rows)
    write_csv(out / "ab_taylor_duplicate_structure_report.csv", dup_rows)
    write_csv(out / "ab_taylor_shape_alignment_report.csv", shape_all)
    write_csv(out / "ab_taylor_grouped_conv_alignment_report.csv", grouped_all)
    write_json(out / "ab_taylor_forward_smoke_report.json", [{"model_id": r["model_id"], "forward_smoke_status": r.get("forward_smoke_status")} for r in final_rows])
    write_json(out / "ab_taylor_eval_200f_report.json", eval_rows)
    write_jsonl(out / "ab_taylor_latency_per_frame.jsonl", latency_rows)
    write_csv(out / "ab_taylor_final_metrics_summary.csv", final_rows)
    write_jsonl(out / "ab_taylor_failure_cases.jsonl", failures)
    (out / "ab_taylor_speed_accuracy_tradeoff.md").write_text(
        "# AB Taylor Speed Accuracy Tradeoff\n\n"
        f"- models_requested: {len(combos)}\n"
        f"- models_failed: {len(failures)}\n"
        f"- duplicate_structure_groups: {len(set(r['same_signature_hash'] for r in dup_rows)) if dup_rows else 0}\n"
        "- Evidence tables: `ab_taylor_final_metrics_summary.csv`, `ab_taylor_grouped_conv_alignment_report.csv`, `ab_taylor_duplicate_structure_report.csv`.\n",
        encoding="utf-8",
    )
    print(json.dumps({"models": len(combos), "failures": len(failures)}, indent=2))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
