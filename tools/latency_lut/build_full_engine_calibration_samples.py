from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from opencood.tools.compression.latency_lut.latency_proxy import LatencyProxy
from opencood.tools.compression.latency_lut.lut_database import LatencyLUTDatabase
from opencood.tools.compression.latency_lut.schema import DEPLOY_MODE, FIXED_K, LatencyRecord, read_record_jsonl


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build full-engine calibration samples for latency LUT calibration.")
    parser.add_argument("--lut", default="outputs/latency_lut/lut_records.jsonl")
    parser.add_argument("--output", default="outputs/latency_lut/full_engine_samples.jsonl")
    parser.add_argument("--report", default="outputs/latency_lut/full_engine_samples_report.json")
    parser.add_argument("--candidate-dir", "--candidate_dir", dest="candidate_dir", default="outputs/latency_lut/full_engine_candidates")
    parser.add_argument(
        "--candidate-preset",
        "--candidate_preset",
        dest="candidate_preset",
        default="lidar_pyramid_fp16_fp32_pruning10",
        choices=["lidar_pyramid_fp16_fp32_pruning10", "lut_records"],
    )
    parser.add_argument("--num-samples", "--num_samples", dest="num_samples", type=int, default=20)
    parser.add_argument("--command-template", "--command_template", dest="command_template", default=None)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true")
    return parser.parse_args(argv)


def _stable_hash(data: Any) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _success_records(path: str | Path) -> list[LatencyRecord]:
    return [record for record in read_record_jsonl(path) if record.status in {"success", "ok"} and record.key.block_type != "precision_boundary"]


def _unit_from_record(record: LatencyRecord) -> dict[str, Any]:
    key = record.key
    return {
        "unit_id": f"{key.module_name}.{key.block_name}",
        "module_name": key.module_name,
        "block_name": key.block_name,
        "block_type": key.block_type,
        "H": key.H,
        "W": key.W,
        "C_in": key.C_in,
        "C_mid": key.C_mid,
        "C_out": key.C_out,
        "kernel_size": key.kernel_size,
        "stride": key.stride,
        "padding": key.padding,
        "dilation": key.dilation,
        "groups": key.groups,
        "precision": key.precision_profile,
        "plugin_name": key.plugin_name,
        "plugin_version": key.plugin_version,
        "metadata": key.metadata,
    }


def _candidate_records(records: list[LatencyRecord], index: int) -> list[LatencyRecord]:
    grouped: dict[tuple[str, str, str], list[LatencyRecord]] = {}
    for record in records:
        grouped.setdefault((record.key.module_name, record.key.block_name, record.key.block_type), []).append(record)
    selected = []
    for group_key in sorted(grouped):
        options = sorted(grouped[group_key], key=lambda item: (item.key.precision_profile, item.key.C_in or 0, item.key.C_out or 0))
        selected.append(options[index % len(options)])
    return selected


def _precision_profile(label: str) -> str:
    value = str(label).upper()
    if value in {"FP16", "TRT_FP16"}:
        return "TRT_FP16"
    if value in {"FP32", "TRT_FP32"}:
        return "TRT_FP32"
    if value in {"INT8", "TRT_INT8_QDQ"}:
        return "TRT_INT8_QDQ"
    raise ValueError(f"unsupported preset precision: {label}")


def _records_by_unit(records: list[LatencyRecord]) -> dict[tuple[str, str, str], list[LatencyRecord]]:
    grouped: dict[tuple[str, str, str], list[LatencyRecord]] = {}
    for record in records:
        if record.status not in {"success", "ok"}:
            continue
        if record.key.block_type == "precision_boundary":
            continue
        grouped.setdefault((record.key.module_name, record.key.block_name, record.key.block_type), []).append(record)
    return grouped


def _record_channel_score(record: LatencyRecord, target_ratio: float, max_channels: int | None) -> tuple[int, float, int]:
    channels = record.key.C_out or record.key.C_mid or record.key.C_in
    if not channels or not max_channels:
        return (0, 0.0, 0)
    target = max(1.0, float(max_channels) * float(target_ratio))
    upward = int(channels < target)
    return (upward, abs(float(channels) - target), -int(channels))


def _select_unit_records(
    records: list[LatencyRecord],
    *,
    default_precision: str,
    keep_ratio: float,
    module_keep_overrides: dict[str, float] | None = None,
    precision_overrides: dict[str, str] | None = None,
) -> list[LatencyRecord]:
    grouped = _records_by_unit(records)
    selected: list[LatencyRecord] = []
    module_keep_overrides = module_keep_overrides or {}
    precision_overrides = precision_overrides or {}
    for group_key in sorted(grouped):
        module_name, _block_name, _block_type = group_key
        precision = _precision_profile(precision_overrides.get(module_name, default_precision))
        options = [record for record in grouped[group_key] if record.key.precision_profile == precision]
        if not options:
            continue
        max_channels = max((record.key.C_out or record.key.C_mid or record.key.C_in or 0) for record in options) or None
        ratio = float(module_keep_overrides.get(module_name, keep_ratio))
        selected.append(sorted(options, key=lambda record: _record_channel_score(record, ratio, max_channels))[0])
    return selected


def _enrich_candidate_with_records(candidate: dict[str, Any], records: list[LatencyRecord]) -> dict[str, Any]:
    if not records:
        candidate.setdefault("units", [])
        candidate.setdefault("channel_config", {})
        candidate.setdefault("precision_config", {"default": candidate.get("precision_config", {}).get("default", "FP16")})
        candidate["channel_config_hash"] = _stable_hash(candidate["channel_config"])
        candidate["quant_config_hash"] = _stable_hash(candidate["precision_config"])
        return candidate
    pruning = dict(candidate.get("pruning") or {})
    keep_ratio = float(pruning.get("target_keep_ratio", 1.0))
    selected = _select_unit_records(
        records,
        default_precision=dict(candidate.get("precision_config") or {}).get("default", "FP16"),
        keep_ratio=keep_ratio,
        module_keep_overrides=dict(candidate.get("module_keep_overrides") or {}),
        precision_overrides=dict(candidate.get("precision_overrides") or {}),
    )
    units = [_unit_from_record(record) for record in selected]
    channel_config = {
        unit["unit_id"]: {"C_in": unit["C_in"], "C_mid": unit["C_mid"], "C_out": unit["C_out"]}
        for unit in units
    }
    precision_config = {"default": dict(candidate.get("precision_config") or {}).get("default", "FP16")}
    precision_config.update({unit["unit_id"]: unit["precision"] for unit in units})
    candidate["units"] = units
    candidate["channel_config"] = channel_config
    candidate["precision_config"] = precision_config
    candidate["channel_config_hash"] = _stable_hash(channel_config)
    candidate["quant_config_hash"] = _stable_hash(precision_config)
    return candidate


def build_named_candidate_preset(
    *,
    num_samples: int = 10,
    records: list[LatencyRecord] | None = None,
) -> list[dict[str, Any]]:
    very_light_protected = [
        "encoder_m1",
        "pillar_vfe",
        "voxel_encoder",
        "backbone_m1.resnet.layer0",
        "shrink_conv",
        "cls_head",
        "reg_head",
        "dir_head",
        "pyramid_backbone.deblocks",
        "pyramid_fusion",
        "fusion_net",
    ]
    specs: list[dict[str, Any]] = [
        {
            "candidate_id": "baseline_like_fp16",
            "pruning": {"enabled": False},
            "precision_config": {"default": "FP16"},
        },
        {
            "candidate_id": "baseline_like_fp32",
            "pruning": {"enabled": False},
            "precision_config": {"default": "FP32"},
        },
        {
            "candidate_id": "light_prune_fp16",
            "pruning": {"enabled": True, "source": "pruning_tool", "importance": "l1", "scope": "global", "target_keep_ratio": 0.97, "min_keep_ratio": 0.875, "align": 8, "respect_group_conv_alignment": True, "extra_protected_prefixes": very_light_protected},
            "precision_config": {"default": "FP16"},
        },
        {
            "candidate_id": "medium_prune_fp16",
            "pruning": {"enabled": True, "source": "pruning_tool", "importance": "l1", "scope": "global", "target_keep_ratio": 0.75, "align": 8, "respect_group_conv_alignment": True},
            "precision_config": {"default": "FP16"},
        },
        {
            "candidate_id": "heavy_prune_fp16",
            "pruning": {"enabled": True, "source": "pruning_tool", "importance": "l1", "scope": "global", "target_keep_ratio": 0.625, "align": 8, "respect_group_conv_alignment": True},
            "precision_config": {"default": "FP16"},
        },
        {
            "candidate_id": "backbone_heavy_prune_fp16",
            "pruning": {"enabled": True, "source": "pruning_tool", "importance": "l1", "scope": "global", "target_keep_ratio": 0.75, "align": 8, "respect_group_conv_alignment": True},
            "module_keep_overrides": {"backbone": 0.5, "shrink": 0.75, "detection_head": 1.0},
            "precision_config": {"default": "FP16"},
        },
        {
            "candidate_id": "fusion_heavy_prune_fp16",
            "pruning": {"enabled": True, "source": "pruning_tool", "importance": "l1", "scope": "global", "target_keep_ratio": 0.75, "align": 8, "respect_group_conv_alignment": True},
            "module_keep_overrides": {"pyramid_fusion": 0.5, "backbone": 0.875, "detection_head": 1.0},
            "precision_config": {"default": "FP16"},
        },
        {
            "candidate_id": "head_kept_medium_prune_fp16",
            "pruning": {"enabled": True, "source": "pruning_tool", "importance": "l1", "scope": "global", "target_keep_ratio": 0.75, "align": 8, "respect_group_conv_alignment": True},
            "module_keep_overrides": {"detection_head": 1.0},
            "precision_config": {"default": "FP16"},
        },
        {
            "candidate_id": "light_prune_fp32",
            "pruning": {"enabled": True, "source": "pruning_tool", "importance": "l1", "scope": "global", "target_keep_ratio": 0.97, "min_keep_ratio": 0.875, "align": 8, "respect_group_conv_alignment": True, "extra_protected_prefixes": very_light_protected},
            "precision_config": {"default": "FP32"},
        },
        {
            "candidate_id": "medium_prune_mixed_fp16_fp32",
            "pruning": {"enabled": True, "source": "pruning_tool", "importance": "l1", "scope": "global", "target_keep_ratio": 0.75, "align": 8, "respect_group_conv_alignment": True},
            "precision_config": {"default": "FP16"},
            "precision_overrides": {"detection_head": "FP32", "pyramid_fusion": "FP32"},
            "full_engine_note": "mixed FP16/FP32 is retained as a calibration candidate spec, but the current full-engine runner will fail it unless a mixed-precision full ONNX/build path is added.",
        },
    ]
    out = []
    for spec in specs[: max(0, int(num_samples))]:
        candidate = {
            "deploy_mode": DEPLOY_MODE,
            "fixed_K": FIXED_K,
            **spec,
        }
        out.append(_enrich_candidate_with_records(candidate, list(records or [])))
    return out


def build_candidate(records: list[LatencyRecord], index: int) -> dict[str, Any]:
    selected = _candidate_records(records, index)
    units = [_unit_from_record(record) for record in selected]
    channel_config = {
        unit["unit_id"]: {"C_in": unit["C_in"], "C_mid": unit["C_mid"], "C_out": unit["C_out"]}
        for unit in units
    }
    precision_config = {unit["unit_id"]: unit["precision"] for unit in units}
    return {
        "candidate_id": f"cand_{index:04d}",
        "deploy_mode": DEPLOY_MODE,
        "fixed_K": FIXED_K,
        "units": units,
        "channel_config": channel_config,
        "precision_config": precision_config,
        "channel_config_hash": _stable_hash(channel_config),
        "quant_config_hash": _stable_hash(precision_config),
    }


def _run_template(command_template: str, *, candidate_path: Path, result_path: Path, candidate_id: str, timeout: int) -> dict[str, Any]:
    rendered = command_template.format(
        candidate_json=str(candidate_path),
        output_json=str(result_path),
        candidate_id=candidate_id,
    )
    cmd = shlex.split(rendered)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=int(timeout), check=False)
    result_data = None
    if result_path.is_file():
        try:
            result_data = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            result_data = None
    if proc.returncode != 0:
        return {
            "success": False,
            "error": f"full-engine command failed with returncode {proc.returncode}",
            "stdout": proc.stdout,
            "command": cmd,
            "result": result_data,
        }
    if not result_path.is_file():
        return {
            "success": False,
            "error": f"full-engine command did not write result JSON: {result_path}",
            "stdout": proc.stdout,
            "command": cmd,
        }
    data = json.loads(result_path.read_text(encoding="utf-8"))
    data.setdefault("command", cmd)
    data.setdefault("stdout", proc.stdout)
    data["success"] = True
    return data


def _sample_record(candidate: dict[str, Any], estimate: Any, real: dict[str, Any]) -> dict[str, Any]:
    p50 = real.get("T_real_p50", real.get("real_engine_p50_ms", real.get("latency_p50_ms")))
    p90 = real.get("T_real_p90", real.get("real_engine_p90_ms", real.get("latency_p90_ms")))
    p95 = real.get("T_real_p95", real.get("real_engine_p95_ms", real.get("latency_p95_ms")))
    mean = real.get("T_real_mean", real.get("real_engine_mean_ms", real.get("latency_mean_ms")))
    std = real.get("T_real_std", real.get("real_engine_std_ms", real.get("latency_std_ms")))
    if p50 is None:
        raise ValueError("full-engine result JSON must contain T_real_p50 or real_engine_p50_ms")
    return {
        "candidate_id": candidate["candidate_id"],
        "deploy_mode": DEPLOY_MODE,
        "fixed_K": FIXED_K,
        "channel_config": candidate["channel_config"],
        "precision_config": candidate["precision_config"],
        "predicted_lut_ms": float(estimate.latency_lut_raw_ms),
        "T_lut_pred": float(estimate.latency_lut_raw_ms),
        "latency_proxy_ms": float(estimate.latency_ms),
        "real_engine_p50_ms": float(p50),
        "real_engine_p90_ms": float(p90) if p90 is not None else None,
        "real_engine_p95_ms": float(p95) if p95 is not None else None,
        "real_engine_mean_ms": float(mean) if mean is not None else None,
        "real_engine_std_ms": float(std) if std is not None else None,
        "T_real_p50": float(p50),
        "T_real_p90": float(p90) if p90 is not None else None,
        "T_real_p95": float(p95) if p95 is not None else None,
        "T_real_mean": float(mean) if mean is not None else None,
        "T_real_std": float(std) if std is not None else None,
        "num_val_frames": real.get("num_val_frames"),
        "status": real.get("status", "success"),
        "mAP": real.get("mAP"),
        "AP_0_70": real.get("AP_0_70", real.get("AP@0.70")),
        "engine_hash": real.get("engine_hash"),
        "onnx_hash": real.get("onnx_hash"),
        "channel_config_hash": candidate["channel_config_hash"],
        "quant_config_hash": candidate["quant_config_hash"],
        "features": estimate.calibration_features,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    records = _success_records(args.lut)
    candidate_dir = Path(args.candidate_dir)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    output = Path(args.output)
    report_path = Path(args.report)
    db = LatencyLUTDatabase.from_jsonl(args.lut)
    proxy = LatencyProxy(db, kappa=0.0)
    if args.candidate_preset == "lut_records":
        candidates = [build_candidate(records, idx) for idx in range(max(0, int(args.num_samples)))] if records else []
    else:
        candidates = build_named_candidate_preset(num_samples=int(args.num_samples), records=records)
    for candidate in candidates:
        estimate = proxy.estimate(candidate)
        candidate["T_lut_pred"] = float(estimate.latency_lut_raw_ms)
        candidate["latency_proxy_ms"] = float(estimate.latency_ms)
        candidate["features"] = estimate.calibration_features
        (candidate_dir / f"{candidate['candidate_id']}.json").write_text(json.dumps(candidate, ensure_ascii=False, indent=2), encoding="utf-8")

    stats: dict[str, Any] = {
        "status": "not_started",
        "lut": str(args.lut),
        "output": str(output),
        "candidate_dir": str(candidate_dir),
        "num_requested": int(args.num_samples),
        "num_candidates": len(candidates),
        "num_samples_written": 0,
        "num_failed": 0,
        "dry_run": bool(args.dry_run),
    }
    if not candidates:
        stats.update({"status": "skipped_no_success_lut_records", "reason": "no successful LUT records are available"})
    elif args.dry_run:
        stats.update({"status": "dry_run", "reason": "candidate configs written; full-engine command was not executed"})
    elif not args.command_template:
        stats.update(
            {
                "status": "skipped_no_engine_builder",
                "reason": "no --command-template was provided; refusing to fabricate full-engine calibration samples",
            }
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("", encoding="utf-8")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        samples = []
        failures = []
        for candidate in candidates:
            candidate_path = candidate_dir / f"{candidate['candidate_id']}.json"
            result_path = candidate_dir / f"{candidate['candidate_id']}.full_engine_result.json"
            real = _run_template(
                str(args.command_template),
                candidate_path=candidate_path,
                result_path=result_path,
                candidate_id=candidate["candidate_id"],
                timeout=int(args.timeout),
            )
            if not real.get("success"):
                failures.append({"candidate_id": candidate["candidate_id"], **real})
                continue
            estimate = proxy.estimate(candidate)
            try:
                samples.append(_sample_record(candidate, estimate, real))
            except Exception as exc:
                failures.append({"candidate_id": candidate["candidate_id"], "error": str(exc), "result": real})
        with output.open("w", encoding="utf-8") as handle:
            for sample in samples:
                handle.write(json.dumps(sample, ensure_ascii=False, sort_keys=True) + "\n")
        if failures:
            (output.with_suffix(".failed.json")).write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
        stats.update(
            {
                "status": "success" if samples else "failed_no_samples",
                "num_samples_written": len(samples),
                "num_failed": len(failures),
            }
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def main(argv: list[str] | None = None) -> int:
    stats = run(parse_args(argv))
    return 0 if stats.get("status") not in {"failed_no_samples"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
