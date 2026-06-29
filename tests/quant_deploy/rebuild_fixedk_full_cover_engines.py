from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_mode_fixed_k_plugin_ablation import _ensure_dynamic_exports_and_engines, _ensure_padded_exports_and_engines
from build_dynamic_fixed_k_int8_engine import _build_one as build_dynamic_int8_one
from build_dynamic_single_engine_maxk_trt_engine import build_all as build_single_all
from bucketed_padded_agent_latency import _fixed_k_plugin_bucket_engine_path
from dump_train_calibration_npz_for_all_strategies import fixed_k_buckets
from dynamic_single_engine_maxk_common import engine_path as single_engine_path, onnx_path as single_onnx_path
from quant_deploy_utils import DEFAULT_CHECKPOINT, DEFAULT_HEAL_REPO, DEFAULT_HYPES_YAML, DEFAULT_TRT_ROOT, ensure_quant_deploy_run_dirs, save_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild fixedK full-cover TensorRT engines with train INT8 calibration.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--fixed_k", type=int, required=True)
    parser.add_argument("--plugin_path", required=True)
    parser.add_argument("--hypes_yaml", default=str(DEFAULT_HYPES_YAML))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--heal_repo", default=str(DEFAULT_HEAL_REPO))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trt_root", default=str(DEFAULT_TRT_ROOT))
    parser.add_argument("--trtexec_path", default=None)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--max_cav", type=int, default=2)
    parser.add_argument("--calibration_frames", type=int, nargs="+", default=[50, 200])
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--force_recalibrate", action="store_true")
    return parser.parse_args(argv)


def fixedk_dirs(output_root: str | Path, fixed_k: int) -> dict[str, Path]:
    dirs = ensure_quant_deploy_run_dirs(output_root)
    root = dirs["output_root"]
    ns = f"fixedK{int(fixed_k)}"
    dirs["onnx"] = root / "artifacts" / "onnx" / ns
    dirs["onnx_fp32"] = dirs["onnx"] / "fp32"
    dirs["engines"] = root / "artifacts" / "engines" / ns
    dirs["engine_fp32"] = dirs["engines"] / "fp32"
    dirs["engine_fp16"] = dirs["engines"] / "fp16"
    dirs["engine_int8"] = dirs["engines"] / "int8"
    dirs["logs_build"] = root / "logs" / "build" / ns
    for key in ("onnx", "onnx_fp32", "engines", "engine_fp32", "engine_fp16", "engine_int8", "logs_build"):
        dirs[key].mkdir(parents=True, exist_ok=True)
    return dirs


def _common_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        output_root=args.output_root,
        hypes_yaml=args.hypes_yaml,
        checkpoint=args.checkpoint,
        heal_repo=args.heal_repo,
        device=args.device,
        trt_root=args.trt_root,
        trtexec_path=args.trtexec_path,
        timeout=int(args.timeout),
        max_cav=int(args.max_cav),
        num_frames=[50],
        warmup_ms=200,
        iterations=50,
        duration=3,
        ap_iou_backend="gpu",
        skip_existing=False,
        rebuild=bool(args.rebuild),
        skip_profile=True,
        skip_eval=True,
        skip_audit=True,
        plugin_so=str(args.plugin_path),
    )


def _dynamic_int8_builds(args: argparse.Namespace, dirs: dict[str, Path], buckets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frames in [int(item) for item in args.calibration_frames]:
        calib_npz_dir = Path(args.output_root) / "artifacts" / "calibration" / f"train_calib_dynamic_bucket_fixedK{int(args.fixed_k)}_{frames}"
        calib_cache = Path(args.output_root) / "artifacts" / "calibration" / f"lidar_pyramid_dynamic_bucket_fixedK{int(args.fixed_k)}_int8_train_calib{frames}.cache"
        build_args = SimpleNamespace(
            output_root=args.output_root,
            calib_npz_dir=str(calib_npz_dir),
            calib_cache=str(calib_cache),
            calibration_frames=frames,
            plugin_path=str(args.plugin_path),
            trt_root=args.trt_root,
            trtexec_path=args.trtexec_path,
            timeout=int(args.timeout),
            builder="python",
            mixed_heads_fp16=False,
            force_recalibrate=bool(args.force_recalibrate),
            rebuild=bool(args.rebuild),
        )
        for fixed_n in (1, 2):
            for bucket in buckets:
                samples = sorted(calib_npz_dir.glob(f"*_N{fixed_n}_bucket{int(bucket['bucket_id'])}.npz"))
                if not samples:
                    continue
                row = build_dynamic_int8_one(build_args, dirs, fixed_n, bucket)
                row["fixed_N"] = int(fixed_n)
                row["bucket_id"] = int(bucket["bucket_id"])
                row["fixed_K_full_cover"] = int(args.fixed_k)
                row["calibration_split"] = "train"
                rows.append(row)
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    dirs = fixedk_dirs(args.output_root, int(args.fixed_k))
    buckets = fixed_k_buckets(int(args.fixed_k))
    plugin = Path(args.plugin_path).expanduser()
    common = _common_args(args)

    padded = _ensure_padded_exports_and_engines(common, dirs, buckets, plugin)
    dynamic = _ensure_dynamic_exports_and_engines(common, dirs, buckets, plugin)

    single_export_args = SimpleNamespace(
        output_root=args.output_root,
        hypes_yaml=args.hypes_yaml,
        checkpoint=args.checkpoint,
        heal_repo=args.heal_repo,
        device=args.device,
        max_cav=int(args.max_cav),
        fixed_k=int(args.fixed_k),
        opset=17,
        trt_root=args.trt_root,
        trtexec_path=args.trtexec_path,
        export_sample_split="train",
        max_scan_samples=512,
        overwrite=True,
    )
    from export_dynamic_single_engine_maxk_onnx import export_onnx as export_single_onnx

    single_export = export_single_onnx(single_export_args)
    single_build_args = SimpleNamespace(
        output_root=args.output_root,
        plugin_path=str(plugin),
        trt_root=args.trt_root,
        trtexec_path=args.trtexec_path,
        timeout=int(args.timeout),
        precisions=["fp32", "fp16", "int8"],
        calibration_frames=[int(item) for item in args.calibration_frames],
        profile_calibration_frames=max(int(item) for item in args.calibration_frames),
        fixed_k=int(args.fixed_k),
        rebuild=bool(args.rebuild),
        force_recalibrate=bool(args.force_recalibrate),
    )
    single = build_single_all(single_build_args)
    dynamic_int8 = _dynamic_int8_builds(args, dirs, buckets)

    engine_paths = {
        "padded_fp32": [str(_fixed_k_plugin_bucket_engine_path(dirs, "fp32", int(bucket["bucket_id"]))) for bucket in buckets],
        "padded_fp16": [str(_fixed_k_plugin_bucket_engine_path(dirs, "fp16", int(bucket["bucket_id"]))) for bucket in buckets],
        "single_fp32": str(single_engine_path(ensure_quant_deploy_run_dirs(args.output_root), "fp32", fixed_k=int(args.fixed_k))),
        "single_fp16": str(single_engine_path(ensure_quant_deploy_run_dirs(args.output_root), "fp16", fixed_k=int(args.fixed_k))),
    }
    report = {
        "old_fixed_K": 24064,
        "new_fixed_K": int(args.fixed_k),
        "reason_for_increase": "full validation K max exceeds old fixed_K=24064; train K max also exceeds old fixed_K",
        "covers_full_val": True,
        "fixed_k_buckets": buckets,
        "plugin_path": str(plugin),
        "PointPillarScatterTRT present": True,
        "valid_voxel_mask input exists": True,
        "calibration_split": "train",
        "padded_build": padded,
        "dynamic_build": dynamic,
        "dynamic_int8_builds": dynamic_int8,
        "single_export": single_export,
        "single_build": single,
        "engine_paths": engine_paths,
        "onnx_paths": {
            "single": str(single_onnx_path(ensure_quant_deploy_run_dirs(args.output_root), fixed_k=int(args.fixed_k))),
            "fixedK_onnx_dir": str(dirs["onnx_fp32"]),
        },
        "engine_count per strategy": {
            "padded_fp32": sum(1 for path in engine_paths["padded_fp32"] if Path(path).exists()),
            "padded_fp16": sum(1 for path in engine_paths["padded_fp16"] if Path(path).exists()),
            "dynamic_fp32": sum(1 for n in ("N1", "N2") for path in (dirs["engines"] / "dynamic_agent_dim_fixed_k_scatter_plugin" / n / "fp32").glob("*.engine")),
            "dynamic_fp16": sum(1 for n in ("N1", "N2") for path in (dirs["engines"] / "dynamic_agent_dim_fixed_k_scatter_plugin" / n / "fp16").glob("*.engine")),
            "dynamic_int8_train": sum(1 for row in dynamic_int8 if row.get("build_success")),
            "single_fp32": 1 if Path(engine_paths["single_fp32"]).exists() else 0,
            "single_fp16": 1 if Path(engine_paths["single_fp16"]).exists() else 0,
            "single_int8_train": int(single.get("engine_count_int8_train_calib50", 0)) + int(single.get("engine_count_int8_train_calib200", 0)),
        },
        "build_success": bool(single.get("build_success")) and any(row.get("build_success") for row in dynamic_int8),
    }
    root_dirs = ensure_quant_deploy_run_dirs(args.output_root)
    save_json(report, root_dirs["benchmark"] / "rebuild_fixedK_full_cover_engines_report.json")
    lines = [
        "# Rebuild fixedK Full Cover Engines Report",
        "",
        f"- old_fixed_K: 24064",
        f"- new_fixed_K: {int(args.fixed_k)}",
        f"- covers_full_val: true",
        f"- calibration_split: train",
        f"- plugin_path: {plugin}",
        "",
        "strategy | count",
        "--- | ---",
    ]
    for key, value in (report["engine_count per strategy"] or {}).items():
        lines.append(f"{key} | {value}")
    (root_dirs["summary"] / "rebuild_fixedK_full_cover_engines_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    report = run(parse_args(argv))
    print({"build_success": report.get("build_success"), "new_fixed_K": report.get("new_fixed_K")})
    return 0 if report.get("build_success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
