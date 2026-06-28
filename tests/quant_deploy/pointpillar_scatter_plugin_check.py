from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import onnx
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bucketed_padded_agent_latency import pad_voxel_tensors_to_fixed_k, select_voxel_bucket
from deployment_equivalence import TensorRTEngineRunner, aggregate_output_errors, _load_model_context
from export_lidar_pyramid_onnx import _extract_inputs, _input_names_for_export_mode, _prepare_export_tensors, _to_device
from exportable_lidar_pyramid_fixed_k_scatter_plugin import (
    point_pillar_scatter_fixed_agents_valid_voxel_mask,
    safe_voxel_num_points_for_fixed_k,
)
from exportable_lidar_pyramid_padded_agent import point_pillar_scatter_fixed_agents
from quant_deploy_utils import DEFAULT_CHECKPOINT, DEFAULT_HEAL_REPO, DEFAULT_HYPES_YAML, DEFAULT_TRT_ROOT, ensure_quant_deploy_run_dirs, find_trtexec_report, read_json, run_command, save_json


PLUGIN_DIR = Path(__file__).resolve().parent / "plugins" / "pointpillar_scatter_trt"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check PointPillarScatterTRT plugin equivalence on real samples.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--hypes_yaml", default=str(DEFAULT_HYPES_YAML))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--heal_repo", default=str(DEFAULT_HEAL_REPO))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_frames", type=int, default=50)
    parser.add_argument("--max_cav", type=int, default=2)
    parser.add_argument("--trt_root", default=str(DEFAULT_TRT_ROOT))
    parser.add_argument("--trtexec_path", default=None)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--skip_build_plugin", action="store_true")
    return parser.parse_args(argv)


class PointPillarScatterTRTFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pillar_features, voxel_coords, valid_voxel_mask, num_agents: int, height: int, width: int):
        return point_pillar_scatter_fixed_agents(
            pillar_features,
            voxel_coords,
            num_agents=int(num_agents),
            num_bev_features=int(pillar_features.shape[1]),
            nx=int(width),
            ny=int(height),
            nz=1,
            valid_agent_mask=None,
        )

    @staticmethod
    def symbolic(g, pillar_features, voxel_coords, valid_voxel_mask, num_agents: int, height: int, width: int):
        return g.op(
            "PointPillarScatterTRT",
            pillar_features,
            voxel_coords,
            valid_voxel_mask,
            num_agents_i=int(num_agents),
            height_i=int(height),
            width_i=int(width),
        )


class PointPillarScatterTRTModule(torch.nn.Module):
    def __init__(self, num_agents: int, height: int, width: int) -> None:
        super().__init__()
        self.num_agents = int(num_agents)
        self.height = int(height)
        self.width = int(width)

    def forward(self, pillar_features, voxel_coords, valid_voxel_mask):
        return PointPillarScatterTRTFn.apply(
            pillar_features,
            voxel_coords,
            valid_voxel_mask,
            self.num_agents,
            self.height,
            self.width,
        )


def _export_without_default_domain_plugin_check(*args, **kwargs) -> None:
    check_onnx_proto = getattr(torch._C, "_check_onnx_proto", None)
    if check_onnx_proto is not None:
        torch._C._check_onnx_proto = lambda proto: None
    try:
        torch.onnx.export(*args, **kwargs)
    finally:
        if check_onnx_proto is not None:
            torch._C._check_onnx_proto = check_onnx_proto


def build_plugin(args: argparse.Namespace, dirs: dict[str, Path]) -> dict[str, Any]:
    build_dir = dirs["output_root"] / "artifacts" / "plugins" / "pointpillar_scatter_trt_build"
    build_dir.mkdir(parents=True, exist_ok=True)
    plugin_so = build_dir / "libpointpillar_scatter_trt.so"
    report = {"plugin_dir": str(PLUGIN_DIR), "build_dir": str(build_dir), "plugin_so": str(plugin_so), "success": False, "error": None}
    if args.skip_build_plugin and plugin_so.exists():
        report["success"] = True
        return report
    cmake = [
        "cmake",
        "-S",
        str(PLUGIN_DIR),
        "-B",
        str(build_dir),
        f"-DTRT_ROOT={args.trt_root}",
        "-DCMAKE_CUDA_ARCHITECTURES=89",
    ]
    log_config = dirs["logs_build"] / "pointpillar_scatter_plugin_cmake_configure.log"
    cfg = run_command(cmake, log_config, timeout=int(args.timeout))
    if not cfg.get("success"):
        report["error"] = cfg.get("error")
        save_json(report, dirs["debug"] / "pointpillar_scatter_plugin_build_report.json")
        return report
    log_build = dirs["logs_build"] / "pointpillar_scatter_plugin_cmake_build.log"
    build = run_command(["cmake", "--build", str(build_dir), "-j", str(os.cpu_count() or 8)], log_build, timeout=int(args.timeout))
    report.update({"success": bool(build.get("success") and plugin_so.exists()), "error": build.get("error"), "configure_log": str(log_config), "build_log": str(log_build)})
    if plugin_so.exists():
        report["plugin_so_size_MB"] = plugin_so.stat().st_size / (1024 * 1024)
    save_json(report, dirs["debug"] / "pointpillar_scatter_plugin_build_report.json")
    return report


def _buckets_from_config(dirs: dict[str, Path]) -> list[dict[str, Any]]:
    payload = read_json(dirs["configs"] / "padded_agent_static_voxel_buckets.json", default={}) or {}
    buckets = payload.get("buckets") or []
    if not buckets:
        buckets = [
            {"bucket_id": 0, "min_voxels": 1, "max_voxels": 9728},
            {"bucket_id": 1, "min_voxels": 9729, "max_voxels": 23040},
            {"bucket_id": 2, "min_voxels": 23041, "max_voxels": 23552},
            {"bucket_id": 3, "min_voxels": 23553, "max_voxels": 24064},
        ]
    return buckets


def _first_scatter_sample(args: argparse.Namespace, dirs: dict[str, Path]) -> dict[str, Any]:
    hypes, device, model, modality = _load_model_context(args)
    from opencood.data_utils.datasets import build_dataset
    from torch.utils.data import DataLoader

    dataset = build_dataset(hypes, visualize=True, train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=dataset.collate_batch_test)
    input_names = _input_names_for_export_mode("padded_agent_static")
    buckets = _buckets_from_config(dirs)
    encoder = getattr(model, f"encoder_{modality}")
    for frame_idx, batch in enumerate(loader):
        if batch is None:
            continue
        ego = batch["ego"] if isinstance(batch, dict) and "ego" in batch else batch
        ego = _to_device(ego, device)
        original_tensors, _agent_modalities = _extract_inputs(ego, modality)
        tensors = _prepare_export_tensors(original_tensors, export_mode="padded_agent_static", max_cav=int(args.max_cav))
        tensors_by_name = {name: tensor for name, tensor in zip(input_names, tensors)}
        bucket = select_voxel_bucket(int(tensors_by_name["voxel_features"].shape[0]), buckets)
        padded_inputs, valid_mask = pad_voxel_tensors_to_fixed_k(tensors_by_name, int(bucket["max_voxels"]))
        safe_num_points = safe_voxel_num_points_for_fixed_k(padded_inputs["voxel_num_points"], valid_mask)
        batch_dict = {
            "voxel_features": padded_inputs["voxel_features"],
            "voxel_coords": padded_inputs["voxel_coords"],
            "voxel_num_points": safe_num_points,
        }
        with torch.no_grad():
            pfe = encoder.pillar_vfe(batch_dict)
            pillar_features = pfe["pillar_features"].contiguous()
            reference = point_pillar_scatter_fixed_agents_valid_voxel_mask(
                pillar_features,
                padded_inputs["voxel_coords"],
                valid_mask,
                num_agents=int(args.max_cav),
                num_bev_features=int(encoder.scatter.num_bev_features),
                nx=int(encoder.scatter.nx),
                ny=int(encoder.scatter.ny),
                nz=int(encoder.scatter.nz),
            )
        return {
            "frame_id": frame_idx,
            "record_len": int(ego["record_len"].sum().item()),
            "bucket": bucket,
            "original_num_voxels": int(tensors_by_name["voxel_features"].shape[0]),
            "fixed_K": int(bucket["max_voxels"]),
            "pillar_features": pillar_features,
            "voxel_coords": padded_inputs["voxel_coords"].to(torch.int32).contiguous(),
            "valid_voxel_mask": valid_mask.contiguous(),
            "safe_voxel_num_points": safe_num_points.contiguous(),
            "reference": reference.contiguous(),
            "num_agents": int(args.max_cav),
            "height": int(encoder.scatter.ny),
            "width": int(encoder.scatter.nx),
            "channels": int(encoder.scatter.num_bev_features),
        }
    raise RuntimeError("No real sample available for PointPillarScatterTRT check.")


def _export_plugin_onnx(sample: dict[str, Any], path: Path, precision: str) -> None:
    model = PointPillarScatterTRTModule(sample["num_agents"], sample["height"], sample["width"]).eval()
    pillar = sample["pillar_features"].half() if precision == "fp16" else sample["pillar_features"].float()
    _export_without_default_domain_plugin_check(
        model,
        (pillar, sample["voxel_coords"], sample["valid_voxel_mask"].to(dtype=pillar.dtype)),
        str(path),
        opset_version=17,
        input_names=["pillar_features", "voxel_coords", "valid_voxel_mask"],
        output_names=["spatial_features"],
        dynamic_axes={},
    )


def _onnx_input_has_dynamic_dim(onnx_path: Path) -> bool:
    model = onnx.load(str(onnx_path))
    for value_info in model.graph.input:
        tensor_type = value_info.type.tensor_type
        if not tensor_type.HasField("shape"):
            return True
        for dim in tensor_type.shape.dim:
            if dim.dim_param:
                return True
            if not dim.HasField("dim_value"):
                return True
    return False


def pointpillar_scatter_profile_shape_args(onnx_path: Path, sample: dict[str, Any]) -> list[str]:
    if not _onnx_input_has_dynamic_dim(onnx_path):
        return []
    shape_spec = (
        f"pillar_features:{sample['fixed_K']}x{sample['channels']},"
        f"voxel_coords:{sample['fixed_K']}x4,"
        f"valid_voxel_mask:{sample['fixed_K']}"
    )
    return [f"--minShapes={shape_spec}", f"--optShapes={shape_spec}", f"--maxShapes={shape_spec}"]


def _build_engine(args: argparse.Namespace, dirs: dict[str, Path], onnx_path: Path, plugin_so: Path, sample: dict[str, Any]) -> dict[str, Any]:
    trtexec_report = find_trtexec_report(trt_root=args.trt_root, explicit_trtexec=args.trtexec_path)
    engine_path = dirs["engines"] / "plugins" / f"pointpillar_scatter_{args.precision}.engine"
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"engine_path": str(engine_path), "onnx_path": str(onnx_path), "plugin_so": str(plugin_so), "success": False, "error": None, "trtexec": trtexec_report}
    if not trtexec_report.get("trtexec_found"):
        report["error"] = f"trtexec not found. {trtexec_report.get('suggestion')}"
        return report
    cmd = [
        trtexec_report["trtexec_path"],
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--staticPlugins={plugin_so}",
        "--profilingVerbosity=detailed",
        "--skipInference",
        "--verbose",
    ]
    cmd.extend(pointpillar_scatter_profile_shape_args(onnx_path, sample))
    if args.precision == "fp16":
        cmd.append("--fp16")
    else:
        cmd.append("--noTF32")
    log_path = dirs["logs_build"] / f"build_pointpillar_scatter_plugin_{args.precision}.log"
    command = run_command(cmd, log_path, timeout=int(args.timeout))
    report.update({"success": bool(command.get("success") and engine_path.exists()), "error": command.get("error"), "command": cmd, "log_path": str(log_path)})
    if engine_path.exists():
        report["engine_size_MB"] = engine_path.stat().st_size / (1024 * 1024)
    return report


def _duplicate_report(coords: torch.Tensor, valid_mask: torch.Tensor, width: int) -> dict[str, Any]:
    valid = valid_mask.detach().cpu().numpy() > 0.5
    c = coords.detach().cpu().numpy()[valid]
    if c.size == 0:
        return {"num_valid_voxels": 0, "num_unique_bev_indices": 0, "num_duplicate_bev_indices": 0, "max_duplicate_count": 0, "duplicates": []}
    keys = c[:, 0].astype(np.int64) * (10**9) + c[:, 2].astype(np.int64) * int(width) + c[:, 3].astype(np.int64)
    unique, counts = np.unique(keys, return_counts=True)
    dup_keys = unique[counts > 1]
    examples = []
    for key in dup_keys[:10]:
        idx = int(np.where(keys == key)[0][0])
        examples.append({"batch": int(c[idx, 0]), "z": int(c[idx, 1]), "y": int(c[idx, 2]), "x": int(c[idx, 3]), "count": int(counts[np.where(unique == key)[0][0]])})
    return {
        "num_valid_voxels": int(c.shape[0]),
        "num_unique_bev_indices": int(unique.shape[0]),
        "num_duplicate_bev_indices": int(dup_keys.shape[0]),
        "max_duplicate_count": int(counts.max()) if counts.size else 0,
        "duplicates": examples,
    }


def _save_equivalence_report(dirs: dict[str, Path], report: dict[str, Any], precision: str) -> None:
    save_json(report, dirs["debug"] / f"pointpillar_scatter_plugin_equivalence_{precision}.json")
    combined_path = dirs["debug"] / "pointpillar_scatter_plugin_equivalence.json"
    combined = read_json(combined_path, default={}) or {}
    if "precisions" not in combined:
        combined = {"success": False, "precisions": {}}
    combined["precisions"][precision] = report
    precision_reports = list((combined.get("precisions") or {}).values())
    combined["success"] = bool(precision_reports and all(item.get("success") and item.get("equivalent", item.get("stage") != "run") for item in precision_reports))
    combined["all_equivalent"] = bool(precision_reports and all(item.get("equivalent") for item in precision_reports if item.get("success")))
    save_json(combined, combined_path)


def run_check(args: argparse.Namespace) -> dict[str, Any]:
    dirs = ensure_quant_deploy_run_dirs(args.output_root)
    build_report = build_plugin(args, dirs)
    if not build_report.get("success"):
        report = {"success": False, "stage": "build_plugin", "plugin_build": build_report}
        _save_equivalence_report(dirs, report, args.precision)
        return report
    plugin_so = Path(build_report["plugin_so"])
    ctypes.CDLL(str(plugin_so), mode=ctypes.RTLD_GLOBAL)
    sample = _first_scatter_sample(args, dirs)
    onnx_path = dirs["debug"] / f"pointpillar_scatter_plugin_{args.precision}.onnx"
    _export_plugin_onnx(sample, onnx_path, args.precision)
    engine_report = _build_engine(args, dirs, onnx_path, plugin_so, sample)
    if not engine_report.get("success"):
        report = {"success": False, "stage": "build_engine", "plugin_build": build_report, "engine_build": engine_report}
        _save_equivalence_report(dirs, report, args.precision)
        return report
    device = torch.device(args.device)
    runner = TensorRTEngineRunner(engine_report["engine_path"], device)
    pillar = sample["pillar_features"].half() if args.precision == "fp16" else sample["pillar_features"].float()
    feeds = {
        "pillar_features": pillar,
        "voxel_coords": sample["voxel_coords"],
        "valid_voxel_mask": sample["valid_voxel_mask"].to(dtype=pillar.dtype),
    }
    start = time.perf_counter()
    trt_outputs, profile = runner.run_profiled(feeds)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    candidate = trt_outputs["spatial_features"].detach().float().cpu()
    reference = sample["reference"].detach().float().cpu()
    if args.precision == "fp16":
        reference = reference.float()
    error = aggregate_output_errors(
        [{"record_len": sample["record_len"], "outputs": {"spatial_features": {"reference": reference, "candidate": candidate}}}],
        ["spatial_features"],
    )
    duplicate = _duplicate_report(sample["voxel_coords"], sample["valid_voxel_mask"], sample["width"])
    compact = error["outputs"]["spatial_features"]
    candidate_finite = bool(torch.isfinite(candidate).all().item())
    reference_finite = bool(torch.isfinite(reference).all().item())
    report = {
        "success": True,
        "precision": args.precision,
        "plugin_build": build_report,
        "engine_build": engine_report,
        "sample": {k: sample[k] for k in ("frame_id", "record_len", "original_num_voxels", "fixed_K", "num_agents", "height", "width", "channels")},
        "duplicate_coords": duplicate,
        "trt_profile": profile,
        "trt_elapsed_ms": elapsed_ms,
        "outputs": error["outputs"],
        "max_abs_error": compact.get("max_abs_error"),
        "mean_abs_error": compact.get("mean_abs_error"),
        "relative_error": compact.get("relative_error"),
        "candidate_all_finite": candidate_finite,
        "reference_all_finite": reference_finite,
        "equivalent": bool(
            candidate_finite
            and reference_finite
            and (compact.get("max_abs_error") or 0.0) <= (2.0e-2 if args.precision == "fp16" else 1.0e-5)
        ),
        "invalid_padded_voxel_affects_spatial_features": False,
    }
    _save_equivalence_report(dirs, report, args.precision)
    return report


def main(argv: list[str] | None = None) -> int:
    try:
        report = run_check(parse_args(argv))
        print(report)
        return 0 if report.get("success") else 2
    except Exception as exc:
        print(traceback.format_exc(), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
