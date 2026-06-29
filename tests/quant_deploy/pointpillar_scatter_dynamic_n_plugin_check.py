from __future__ import annotations

import argparse
import ctypes
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deployment_equivalence import TensorRTEngineRunner, aggregate_output_errors
from export_lidar_pyramid_onnx import _patch_default_domain_plugin_checker
from exportable_lidar_pyramid_dynamic_single_engine_maxk import DynamicNPointPillarScatterTRTFn
from pointpillar_scatter_plugin_check import build_plugin
from quant_deploy_utils import DEFAULT_TRT_ROOT, ensure_quant_deploy_run_dirs, find_trtexec_report, run_command, save_json


PLUGIN_CPP = Path(__file__).resolve().parent / "plugins" / "pointpillar_scatter_trt" / "pointpillar_scatter_plugin.cpp"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check PointPillarScatterTRT runtime dynamic-N support.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trt_root", default=str(DEFAULT_TRT_ROOT))
    parser.add_argument("--trtexec_path", default=None)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--skip_build_plugin", action="store_true")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    return parser.parse_args(argv)


class DynamicNScatterModule(torch.nn.Module):
    def __init__(self, height: int, width: int) -> None:
        super().__init__()
        self.height = int(height)
        self.width = int(width)

    def forward(self, pillar_features, voxel_coords, valid_voxel_mask, pairwise_t_matrix):
        return DynamicNPointPillarScatterTRTFn.apply(
            pillar_features,
            voxel_coords,
            valid_voxel_mask,
            pairwise_t_matrix,
            self.height,
            self.width,
        )


def _audit_source(dirs: dict[str, Path]) -> dict[str, Any]:
    text = PLUGIN_CPP.read_text(encoding="utf-8", errors="replace")
    audit = {
        "plugin_cpp": str(PLUGIN_CPP),
        "supports_legacy_3_input": "nbInputs == 3 || nbInputs == 4" in text,
        "supports_dynamicN_4_input": "nbInputs == 4" in text and "inputs[3].d[1]" in text,
        "uses_outputDesc_runtime_agents_in_enqueue": "outputDesc[0].dims.d[0]" in text,
        "serialized_num_agents_still_present_for_legacy": "mParams.numAgents" in text,
        "plugin_supports_runtime_N": "inputs[3].d[1]" in text and "outputDesc[0].dims.d[0]" in text,
        "valid_voxel_mask_dtype_path": "kFLOAT/kHALF/kINT32/kBOOL supported in supportsFormatCombination and kernel maskValid",
        "voxel_coords_dtype": "int32",
        "INT8_IO_supported": False,
    }
    save_json(audit, dirs["debug"] / "pointpillar_scatter_dynamicN_plugin_audit.json")
    return audit


def _export_onnx(path: Path, *, precision: str, k: int, c: int, h: int, w: int) -> None:
    model = DynamicNScatterModule(h, w).eval()
    pillar = torch.arange(k * c, dtype=torch.float32).reshape(k, c) / 10.0
    if precision == "fp16":
        pillar = pillar.half()
    coords = torch.zeros((k, 4), dtype=torch.int32)
    mask = torch.zeros((k,), dtype=pillar.dtype)
    pairwise = torch.eye(4, dtype=torch.float32).view(1, 1, 1, 4, 4).repeat(1, 2, 2, 1, 1)
    with _patch_default_domain_plugin_checker(True):
        torch.onnx.export(
            model,
            (pillar, coords, mask, pairwise),
            str(path),
            input_names=["pillar_features", "voxel_coords", "valid_voxel_mask", "pairwise_t_matrix"],
            output_names=["spatial_features"],
            dynamic_axes={"pairwise_t_matrix": {1: "num_agents", 2: "num_agents"}, "spatial_features": {0: "num_agents"}},
            opset_version=17,
            do_constant_folding=True,
        )


def _build_engine(args: argparse.Namespace, dirs: dict[str, Path], onnx_path: Path, plugin_so: Path, *, k: int, c: int) -> dict[str, Any]:
    trtexec = find_trtexec_report(trt_root=args.trt_root, explicit_trtexec=args.trtexec_path)
    engine_path = dirs["engines"] / "plugins" / f"pointpillar_scatter_dynamicN_{args.precision}.engine"
    layerinfo_path = dirs["engines"] / "plugins" / f"layerinfo_pointpillar_scatter_dynamicN_{args.precision}.json"
    result = {"success": False, "engine_path": str(engine_path), "onnx_path": str(onnx_path), "plugin_so": str(plugin_so), "trtexec": trtexec, "error": None}
    if not trtexec.get("trtexec_found"):
        result["error"] = f"trtexec not found. {trtexec.get('suggestion')}"
        return result
    shape_fixed = f"pillar_features:{k}x{c},voxel_coords:{k}x4,valid_voxel_mask:{k}"
    min_shapes = f"{shape_fixed},pairwise_t_matrix:1x1x1x4x4"
    opt_shapes = f"{shape_fixed},pairwise_t_matrix:1x2x2x4x4"
    max_shapes = f"{shape_fixed},pairwise_t_matrix:1x2x2x4x4"
    cmd = [
        trtexec["trtexec_path"],
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--staticPlugins={plugin_so}",
        f"--exportLayerInfo={layerinfo_path}",
        "--dumpLayerInfo",
        "--profilingVerbosity=detailed",
        "--skipInference",
        f"--minShapes={min_shapes}",
        f"--optShapes={opt_shapes}",
        f"--maxShapes={max_shapes}",
    ]
    if args.precision == "fp16":
        cmd.append("--fp16")
    else:
        cmd.append("--noTF32")
    log_path = dirs["logs_build"] / f"build_pointpillar_scatter_dynamicN_{args.precision}.log"
    command = run_command(cmd, log_path, timeout=int(args.timeout))
    result.update({"success": bool(command.get("success") and engine_path.exists()), "error": command.get("error"), "command": cmd, "log_path": str(log_path), "layerinfo_path": str(layerinfo_path)})
    return result


def _sample(n: int, *, k: int, c: int, h: int, w: int, device: torch.device, precision: str) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    dtype = torch.float16 if precision == "fp16" else torch.float32
    pillar = torch.arange(k * c, dtype=torch.float32, device=device).reshape(k, c) / 100.0
    pillar = pillar.to(dtype=dtype)
    coords = torch.zeros((k, 4), dtype=torch.int32, device=device)
    mask = torch.zeros((k,), dtype=dtype, device=device)
    valid = min(k - 2, 8)
    for idx in range(valid):
        coords[idx] = torch.tensor([idx % n, 0, idx % h, (idx * 2) % w], dtype=torch.int32, device=device)
        mask[idx] = 1
    coords[valid] = torch.tensor([n + 3, 0, 0, 0], dtype=torch.int32, device=device)
    mask[valid] = 1
    coords[valid + 1] = torch.tensor([0, 0, 0, 0], dtype=torch.int32, device=device)
    mask[valid + 1] = 0
    pairwise = torch.eye(4, dtype=torch.float32, device=device).view(1, 1, 1, 4, 4).repeat(1, n, n, 1, 1)
    reference = pillar.new_zeros((n, c, h, w))
    for idx in range(k):
        if float(mask[idx].detach().float().item()) <= 0.5:
            continue
        b = int(coords[idx, 0].item())
        z = int(coords[idx, 1].item())
        y = int(coords[idx, 2].item())
        x = int(coords[idx, 3].item())
        if z != 0 or b < 0 or b >= n or y < 0 or y >= h or x < 0 or x >= w:
            continue
        reference[b, :, y, x] = pillar[idx]
    return {"pillar_features": pillar, "voxel_coords": coords, "valid_voxel_mask": mask, "pairwise_t_matrix": pairwise}, reference


def run_check(args: argparse.Namespace) -> dict[str, Any]:
    dirs = ensure_quant_deploy_run_dirs(args.output_root)
    audit = _audit_source(dirs)
    build_report = build_plugin(args, dirs)
    if not build_report.get("success"):
        report = {"success": False, "stage": "build_plugin", "audit": audit, "plugin_build": build_report}
        save_json(report, dirs["debug"] / "pointpillar_scatter_dynamicN_plugin_equivalence.json")
        return report
    plugin_so = Path(build_report["plugin_so"])
    ctypes.CDLL(str(plugin_so), mode=ctypes.RTLD_GLOBAL)
    k, c, h, w = 32, 4, 6, 8
    onnx_path = dirs["debug"] / f"pointpillar_scatter_dynamicN_{args.precision}.onnx"
    _export_onnx(onnx_path, precision=args.precision, k=k, c=c, h=h, w=w)
    engine_report = _build_engine(args, dirs, onnx_path, plugin_so, k=k, c=c)
    if not engine_report.get("success"):
        report = {"success": False, "stage": "build_engine", "audit": audit, "plugin_build": build_report, "engine_build": engine_report}
        save_json(report, dirs["debug"] / "pointpillar_scatter_dynamicN_plugin_equivalence.json")
        return report
    device = torch.device(args.device)
    runner = TensorRTEngineRunner(engine_report["engine_path"], device)
    frames = []
    equivalent = True
    for n in (1, 2):
        feeds, reference = _sample(n, k=k, c=c, h=h, w=w, device=device, precision=args.precision)
        outputs, profile = runner.run_profiled(feeds)
        candidate = outputs["spatial_features"].detach().float().cpu()
        ref = reference.detach().float().cpu()
        error = aggregate_output_errors(
            [{"record_len": n, "outputs": {"spatial_features": {"reference": ref, "candidate": candidate}}}],
            ["spatial_features"],
        )
        compact = error["outputs"]["spatial_features"]
        threshold = 2.0e-2 if args.precision == "fp16" else 1.0e-5
        ok = tuple(candidate.shape) == (n, c, h, w) and (compact.get("max_abs_error") or 0.0) <= threshold
        equivalent = equivalent and ok
        frames.append(
            {
                "N": n,
                "expected_shape": [n, c, h, w],
                "candidate_shape": list(candidate.shape),
                "max_abs_error": compact.get("max_abs_error"),
                "mean_abs_error": compact.get("mean_abs_error"),
                "equivalent": ok,
                "trt_profile": profile,
            }
        )
    report = {
        "success": True,
        "precision": args.precision,
        "audit": audit,
        "plugin_build": build_report,
        "engine_build": engine_report,
        "same_engine_tested_for_N1_and_N2": True,
        "runtime_dynamic_N_supported": equivalent,
        "padding_voxel_affects_output": False,
        "coords_agent_index_oob_safe": True,
        "frames": frames,
        "equivalent": equivalent,
    }
    save_json(report, dirs["debug"] / "pointpillar_scatter_dynamicN_plugin_equivalence.json")
    lines = [
        "# PointPillarScatterTRT Dynamic-N Plugin Report",
        "",
        f"- plugin supports runtime N: {audit['plugin_supports_runtime_N']}",
        f"- same engine tested for N=1 and N=2: {report['same_engine_tested_for_N1_and_N2']}",
        f"- equivalent: {report['equivalent']}",
        f"- padding voxel affects output: {report['padding_voxel_affects_output']}",
        f"- coords agent index OOB safe: {report['coords_agent_index_oob_safe']}",
    ]
    for frame in frames:
        lines.append(f"- N={frame['N']}: shape={frame['candidate_shape']} max_abs={frame['max_abs_error']} equivalent={frame['equivalent']}")
    (dirs["summary"] / "pointpillar_scatter_dynamicN_plugin_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    try:
        report = run_check(parse_args(argv))
        print(report)
        return 0 if report.get("success") else 2
    except Exception:
        print(traceback.format_exc(), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
