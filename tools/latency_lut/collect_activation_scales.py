from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any


DEFAULT_CACHE = Path(
    "tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/calibration/"
    "lidar_pyramid_dynamic_agent_single_engine_maxK_int8_train_calib200.cache"
)
DEFAULT_CALIB_DIR = Path(
    "tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/calibration/"
    "train_calib_single_engine_maxK29696_200"
)


def _parse_trt_cache(path: Path) -> dict[str, float]:
    entries: dict[str, float] = {}
    for line in path.read_text(errors="ignore").splitlines()[1:]:
        if ": " not in line:
            continue
        name, hex_value = line.split(": ", 1)
        try:
            raw = bytes.fromhex(hex_value.strip())
            if len(raw) == 4:
                entries[name] = float(struct.unpack(">f", raw)[0])
        except Exception:
            continue
    return entries


def _onnx_tensor_names(onnx_path: Path) -> set[str]:
    import onnx

    model = onnx.load(str(onnx_path))
    names: set[str] = set()
    for node in model.graph.node:
        names.update(str(v) for v in node.input)
        names.update(str(v) for v in node.output)
    names.update(str(v.name) for v in model.graph.initializer)
    return names


def _load_inventory(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("units") or data)


def collect_scales(args: argparse.Namespace) -> dict[str, Any]:
    calib_dir = Path(args.calib200_dir)
    if not calib_dir.is_dir() or not list(calib_dir.glob("*.npz")):
        return {"success": False, "status": "calib200_frame_list_not_found", "calib200_dir": str(calib_dir)}
    cache = Path(args.trt_calibration_cache)
    if not cache.is_file():
        return {"success": False, "status": "calib200_trt_cache_not_found", "cache": str(cache)}
    onnx_path = Path(args.onnx)
    if not onnx_path.is_file():
        return {"success": False, "status": "onnx_not_found", "onnx": str(onnx_path)}
    inventory_path = Path(args.inventory)
    if not inventory_path.is_file():
        return {"success": False, "status": "activation_scale_unit_mapping_failed", "inventory": str(inventory_path)}

    tensor_names = _onnx_tensor_names(onnx_path)
    cache_entries = _parse_trt_cache(cache)
    units = _load_inventory(inventory_path)
    requested = set(args.units or [])
    out_units: dict[str, Any] = {}
    missing: list[dict[str, Any]] = []
    num_frames = len(list(calib_dir.glob("*.npz")))

    for unit in units:
        unit_id = str(unit.get("unit_id"))
        if requested and unit_id not in requested and str(unit.get("module_path")) not in requested:
            continue
        if not unit.get("int8_supported"):
            continue
        input_scales = {}
        output_scales = {}
        for name in unit.get("input_tensors") or []:
            if name in cache_entries and name in tensor_names:
                scale = float(cache_entries[name])
                input_scales[name] = {"scale": scale, "amax": scale * 127.0, "dtype": "FP32", "num_samples": num_frames}
        for name in unit.get("output_tensors") or []:
            if name in cache_entries and name in tensor_names:
                scale = float(cache_entries[name])
                output_scales[name] = {"scale": scale, "amax": scale * 127.0, "dtype": "FP32", "num_samples": num_frames}
        weight_scales = {}
        import onnx
        from onnx import numpy_helper

        model = onnx.load(str(onnx_path))
        inits = {init.name: init for init in model.graph.initializer}
        for name in unit.get("weight_tensors") or []:
            init = inits.get(name)
            if init is None:
                continue
            arr = numpy_helper.to_array(init)
            if arr.ndim >= 1:
                # Per-output-channel symmetric scale for Conv/GEMM weights.
                flat = abs(arr.reshape(arr.shape[0], -1)).max(axis=1)
                scale = [max(float(v) / 127.0, 1.0e-8) for v in flat]
                weight_scales[name] = {"scale": scale, "per_channel": True, "axis": 0}
        if not input_scales or not weight_scales:
            missing.append({"unit_id": unit_id, "missing_input_scale": not bool(input_scales), "missing_weight_scale": not bool(weight_scales)})
            continue
        out_units[unit_id] = {
            "unit_id": unit_id,
            "scale_source": "train_calib200_trt_cache_tensor_name_verified",
            "num_calib_frames": min(int(args.num_frames), num_frames),
            "input_activation_tensors": input_scales,
            "output_activation_tensors": output_scales,
            "weight_tensors": weight_scales,
            "scale_valid": True,
            "covered_onnx_nodes": unit.get("covered_onnx_nodes") or [],
        }

    status = "success" if out_units and not missing else ("activation_scale_unit_mapping_failed" if not out_units else "partial_success")
    return {
        "success": bool(out_units),
        "status": status,
        "scale_source": "train_calib200_trt_cache_tensor_name_verified",
        "num_calib_frames": min(int(args.num_frames), num_frames),
        "calib200_dir": str(calib_dir),
        "trt_calibration_cache": str(cache),
        "mapping_proof": "cache tensor names were matched exactly against the current full-engine ONNX graph tensor names",
        "units": out_units,
        "missing_units": missing,
    }


def write_report(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Activation Scale Collection Report",
        "",
        f"- status: {payload.get('status')}",
        f"- success: {payload.get('success')}",
        f"- scale_source: {payload.get('scale_source')}",
        f"- num_calib_frames: {payload.get('num_calib_frames')}",
        f"- calib200_dir: `{payload.get('calib200_dir')}`",
        f"- mapping_proof: {payload.get('mapping_proof')}",
        "",
        "No random or dummy scales are generated. If a requested INT8 unit has no exact tensor-name scale match, Q/DQ insertion must fail.",
        "",
        f"- units_with_scales: {len(payload.get('units') or {})}",
        f"- missing_units: {len(payload.get('missing_units') or [])}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--inventory", default="outputs/latency_lut/atomic_deployment_unit_inventory.json")
    parser.add_argument("--trt-calibration-cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--calib200-dir", default=str(DEFAULT_CALIB_DIR))
    parser.add_argument("--num-frames", type=int, default=200)
    parser.add_argument("--units", nargs="*", default=None)
    parser.add_argument("--output", default="outputs/latency_lut/activation_scale_cache_calib200.json")
    parser.add_argument("--report", default="outputs/latency_lut/activation_scale_collection_report.md")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = collect_scales(args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(payload, Path(args.report))
    print(json.dumps({k: payload.get(k) for k in ["success", "status", "num_calib_frames", "scale_source"]}, indent=2))
    return 0 if payload.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
