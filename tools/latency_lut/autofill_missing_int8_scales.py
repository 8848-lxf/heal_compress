from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.latency_lut.collect_activation_scales import collect_scales, write_report


DEFAULT_ONNX = Path(
    "tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/onnx/"
    "fixedK29696/dynamic_agent_single_engine_maxK/lidar_pyramid_dynamic_agent_single_engine_maxK.onnx"
)
DEFAULT_CACHE = Path("outputs/latency_lut/activation_scale_cache_calib200.json")
DEFAULT_INVENTORY = Path("outputs/latency_lut/atomic_deployment_unit_inventory_v2.json")
DEFAULT_TRT_CACHE = Path(
    "tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/calibration/"
    "lidar_pyramid_dynamic_agent_single_engine_maxK_int8_train_calib200.cache"
)
DEFAULT_CALIB_DIR = Path(
    "tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/calibration/"
    "train_calib_single_engine_maxK29696_200"
)


def _load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _int8_units_from_key_space(path: str | Path) -> dict[str, int]:
    data = _load_json(path)
    rows = data.get("keys") or data.get("key_space") or data if isinstance(data, dict) else []
    counts: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("precision")).upper() not in {"INT8", "INT8_QDQ", "TRT_INT8_QDQ"}:
            continue
        unit_id = str(row.get("unit_id"))
        counts[unit_id] = counts.get(unit_id, 0) + 1
    return counts


def _has_scale(unit_id: str, cache: dict[str, Any]) -> bool:
    units = cache.get("units") or {}
    if unit_id in units and units[unit_id].get("scale_valid", True):
        return True
    return any((unit_id == key or key.startswith(unit_id + ".") or unit_id in key) and value.get("scale_valid", True) for key, value in units.items())


def autofill(args: argparse.Namespace) -> dict[str, Any]:
    key_counts = _int8_units_from_key_space(args.key_space)
    cache = _load_json(args.scale_cache)
    missing = [unit for unit in sorted(key_counts) if not _has_scale(unit, cache)]
    statuses = []
    payload = {"success": False, "units": {}}
    if missing:
        collect_args = SimpleNamespace(
            onnx=str(args.onnx),
            inventory=str(args.inventory),
            trt_calibration_cache=str(args.trt_calibration_cache),
            calib200_dir=str(args.calib200_dir),
            num_frames=int(args.num_frames),
            units=missing,
            output=str(args.scale_cache),
            report=str(Path(args.report).with_suffix(".collect.md")),
        )
        payload = collect_scales(collect_args)
        existing_units = dict((cache.get("units") or {}))
        existing_units.update(payload.get("units") or {})
        cache.update({k: v for k, v in payload.items() if k != "units"})
        cache["units"] = existing_units
        cache["success"] = bool(existing_units)
        _write_json(args.scale_cache, cache)
        write_report(payload, Path(collect_args.report))
    refreshed = _load_json(args.scale_cache)
    for unit_id, count in sorted(key_counts.items()):
        before = _has_scale(unit_id, _load_json(args.scale_cache)) if not missing else unit_id not in missing
        after = _has_scale(unit_id, refreshed)
        status = "success" if after else "failed"
        reason = "" if after else "unit_mapping_failed"
        statuses.append(
            {
                "unit_id": unit_id,
                "requested_by_int8_keys": count,
                "scale_before": bool(before),
                "scale_after": bool(after),
                "status": status,
                "failed_stage": "" if after else "activation_scale_autofill",
                "reason": reason,
            }
        )
    report = {
        "success": all(row["scale_after"] for row in statuses),
        "key_space": str(args.key_space),
        "scale_cache": str(args.scale_cache),
        "num_int8_units": len(key_counts),
        "num_missing_before": len(missing),
        "num_missing_after": sum(1 for row in statuses if not row["scale_after"]),
        "unit_status": statuses,
        "collect_status": payload.get("status"),
    }
    lines = [
        "# INT8 Scale Autofill Report",
        "",
        f"- success: {report['success']}",
        f"- int8 units: {report['num_int8_units']}",
        f"- missing before autofill: {report['num_missing_before']}",
        f"- missing after autofill: {report['num_missing_after']}",
        f"- collect_status: {report.get('collect_status')}",
        "",
        "No dummy scale is generated. Missing scales are attempted through train_calib200 cache/tensor-name mapping.",
    ]
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_json(Path(args.report).with_suffix(".json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key-space", "--key_space", dest="key_space", default="outputs/latency_lut/layer_width_precision_key_space_v4.json")
    parser.add_argument("--scale-cache", "--scale_cache", dest="scale_cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--onnx", default=str(DEFAULT_ONNX))
    parser.add_argument("--inventory", default=str(DEFAULT_INVENTORY))
    parser.add_argument("--trt-calibration-cache", "--trt_calibration_cache", dest="trt_calibration_cache", default=str(DEFAULT_TRT_CACHE))
    parser.add_argument("--calib200-dir", "--calib200_dir", dest="calib200_dir", default=str(DEFAULT_CALIB_DIR))
    parser.add_argument("--num-frames", "--num_frames", dest="num_frames", type=int, default=200)
    parser.add_argument("--report", default="outputs/latency_lut/int8_scale_autofill_report.md")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    report = autofill(parse_args(argv))
    return 0 if report.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
