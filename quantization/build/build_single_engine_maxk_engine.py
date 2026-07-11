"""Deprecated CLI wrapper around the formal TensorRT build API."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

from quantization.api import build_trt_engine
from quantization.artifacts.io import atomic_write_json, load_json
from quantization.config import TensorRTBuildConfig
from quantization.types import CanonicalPrecisionEntry, CanonicalPrecisionMappingResult
from quantization.utils.calibration import load_observed_shapes, profile_from_observed_shapes
from quantization.utils.engine_io import file_info
from quantization.utils.paths import DEFAULT_FIXED_K, DEFAULT_PLUGIN, DEFAULT_PRECISION, DEFAULT_TRT_ROOT, ensure_dir, formal_engine_name
from quantization.utils.trt_runtime import find_trtexec


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--precision", default=DEFAULT_PRECISION, choices=["fp32", "fp16", "int8"])
    parser.add_argument("--precision-mapping", required=True, help="Formal canonical_precision_mapping.json")
    parser.add_argument("--fixed-k", "--fixed_k", dest="fixed_k", type=int, default=DEFAULT_FIXED_K)
    parser.add_argument("--trt-root", "--trt_root", dest="trt_root", default=str(DEFAULT_TRT_ROOT or ""))
    parser.add_argument("--trtexec-path", "--trtexec_path", dest="trtexec_path", default=None)
    parser.add_argument("--plugin", default=str(DEFAULT_PLUGIN or ""))
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True)
    parser.add_argument("--calibration-npz-dir", "--calibration_npz_dir", dest="calibration_npz_dir", default=None)
    parser.add_argument("--calibration-cache", "--calibration_cache", dest="calibration_cache", default=None)
    parser.add_argument("--calibration-frames", "--calibration_frames", dest="calibration_frames", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--skip-existing", "--skip_existing", dest="skip_existing", action="store_true")
    return parser.parse_args(argv)


def _profile(args: argparse.Namespace) -> dict[str, dict[str, tuple[int, ...]]]:
    shapes = load_observed_shapes(args.calibration_npz_dir) if getattr(args, "calibration_npz_dir", None) else []
    raw = profile_from_observed_shapes(shapes, fixed_k=int(args.fixed_k))
    return {
        str(name): {str(kind): tuple(int(value) for value in dims) for kind, dims in profile.items()}
        for name, profile in raw.items()
    }


def _load_mapping(path: str | Path) -> CanonicalPrecisionMappingResult:
    payload = load_json(path)
    return CanonicalPrecisionMappingResult(
        entries=[CanonicalPrecisionEntry(**row) for row in payload.get("entries", [])],
        profile_id=payload.get("profile_id", ""),
        profile_hash=payload.get("profile_hash", ""),
        origin_map_hash=payload.get("origin_map_hash", ""),
        policy_version=payload.get("policy_version", "canonical-precision-mapping-v1"),
        mapping_hash=payload.get("mapping_hash", ""),
        schema_version=payload.get("schema_version", "canonical-precision-mapping-v1"),
    )


def build_engine(args: argparse.Namespace) -> dict[str, Any]:
    """Validate compatibility arguments and delegate the build transaction."""

    precision = str(args.precision).lower()
    if precision not in {"fp32", "fp16", "int8"}:
        raise ValueError(f"unsupported strict precision: {precision}")
    onnx = Path(args.onnx).expanduser()
    plugin = Path(args.plugin).expanduser()
    mapping_path = Path(getattr(args, "precision_mapping", "")).expanduser()
    output_dir = ensure_dir(args.output_dir)
    engine = output_dir / formal_engine_name("fp16" if precision == "fp32" else precision, int(args.fixed_k), args.calibration_frames)
    if precision == "fp32":
        engine = engine.with_name(engine.name.replace("_fp16.engine", "_fp32.engine"))
    layerinfo = engine.with_suffix(".layerinfo.json")
    log_path = engine.with_suffix(".build.log")
    report: dict[str, Any] = {
        "formal_tool": "quantization.build.build_single_engine_maxk_engine",
        "implementation": "quantization.api.build_trt_engine",
        "strategy": "single_engine_maxK",
        "fixed_K": int(args.fixed_k),
        "precision": precision,
        "onnx": str(onnx),
        "plugin": str(plugin),
        "precision_mapping": str(mapping_path),
        "engine": str(engine),
        "layerinfo": str(layerinfo),
        "success": False,
        "status": "not_started",
    }
    destination = output_dir / "formal_build_report.json"
    if engine.exists() and bool(getattr(args, "skip_existing", False)):
        report.update({"success": True, "status": "skipped_existing", "engine_info": file_info(engine)})
        atomic_write_json(destination, report)
        return report
    missing = [str(path) for path in (onnx, plugin, mapping_path) if not path.is_file()]
    if missing:
        report.update({"status": "missing_required_artifact", "error": f"missing required artifacts: {missing}"})
        atomic_write_json(destination, report)
        return report
    trtexec = find_trtexec(getattr(args, "trt_root", None), getattr(args, "trtexec_path", None))
    if not trtexec.get("trtexec_found"):
        report.update({"status": "missing_trtexec", "error": "trtexec not found"})
        atomic_write_json(destination, report)
        return report
    mapping = _load_mapping(mapping_path)
    config = TensorRTBuildConfig(
        trtexec_path=Path(str(trtexec["trtexec_path"])),
        plugin_path=plugin,
        shape_profiles=_profile(args),
        timeout_seconds=int(args.timeout),
        precision_constraints="obey",
        enable_fp16=precision in {"fp16", "int8"},
        enable_int8=precision == "int8",
        no_tf32=True,
        skip_inference=True,
        export_layer_info=True,
        policy_version=f"deprecated-wrapper-strict-{precision}-v1",
    )
    result = build_trt_engine(
        onnx,
        engine,
        mapping,
        config=config,
        layer_info_path=layerinfo,
        log_path=log_path,
        raise_on_failure=False,
    )
    report.update(result.to_dict())
    report.update(
        {
            "formal_tool": "quantization.build.build_single_engine_maxk_engine",
            "implementation": "quantization.api.build_trt_engine",
            "status": "success" if result.success else "trtexec_failed",
            "success": bool(result.success),
            "engine": str(engine),
            "engine_info": file_info(engine),
        }
    )
    atomic_write_json(destination, report)
    return report


def build_single_engine_maxk_engine(args: argparse.Namespace) -> dict[str, Any]:
    """Compatibility entry point; new callers should use ``quantization.api``."""

    warnings.warn(
        "quantization.build.build_single_engine_maxk_engine is deprecated; use quantization.api.build_trt_engine",
        DeprecationWarning,
        stacklevel=2,
    )
    return build_engine(args)


def main(argv: list[str] | None = None) -> int:
    report = build_single_engine_maxk_engine(parse_args(argv))
    print(json.dumps({"success": report.get("success"), "status": report.get("status"), "engine": report.get("engine")}, indent=2))
    return 0 if report.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
