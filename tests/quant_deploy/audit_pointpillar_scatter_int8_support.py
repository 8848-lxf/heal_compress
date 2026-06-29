from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from quant_deploy_utils import ensure_quant_deploy_run_dirs, save_json


PLUGIN_SRC_DIR = Path(__file__).resolve().parent / "plugins" / "pointpillar_scatter_trt"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit PointPillarScatterTRT INT8 support from plugin source.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--plugin_src_dir", default=str(PLUGIN_SRC_DIR))
    return parser.parse_args(argv)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def _extract_function(text: str, name: str) -> str:
    match = re.search(rf"{re.escape(name)}\s*\([^)]*\).*?\{{", text, flags=re.DOTALL)
    if not match:
        return ""
    start = match.end()
    depth = 1
    pos = start
    while pos < len(text) and depth:
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
        pos += 1
    return text[match.start() : pos]


def audit(plugin_src_dir: str | Path) -> dict[str, Any]:
    src_dir = Path(plugin_src_dir)
    cpp = _read(src_dir / "pointpillar_scatter_plugin.cpp")
    cu = _read(src_dir / "pointpillar_scatter_kernel.cu")
    header = _read(src_dir / "pointpillar_scatter_plugin.h")
    common = _read(src_dir / "plugin_common.h")
    all_text = "\n".join([cpp, cu, header, common])

    supports = _extract_function(cpp, "PointPillarScatterPlugin::supportsFormatCombination")
    enqueue = _extract_function(cpp, "PointPillarScatterPlugin::enqueue")
    configure = _extract_function(cpp, "PointPillarScatterPlugin::configurePlugin")
    out_dtype = _extract_function(cpp, "PointPillarScatterPlugin::getOutputDataType")
    out_dims = _extract_function(cpp, "PointPillarScatterPlugin::getOutputDimensions")

    feature_supports_fp32 = "DataType::kFLOAT" in supports or "kFLOAT" in supports or "isFp" in supports
    feature_supports_fp16 = "DataType::kHALF" in supports or "kHALF" in supports or "isFp" in supports
    feature_supports_int8 = "DataType::kINT8" in supports or "kINT8" in supports
    kernel_has_int8_path = "kINT8" in cu or re.search(r"launchTyped\s*<\s*(int8_t|int8)", cu) is not None
    coords_int32 = "inOut[pos].type == DataType::kINT32" in supports and "coords" in configure.lower()
    mask_float = "DataType::kFLOAT" in supports and "mask" in configure.lower()
    mask_half = "DataType::kHALF" in supports and "mask" in configure.lower()
    mask_int32 = "DataType::kINT32" in supports and "mask" in configure.lower()
    mask_bool = "DataType::kBOOL" in supports and "mask" in configure.lower()
    output_follows_input = "return inputTypes[0]" in out_dtype
    serialized_num_agents = "mParams.numAgents" in out_dims

    plugin_supports_int8_io = bool(feature_supports_int8 and kernel_has_int8_path)
    plugin_supports_fp16_io = bool(feature_supports_fp16 and "launchTyped<half>" in cu and output_follows_input)
    plugin_supports_fp32_io = bool(feature_supports_fp32 and "launchTyped<float>" in cu and output_follows_input)
    expected_precision = "int8" if plugin_supports_int8_io else ("fp16" if plugin_supports_fp16_io else ("fp32" if plugin_supports_fp32_io else "unknown"))
    reformat_required: bool | str = True if not plugin_supports_int8_io else "unknown"
    risk_level = "medium" if plugin_supports_fp16_io or plugin_supports_fp32_io else "high"
    strategy = (
        "Use native TensorRT INT8 calibration first and allow this plugin to run as an FP16/FP32 precision island. "
        "Audit the INT8 engine layer info for Reformat/Quantize/Dequantize around PointPillarScatterTRT. "
        "If AP drops, protect heads and the plugin in FP16 before moving to Q/DQ or ModelOpt."
    )
    return {
        "plugin_src_dir": str(src_dir),
        "source_files": {
            "cpp": str(src_dir / "pointpillar_scatter_plugin.cpp"),
            "cu": str(src_dir / "pointpillar_scatter_kernel.cu"),
            "header": str(src_dir / "pointpillar_scatter_plugin.h"),
        },
        "audited_methods": {
            "supportsFormatCombination": bool(supports),
            "enqueue": bool(enqueue),
            "configurePlugin": bool(configure),
            "getOutputDataType": bool(out_dtype),
            "getOutputDimensions": bool(out_dims),
        },
        "plugin_supports_int8_io": plugin_supports_int8_io,
        "plugin_supports_fp16_io": plugin_supports_fp16_io,
        "plugin_supports_fp32_io": plugin_supports_fp32_io,
        "expected_plugin_precision_in_int8_engine": expected_precision,
        "int8_engine_requires_reformat_around_plugin": reformat_required,
        "risk_level": risk_level,
        "recommended_int8_strategy": strategy,
        "valid_voxel_mask_dtype": {
            "onnx_expected": "float32 in the dynamic fixed-K ONNX inputs",
            "plugin_accepts": [dtype for dtype, ok in (("fp32", mask_float), ("fp16", mask_half), ("int32", mask_int32), ("bool", mask_bool)) if ok],
        },
        "voxel_coords_dtype": "int32" if coords_int32 else "unknown",
        "int8_fallback_assessment": {
            "plugin_has_int8_kernel": kernel_has_int8_path,
            "output_dtype_follows_feature_input": output_follows_input,
            "tensorRT_can_fallback": "expected, if the builder can insert quantize/dequantize or reformat tensors around a plugin with FP32/FP16 formats",
            "needs_layer_audit": True,
        },
        "implementation_notes": {
            "uses_serialized_num_agents_for_output_dim": serialized_num_agents,
            "true_dynamic_N_single_engine_supported": False,
            "enqueue_feature_dispatch": "half/float only",
        },
        "snippets": {
            "supportsFormatCombination": supports,
            "getOutputDataType": out_dtype,
            "getOutputDimensions": out_dims,
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# PointPillarScatterTRT INT8 Support Audit",
        "",
        f"- plugin_supports_int8_io: {report['plugin_supports_int8_io']}",
        f"- plugin_supports_fp16_io: {report['plugin_supports_fp16_io']}",
        f"- plugin_supports_fp32_io: {report['plugin_supports_fp32_io']}",
        f"- expected_plugin_precision_in_int8_engine: {report['expected_plugin_precision_in_int8_engine']}",
        f"- int8_engine_requires_reformat_around_plugin: {report['int8_engine_requires_reformat_around_plugin']}",
        f"- risk_level: {report['risk_level']}",
        "",
        "## Dtype Findings",
        "",
        f"- valid_voxel_mask dtype: {report['valid_voxel_mask_dtype']}",
        f"- voxel_coords dtype: {report['voxel_coords_dtype']}",
        "- feature/output INT8: not supported unless both supportsFormatCombination and CUDA enqueue add kINT8 handling.",
        "",
        "## Recommendation",
        "",
        report["recommended_int8_strategy"],
        "",
    ]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    dirs = ensure_quant_deploy_run_dirs(args.output_root)
    report = audit(args.plugin_src_dir)
    save_json(report, dirs["debug"] / "pointpillar_scatter_int8_support_audit.json")
    (dirs["summary"] / "pointpillar_scatter_int8_support_audit.md").write_text(_markdown(report), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    report = run(parse_args(argv))
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
