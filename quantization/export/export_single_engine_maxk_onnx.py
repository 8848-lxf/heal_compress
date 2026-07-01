from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from quantization.utils.logging import save_json, status_record
from quantization.utils.paths import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_FIXED_K,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PRECISION,
    DEFAULT_TRT_ROOT,
    ensure_dir,
    formal_onnx_name,
    infer_output_root,
    load_quant_deploy_module,
    validate_precision,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export formal single_engine_maxK fixedK ONNX.")
    parser.add_argument("--config", "--hypes-yaml", "--hypes_yaml", dest="config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--fixed-k", "--fixed_k", dest="fixed_k", type=int, default=DEFAULT_FIXED_K)
    parser.add_argument("--precision", default=DEFAULT_PRECISION, choices=["fp32", "fp16", "int8"])
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default=str(DEFAULT_OUTPUT_ROOT / "artifacts/onnx/fixedK29696/single_engine_maxK"))
    parser.add_argument("--heal-repo", "--heal_repo", dest="heal_repo", default="/home/lixingfeng/UniAD_examine/HEAL")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-cav", "--max_cav", dest="max_cav", type=int, default=2)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--trt-root", "--trt_root", dest="trt_root", default=str(DEFAULT_TRT_ROOT))
    parser.add_argument("--trtexec-path", "--trtexec_path", dest="trtexec_path", default=None)
    parser.add_argument("--export-sample-split", "--export_sample_split", dest="export_sample_split", default="train", choices=["train", "val"])
    parser.add_argument("--max-scan-samples", "--max_scan_samples", dest="max_scan_samples", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def export_single_engine_maxk_onnx(args: argparse.Namespace) -> dict[str, Any]:
    precision = validate_precision(args.precision)
    if precision == "int8":
        # ONNX export is precision-agnostic; INT8 remains an optional build path.
        precision = "fp16"
    output_dir = ensure_dir(args.output_dir)
    output_root = infer_output_root(output_dir)
    legacy = load_quant_deploy_module("export_dynamic_single_engine_maxk_onnx")
    legacy_common = load_quant_deploy_module("dynamic_single_engine_maxk_common")
    legacy_args = SimpleNamespace(
        output_root=str(output_root),
        hypes_yaml=str(args.config),
        checkpoint=str(args.checkpoint),
        heal_repo=str(args.heal_repo),
        device=str(args.device),
        max_cav=int(args.max_cav),
        fixed_k=int(args.fixed_k),
        opset=int(args.opset),
        trt_root=str(args.trt_root),
        trtexec_path=args.trtexec_path,
        export_sample_split=str(args.export_sample_split),
        max_scan_samples=int(args.max_scan_samples),
        overwrite=bool(args.overwrite),
    )
    report = legacy.export_onnx(legacy_args)
    legacy_dirs = load_quant_deploy_module("quant_deploy_utils").ensure_quant_deploy_run_dirs(str(output_root))
    legacy_path = legacy_common.onnx_path(legacy_dirs, fixed_k=int(args.fixed_k))
    formal_path = output_dir / legacy_path.name
    named_formal_path = output_dir / formal_onnx_name(int(args.fixed_k))
    if legacy_path.exists():
        if formal_path.resolve() != legacy_path.resolve():
            shutil.copyfile(legacy_path, formal_path)
        if named_formal_path != formal_path:
            shutil.copyfile(legacy_path, named_formal_path)
    report.update(
        {
            "formal_tool": "quantization.export.export_single_engine_maxk_onnx",
            "formal_strategy": "single_engine_maxK",
            "default_precision": "fp16",
            "requested_precision": args.precision,
            "effective_export_precision": precision,
            "fixed_K": int(args.fixed_k),
            "formal_output_dir": str(output_dir),
            "formal_onnx_path": str(named_formal_path if named_formal_path.exists() else formal_path),
            "legacy_output_root": str(output_root),
        }
    )
    save_json(report, output_dir / "formal_export_report.json")
    return report


def main(argv: list[str] | None = None) -> int:
    report = export_single_engine_maxk_onnx(parse_args(argv))
    print(json.dumps({"success": report.get("success"), "onnx_path": report.get("formal_onnx_path")}, indent=2))
    return 0 if report.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
