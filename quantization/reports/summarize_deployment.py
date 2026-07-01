from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from quantization.utils.engine_io import file_info
from quantization.utils.logging import save_json, save_markdown, status_record
from quantization.utils.paths import (
    DEFAULT_FIXED_K,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PRECISION,
    DEFAULT_STRATEGY,
    ensure_dir,
    normalize_strategy,
)

MIGRATED_SCRIPTS = [
    "tests/quant_deploy/export_dynamic_single_engine_maxk_onnx.py",
    "tests/quant_deploy/exportable_lidar_pyramid_dynamic_single_engine_maxk.py",
    "tests/quant_deploy/dynamic_single_engine_maxk_common.py",
    "tests/quant_deploy/rebuild_fixedk_full_cover_engines.py",
    "tests/quant_deploy/dump_train_calibration_npz_for_all_strategies.py",
    "tests/quant_deploy/evaluate_all_deployment_engines_full_val_idle_gpu.py",
    "tests/quant_deploy/summarize_engine_file_sizes.py",
    "tests/quant_deploy/summarize_final_fixedk_full_cover_traincalib.py",
    "tests/quant_deploy/select_idle_gpu.py",
    "tests/quant_deploy/deployment_equivalence.py",
    "tests/quant_deploy/plugins/pointpillar_scatter_trt/",
]

TESTS_RETAINED_AS_REGRESSION = [
    "tests/test_quant_deploy_utils.py",
    "tests/quant_deploy/*.py historical CLI scripts",
    "tests/quant_deploy/plugins/pointpillar_scatter_trt/ source mirror",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize formal quantization deployment migration.")
    parser.add_argument("--root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY)
    parser.add_argument("--fixed-k", "--fixed_k", dest="fixed_k", type=int, default=DEFAULT_FIXED_K)
    return parser.parse_args(argv)


def build_migration_report(root: str | Path, *, strategy: str = DEFAULT_STRATEGY, fixed_k: int = DEFAULT_FIXED_K) -> dict[str, Any]:
    root_path = ensure_dir(root)
    strategy = normalize_strategy(strategy)
    fp16_engine_dir = root_path / "artifacts" / "engines" / f"fixedK{int(fixed_k)}" / "dynamic_agent_single_engine_maxK" / "fp16"
    fp16_engines = sorted(fp16_engine_dir.glob("*.engine"))
    report = status_record(
        success=True,
        status="migration_documented",
        formal_tool="quantization.reports.summarize_deployment",
        default_strategy="single_engine_maxK",
        requested_strategy=strategy,
        fixed_K=int(fixed_k),
        default_precision=DEFAULT_PRECISION,
        calibration_split="train",
        eval_split="val",
        int8_role="optional_build_and_evaluation_path_not_default_recommendation",
        padded_agent_static_role="legacy_baseline_only",
        dynamic_bucket_role="legacy_comparison_only",
        heal_opencood_source_modified=False,
        dynamic_k_frontend_implemented=False,
        dynamic_pillar_vfe_scatter_trt_plugin_implemented=False,
        migrated_scripts=MIGRATED_SCRIPTS,
        tests_retained_as_regression=TESTS_RETAINED_AS_REGRESSION,
        formal_modules={
            "export": "quantization.export.export_single_engine_maxk_onnx",
            "build": "quantization.build.build_single_engine_maxk_engine",
            "calibrate": "quantization.calibrate.dump_train_calibration_npz",
            "eval": "quantization.eval.evaluate_single_engine_maxk",
            "benchmark": "quantization.benchmark.benchmark_engine_latency",
        },
        pointpillar_scatter_trt_plugin="quantization/plugins/pointpillar_scatter_trt/",
        existing_fp16_engines=[file_info(path) for path in fp16_engines],
    )
    return report


def write_migration_report(report: dict[str, Any], root: str | Path) -> tuple[Path, Path]:
    root_path = ensure_dir(root)
    md_path = root_path / "summary" / "formal_quantization_tool_migration_report.md"
    json_path = root_path / "debug" / "formal_quantization_tool_migration_report.json"
    save_json(report, json_path)
    lines = [
        "# Formal Quantization Tool Migration Report",
        "",
        f"- default deployment strategy: {report['default_strategy']} fixedK{report['fixed_K']} {report['default_precision'].upper()}",
        f"- INT8 default recommendation: no ({report['int8_role']})",
        f"- padded_agent_static role: {report['padded_agent_static_role']}",
        f"- dynamic_bucket role: {report['dynamic_bucket_role']}",
        f"- PointPillarScatterTRT plugin: {report['pointpillar_scatter_trt_plugin']}",
        f"- modified HEAL/OpenCOOD source: {'yes' if report['heal_opencood_source_modified'] else 'no'}",
        f"- external dynamic-K frontend implemented: {'yes' if report['dynamic_k_frontend_implemented'] else 'no'}",
        f"- DynamicPillarVFEAndScatterTRT plugin implemented: {'yes' if report['dynamic_pillar_vfe_scatter_trt_plugin_implemented'] else 'no'}",
        "",
        "## Migrated Sources",
    ]
    lines.extend(f"- {item}" for item in report["migrated_scripts"])
    lines.extend(
        [
            "",
            "## Regression Tests / Historical Scripts Kept",
            *[f"- {item}" for item in report["tests_retained_as_regression"]],
            "",
            "## CLI Examples",
            "",
            "```bash",
            "python -m quantization.export.export_single_engine_maxk_onnx --config <config.yaml> --checkpoint <ckpt.pth> --fixed-k 29696 --precision fp16 --output-dir tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/onnx/fixedK29696/single_engine_maxK/",
            "python -m quantization.build.build_single_engine_maxk_engine --onnx <onnx_path> --precision fp16 --fixed-k 29696 --plugin quantization/plugins/pointpillar_scatter_trt/build/libpointpillar_scatter_trt.so --output-dir tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/engines/fixedK29696/dynamic_agent_single_engine_maxK/fp16/",
            "python -m quantization.calibrate.dump_train_calibration_npz --config <config.yaml> --checkpoint <ckpt.pth> --fixed-k 29696 --num-frames 200 --split train --strategy single_engine_maxK --output-dir tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/artifacts/calibration/train_calib_single_engine_maxK29696_200/",
            "python -m quantization.eval.evaluate_single_engine_maxk --config <config.yaml> --checkpoint <ckpt.pth> --engine <engine_path> --precision fp16 --fixed-k 29696 --split val --output-dir tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/evaluation/single_engine_maxK_fixedK29696_fp16_formal_tool/",
            "python -m quantization.reports.summarize_deployment --root tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/ --strategy single_engine_maxK --fixed-k 29696",
            "```",
        ]
    )
    save_markdown(lines, md_path)
    return md_path, json_path


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    report = build_migration_report(args.root, strategy=args.strategy, fixed_k=int(args.fixed_k))
    md_path, json_path = write_migration_report(report, args.root)
    report["markdown_path"] = str(md_path)
    report["json_path"] = str(json_path)
    return report


def main(argv: list[str] | None = None) -> int:
    report = summarize(parse_args(argv))
    print(json.dumps({"success": report.get("success"), "markdown_path": report.get("markdown_path")}, indent=2))
    return 0 if report.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
