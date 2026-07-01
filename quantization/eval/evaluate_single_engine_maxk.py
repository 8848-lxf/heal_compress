from __future__ import annotations

import argparse
import json
from types import SimpleNamespace
from typing import Any

from quantization.utils.logging import save_json
from quantization.utils.paths import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_FIXED_K,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PLUGIN,
    DEFAULT_PRECISION,
    DEFAULT_TRT_ROOT,
    ensure_dir,
    infer_output_root,
    infer_output_tag,
    load_quant_deploy_module,
    validate_precision,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate formal single_engine_maxK TensorRT engine on full validation.")
    parser.add_argument("--config", "--hypes-yaml", "--hypes_yaml", dest="config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--engine", required=True)
    parser.add_argument("--precision", default=DEFAULT_PRECISION, choices=["fp32", "fp16", "int8"])
    parser.add_argument("--fixed-k", "--fixed_k", dest="fixed_k", type=int, default=DEFAULT_FIXED_K)
    parser.add_argument("--split", default="val", choices=["val"])
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", default=str(DEFAULT_OUTPUT_ROOT / "evaluation/single_engine_maxK_fixedK29696_fp16_formal_tool"))
    parser.add_argument("--plugin", default=str(DEFAULT_PLUGIN))
    parser.add_argument("--heal-repo", "--heal_repo", dest="heal_repo", default="/home/lixingfeng/UniAD_examine/HEAL")
    parser.add_argument("--trt-root", "--trt_root", dest="trt_root", default=str(DEFAULT_TRT_ROOT))
    parser.add_argument("--gpu-index", "--gpu_index", dest="gpu_index", type=int, default=None)
    parser.add_argument("--summarize-existing", "--summarize_existing", dest="summarize_existing", action="store_true")
    return parser.parse_args(argv)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    precision = validate_precision(args.precision)
    output_dir = ensure_dir(args.output_dir)
    output_root = infer_output_root(output_dir)
    output_tag = infer_output_tag(output_dir, f"single_engine_maxK_fixedK{int(args.fixed_k)}_{precision}_formal_tool")
    mode = f"single_engine_maxK_{precision}"
    if precision == "int8":
        mode = "single_engine_maxK_int8_train_calib200"
    legacy = load_quant_deploy_module("evaluate_all_deployment_engines_full_val_idle_gpu")
    legacy_args = SimpleNamespace(
        output_root=str(output_root),
        plugin_path=str(args.plugin),
        hypes_yaml=str(args.config),
        checkpoint=str(args.checkpoint),
        heal_repo=str(args.heal_repo),
        trt_root=str(args.trt_root),
        max_cav=2,
        eval_split="val",
        eval_all=False,
        latency_warmup_frames=20,
        latency_repeat=1,
        ap_iou_backend="gpu",
        gpu_idle_util_threshold=5,
        gpu_idle_mem_threshold_mb=2000,
        gpu_wait_timeout_sec=3600,
        gpu_poll_interval_sec=30,
        gpu_index=args.gpu_index,
        allow_busy_gpu=False,
        schemes=[mode],
        summarize_existing=bool(args.summarize_existing),
        fixed_k=int(args.fixed_k),
        output_tag=output_tag,
        fixedk_engine_namespace=True,
        dynamic_int8_calibration_split="train" if precision == "int8" else None,
        include_mixed_heads_fp16=False,
        force_gpu_index_no_nvidia_smi=None,
        excluded_gpu_indices="",
        progress_every_frames=1,
        no_progress_stdout=False,
    )
    if args.summarize_existing:
        summary = legacy.summarize_existing(legacy_args)
        report = {"success": bool(summary.get("success")), "summary": summary}
    else:
        report = legacy.run(legacy_args)
    report.update(
        {
            "formal_tool": "quantization.eval.evaluate_single_engine_maxk",
            "strategy": "single_engine_maxK",
            "fixed_K": int(args.fixed_k),
            "precision": precision,
            "engine_argument": str(args.engine),
            "output_dir": str(output_dir),
            "output_tag": output_tag,
        }
    )
    save_json(report, output_dir / "formal_evaluation_report.json")
    return report


def main(argv: list[str] | None = None) -> int:
    report = evaluate(parse_args(argv))
    print(json.dumps({"success": report.get("success"), "strategy": report.get("strategy")}, indent=2))
    return 0 if report.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
