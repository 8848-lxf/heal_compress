#!/usr/bin/env python3
"""Physical pruning + ONNX export + TRT engine build entry point.

Usage:
    python scripts/run_prune_export.py \
        --config configs/default_config.yaml \
        --checkpoint ./finetune_results/proxy_model_finetuned.pth \
        --heal-config /path/to/heal_config.yaml \
        --export-script /path/to/export_dynamic_onnx.py \
        --subnet-config ./search_results/best_subnet_config.json \
        --groups ./search_results/coupled_channel_groups.json \
        --example-npz /path/to/frame_000000.npz \
        --output-dir ./deploy/

Flow:
    1. PhysicalPruner executes physical channel pruning
    2. StructureLegalityChecker validates pruned structure
    3. ChannelAlignmentChecker checks hardware alignment
    4. Export pruned ONNX via install_plugin_patches + DynamicAgentExportWrapper
    5. QDQDeploymentBuilder inserts weight Q/DQ nodes
    6. TRTBuilder constructs TensorRT engine
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from heal_compress.utils.io_utils import load_yaml, load_json, ensure_dir
from heal_compress.utils.model_utils import import_export_module, load_calibration_npz
from heal_compress.pruning.physical_pruner import PhysicalPruner
from heal_compress.pruning.legality_checker import StructureLegalityChecker
from heal_compress.pruning.alignment_checker import ChannelAlignmentChecker
from heal_compress.quantization.qdq_builder import QDQDeploymentBuilder
from heal_compress.deploy.trt_builder import TRTBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Physical pruning + ONNX export + TRT build.")
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--heal-config", default=None)
    parser.add_argument("--export-script", required=True)
    parser.add_argument("--subnet-config", required=True)
    parser.add_argument("--groups", required=True,
                        help="Path to coupled_channel_groups.json.")
    parser.add_argument("--example-npz", required=True)
    parser.add_argument("--output-dir", default="./deploy")
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-trt", action="store_true",
                        help="Skip TRT engine build.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    out_dir = ensure_dir(args.output_dir)

    import torch
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    export_mod = import_export_module(args.export_script)

    # Step 1: Load model
    logger.info("Loading model...")
    model, hypes = export_mod.load_model_and_hypes(
        Path(args.checkpoint), args.heal_config, torch.device(device)
    )
    export_mod.install_plugin_patches(model)

    # Step 2: Physical pruning
    logger.info("Executing physical pruning...")
    pruner = PhysicalPruner(model)
    pruned_ckpt = pruner.apply(
        args.subnet_config, args.groups, str(out_dir / "pruning"),
    )

    # Step 3: Legality check
    logger.info("Checking structure legality...")
    calib_inputs = load_calibration_npz(args.example_npz, device)
    checker = StructureLegalityChecker(model)
    report = checker.check(
        str(out_dir / "legality"),
        calibration_inputs=calib_inputs,
        export_module=export_mod,
    )
    if not report["legal"]:
        logger.error(f"Legality check failed with {report['num_issues']} issues!")
        for issue in report["issues"][:5]:
            logger.error(f"  {issue}")

    # Step 4: Alignment check
    logger.info("Checking channel alignment...")
    subnet = load_json(args.subnet_config)
    aligner = ChannelAlignmentChecker(
        default_align=cfg.get("search", {}).get("channel_alignment", 8)
    )
    align_report = aligner.check(
        model, subnet.get("bitwidth_vars"), str(out_dir / "alignment"),
    )

    # Step 5: ONNX export
    logger.info("Exporting pruned ONNX...")
    modality = cfg.get("model", {}).get("modality")
    if modality is None:
        for name in getattr(model, "modality_name_list", []):
            if hasattr(model, f"encoder_{name}"):
                modality = name
                break

    crop_params = export_mod._record_crop_params(model, modality, calib_inputs)
    wrapper = export_mod.DynamicAgentExportWrapper(
        model, modality_name=modality, crop_params=crop_params,
        insert_se3_inverse=True,
    ).to(device).eval()

    onnx_path = str(out_dir / "model_pruned.onnx")
    input_names = ["imgs", "rots", "trans", "intrins", "post_rots", "post_trans", "pairwise_t_matrix"]
    output_names = ["cls_preds", "reg_preds", "dir_preds", "occ_single_0", "occ_single_1", "occ_single_2"]

    with torch.no_grad():
        torch.onnx.export(
            wrapper, calib_inputs, onnx_path,
            opset_version=17,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=export_mod._input_dynamic_axes(),
            do_constant_folding=True,
        )
    export_mod._normalize_plugin_domains_for_trt(onnx_path)
    logger.info(f"ONNX exported to {onnx_path}")

    # Step 6: Insert Q/DQ nodes
    logger.info("Inserting Q/DQ nodes...")
    qdq_builder = QDQDeploymentBuilder(
        args.subnet_config,
        granularity=cfg.get("deploy", {}).get("quant_granularity", "per_channel"),
    )
    qdq_result = qdq_builder.insert_qdq(onnx_path, str(out_dir))
    logger.info(f"Q/DQ ONNX: {qdq_result['qdq_onnx']}")

    # Step 7: TRT build
    if not args.skip_trt:
        logger.info("Building TRT engine...")
        deploy_cfg = cfg.get("deploy", {})
        trt = TRTBuilder(
            trt_root=deploy_cfg.get("trt_root"),
            plugin_so=deploy_cfg.get("plugin_so"),
            agent_profile=deploy_cfg.get("agent_profile", {"min": 1, "opt": 2, "max": 2}),
            trtexec_path=deploy_cfg.get("trtexec_path", "trtexec"),
        )
        build_result = trt.build(qdq_result["qdq_onnx"], str(out_dir))
        if build_result["success"]:
            logger.info(f"TRT engine built: {build_result['engine_path']}")
        else:
            logger.error("TRT engine build failed. See log for details.")
    else:
        logger.info("Skipping TRT build (--skip-trt)")

    logger.info(f"Deploy pipeline complete. Output: {out_dir}")


if __name__ == "__main__":
    main()
