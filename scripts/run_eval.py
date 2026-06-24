#!/usr/bin/env python3
"""Evaluation entry point: validate pruned/quantized model on test data.

Usage:
    python scripts/run_eval.py \
        --checkpoint /path/to/pruned_model.pth \
        --heal-config /path/to/heal_config.yaml \
        --export-script /path/to/export_dynamic_onnx.py \
        --example-npz /path/to/frame_000000.npz \
        --output-dir ./eval_results/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from heal_compress.utils.io_utils import load_yaml, save_json, ensure_dir
from heal_compress.utils.model_utils import import_export_module, load_calibration_npz

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate pruned/quantized HEAL model.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--heal-config", default=None)
    parser.add_argument("--export-script", required=True)
    parser.add_argument("--example-npz", required=True)
    parser.add_argument("--output-dir", default="./eval_results")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)

    import torch
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    export_mod = import_export_module(args.export_script)

    # Load model
    logger.info("Loading model...")
    model, hypes = export_mod.load_model_and_hypes(
        Path(args.checkpoint), args.heal_config, torch.device(device)
    )
    export_mod.install_plugin_patches(model)
    model.eval()

    # Load example inputs
    inputs = load_calibration_npz(args.example_npz, device)

    # Detect modality and run forward
    modality = None
    for name in getattr(model, "modality_name_list", []):
        if hasattr(model, f"encoder_{name}"):
            modality = name
            break

    if modality is None:
        logger.error("Cannot detect model modality.")
        return

    crop_params = export_mod._record_crop_params(model, modality, inputs)
    wrapper = export_mod.DynamicAgentExportWrapper(
        model, modality_name=modality, crop_params=crop_params,
        insert_se3_inverse=True,
    ).to(device).eval()

    logger.info("Running forward pass...")
    with torch.no_grad():
        outputs = wrapper(*inputs)

    output_names = ["cls_preds", "reg_preds", "dir_preds",
                     "occ_single_0", "occ_single_1", "occ_single_2"]

    # Report output shapes
    results = {}
    for name, tensor in zip(output_names, outputs):
        shape = list(tensor.shape)
        results[name] = {
            "shape": shape,
            "min": float(tensor.min().cpu()),
            "max": float(tensor.max().cpu()),
            "mean": float(tensor.mean().cpu()),
        }
        logger.info(f"  {name}: shape={shape}")

    # Model stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024

    stats = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "model_size_mb": round(model_size_mb, 2),
        "outputs": results,
    }

    save_json(stats, str(Path(out_dir) / "eval_results.json"))
    logger.info(f"Model: {total_params:,} params, {model_size_mb:.2f} MB")
    logger.info(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()
