#!/usr/bin/env python3
"""Distillation finetuning entry point.

Usage:
    python scripts/run_finetune.py \
        --config configs/default_config.yaml \
        --checkpoint /path/to/model.pth \
        --heal-config /path/to/heal_config.yaml \
        --export-script /path/to/export_dynamic_onnx.py \
        --subnet-config ./search_results/best_subnet_config.json \
        --calibration-npz /path/to/calib_npz/ \
        --output-dir ./finetune_results/

Flow:
    1. Load original model (teacher) and create proxy model (student)
    2. Apply channel masks and pseudo-quantization per subnet config
    3. BEVKLDistillationTrainer finetunes with BEV KL loss
    4. Save finetuned proxy model
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
from heal_compress.utils.model_utils import import_export_module
from heal_compress.distillation.bev_kl_distill import BEVKLDistillationTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="BEV KL distillation finetuning.")
    parser.add_argument("--config", default="configs/default_config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--heal-config", default=None)
    parser.add_argument("--export-script", required=True)
    parser.add_argument("--subnet-config", required=True,
                        help="Path to best_subnet_config.json from search.")
    parser.add_argument("--calibration-npz", required=True)
    parser.add_argument("--output-dir", default="./finetune_results")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    subnet = load_json(args.subnet_config)
    out_dir = ensure_dir(args.output_dir)

    import torch
    import copy
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    export_mod = import_export_module(args.export_script)

    # Load teacher (original)
    logger.info("Loading teacher model...")
    teacher, _ = export_mod.load_model_and_hypes(
        Path(args.checkpoint), args.heal_config, torch.device(device)
    )
    export_mod.install_plugin_patches(teacher)

    # Load student (copy)
    logger.info("Loading student model...")
    student, _ = export_mod.load_model_and_hypes(
        Path(args.checkpoint), args.heal_config, torch.device(device)
    )
    export_mod.install_plugin_patches(student)

    # Build calibration data loader
    from heal_compress.utils.model_utils import load_calibration_npz
    calib_inputs = load_calibration_npz(args.calibration_npz, device)
    # Wrap as simple list of batches
    train_loader = [calib_inputs]

    dist_cfg = cfg.get("distillation", {})
    modality = cfg.get("model", {}).get("modality")
    if modality is None:
        for name in getattr(teacher, "modality_name_list", []):
            if hasattr(teacher, f"encoder_{name}"):
                modality = name
                break

    trainer = BEVKLDistillationTrainer(
        teacher=teacher,
        student=student,
        export_module=export_mod,
        modality=modality,
        epochs=dist_cfg.get("epochs", 10),
        lr=dist_cfg.get("lr", 1e-4),
        tau=dist_cfg.get("tau", 4.0),
        lambda_bev=dist_cfg.get("lambda_bev", 0.5),
        device=device,
    )

    ckpt = trainer.run(train_loader, subnet, str(out_dir))
    logger.info(f"Finetuning complete. Model saved to {ckpt}")


if __name__ == "__main__":
    main()
