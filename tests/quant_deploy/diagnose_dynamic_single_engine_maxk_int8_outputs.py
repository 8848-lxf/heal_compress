from __future__ import annotations

import argparse
import ctypes
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deployment_equivalence import TensorRTEngineRunner, _load_model_context
from dynamic_single_engine_maxk_common import engine_path
from quant_deploy_utils import DEFAULT_CHECKPOINT, DEFAULT_HEAL_REPO, DEFAULT_HYPES_YAML, ensure_quant_deploy_run_dirs, save_json
from run_dynamic_single_engine_maxk import _prepare_inputs
from export_lidar_pyramid_onnx import _to_device


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose dynamic single-engine maxK INT8 output drift.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--plugin_path", required=True)
    parser.add_argument("--hypes_yaml", default=str(DEFAULT_HYPES_YAML))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--heal_repo", default=str(DEFAULT_HEAL_REPO))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_frames", type=int, default=10)
    parser.add_argument("--calibration_frames", type=int, nargs="+", default=[50, 200])
    return parser.parse_args(argv)


def _metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    ref = reference.detach().float().reshape(-1)
    cand = candidate.detach().float().reshape(-1)
    if ref.numel() != cand.numel():
        return {"shape_mismatch": True, "reference_shape": list(reference.shape), "candidate_shape": list(candidate.shape)}
    diff = (ref - cand).abs()
    cosine = torch.nn.functional.cosine_similarity(ref, cand, dim=0).item() if ref.numel() else None
    return {
        "shape_mismatch": False,
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
        "cosine": float(cosine) if cosine is not None else None,
        "numel": int(ref.numel()),
    }


def _topk_scores(cls: torch.Tensor, k: int = 100) -> dict[str, Any]:
    values = torch.sigmoid(cls.detach().float()).reshape(-1)
    if values.numel() == 0:
        return {"count": 0}
    topk = torch.topk(values, min(k, values.numel())).values.detach().cpu().numpy().astype(np.float64)
    return {
        "count": int(values.numel()),
        "topk": int(topk.size),
        "mean": float(topk.mean()) if topk.size else None,
        "min": float(topk.min()) if topk.size else None,
        "max": float(topk.max()) if topk.size else None,
        "p50": float(np.percentile(topk, 50)) if topk.size else None,
    }


def _pre_nms_count(cls: torch.Tensor, threshold: float = 0.2) -> int:
    return int((torch.sigmoid(cls.detach().float()) > threshold).sum().item())


def _loader(args: argparse.Namespace):
    hypes, device, model, modality = _load_model_context(args)
    from opencood.data_utils.datasets import build_dataset

    dataset = build_dataset(hypes, visualize=True, train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=dataset.collate_batch_test)
    return device, model, modality, loader


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    dirs = ensure_quant_deploy_run_dirs(args.output_root)
    ctypes.CDLL(str(Path(args.plugin_path).expanduser().resolve()), mode=ctypes.RTLD_GLOBAL)
    device, model, modality, loader = _loader(args)
    runners: dict[str, TensorRTEngineRunner] = {
        "fp32": TensorRTEngineRunner(engine_path(dirs, "fp32"), device),
        "fp16": TensorRTEngineRunner(engine_path(dirs, "fp16"), device),
    }
    for frames in args.calibration_frames:
        path = engine_path(dirs, "int8", int(frames))
        if path.exists():
            runners[f"int8_train_calib{int(frames)}"] = TensorRTEngineRunner(path, device)
    frames_out: list[dict[str, Any]] = []
    output_names = ["cls_preds", "reg_preds", "dir_preds"]
    actual = 0
    for frame_idx, batch in enumerate(loader):
        if actual >= int(args.max_frames):
            break
        if batch is None:
            continue
        try:
            ego = batch["ego"] if isinstance(batch, dict) and "ego" in batch else batch
            ego = _to_device(ego, device)
            with torch.no_grad():
                raw = model(ego)
            output_names = [name for name in output_names if name in raw and torch.is_tensor(raw[name])]
            inputs, meta = _prepare_inputs(ego, modality)
            outputs = {name: runner.run(inputs) for name, runner in runners.items()}
            comparisons: dict[str, Any] = {}
            pairs = [("fp32", "fp16")] + [("fp16", name) for name in outputs if name.startswith("int8")]
            for ref_name, cand_name in pairs:
                if ref_name not in outputs or cand_name not in outputs:
                    continue
                comparisons[f"{ref_name}_vs_{cand_name}"] = {
                    out_name: _metrics(outputs[ref_name][out_name], outputs[cand_name][out_name])
                    for out_name in output_names
                    if out_name in outputs[ref_name] and out_name in outputs[cand_name]
                }
            score_stats = {name: _topk_scores(out["cls_preds"]) for name, out in outputs.items() if "cls_preds" in out}
            pre_nms_counts = {name: _pre_nms_count(out["cls_preds"]) for name, out in outputs.items() if "cls_preds" in out}
            frames_out.append(
                {
                    "frame_id": int(frame_idx),
                    "sample_idx": int(actual),
                    "record_len": int(meta["record_len"]),
                    "original_num_voxels": int(meta["original_num_voxels"]),
                    "comparisons": comparisons,
                    "topk_score_distribution": score_stats,
                    "pre_nms_boxes_count_proxy": pre_nms_counts,
                }
            )
            actual += 1
        except Exception as exc:
            frames_out.append({"frame_id": int(frame_idx), "error": str(exc), "traceback": traceback.format_exc()})
    report = {
        "strategy": "dynamic_agent_single_engine_maxK",
        "calibration_split": "train",
        "evaluation_split": "val",
        "calibration_eval_overlap": False,
        "num_frames": actual,
        "engines": {name: runner.engine_path for name, runner in runners.items()},
        "frames": frames_out,
        "AP@0.70_drop_reason_inference": "cls/reg/dir drift should be inferred from per-head cosine and top-k score changes above.",
    }
    save_json(report, dirs["debug"] / "dynamic_single_engine_maxK_int8_output_diagnosis.json")
    lines = [
        "# Dynamic Single Engine maxK INT8 Output Diagnosis",
        "",
        f"- num_frames: {actual}",
        "- calibration_split: train",
        "- evaluation_split: val",
        "- calibration_eval_overlap: false",
        "",
        "frame | comparison | cls cosine | reg cosine | dir cosine",
        "--- | --- | --- | --- | ---",
    ]
    for frame in frames_out:
        for comp, outputs in (frame.get("comparisons") or {}).items():
            lines.append(
                f"{frame.get('frame_id')} | {comp} | "
                f"{(outputs.get('cls_preds') or {}).get('cosine')} | "
                f"{(outputs.get('reg_preds') or {}).get('cosine')} | "
                f"{(outputs.get('dir_preds') or {}).get('cosine')}"
            )
    (dirs["summary"] / "dynamic_single_engine_maxK_int8_output_diagnosis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    report = diagnose(parse_args(argv))
    print(report)
    return 0 if report.get("num_frames", 0) else 2


if __name__ == "__main__":
    raise SystemExit(main())
