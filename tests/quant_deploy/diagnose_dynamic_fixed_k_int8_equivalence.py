from __future__ import annotations

import argparse
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent_mode_fixed_k_plugin_ablation import FIXED_K_BUCKETS, AgentModeFixedKRouter, _load_dataset_context
from deployment_equivalence import _record_len_value
from export_lidar_pyramid_onnx import _extract_inputs, _input_names_for_export_mode, _prepare_export_tensors, _to_device
from quant_deploy_utils import DEFAULT_CHECKPOINT, DEFAULT_HEAL_REPO, DEFAULT_HYPES_YAML, ensure_quant_deploy_run_dirs, save_json


def _precision_tag(calibration_frames: int) -> str:
    return f"int8_calib{int(calibration_frames)}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose FP32/FP16/INT8 output equivalence for dynamic fixed-K plugin engines.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--calibration_frames", type=int, required=True)
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--plugin_path", required=True)
    parser.add_argument("--hypes_yaml", default=str(DEFAULT_HYPES_YAML))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--heal_repo", default=str(DEFAULT_HEAL_REPO))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_cav", type=int, default=2)
    return parser.parse_args(argv)


def _stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    af = a.detach().float().reshape(-1).cpu().numpy()
    bf = b.detach().float().reshape(-1).cpu().numpy()
    if af.shape != bf.shape:
        return {"shape_match": False, "a_shape": list(a.shape), "b_shape": list(b.shape)}
    diff = np.abs(af - bf)
    denom = max(float(np.linalg.norm(af) * np.linalg.norm(bf)), 1.0e-12)
    return {
        "shape_match": True,
        "max_abs": float(diff.max()) if diff.size else 0.0,
        "mean_abs": float(diff.mean()) if diff.size else 0.0,
        "cosine": float(np.dot(af, bf) / denom) if diff.size else 1.0,
    }


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    dirs = ensure_quant_deploy_run_dirs(args.output_root)
    try:
        hypes, device, model, modality, dataset, loader = _load_dataset_context(args)
        if device.type != "cuda":
            raise RuntimeError("TensorRT equivalence diagnosis requires CUDA.")
        torch.cuda.set_device(device)
        import ctypes

        ctypes.CDLL(str(args.plugin_path), mode=ctypes.RTLD_GLOBAL)
        int8_precision = _precision_tag(int(args.calibration_frames))
        routers = {
            "fp32": AgentModeFixedKRouter(agent_export_mode="dynamic_agent_dim_fixed_k_plugin", precision="fp32", buckets=FIXED_K_BUCKETS, device=device, dirs=dirs),
            "fp16": AgentModeFixedKRouter(agent_export_mode="dynamic_agent_dim_fixed_k_plugin", precision="fp16", buckets=FIXED_K_BUCKETS, device=device, dirs=dirs),
            "int8": AgentModeFixedKRouter(agent_export_mode="dynamic_agent_dim_fixed_k_plugin", precision=int8_precision, buckets=FIXED_K_BUCKETS, device=device, dirs=dirs),
        }
        input_names = _input_names_for_export_mode("padded_agent_static")
        output_names = None
        frames = []
        for frame_idx, batch in enumerate(loader):
            if len(frames) >= int(args.num_frames):
                break
            if batch is None:
                continue
            ego = batch["ego"] if isinstance(batch, dict) and "ego" in batch else batch
            ego = _to_device(ego, device)
            if output_names is None:
                with torch.no_grad():
                    raw = model(ego)
                output_names = [name for name in ("cls_preds", "reg_preds", "dir_preds") if name in raw and torch.is_tensor(raw[name])]
            original_tensors, _ = _extract_inputs(ego, modality)
            tensors = _prepare_export_tensors(original_tensors, export_mode="padded_agent_static", max_cav=int(args.max_cav))
            tensors_by_name = {name: tensor for name, tensor in zip(input_names, tensors)}
            record_len = int(_record_len_value(ego))
            outputs = {precision: routers[precision].run_profiled(tensors_by_name, record_len=record_len)[0] for precision in routers}
            comparisons = {}
            for name in output_names or []:
                comparisons[name] = {
                    "FP32 vs FP16": _stats(outputs["fp32"][name], outputs["fp16"][name]),
                    "FP32 vs INT8": _stats(outputs["fp32"][name], outputs["int8"][name]),
                    "FP16 vs INT8": _stats(outputs["fp16"][name], outputs["int8"][name]),
                }
            cls = outputs["int8"].get("cls_preds")
            score_stats = {}
            if cls is not None:
                values = torch.sigmoid(cls.float()).detach().cpu().numpy().reshape(-1)
                score_stats = {
                    "int8_topk_score_mean": float(np.sort(values)[-min(100, values.size) :].mean()),
                    "int8_max_score": float(values.max()),
                    "int8_scores_over_0_1": int((values > 0.1).sum()),
                }
            frames.append(
                {
                    "frame_id": frame_idx,
                    "record_len": record_len,
                    "engine_precision_tag": int8_precision,
                    "outputs": comparisons,
                    "topk_score_distribution": score_stats,
                }
            )
        report = {
            "success": True,
            "calibration_frames": int(args.calibration_frames),
            "engine_precision_tag": int8_precision,
            "num_frames": len(frames),
            "frames": frames,
            "diagnosis": "Inspect per-head FP32/FP16/INT8 drift; large cls drift suggests activation calibration or head precision protection.",
        }
    except Exception as exc:
        report = {"success": False, "calibration_frames": int(args.calibration_frames), "error": str(exc), "traceback": traceback.format_exc()}
    save_json(report, dirs["debug"] / f"int8_output_equivalence_diagnosis_calib{int(args.calibration_frames)}.json")
    lines = ["# INT8 Output Equivalence Diagnosis", "", f"- success: {report.get('success')}", f"- calibration_frames: {args.calibration_frames}", f"- error: {report.get('error')}"]
    (dirs["summary"] / "int8_output_equivalence_diagnosis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    report = diagnose(parse_args(argv))
    print(report)
    return 0 if report.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
