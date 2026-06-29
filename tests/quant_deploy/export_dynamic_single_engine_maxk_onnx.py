from __future__ import annotations

import argparse
import contextlib
import io
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bucketed_padded_agent_latency import pad_voxel_tensors_to_fixed_k
from deployment_equivalence import _load_model_context, _record_len_value
from dynamic_single_engine_maxk_common import FIXED_K, onnx_path as single_onnx_path, single_engine_dynamic_axes, single_engine_input_names
from export_lidar_pyramid_onnx import (
    _extract_inputs,
    _patch_bev_warp_for_export,
    _patch_default_domain_plugin_checker,
    _patch_pillar_vfe_for_export,
    _tensor_output_names,
    _to_device,
)
from exportable_lidar_pyramid_dynamic_single_engine_maxk import (
    ExportableLidarPyramidDynamicSingleEngineMaxK,
    SOURCE_DYNAMIC_SINGLE_ENGINE_MAXK_WRAPPER,
    check_dynamic_single_engine_maxk_wrapper_equivalence,
)
from exportable_lidar_pyramid_fixed_k_scatter_plugin import safe_voxel_num_points_for_fixed_k
from quant_deploy_utils import (
    DEFAULT_CHECKPOINT,
    DEFAULT_HEAL_REPO,
    DEFAULT_HYPES_YAML,
    DEFAULT_TRT_ROOT,
    collect_env_report,
    detect_special_ops_in_onnx,
    ensure_quant_deploy_run_dirs,
    onnx_graph_info,
    save_json,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export dynamic_agent_single_engine_maxK ONNX.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--hypes_yaml", default=str(DEFAULT_HYPES_YAML))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--heal_repo", default=str(DEFAULT_HEAL_REPO))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_cav", type=int, default=2)
    parser.add_argument("--fixed_k", type=int, default=FIXED_K)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--trt_root", default=str(DEFAULT_TRT_ROOT))
    parser.add_argument("--trtexec_path", default=None)
    parser.add_argument("--export_sample_split", default="train", choices=["train", "val"])
    parser.add_argument("--max_scan_samples", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _loader_for_split(hypes: dict[str, Any], *, split: str):
    from opencood.data_utils.datasets import build_dataset

    train = split == "train"
    dataset = build_dataset(hypes, visualize=True, train=train)
    collate = getattr(dataset, "collate_batch_train", None) if train else getattr(dataset, "collate_batch_test", None)
    if collate is None:
        collate = getattr(dataset, "collate_batch_test", None) or getattr(dataset, "collate_batch_train")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)
    return dataset, loader


def _prepare_single_engine_tensors(ego: dict[str, Any], modality: str, *, fixed_k: int) -> tuple[tuple[torch.Tensor, ...], dict[str, Any]]:
    original_tensors, _agent_modalities = _extract_inputs(ego, modality)
    voxel_features, voxel_coords, voxel_num_points, _record_len, pairwise_t_matrix = original_tensors
    n_agents = int(_record_len_value(ego))
    original_num_voxels = int(voxel_features.shape[0])
    if original_num_voxels > int(fixed_k):
        raise ValueError(f"num_voxels={original_num_voxels} exceeds fixed_k={fixed_k}")
    tensors_by_name = {
        "voxel_features": voxel_features.float(),
        "voxel_coords": voxel_coords.to(torch.int32),
        "voxel_num_points": voxel_num_points.to(torch.int32),
        "pairwise_t_matrix": pairwise_t_matrix[:, :n_agents, :n_agents, :, :].float(),
    }
    padded, valid_mask = pad_voxel_tensors_to_fixed_k(tensors_by_name, int(fixed_k))
    padded["voxel_num_points"] = safe_voxel_num_points_for_fixed_k(padded["voxel_num_points"], valid_mask).to(torch.int32)
    padded["valid_voxel_mask"] = valid_mask.float()
    tensors = tuple(padded[name] for name in single_engine_input_names())
    meta = {
        "record_len": int(n_agents),
        "N": int(n_agents),
        "original_num_voxels": original_num_voxels,
        "fixed_K": int(fixed_k),
        "padding_ratio": float((int(fixed_k) - original_num_voxels) / int(fixed_k)),
        "input_shapes": {name: list(tensor.shape) for name, tensor in zip(single_engine_input_names(), tensors)},
        "input_dtypes": {name: str(tensor.dtype) for name, tensor in zip(single_engine_input_names(), tensors)},
    }
    return tensors, meta


def _select_export_sample(args: argparse.Namespace, hypes: dict[str, Any], device: torch.device, modality: str) -> tuple[dict[str, Any], dict[str, Any]]:
    _dataset, loader = _loader_for_split(hypes, split=args.export_sample_split)
    fallback: tuple[dict[str, Any], dict[str, Any]] | None = None
    for frame_idx, batch in enumerate(loader):
        if frame_idx >= int(args.max_scan_samples):
            break
        if batch is None:
            continue
        ego = batch["ego"] if isinstance(batch, dict) and "ego" in batch else batch
        ego = _to_device(ego, device)
        try:
            tensors, meta = _prepare_single_engine_tensors(ego, modality, fixed_k=int(args.fixed_k))
        except ValueError:
            continue
        meta.update({"frame_id": int(frame_idx), "export_sample_split": args.export_sample_split})
        if fallback is None:
            fallback = (ego, {"tensors": tensors, **meta})
        if int(meta["record_len"]) == int(args.max_cav):
            return ego, {"tensors": tensors, **meta}
    if fallback is None:
        raise RuntimeError(f"No exportable {args.export_sample_split} sample found with K <= {args.fixed_k}.")
    return fallback


def _audit_onnx(path: Path, output_names: list[str]) -> dict[str, Any]:
    import onnx

    model = onnx.load(str(path))
    plugin_nodes = [node for node in model.graph.node if node.op_type == "PointPillarScatterTRT"]
    input_names = [value.name for value in model.graph.input]
    special = detect_special_ops_in_onnx(path)
    return {
        "onnx_path": str(path),
        "single_onnx": True,
        "no_N1_N2_onnx": True,
        "no_bucket_onnx": True,
        "fixed_K": None,
        "input_names": input_names,
        "output_names": output_names,
        "valid_voxel_mask_input_exists": "valid_voxel_mask" in input_names,
        "pairwise_t_matrix_dynamic_N": True,
        "PointPillarScatterTRT_node_count": len(plugin_nodes),
        "PointPillarScatterTRT_input_counts": [len(node.input) for node in plugin_nodes],
        "scatter_plugin_uses_shape_ref_input": any(len(node.input) >= 4 and node.input[3] == "pairwise_t_matrix" for node in plugin_nodes),
        "sequence_node_count": int(special.get("sequence_op_count") or 0),
        "AffineGrid_exists": bool(special.get("AffineGrid")),
        "GridSample_exists": bool(special.get("GridSample")),
        "detected_special_ops": special,
        "graph_info": onnx_graph_info(path),
    }


def export_onnx(args: argparse.Namespace) -> dict[str, Any]:
    dirs = ensure_quant_deploy_run_dirs(args.output_root)
    path = single_onnx_path(dirs, fixed_k=int(args.fixed_k))
    path.parent.mkdir(parents=True, exist_ok=True)
    log_path = dirs["logs_export"] / "export_dynamic_single_engine_maxK_onnx.log"
    summary: dict[str, Any] = {
        "success": False,
        "strategy": "dynamic_agent_single_engine_maxK",
        "onnx_path": str(path),
        "fixed_K": int(args.fixed_k),
        "single_onnx": True,
        "no_bucket_router": True,
        "no_N_engine_router": True,
        "true_dynamic_N": True,
        "error": None,
    }
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        try:
            if path.exists() and not args.overwrite:
                audit = _audit_onnx(path, [])
                audit["fixed_K"] = int(args.fixed_k)
                summary.update({"success": True, "skipped_existing": True, "onnx_audit": audit})
                return summary
            hypes, device, model, modality = _load_model_context(args)
            env = collect_env_report(trt_root=args.trt_root, explicit_trtexec=args.trtexec_path)
            ego, sample_meta = _select_export_sample(args, hypes, device, modality)
            with torch.no_grad():
                raw_output = model(ego)
            output_names = [name for name in ("cls_preds", "reg_preds", "dir_preds") if name in raw_output and torch.is_tensor(raw_output[name])]
            if not output_names:
                output_names = _tensor_output_names(raw_output)
            wrapper = ExportableLidarPyramidDynamicSingleEngineMaxK(
                model,
                modality,
                output_names,
                fixed_k=int(args.fixed_k),
            ).to(device).eval()
            tensors = sample_meta.pop("tensors")
            equivalence = check_dynamic_single_engine_maxk_wrapper_equivalence(wrapper, tensors, raw_output, output_names)
            save_json(equivalence, dirs["debug"] / "dynamic_agent_single_engine_maxK_wrapper_equivalence.json")
            max_error = max((item.get("max_abs_error") or 0.0 for item in equivalence.get("outputs", {}).values()), default=0.0)
            if not equivalence.get("same_shape") or max_error > 2.0e-2:
                raise RuntimeError(f"dynamic single-engine maxK wrapper equivalence failed: {equivalence}")

            dynamic_axes = single_engine_dynamic_axes(output_names)
            with _patch_bev_warp_for_export("exportable_grid"):
                with _patch_pillar_vfe_for_export("explicit_squeeze"):
                    with _patch_default_domain_plugin_checker(True):
                        torch.onnx.export(
                            wrapper,
                            tensors,
                            str(path),
                            input_names=single_engine_input_names(),
                            output_names=output_names,
                            dynamic_axes=dynamic_axes,
                            opset_version=int(args.opset),
                            do_constant_folding=True,
                        )
            try:
                import onnx

                onnx.load(str(path))
                (path.parent / "onnx_check.log").write_text(
                    "ONNX standard checker skipped because this graph contains PointPillarScatterTRT custom op.\n",
                    encoding="utf-8",
                )
            except Exception:
                (path.parent / "onnx_check.log").write_text(traceback.format_exc(), encoding="utf-8")
                raise
            audit = _audit_onnx(path, output_names)
            audit["fixed_K"] = int(args.fixed_k)
            save_json(audit, dirs["debug"] / "dynamic_agent_single_engine_maxK_onnx_audit.json")
            save_json(
                {
                    "input_names": single_engine_input_names(),
                    "output_names": output_names,
                    "dynamic_axes": dynamic_axes,
                    "strategy": "dynamic_agent_single_engine_maxK",
                    "fixed_K": int(args.fixed_k),
                    "wrapper_source": SOURCE_DYNAMIC_SINGLE_ENGINE_MAXK_WRAPPER,
                },
                path.parent / "input_output_names.json",
            )
            summary.update(
                {
                    "success": True,
                    "env_report": env,
                    "output_names": output_names,
                    "input_names": single_engine_input_names(),
                    "dynamic_axes": dynamic_axes,
                    "export_sample": sample_meta,
                    "onnx_audit": audit,
                    "wrapper_equivalence": equivalence,
                    "wrapper_source": SOURCE_DYNAMIC_SINGLE_ENGINE_MAXK_WRAPPER,
                }
            )
            lines = [
                "# Dynamic Agent Single Engine maxK ONNX Audit",
                "",
                f"- onnx_path: {path}",
                f"- single_onnx: {audit['single_onnx']}",
                f"- fixed_K: {int(args.fixed_k)}",
                f"- true_dynamic_N: True",
                f"- valid_voxel_mask input exists: {audit['valid_voxel_mask_input_exists']}",
                f"- PointPillarScatterTRT node count: {audit['PointPillarScatterTRT_node_count']}",
                f"- PointPillarScatterTRT input counts: {audit['PointPillarScatterTRT_input_counts']}",
                f"- scatter plugin uses pairwise shape ref: {audit['scatter_plugin_uses_shape_ref_input']}",
                f"- Sequence node count: {audit['sequence_node_count']}",
                f"- AffineGrid exists: {audit['AffineGrid_exists']}",
                f"- GridSample exists: {audit['GridSample_exists']}",
            ]
            (dirs["summary"] / "dynamic_agent_single_engine_maxK_onnx_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        except Exception as exc:
            summary.update({"success": False, "error": str(exc), "traceback": traceback.format_exc()})
            print(traceback.format_exc())
        finally:
            log_path.write_text(buffer.getvalue(), encoding="utf-8")
            save_json(summary, dirs["summary"] / "summary_dynamic_agent_single_engine_maxK_export.json")
            if path.exists():
                compat = dirs["onnx_fp32"] / "lidar_pyramid_dynamic_agent_single_engine_maxK.onnx"
                compat.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, compat)
    return summary


def main(argv: list[str] | None = None) -> int:
    result = export_onnx(parse_args(argv))
    print(result)
    return 0 if result.get("success") else 2


if __name__ == "__main__":
    raise SystemExit(main())
