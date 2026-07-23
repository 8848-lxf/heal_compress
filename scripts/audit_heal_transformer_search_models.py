#!/usr/bin/env python3
"""Audit real HEAL model structures for unified Transformer/CNN search.

This command loads one strict checkpoint, executes the first real validation
sample, records module and functional tensor shapes, and identifies only
attention/FFN patterns that exist in the live model.  It does not prune,
quantize, export, or mutate the checkpoint directory.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import torch
import torch.nn as nn


HEAL_ROOT = Path("/home/lixingfeng/UniAD_examine/HEAL")
MODEL_ROOT = Path(
    "/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly"
)
MODEL_SPECS = {
    "v2xvit": {
        "canonical_name": "HeterBaseline_DAIR_lidar_v2xvit",
        "config": MODEL_ROOT / "lidar_v2xvit/config.yaml",
        "checkpoint": MODEL_ROOT / "lidar_v2xvit/net_epoch_bestval_at27.pth",
        "fusion_method": "v2xvit",
    },
    "cobevt": {
        "canonical_name": "HeterBaseline_DAIR_lidar_cobevt",
        "config": MODEL_ROOT / "lidar_cobevt/config.yaml",
        "checkpoint": MODEL_ROOT / "lidar_cobevt/net_epoch_bestval_at19.pth",
        "fusion_method": "cobevt",
    },
    "attfusion": {
        "canonical_name": "HeterBaseline_DAIR_lidar_attfuse",
        "config": MODEL_ROOT / "lidar_attfuse/config.yaml",
        "checkpoint": MODEL_ROOT / "lidar_attfuse/net_epoch_bestval_at33.pth",
        "fusion_method": "att",
    },
    "coalign": {
        "canonical_name": "HeterBaseline_DAIR_lidar_coalign",
        "config": MODEL_ROOT / "lidar_coalign/config.yaml",
        "checkpoint": MODEL_ROOT / "lidar_coalign/net_epoch_bestval_at25.pth",
        "fusion_method": "att",
    },
}

WEIGHTED_TYPES = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.Linear,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _iter_tensors(value: Any) -> Iterable[torch.Tensor]:
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)


def _tensor_descriptors(value: Any) -> list[dict[str, Any]]:
    return [
        {
            "shape": [int(item) for item in tensor.shape],
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "finite": bool(torch.isfinite(tensor).all().item())
            if tensor.is_floating_point()
            else True,
        }
        for tensor in _iter_tensors(value)
    ]


def _module_rows(model: nn.Module) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path, module in model.named_modules():
        direct_parameters = sum(
            int(parameter.numel()) for parameter in module.parameters(recurse=False)
        )
        row: dict[str, Any] = {
            "module_path": path,
            "module_type": type(module).__name__,
            "direct_parameter_count": direct_parameters,
            "recursive_parameter_count": sum(
                int(parameter.numel()) for parameter in module.parameters()
            ),
            "is_weighted": isinstance(module, WEIGHTED_TYPES),
            "training": bool(module.training),
        }
        for name in (
            "in_features",
            "out_features",
            "in_channels",
            "out_channels",
            "groups",
            "heads",
            "num_heads",
            "head_dim",
            "d_qk",
            "d_v",
            "window_size",
            "scale",
        ):
            if hasattr(module, name):
                value = getattr(module, name)
                if isinstance(value, torch.Tensor):
                    continue
                row[name] = value
        weight = getattr(module, "weight", None)
        bias = getattr(module, "bias", None)
        if torch.is_tensor(weight):
            row["weight_shape"] = [int(value) for value in weight.shape]
        if torch.is_tensor(bias):
            row["bias_shape"] = [int(value) for value in bias.shape]
        if isinstance(module, nn.LayerNorm):
            row["normalized_shape"] = list(module.normalized_shape)
        rows.append(row)
    return rows


def _nearest_block(path: str, modules: Mapping[str, nn.Module]) -> str:
    parts = path.split(".")
    for stop in range(len(parts) - 1, 0, -1):
        candidate = ".".join(parts[:stop])
        module = modules.get(candidate)
        if module is not None and (
            "block" in type(module).__name__.lower()
            or type(module).__name__ in {"V2XTEncoder", "SwapFusionEncoder"}
        ):
            return candidate
    return path.rsplit(".", 1)[0] if "." in path else path


def _family(model_name: str, path: str, module: nn.Module) -> str:
    lower = path.lower()
    if model_name == "cobevt":
        if "window_attention" in lower:
            return "cobevt_window"
        if "grid_attention" in lower:
            return "cobevt_grid"
    if model_name == "v2xvit":
        if type(module).__name__ in {"HGTCavAttention", "CavAttention"}:
            return "v2xvit_agent_relation"
        if type(module).__name__ == "BaseWindowAttention":
            return f"v2xvit_spatial_window_w{int(module.window_size)}"
    if model_name == "attfusion":
        return "attfusion_functional_agent_attention"
    if model_name == "coalign":
        return "coalign_multiscale_functional_agent_attention"
    return f"{model_name}_unclassified_attention"


def _attention_instances(model_name: str, model: nn.Module) -> list[dict[str, Any]]:
    modules = dict(model.named_modules())
    rows: list[dict[str, Any]] = []
    for path, module in modules.items():
        module_type = type(module).__name__
        if hasattr(module, "to_qkv") and isinstance(getattr(module, "to_qkv"), nn.Linear):
            qkv = module.to_qkv
            out = getattr(module, "to_out", None)
            out_linear = out[0] if isinstance(out, nn.Sequential) and out and isinstance(out[0], nn.Linear) else None
            heads = int(getattr(module, "heads", 0))
            if qkv.out_features % 3 or heads <= 0 or (qkv.out_features // 3) % heads:
                supported = False
                reason = "fused_qkv_dimensions_not_head_factorable"
                d_h = 0
            elif out_linear is None:
                supported = False
                reason = "fused_qkv_output_projection_missing"
                d_h = qkv.out_features // 3 // heads
            else:
                supported = True
                reason = ""
                d_h = qkv.out_features // 3 // heads
            rows.append(
                {
                    "instance_id": f"attention::{path}",
                    "module_path": path,
                    "module_type": module_type,
                    "block_path": _nearest_block(path, modules),
                    "family": _family(model_name, path, module),
                    "qkv_layout": "fused_qkv",
                    "q_projection_paths": [f"{path}.to_qkv"],
                    "k_projection_paths": [f"{path}.to_qkv"],
                    "v_projection_paths": [f"{path}.to_qkv"],
                    "output_projection_paths": [f"{path}.to_out.0"] if out_linear is not None else [],
                    "softmax_paths": [f"{path}.attend"] if hasattr(module, "attend") else [],
                    "heads": heads,
                    "original_d_h": int(d_h),
                    "inner_dim": heads * int(d_h),
                    "d_model": int(qkv.in_features),
                    "scale": float(getattr(module, "scale", float("nan"))),
                    "expected_scale": float(d_h**-0.5) if d_h else None,
                    "attention_dh_domain": supported,
                    "unsupported_reason": reason,
                    "shared_qkvo_index_initial_mode": False,
                    "independent_domain_default": True,
                }
            )
            continue
        if module_type == "HGTCavAttention" and hasattr(module, "q_linears"):
            heads = int(module.heads)
            q_linears = list(module.q_linears)
            k_linears = list(module.k_linears)
            v_linears = list(module.v_linears)
            a_linears = list(module.a_linears)
            d_h = int(q_linears[0].out_features) // heads if q_linears and heads else 0
            supported = bool(
                heads
                and q_linears
                and len(q_linears) == len(k_linears) == len(v_linears) == len(a_linears)
                and all(value.out_features == heads * d_h for value in [*q_linears, *k_linears, *v_linears])
                and all(value.in_features == heads * d_h for value in a_linears)
            )
            rows.append(
                {
                    "instance_id": f"attention::{path}",
                    "module_path": path,
                    "module_type": module_type,
                    "block_path": _nearest_block(path, modules),
                    "family": _family(model_name, path, module),
                    "qkv_layout": "separate_typed_qkv",
                    "q_projection_paths": [f"{path}.q_linears.{index}" for index in range(len(q_linears))],
                    "k_projection_paths": [f"{path}.k_linears.{index}" for index in range(len(k_linears))],
                    "v_projection_paths": [f"{path}.v_linears.{index}" for index in range(len(v_linears))],
                    "output_projection_paths": [f"{path}.a_linears.{index}" for index in range(len(a_linears))],
                    "relation_parameter_paths": [f"{path}.relation_att", f"{path}.relation_msg"],
                    "softmax_paths": [f"{path}.attend"],
                    "heads": heads,
                    "original_d_h": d_h,
                    "inner_dim": heads * d_h,
                    "d_model": int(q_linears[0].in_features) if q_linears else 0,
                    "scale": float(module.scale),
                    "expected_scale": float(d_h**-0.5) if d_h else None,
                    "attention_dh_domain": supported,
                    "unsupported_reason": "" if supported else "typed_qkvo_projection_contract_mismatch",
                    "shared_qkvo_index_initial_mode": False,
                    "independent_domain_default": True,
                }
            )
            continue
        if module_type == "ScaledDotProductAttention":
            parent_path = path.rsplit(".", 1)[0] if "." in path else ""
            parent = modules.get(parent_path)
            if parent is None or type(parent).__name__ != "AttFusion":
                continue
            scale = float(getattr(module, "sqrt_dim", float("nan")))
            rows.append(
                {
                    "instance_id": f"attention::{path}",
                    "module_path": path,
                    "module_type": module_type,
                    "block_path": parent_path,
                    "family": _family(model_name, path, module),
                    "qkv_layout": "projection_free_functional_qkv",
                    "q_projection_paths": [],
                    "k_projection_paths": [],
                    "v_projection_paths": [],
                    "output_projection_paths": [],
                    "softmax_paths": [],
                    "heads": 1,
                    "original_d_h": int(round(scale * scale)) if math.isfinite(scale) else 0,
                    "d_model": int(round(scale * scale)) if math.isfinite(scale) else 0,
                    "attention_dh_domain": False,
                    "unsupported_reason": "no_trainable_qkv_or_output_projection; functional full_feature_attention_only",
                    "independent_domain_default": False,
                }
            )
    rows.sort(key=lambda row: str(row["module_path"]))
    return rows


def _ffn_instances(model_name: str, model: nn.Module) -> list[dict[str, Any]]:
    del model_name
    modules = dict(model.named_modules())
    rows: list[dict[str, Any]] = []
    for path, module in modules.items():
        if type(module).__name__ != "FeedForward" or not hasattr(module, "net"):
            continue
        linears = [
            (index, child)
            for index, child in enumerate(module.net)
            if isinstance(child, nn.Linear)
        ]
        if len(linears) != 2:
            rows.append(
                {
                    "instance_id": f"ffn::{path}",
                    "module_path": path,
                    "block_path": _nearest_block(path, modules),
                    "ffn_type": "unknown",
                    "ffn_hidden_domain": False,
                    "unsupported_reason": f"expected_two_linears_observed_{len(linears)}",
                }
            )
            continue
        (first_index, first), (second_index, second) = linears
        supported = bool(first.out_features == second.in_features and first.in_features == second.out_features)
        rows.append(
            {
                "instance_id": f"ffn::{path}",
                "module_path": path,
                "module_type": type(module).__name__,
                "block_path": _nearest_block(path, modules),
                "family": "standard_transformer_ffn",
                "ffn_type": "standard",
                "first_projection_path": f"{path}.net.{first_index}",
                "second_projection_path": f"{path}.net.{second_index}",
                "activation_path": f"{path}.net.{first_index + 1}",
                "d_model": int(first.in_features),
                "original_d_ff": int(first.out_features),
                "ffn_hidden_domain": supported,
                "unsupported_reason": "" if supported else "ffn_projection_shape_contract_mismatch",
                "independent_domain_default": True,
            }
        )
    rows.sort(key=lambda row: str(row["module_path"]))
    return rows


def _shared_parameters(model: nn.Module) -> dict[str, Any]:
    try:
        module_rows = list(model.named_modules(remove_duplicate=False))
    except TypeError:
        module_rows = list(model.named_modules())
    by_module_object: dict[int, list[str]] = defaultdict(list)
    for path, module in module_rows:
        by_module_object[id(module)].append(path)

    try:
        parameter_rows = list(model.named_parameters(remove_duplicate=False))
    except TypeError:
        parameter_rows = list(model.named_parameters())
    by_parameter_object: dict[int, list[str]] = defaultdict(list)
    by_storage: dict[tuple[int, str], list[str]] = defaultdict(list)
    for path, parameter in parameter_rows:
        by_parameter_object[id(parameter)].append(path)
        storage = parameter.untyped_storage()
        by_storage[(int(storage.data_ptr()), str(parameter.device))].append(path)
    return {
        "shared_module_objects": [
            {"paths": sorted(paths), "count": len(paths)}
            for paths in by_module_object.values()
            if len(set(paths)) > 1
        ],
        "shared_parameter_objects": [
            {"paths": sorted(paths), "count": len(paths)}
            for paths in by_parameter_object.values()
            if len(set(paths)) > 1
        ],
        "shared_parameter_storage": [
            {"paths": sorted(paths), "count": len(paths)}
            for paths in by_storage.values()
            if len(set(paths)) > 1
        ],
    }


def _traceability(model: nn.Module, ego_batch: Mapping[str, Any], model_name: str) -> dict[str, Any]:
    def error_text(exc: Exception) -> str:
        text = f"{type(exc).__name__}:{exc}"
        if len(text) <= 2000:
            return text
        return text[:2000] + f"...<truncated {len(text) - 2000} characters>"

    result: dict[str, Any] = {}
    try:
        torch.fx.symbolic_trace(model)
    except Exception as exc:  # noqa: BLE001 - exact framework error is evidence
        result["fx"] = {"passed": False, "error": error_text(exc)}
    else:
        result["fx"] = {"passed": True, "error": ""}
    try:
        torch.jit.trace(model, (ego_batch,), strict=False, check_trace=False)
    except Exception as exc:  # noqa: BLE001
        result["torchscript"] = {"passed": False, "error": error_text(exc)}
    else:
        result["torchscript"] = {"passed": True, "error": ""}
    result["onnx"] = {
        "attempted_in_inventory": False,
        "project_export_adapter_available": True,
        "adapter": (
            "search.model_family.export.heal_v2xvit"
            if model_name == "v2xvit"
            else "search.model_family.export.heal_lidar_baselines"
        ),
        "status": "deferred_to_physical_candidate_stage2_export_and_checker",
    }
    return result


def _load(model_name: str, device: torch.device) -> tuple[nn.Module, Any, dict[str, Any], Any]:
    spec = MODEL_SPECS[model_name]
    for path in (str(HEAL_ROOT.parent), str(HEAL_ROOT), str(Path(__file__).resolve().parents[1])):
        if path not in sys.path:
            sys.path.insert(0, path)
    from adapters.heal_lidar_adapter import HEALLiDARAdapter
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml import yaml_utils
    from opencood.tools import train_utils
    from search.integration.data_provider import move_batch_to_device

    hypes = yaml_utils.load_yaml(str(spec["config"]))
    adapter = HEALLiDARAdapter(
        heal_repo=str(HEAL_ROOT), config={"model": {"hypes_yaml": str(spec["config"])}}
    )
    model = train_utils.create_model(hypes)
    raw = torch.load(spec["checkpoint"], map_location="cpu")
    state = raw.get("model", raw)
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint_state_dict_missing")
    incompatibility = model.load_state_dict(state, strict=True)
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise RuntimeError(f"strict_checkpoint_mismatch:{incompatibility}")
    model = model.to(device).eval()
    dataset_hypes = adapter._absolutize_dataset_paths(hypes)
    dataset = build_dataset(dataset_hypes, visualize=True, train=False)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=dataset.collate_batch_test,
        num_workers=0,
        pin_memory=False,
    )
    batch = next(iter(loader))
    if batch is None:
        raise RuntimeError("first_validation_batch_empty")
    batch = move_batch_to_device(batch, device)
    return model, adapter, hypes, batch


def run_model(model_name: str, output_root: Path, device: torch.device) -> dict[str, Any]:
    spec = MODEL_SPECS[model_name]
    model, adapter, hypes, batch = _load(model_name, device)
    module_inventory = _module_rows(model)
    attention = _attention_instances(model_name, model)
    ffn = _ffn_instances(model_name, model)
    shared = _shared_parameters(model)

    calls: list[dict[str, Any]] = []
    handles = []
    interesting = (
        nn.Linear,
        nn.Conv2d,
        nn.ConvTranspose2d,
        nn.LayerNorm,
        nn.Softmax,
    )
    attention_paths = {str(row["module_path"]) for row in attention}
    ffn_paths = {str(row["module_path"]) for row in ffn}
    call_counts: dict[str, int] = defaultdict(int)

    def make_hook(path: str):
        def hook(_module: nn.Module, inputs: Any, outputs: Any) -> None:
            call_index = call_counts[path]
            call_counts[path] += 1
            calls.append(
                {
                    "module_path": path,
                    "module_type": type(_module).__name__,
                    "call_index": call_index,
                    "inputs": _tensor_descriptors(inputs),
                    "outputs": _tensor_descriptors(outputs),
                }
            )

        return hook

    for path, module in model.named_modules():
        if isinstance(module, interesting) or path in attention_paths or path in ffn_paths:
            handles.append(module.register_forward_hook(make_hook(path)))
    try:
        with torch.inference_mode():
            outputs = adapter.forward_for_task(model, batch)
    finally:
        for handle in handles:
            handle.remove()
    output_tensors = _tensor_descriptors(outputs)
    if not output_tensors or not all(row["finite"] for row in output_tensors):
        raise RuntimeError("real_validation_forward_nonfinite")

    from tracer.generic_tracer import GenericTracer

    functional_trace = GenericTracer(model, forward_fn=adapter.forward_for_task).trace(batch)
    relevant_ops = []
    for path, row in functional_trace["nodes"].items():
        row_type = str(row.get("type", ""))
        operation = str(row.get("op", row_type))
        if path.startswith("op::") and any(
            token in operation.lower()
            for token in (
                "einsum",
                "bmm",
                "matmul",
                "softmax",
                "reshape",
                "view",
                "permute",
                "transpose",
                "add",
                "cat",
                "chunk",
            )
        ):
            relevant_ops.append({"node": path, **row})

    traceability = _traceability(model, batch["ego"], model_name)
    unsupported = [
        {
            "module_path": row["module_path"],
            "graph_pattern": row["qkv_layout"],
            "unsupported_reason": row["unsupported_reason"],
            "required_adapter": (
                "none; retain functional attention and expose CNN domains only"
                if row["qkv_layout"] == "projection_free_functional_qkv"
                else "transformer_attention_instance_adapter"
            ),
        }
        for row in attention
        if not row["attention_dh_domain"]
    ]
    if not ffn:
        unsupported.append(
            {
                "module_path": "fusion_net",
                "graph_pattern": "no_transformer_ffn",
                "unsupported_reason": "no standard or gated FFN exists in the real model",
                "required_adapter": "none; expose only real CNN/fusion domains",
            }
        )

    config_name = str(hypes.get("name", ""))
    if config_name != str(spec["canonical_name"]):
        raise RuntimeError(f"canonical_model_name_mismatch:{config_name}:{spec['canonical_name']}")
    common = {
        "schema_version": "heal-transformer-model-inventory-v1",
        "model": model_name,
        "canonical_name": config_name,
        "model_type": type(model).__name__,
        "fusion_method": str(hypes["model"]["args"]["fusion_method"]),
        "config_path": str(spec["config"]),
        "checkpoint_path": str(spec["checkpoint"]),
        "config_sha256": _sha256(spec["config"]),
        "checkpoint_sha256": _sha256(spec["checkpoint"]),
        "strict_checkpoint_load": True,
        "parameter_count": sum(int(parameter.numel()) for parameter in model.parameters()),
        "real_validation_dataset_index": 0,
        "real_forward_finite": True,
        "real_forward_outputs": output_tensors,
        "traceability": traceability,
    }
    _write_json(output_root / f"{model_name}_module_inventory.json", {**common, "modules": module_inventory})
    _write_json(
        output_root / f"{model_name}_shape_trace.json",
        {
            **common,
            "module_calls": calls,
            "functional_nodes": relevant_ops,
            "functional_edge_count": len(functional_trace["edges"]),
        },
    )
    _write_json(output_root / f"{model_name}_attention_instances.json", {**common, "instances": attention})
    _write_json(output_root / f"{model_name}_ffn_instances.json", {**common, "instances": ffn})
    _write_json(output_root / f"{model_name}_shared_parameters.json", {**common, **shared})
    _write_json(output_root / f"{model_name}_unsupported_patterns.json", {**common, "patterns": unsupported})
    return {
        "model": model_name,
        "canonical_name": config_name,
        "model_type": type(model).__name__,
        "parameter_count": common["parameter_count"],
        "attention_instance_count": len(attention),
        "attention_dh_domain_count": sum(bool(row["attention_dh_domain"]) for row in attention),
        "ffn_instance_count": len(ffn),
        "ffn_hidden_domain_count": sum(bool(row["ffn_hidden_domain"]) for row in ffn),
        "unsupported_pattern_count": len(unsupported),
        "traceability": traceability,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    result = run_model(args.model, args.output_root.resolve(), device)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
