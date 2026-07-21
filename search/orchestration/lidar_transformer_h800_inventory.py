"""Fresh dual-model H800 Transformer export and canonical-role inventory.

This command is intentionally independent from the Pyramid search runner.  It
loads the strict checkpoint, applies only an exact Q/K/V decomposition needed
for role-addressable precision, proves real-batch parity, and emits a fresh
fixed-K ONNX plus its origin/role inventory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

from search.integration.runtime_environment import runtime_cuda_index_for_physical


MODEL_SPECS = {
    "lidar_cobevt": {
        "checkpoint": "/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_cobevt/net_epoch_bestval_at19.pth",
        "config": "/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_cobevt/config.yaml",
        # Frozen by ``lidar_transformer_h800_manifests`` over train200 plus
        # every one of the 1789 validation samples (max=29164, align=256).
        "fixed_k": 29184,
    },
    "lidar_v2xvit": {
        "checkpoint": "/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_v2xvit/net_epoch_bestval_at27.pth",
        "config": "/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_v2xvit/config.yaml",
        "fixed_k": 27904,
    },
}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(model_name: str, device: torch.device) -> tuple[Any, Any]:
    spec = MODEL_SPECS[model_name]
    if model_name == "lidar_cobevt":
        from search.model_families.lidar_cobevt.model_capability import CobevtModelCapability

        bundle = CobevtModelCapability(
            spec["checkpoint"], spec["config"], "/home/lixingfeng/UniAD_examine/HEAL"
        ).load(device=device)
        if not bundle.load_report.full_weight_coverage:
            raise RuntimeError(f"cobevt_checkpoint_not_strict:{bundle.load_report}")
        return bundle, bundle.hypes
    from search.model_family.model_provider import load_heal_model_family

    bundle = load_heal_model_family(
        config_path=spec["config"],
        checkpoint_path=spec["checkpoint"],
        heal_root="/home/lixingfeng/UniAD_examine/HEAL",
        device=str(device),
        family_id="heal_lidar_v2xvit",
    )
    return bundle, bundle.config


def _real_batch(bundle: Any, config_path: str, device: torch.device) -> dict[str, Any]:
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml import yaml_utils
    from search.integration.data_provider import move_batch_to_device

    hypes = yaml_utils.load_yaml(config_path)
    hypes = bundle.adapter._absolutize_dataset_paths(hypes)
    dataset = build_dataset(hypes, visualize=True, train=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_batch_test,
    )
    batch = next(iter(loader))
    if batch is None:
        raise RuntimeError("transformer_inventory_first_validation_batch_empty")
    return move_batch_to_device(batch, device)


def _parity(reference: Mapping[str, torch.Tensor], actual: Mapping[str, torch.Tensor] | tuple[torch.Tensor, ...]) -> dict[str, Any]:
    names = ("cls_preds", "reg_preds", "dir_preds")
    observed = dict(zip(names, actual)) if isinstance(actual, tuple) else actual
    rows: dict[str, Any] = {}
    for name in names:
        delta = (reference[name].float() - observed[name].float()).abs()
        rows[name] = {
            "shape_reference": list(reference[name].shape),
            "shape_actual": list(observed[name].shape),
            "max_abs": float(delta.max().item()),
            "mean_abs": float(delta.mean().item()),
            "allclose": bool(
                torch.allclose(reference[name].float(), observed[name].float(), atol=5e-3, rtol=1e-4)
                and float(delta.max().item()) <= 5e-3
                and float(delta.mean().item()) <= 5e-5
            ),
        }
    return {"passed": all(row["allclose"] for row in rows.values()), "outputs": rows}


def _export_cobevt(bundle: Any, batch: Mapping[str, Any], output: Path, fixed_k: int) -> tuple[dict[str, Any], dict[str, Any]]:
    from quantization.export.heal_lidar_cobevt import prepare_cobevt_maxk_inputs
    from search.model_families.lidar_cobevt.export_recipe import CobevtExportRecipe

    inputs = prepare_cobevt_maxk_inputs(batch["ego"], fixed_k=fixed_k, max_cav=2)
    wrapper = CobevtExportRecipe(fixed_k=fixed_k, max_cav=2).build_module(bundle.model).eval()
    with torch.inference_mode():
        wrapper_output = wrapper(*(inputs[name] for name in (
            "voxel_features", "voxel_coords", "voxel_num_points", "pairwise_t_matrix", "valid_voxel_mask", "record_len"
        )))
    report = CobevtExportRecipe(fixed_k=fixed_k, max_cav=2).export(
        bundle.model, inputs, output / "base_fp32_canonical.onnx"
    )
    return report.origin_map, {"report": report.__dict__, "wrapper_output": wrapper_output}


def _export_v2xvit(bundle: Any, batch: Mapping[str, Any], output: Path, fixed_k: int) -> tuple[Any, dict[str, Any]]:
    import onnx
    from quantization.export.signal_maxk import capture_weighted_module_calls
    from search.model_family.deployment import canonicalize_v2xvit_onnx
    from search.model_family.export.heal_v2xvit import (
        HealV2XViTExportPolicy,
        build_heal_v2xvit_export_module,
        prepare_v2xvit_fixed_k_inputs,
    )

    policy = HealV2XViTExportPolicy(fixed_k=fixed_k, max_agents=2)
    inputs = prepare_v2xvit_fixed_k_inputs(batch["ego"], policy=policy)
    wrapper = build_heal_v2xvit_export_module(bundle.model, policy=policy).eval()
    input_names = tuple(inputs)
    tensors = tuple(inputs[name] for name in input_names)
    with torch.inference_mode():
        wrapper_output = wrapper(*tensors)
    onnx_path = output / "base_fp32_canonical.onnx"
    with capture_weighted_module_calls(wrapper) as calls:
        torch.onnx.export(
            wrapper,
            tensors,
            str(onnx_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=list(input_names),
            output_names=["cls_preds", "reg_preds", "dir_preds"],
            custom_opsets={"trt": 1},
        )
    origin = canonicalize_v2xvit_onnx(onnx_path, calls, output_path=onnx_path)
    graph = onnx.load(str(onnx_path), load_external_data=False)
    onnx.checker.check_model(graph)
    report = {
        "onnx_path": str(onnx_path),
        "onnx_sha256": _sha256(onnx_path),
        "checker_passed": True,
        "node_count": len(graph.graph.node),
        "weighted_entry_count": len(origin.entries),
        "functional_compute_count": len(origin.functional_compute_groups),
        "origin_map_hash": origin.origin_map_hash,
    }
    return origin, {"report": report, "wrapper_output": wrapper_output}


def run_model(model_name: str, output_root: Path, physical_gpu: int) -> dict[str, Any]:
    from search.model_family.deployment import build_physical_structure_snapshot_v2
    from search.model_families.transformer.model_inventory import build_model_inventory
    from search.model_families.transformer.projection_rewrite import (
        split_cobevt_fused_qkv,
        split_v2xvit_fused_qkv,
    )

    spec = MODEL_SPECS[model_name]
    runtime_gpu = runtime_cuda_index_for_physical(physical_gpu)
    device = torch.device(f"cuda:{runtime_gpu}")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    bundle, _ = _load(model_name, device)
    batch = _real_batch(bundle, spec["config"], device)
    original_parameter_count = sum(int(value.numel()) for value in bundle.model.parameters())
    with torch.inference_mode():
        original_output = bundle.adapter.forward_for_task(bundle.model, batch)
    rewrite = (split_cobevt_fused_qkv if model_name == "lidar_cobevt" else split_v2xvit_fused_qkv)(bundle.model)
    rewritten_parameter_count = sum(int(value.numel()) for value in bundle.model.parameters())
    with torch.inference_mode():
        rewritten_output = bundle.adapter.forward_for_task(bundle.model, batch)
    rewrite_parity = _parity(original_output, rewritten_output)
    if original_parameter_count != rewritten_parameter_count or not rewrite_parity["passed"]:
        raise RuntimeError("identity_projection_rewrite_failed")
    output = output_root / "inventory" / model_name
    output.mkdir(parents=True, exist_ok=True)
    origin, export = (
        _export_cobevt(bundle, batch, output, int(spec["fixed_k"]))
        if model_name == "lidar_cobevt"
        else _export_v2xvit(bundle, batch, output, int(spec["fixed_k"]))
    )
    wrapper_parity = _parity(rewritten_output, export.pop("wrapper_output"))
    if not wrapper_parity["passed"]:
        raise RuntimeError(f"fixed_k_wrapper_parity_failed:{wrapper_parity}")
    inventory = build_model_inventory(
        model_family=model_name,
        model=bundle.model,
        onnx_path=output / "base_fp32_canonical.onnx",
        origin_map=origin,
    )
    origin_payload = origin if isinstance(origin, Mapping) else origin.to_dict()
    _write_json(output / "canonical_origin_map.json", origin_payload)
    _write_json(output / "projection_rewrite.json", rewrite.to_dict())
    _write_json(output / "rewrite_parity.json", rewrite_parity)
    _write_json(output / "fixed_k_wrapper_parity.json", wrapper_parity)
    _write_json(output / "export_report.json", export["report"])
    _write_csv(output_root / "inventory" / f"{model_name.removeprefix('lidar_')}_module_inventory.csv", inventory["rows"])
    _write_json(output_root / "inventory" / f"{model_name.removeprefix('lidar_')}_attention_role_map.json", inventory["attention_role_map"])
    _write_json(output / "inventory.json", inventory)
    snapshot = build_physical_structure_snapshot_v2(
        bundle.model, model_family=model_name
    )
    _write_json(output / "physical_structure_snapshot_v2.json", snapshot)
    return {
        "status": "ok",
        "model": model_name,
        "physical_gpu": int(physical_gpu),
        "original_parameter_count": original_parameter_count,
        "rewritten_parameter_count": rewritten_parameter_count,
        "structure_frozen": True,
        "fixed_k": int(spec["fixed_k"]),
        "onnx": export["report"],
        "role_counts": inventory["role_counts"],
        "unsupported_role_mapping_count": inventory["unsupported_role_mapping_count"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model", choices=tuple(MODEL_SPECS) + ("all",), default="all")
    parser.add_argument("--physical-gpu", type=int, default=0)
    args = parser.parse_args(argv)
    output = Path(args.output_root).resolve()
    models = tuple(MODEL_SPECS) if args.model == "all" else (args.model,)
    result = [run_model(name, output, args.physical_gpu) for name in models]
    _write_json(output / "inventory" / "inventory_acceptance.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
