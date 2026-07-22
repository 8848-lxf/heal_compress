"""Inventory, Taylor ranking and deterministic structure materialization for d_h sweeps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from search.integration.runtime_environment import runtime_cuda_index_for_physical
from search.model_families.transformer.dh_candidate_grid import dense_head_dimension_grid
from search.model_families.transformer.dh_physical_rewrite import discover_attention_families
from search.model_families.transformer.dh_pruning_contract import (
    audit_nested_masks,
    masks_from_rankings,
    stable_hash,
)
from search.orchestration.lidar_transformer_h800_inventory import MODEL_SPECS, _load


TRT_ROOT = Path("/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118")
MODEL_OPT = Path("/home/lixingfeng/miniconda3/envs/modelopt")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
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


def _git(workdir: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=workdir, text=True).strip()


def write_run_manifest(output_root: Path, *, physical_gpu: int) -> dict[str, Any]:
    workdir = Path(__file__).resolve().parents[2]
    manifests = output_root / "evaluation" / "manifests"
    payload = {
        "schema_version": "h800-transformer-dh-alignment-run-v1",
        "branch": _git(workdir, "branch", "--show-current"),
        "base_head": _git(workdir, "rev-parse", "HEAD"),
        "physical_gpu": int(physical_gpu),
        "modelopt_prefix": str(MODEL_OPT),
        "python": str(MODEL_OPT / "bin/python"),
        "nvcc": str(MODEL_OPT / "bin/nvcc"),
        "gxx": str(MODEL_OPT / "bin/g++"),
        "tensorrt_root": str(TRT_ROOT),
        "cuda_home": str(MODEL_OPT),
        "torch_cuda_arch_list": "9.0",
        "models": MODEL_SPECS,
        "manifest_hashes": {
            str(path.relative_to(manifests)): _sha256(path)
            for path in sorted(manifests.rglob("*.json"))
        },
        "constraints": {
            "ga": False,
            "greedy": False,
            "full1789": False,
            "head_count_frozen": True,
            "qkv_unified_head_dimension": True,
            "qk_contract": "F32A32O32",
            "cross_width_cache_reuse": False,
            "rtx4090_engine_reuse": False,
        },
    }
    payload["run_hash"] = stable_hash(payload)
    _write_json(output_root / "run_manifest.json", payload)
    write_dataset_manifest(output_root)
    return payload


def write_dataset_manifest(output_root: Path) -> dict[str, Any]:
    models: dict[str, Any] = {}
    heal_root = Path("/home/lixingfeng/UniAD_examine/HEAL")
    for model_name, spec in MODEL_SPECS.items():
        config_path = Path(spec["config"])
        checkpoint_path = Path(spec["checkpoint"])
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        protocols: dict[str, Any] = {}
        manifest_root = output_root / "evaluation" / "manifests" / model_name
        for path in sorted(manifest_root.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            metadata = payload if isinstance(payload, Mapping) else {}
            protocols[path.stem] = {
                "path": str(path),
                "sha256": _sha256(path),
                "manifest_hash": metadata.get("manifest_hash"),
                "split": metadata.get("split"),
                "frame_count": len(metadata.get("frame_ids", metadata.get("samples", payload if isinstance(payload, list) else ()))),
                "warmup_count": len(metadata.get("warmup_frame_ids", ())),
                "reset_after_warmup": metadata.get("reset_after_warmup"),
            }
        models[model_name] = {
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "dataset": {
                "root_dir": str((heal_root / str(config.get("root_dir"))).resolve()) if config.get("root_dir") and not Path(str(config.get("root_dir"))).is_absolute() else config.get("root_dir"),
                "validate_dir": str((heal_root / str(config.get("validate_dir"))).resolve()) if config.get("validate_dir") and not Path(str(config.get("validate_dir"))).is_absolute() else config.get("validate_dir"),
            },
            "fixed_k": int(spec["fixed_k"]),
            "max_cav": 2,
            "workers": 8,
            "ap_iou_backend": "gpu",
            "postprocess": config.get("postprocess"),
            "manifests": protocols,
        }
    result = {
        "models": models,
        "cross_model_manifest_reuse": False,
        "same_manifest_within_model": True,
        "zero_skip_required": True,
    }
    result["dataset_contract_hash"] = stable_hash(result)
    _write_json(output_root / "dataset_manifest.json", result)
    return result


def inventory_model(model_name: str, output_root: Path, physical_gpu: int) -> dict[str, Any]:
    runtime = runtime_cuda_index_for_physical(physical_gpu)
    device = torch.device(f"cuda:{runtime}")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    bundle, _ = _load(model_name, device)
    families = discover_attention_families(model_name, bundle.model)
    modules = dict(bundle.model.named_modules())
    rows: list[dict[str, Any]] = []
    dependency: dict[str, Any] = {}
    for family in families:
        grid = dense_head_dimension_grid(
            family.original_d_h,
            heads=family.heads,
            low_width_extension=family.original_d_h <= 16,
        )
        dependency[family.family_id] = {
            "family": family.to_dict(),
            "candidate_grid": [row.to_dict() for row in grid],
            "dependencies": [],
        }
        for path in family.module_paths:
            module = modules[path]
            if family.attention_kind == "agent_relation":
                projections = {
                    "q_projection": [f"{path}.q_linears.{index}" for index in range(len(module.q_linears))],
                    "k_projection": [f"{path}.k_linears.{index}" for index in range(len(module.k_linears))],
                    "v_projection": [f"{path}.v_linears.{index}" for index in range(len(module.v_linears))],
                    "output_projection": [f"{path}.a_linears.{index}" for index in range(len(module.a_linears))],
                }
                extra = [f"{path}.relation_att", f"{path}.relation_msg"]
            else:
                projections = {
                    "q_projection": [f"{path}.to_qkv[Q]"],
                    "k_projection": [f"{path}.to_qkv[K]"],
                    "v_projection": [f"{path}.to_qkv[V]"],
                    "output_projection": [f"{path}.to_out.0"],
                }
                extra = []
            dependency[family.family_id]["dependencies"].append(
                {"module_path": path, "projection_paths": projections, "shared_parameters": extra}
            )
            rows.append(
                {
                    "model": model_name,
                    "attention_family": family.family_id,
                    "block": path,
                    "module_path": path,
                    "q_projection": projections["q_projection"],
                    "k_projection": projections["k_projection"],
                    "v_projection": projections["v_projection"],
                    "out_projection": projections["output_projection"],
                    "fused_qkv": family.attention_kind != "agent_relation",
                    "num_heads": family.heads,
                    "original_d_q": family.original_d_h,
                    "original_d_k": family.original_d_h,
                    "original_d_v": family.original_d_h,
                    "original_d_h": family.original_d_h,
                    "projection_output": family.heads * family.original_d_h,
                    "qk_reduction_dimension": family.original_d_h,
                    "av_value_dimension": family.original_d_h,
                    "w_o_input_dimension": family.heads * family.original_d_h,
                    "reshape_view": f"heads={family.heads},d_h={family.original_d_h}",
                    "scale": family.original_d_h**-0.5,
                    "shared_module": bool(extra),
                    "residual_dependency": f"output remains embed_dim={family.embed_dim}",
                    "onnx_role": family.attention_kind,
                    "tensorrt_role": "primitive_QK_AV",
                    "window_size": family.window_size,
                }
            )
    prefix = model_name.removeprefix("lidar_")
    _write_csv(output_root / "inventory" / f"{prefix}_attention_dh_inventory.csv", rows)
    _write_json(output_root / "inventory" / f"{prefix}_attention_families.json", [value.to_dict() for value in families])
    _write_json(
        output_root / "inventory" / f"{prefix}_candidate_grids.json",
        {
            family.family_id: [
                row.to_dict()
                for row in dense_head_dimension_grid(
                    family.original_d_h,
                    heads=family.heads,
                    low_width_extension=family.original_d_h <= 16,
                )
            ]
            for family in families
        },
    )
    _write_json(output_root / "inventory" / f"{prefix}_structural_dependency_map.json", dependency)
    del bundle
    torch.cuda.empty_cache()
    return {"model": model_name, "families": len(families), "modules": len(rows), "family_ids": [row.family_id for row in families]}


def _parameter_local_score(parameter: torch.Tensor, grad: torch.Tensor | None, index: tuple[Any, ...]) -> float:
    if grad is None:
        # Conditional agent-type projections are not invoked in every frame.
        # A missing gradient contributes zero to this sample; the caller still
        # requires every attention module to be observed at least once across
        # the complete frozen manifest.
        return 0.0
    return float((parameter[index].detach() * grad[index].detach()).abs().sum().item())


def _module_scores(module: nn.Module, original_d_h: int) -> list[list[float]]:
    heads = int(module.heads)
    scores = [[0.0 for _ in range(original_d_h)] for _ in range(heads)]
    if module.__class__.__name__ == "HGTCavAttention":
        for head in range(heads):
            for local in range(original_d_h):
                flat = head * original_d_h + local
                value = 0.0
                for collection_name in ("q_linears", "k_linears", "v_linears"):
                    for linear in getattr(module, collection_name):
                        value += _parameter_local_score(linear.weight, linear.weight.grad, (flat, slice(None)))
                        if linear.bias is not None:
                            value += _parameter_local_score(linear.bias, linear.bias.grad, (flat,))
                for linear in module.a_linears:
                    value += _parameter_local_score(linear.weight, linear.weight.grad, (slice(None), flat))
                for parameter in (module.relation_att, module.relation_msg):
                    row = _parameter_local_score(parameter, parameter.grad, (slice(None), head, local, slice(None)))
                    column = _parameter_local_score(parameter, parameter.grad, (slice(None), head, slice(None), local))
                    diagonal = _parameter_local_score(parameter, parameter.grad, (slice(None), head, local, local))
                    value += row + column - diagonal
                scores[head][local] = value
        return scores
    fused = module.to_qkv
    projection = int(fused.out_features) // 3
    out = module.to_out[0]
    for head in range(heads):
        for local in range(original_d_h):
            flat = head * original_d_h + local
            value = sum(
                _parameter_local_score(fused.weight, fused.weight.grad, (offset + flat, slice(None)))
                for offset in (0, projection, 2 * projection)
            )
            if fused.bias is not None:
                value += sum(
                    _parameter_local_score(fused.bias, fused.bias.grad, (offset + flat,))
                    for offset in (0, projection, 2 * projection)
                )
            value += _parameter_local_score(out.weight, out.weight.grad, (slice(None), flat))
            scores[head][local] = value
    return scores


def collect_taylor_rankings(
    model_name: str,
    output_root: Path,
    physical_gpu: int,
    *,
    num_samples: int = 200,
) -> dict[str, Any]:
    runtime = runtime_cuda_index_for_physical(physical_gpu)
    device = torch.device(f"cuda:{runtime}")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    bundle, _ = _load(model_name, device)
    # ``_load`` installs the real HEAL tree before importing ``opencood``.
    # Importing these names earlier would bind the repository's intentionally
    # minimal compatibility namespace instead of HEAL's training package.
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml import yaml_utils
    from search.integration.data_provider import move_batch_to_device
    families = discover_attention_families(model_name, bundle.model)
    family_by_path = {path: family for family in families for path in family.module_paths}
    modules = dict(bundle.model.named_modules())
    manifest_path = output_root / "evaluation" / "manifests" / model_name / "calibration200.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = list(manifest["samples"][: int(num_samples)])
    if len(samples) != int(num_samples) or manifest.get("split") != "train":
        raise RuntimeError("taylor_ranking_manifest_invalid")
    hypes = yaml_utils.load_yaml(MODEL_SPECS[model_name]["config"])
    hypes = bundle.adapter._absolutize_dataset_paths(hypes)
    dataset = build_dataset(hypes, visualize=False, train=True)
    accumulated = {
        path: [[0.0 for _ in range(family.original_d_h)] for _ in range(family.heads)]
        for path, family in family_by_path.items()
    }
    observed_batches = {path: 0 for path in family_by_path}
    losses: list[float] = []
    bundle.model.eval()
    for ordinal, row in enumerate(samples):
        seed = int(row["sample_seed"])
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        batch = dataset.collate_batch_train([dataset[int(row["dataset_index"])]])
        if batch is None:
            raise RuntimeError(f"taylor_ranking_empty_batch:{row['dataset_index']}")
        batch = move_batch_to_device(batch, device)
        bundle.model.zero_grad(set_to_none=True)
        output = bundle.adapter.forward_for_task(bundle.model, batch)
        loss = bundle.adapter.compute_task_loss(output, batch)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"taylor_ranking_nonfinite_loss:{ordinal}")
        loss.backward()
        losses.append(float(loss.detach().item()))
        for path, family in family_by_path.items():
            module = modules[path]
            if any(parameter.grad is not None for parameter in module.parameters()):
                observed_batches[path] += 1
            score = _module_scores(modules[path], family.original_d_h)
            for head in range(family.heads):
                for local in range(family.original_d_h):
                    accumulated[path][head][local] += score[head][local]
    rows: list[dict[str, Any]] = []
    rankings: dict[str, list[list[int]]] = {}
    for path, per_head in accumulated.items():
        if observed_batches[path] <= 0:
            raise RuntimeError(f"first_order_taylor_module_never_observed:{path}")
        family = family_by_path[path]
        rankings[path] = []
        for head, values in enumerate(per_head):
            normalized = [value / len(samples) for value in values]
            ranking = sorted(range(len(normalized)), key=lambda local: (normalized[local], local))
            rankings[path].append(ranking)
            for local, value in enumerate(normalized):
                rows.append(
                    {
                        "model": model_name,
                        "attention_family": family.family_id,
                        "module_path": path,
                        "head": head,
                        "head_local_position": local,
                        "first_order_taylor": value,
                        "rank_low_to_high": ranking.index(local),
                    }
                )
    payload = {
        "model": model_name,
        "method": "first_order_task_loss_taylor_abs_w_times_g",
        "checkpoint": MODEL_SPECS[model_name]["checkpoint"],
        "checkpoint_sha256": _sha256(Path(MODEL_SPECS[model_name]["checkpoint"])),
        "calibration_manifest": str(manifest_path),
        "calibration_manifest_hash": str(manifest["manifest_hash"]),
        "sample_count": len(samples),
        "frame_ids": [str(row["frame_id"]) for row in samples],
        "mean_task_loss": sum(losses) / len(losses),
        "rank_order": "low_to_high_importance_keep_suffix_for_nested_masks",
        "observed_batches_by_module": observed_batches,
        "rankings": rankings,
    }
    payload["ranking_hash"] = stable_hash(payload)
    prefix = model_name.removeprefix("lidar_")
    nested_rows: list[dict[str, Any]] = []
    for family in families:
        masks_by_width = {
            candidate.d_h: masks_from_rankings(family, rankings, candidate.d_h)
            for candidate in dense_head_dimension_grid(
                family.original_d_h, heads=family.heads
            )
        }
        nested_rows.extend(
            {
                "model": model_name,
                "attention_family": family.family_id,
                **row,
            }
            for row in audit_nested_masks(masks_by_width)
        )
    _write_csv(output_root / "importance" / f"{prefix}_per_block_per_head_importance.csv", rows)
    _write_csv(output_root / "importance" / f"{prefix}_nested_mask_audit.csv", nested_rows)
    _write_json(output_root / "importance" / f"{prefix}_ranking_manifest.json", payload)
    _write_json(
        output_root / "importance" / f"{prefix}_ranking_hash.json",
        {
            "model": model_name,
            "ranking_hash": payload["ranking_hash"],
            "nested_mask_rows": len(nested_rows),
            "all_nested": all(row["nested"] for row in nested_rows),
        },
    )
    del bundle
    torch.cuda.empty_cache()
    return {"model": model_name, "ranking_hash": payload["ranking_hash"], "samples": len(samples), "mean_task_loss": payload["mean_task_loss"]}


def write_nested_audit_from_manifest(model_name: str, output_root: Path) -> dict[str, Any]:
    prefix = model_name.removeprefix("lidar_")
    family_rows = json.loads(
        (output_root / "inventory" / f"{prefix}_attention_families.json").read_text(encoding="utf-8")
    )
    ranking = json.loads(
        (output_root / "importance" / f"{prefix}_ranking_manifest.json").read_text(encoding="utf-8")
    )
    from search.model_families.transformer.dh_pruning_contract import AttentionFamilyRecord

    rows: list[dict[str, Any]] = []
    for payload in family_rows:
        payload = dict(payload)
        payload["module_paths"] = tuple(payload["module_paths"])
        if isinstance(payload.get("window_size"), list):
            payload["window_size"] = tuple(payload["window_size"])
        family = AttentionFamilyRecord(**payload)
        masks_by_width = {
            candidate.d_h: masks_from_rankings(family, ranking["rankings"], candidate.d_h)
            for candidate in dense_head_dimension_grid(family.original_d_h, heads=family.heads)
        }
        rows.extend(
            {"model": model_name, "attention_family": family.family_id, **row}
            for row in audit_nested_masks(masks_by_width)
        )
    _write_csv(output_root / "importance" / f"{prefix}_nested_mask_audit.csv", rows)
    summary = {
        "model": model_name,
        "ranking_hash": ranking["ranking_hash"],
        "nested_mask_rows": len(rows),
        "all_nested": all(row["nested"] for row in rows),
    }
    _write_json(output_root / "importance" / f"{prefix}_ranking_hash.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--physical-gpu", required=True, type=int)
    parser.add_argument("--model", choices=("lidar_cobevt", "lidar_v2xvit", "all"), default="all")
    parser.add_argument("--phase", choices=("inventory", "ranking", "nested", "all"), default="all")
    parser.add_argument("--ranking-samples", type=int, default=200)
    args = parser.parse_args(argv)
    output = Path(args.output_root).resolve()
    write_run_manifest(output, physical_gpu=args.physical_gpu)
    models = tuple(MODEL_SPECS) if args.model == "all" else (args.model,)
    result: dict[str, Any] = {"inventory": [], "ranking": []}
    if args.phase in {"inventory", "all"}:
        result["inventory"] = [inventory_model(name, output, args.physical_gpu) for name in models]
    if args.phase in {"ranking", "all"}:
        result["ranking"] = [collect_taylor_rankings(name, output, args.physical_gpu, num_samples=args.ranking_samples) for name in models]
    if args.phase == "nested":
        result["nested"] = [write_nested_audit_from_manifest(name, output) for name in models]
    _write_json(output / "inventory" / "dh_sweep_preparation.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
