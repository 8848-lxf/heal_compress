"""Numerical parity of D0 physical decomposition against the original model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from search.integration.runtime_environment import runtime_cuda_index_for_physical
from search.model_families.transformer.dh_physical_rewrite import (
    discover_attention_families,
    materialize_family_head_dimension,
)
from search.model_families.transformer.dh_pruning_contract import masks_from_rankings
from search.orchestration.lidar_transformer_h800_inventory import MODEL_SPECS, _load, _real_batch


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _outputs(value: Any) -> dict[str, torch.Tensor]:
    if isinstance(value, Mapping):
        return {str(key): tensor.detach() for key, tensor in value.items() if torch.is_tensor(tensor)}
    if isinstance(value, (list, tuple)):
        return {str(index): tensor.detach() for index, tensor in enumerate(value) if torch.is_tensor(tensor)}
    return {"output": value.detach()} if torch.is_tensor(value) else {}


def _compare(reference: Mapping[str, torch.Tensor], actual: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    issues: list[str] = []
    rows: list[dict[str, Any]] = []
    if set(reference) != set(actual):
        issues.append("output_key_mismatch")
    for name in sorted(set(reference) & set(actual)):
        left = reference[name].float()
        right = actual[name].float()
        if left.shape != right.shape:
            issues.append(f"output_shape_mismatch:{name}")
            continue
        difference = (left - right).abs()
        flat_left = left.reshape(-1)
        flat_right = right.reshape(-1)
        cosine = float(torch.nn.functional.cosine_similarity(flat_left, flat_right, dim=0).item()) if flat_left.numel() else 1.0
        row = {
            "tensor": name,
            "shape": list(left.shape),
            "exact_equal": bool(torch.equal(left, right)),
            "allclose_1e_5": bool(torch.allclose(left, right, rtol=1e-5, atol=1e-5)),
            "allclose_1e_4": bool(torch.allclose(left, right, rtol=1e-4, atol=1e-4)),
            "mae": float(difference.mean().item()),
            "max_abs_error": float(difference.max().item()),
            "cosine": cosine,
            "finite": bool(torch.isfinite(right).all()),
        }
        if not row["allclose_1e_4"] or not row["finite"]:
            issues.append(f"output_parity_failed:{name}")
        rows.append(row)
    return {"passed": bool(rows) and not issues, "issues": issues, "tensors": rows, "tolerance": {"rtol": 1e-4, "atol": 1e-4}}


def run(*, output_root: Path, model_name: str, physical_gpu: int) -> list[dict[str, Any]]:
    runtime = runtime_cuda_index_for_physical(physical_gpu)
    device = torch.device(f"cuda:{runtime}")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    original_bundle, _ = _load(model_name, device)
    batch = _real_batch(original_bundle, str(MODEL_SPECS[model_name]["config"]), device)
    with torch.inference_mode():
        reference = _outputs(original_bundle.adapter.forward_for_task(original_bundle.model, batch))
    family_records = discover_attention_families(model_name, original_bundle.model)
    ranking = json.loads((output_root / "importance" / f"{model_name.removeprefix('lidar_')}_ranking_manifest.json").read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = []
    del original_bundle
    torch.cuda.empty_cache()
    for family in family_records:
        bundle, _ = _load(model_name, device)
        masks = masks_from_rankings(family, ranking["rankings"], family.original_d_h)
        report = materialize_family_head_dimension(model_name, bundle.model, family, masks)
        with torch.inference_mode():
            actual = _outputs(bundle.adapter.forward_for_task(bundle.model, batch))
        parity = _compare(reference, actual)
        result = {
            "model": model_name,
            "attention_family": family.family_id,
            "d_h": family.original_d_h,
            "head_count": family.heads,
            "physical_rewrite_passed": report.passed,
            "original_parameter_count": report.original_parameter_count,
            "physical_parameter_count": report.physical_parameter_count,
            "parameter_count_equal": report.original_parameter_count == report.physical_parameter_count,
            "structure_hash": report.structure_hash,
            **parity,
        }
        directory = output_root / "structures" / model_name / family.family_id / f"dh_{family.original_d_h:03d}"
        _write(directory / "original_forward_parity.json", result)
        structure_path = directory / "structure_result.json"
        if structure_path.is_file():
            structure = json.loads(structure_path.read_text(encoding="utf-8"))
            structure["d0_original_forward_parity_passed"] = result["passed"]
            structure["d0_parameter_count_equal"] = result["parameter_count_equal"]
            _write(structure_path, structure)
        results.append(result)
        del bundle, actual
        torch.cuda.empty_cache()
    _write(output_root / "reports" / f"{model_name}_d0_original_forward_parity.json", results)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    args = parser.parse_args(argv)
    results = run(output_root=Path(args.output_root).resolve(), model_name=args.model, physical_gpu=args.physical_gpu)
    print(json.dumps(results, sort_keys=True))
    return 0 if all(row["passed"] and row["parameter_count_equal"] for row in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
