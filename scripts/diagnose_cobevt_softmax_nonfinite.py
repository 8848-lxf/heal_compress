#!/usr/bin/env python3
"""Diagnose non-finite CoBEVT attention probabilities without changing search.

This is a read-only model/data diagnostic.  It records the exact Softmax
input/output finiteness for the frozen Taylor sample prefix and proves whether
an offending row is fully masked before any remediation is considered.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.audit_heal_transformer_search_models import _load
from scripts.run_v2xvit_greedy005_full import _build_full_space
from scripts.run_heal_transformer_six_budget_proxy import FrozenTrainPrefix
from scripts.run_v2xvit_greedy005_weight_only_abs import _type_coverage_forward
from search.proxy.conservative_gate_activation_taylor import build_activation_units
from search.proxy.joint_weight_activation_taylor import taylor_units_from_transformer_precision


def _tensors(value: Any) -> Iterable[torch.Tensor]:
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item)


def _tensor_stats(value: torch.Tensor) -> dict[str, Any]:
    floating = value.is_floating_point() or value.is_complex()
    finite = torch.isfinite(value) if floating else torch.ones_like(value, dtype=torch.bool)
    result: dict[str, Any] = {
        "shape": [int(item) for item in value.shape],
        "dtype": str(value.dtype),
        "element_count": int(value.numel()),
        "finite_count": int(finite.sum().item()),
        "nonfinite_count": int((~finite).sum().item()),
        "nan_count": int(torch.isnan(value).sum().item()) if floating else 0,
        "posinf_count": int(torch.isposinf(value).sum().item()) if floating else 0,
        "neginf_count": int(torch.isneginf(value).sum().item()) if floating else 0,
    }
    if value.ndim and floating:
        flat = value.reshape(-1, int(value.shape[-1]))
        result["row_count"] = int(flat.shape[0])
        result["all_neginf_row_count"] = int(torch.isneginf(flat).all(dim=-1).sum().item())
        result["any_nonfinite_row_count"] = int((~torch.isfinite(flat)).any(dim=-1).sum().item())
    finite_values = value[finite]
    if finite_values.numel():
        result["finite_min"] = float(finite_values.min().item())
        result["finite_max"] = float(finite_values.max().item())
    return result


def run(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing_to_overwrite:{output}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    device = torch.device("cuda:0")
    model, adapter, hypes, representative = _load("cobevt", device)
    prefix = FrozenTrainPrefix(
        adapter=adapter,
        hypes=hypes,
        device=device,
        manifest=manifest,
        count=int(args.samples),
    )

    current: dict[str, list[dict[str, Any]]] = {}
    handles = []
    target_paths = []
    for path, module in model.named_modules():
        if not isinstance(module, torch.nn.Softmax) or "fusion_net.layers" not in path:
            continue
        target_paths.append(path)

        def hook(_module: torch.nn.Module, inputs: tuple[Any, ...], result: Any, *, _path=path) -> None:
            if not inputs or not torch.is_tensor(inputs[0]) or not torch.is_tensor(result):
                raise RuntimeError(f"cobevt_softmax_boundary_not_tensor:{_path}")
            current.setdefault(_path, []).append({
                "input": _tensor_stats(inputs[0].detach()),
                "output": _tensor_stats(result.detach()),
            })

        handles.append(module.register_forward_hook(hook))

    if not target_paths:
        raise RuntimeError("cobevt_softmax_modules_missing")
    samples = []
    try:
        for index, batch in enumerate(prefix):
            current.clear()
            with torch.no_grad():
                task_output = _type_coverage_forward(adapter, model, batch)
                loss = adapter.compute_task_loss(task_output, batch)
            task_tensors = list(_tensors(task_output))
            samples.append({
                "sample_index": int(index),
                "dataset_index": int(manifest["samples"][index]["dataset_index"]),
                "task_loss": float(loss.detach().cpu()),
                "task_loss_finite": bool(torch.isfinite(loss).item()),
                "task_output_nonfinite_count": int(sum(
                    (~torch.isfinite(tensor)).sum().item()
                    for tensor in task_tensors
                    if tensor.is_floating_point()
                )),
                "softmax": {path: list(rows) for path, rows in sorted(current.items())},
            })
    finally:
        for handle in handles:
            handle.remove()

    # Reproduce the production activation-unit mapper.  Protected/fixed
    # Softmax groups must have no Stage-1 Taylor transition; if a future
    # Softmax profile becomes mutable, its dedicated regression test requires
    # a module-output (post-probability) boundary.
    identity = _build_full_space(model, adapter, hypes, representative)
    mapped_units = taylor_units_from_transformer_precision(
        model,
        identity["components"].precision_units,
        active_module_paths=sorted(
            {row.module_path for row in identity["runtime"].shapes}
        ),
    )
    activation_units, group_to_units = build_activation_units(
        model, identity["space"], mapped_units
    )
    softmax_groups = tuple(
        str(group.group_id)
        for group in identity["space"].quantization_groups
        if str(group.metadata.get("transformer_role", "")) == "softmax"
    )
    mutable = set(identity["space"].precision_gene_ids)
    active_softmax_taylor_groups = sorted(
        group_id for group_id in softmax_groups if group_id in group_to_units
    )
    if any(group_id in mutable for group_id in softmax_groups):
        raise RuntimeError("cobevt_softmax_unexpectedly_mutable")
    if active_softmax_taylor_groups:
        raise RuntimeError(
            f"cobevt_fixed_softmax_taylor_units_present:"
            f"{active_softmax_taylor_groups}"
        )
    mapped_capture_audit = {
        "softmax_precision_group_ids": list(softmax_groups),
        "softmax_mutable_group_ids": sorted(set(softmax_groups) & mutable),
        "softmax_activation_taylor_group_ids": active_softmax_taylor_groups,
        "activation_unit_count": len(activation_units),
        "fixed_softmax_excluded_from_stage1_taylor": True,
    }

    offenders = []
    for sample in samples:
        for path, calls in sample["softmax"].items():
            for call_index, call in enumerate(calls):
                if int(call["output"]["nonfinite_count"]):
                    offenders.append({
                        "sample_index": sample["sample_index"],
                        "dataset_index": sample["dataset_index"],
                        "module_path": path,
                        "call_index": int(call_index),
                        **call,
                    })
    payload = {
        "schema_version": "cobevt-softmax-nonfinite-diagnostic-v1",
        "model": "cobevt",
        "sample_count": len(samples),
        "manifest_hash": manifest.get("manifest_hash"),
        "softmax_module_paths": sorted(target_paths),
        "offender_count": len(offenders),
        "offenders": offenders,
        "production_boundary_capture": mapped_capture_audit,
        "samples": samples,
        "search_semantics_changed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "sample_count": len(samples),
        "offender_count": len(offenders),
    }, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=32)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
