#!/usr/bin/env python3
"""Run a bounded real-checkpoint physical Transformer-width smoke.

This diagnostic uses deterministic identity rankings only to validate shape
rewrites.  Formal search rankings are produced separately from the common task
loss and cannot silently reuse these manifests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from scripts.audit_heal_transformer_search_models import MODEL_SPECS, _load, _sha256
from search.adapters.transformer_models import build_transformer_search_components
from search.pruning_space.transformer_physical_pruner import materialize_transformer_widths


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _iter_tensors(value: Any) -> Iterable[torch.Tensor]:
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)


def _next_lower(domain: Any) -> int:
    values = tuple(int(value) for value in domain.legal_widths)
    position = values.index(int(domain.original_width))
    if position <= 0:
        raise RuntimeError(f"domain_has_no_lower_width:{domain.domain_id}:{values}")
    return values[position - 1]


def run(model_name: str, output_root: Path, device: torch.device, mode: str) -> dict[str, Any]:
    model, adapter, hypes, batch = _load(model_name, device)
    components = build_transformer_search_components(
        model,
        hypes,
        allow_identity_ranking=True,
    )
    domains = list(components.transformer_domains)
    attention = list(components.attention_instances)
    ffn = list(components.ffn_instances)
    manifest = {
        "schema_version": "heal-transformer-domain-manifest-v1",
        "model": model_name,
        "config_path": str(MODEL_SPECS[model_name]["config"]),
        "checkpoint_path": str(MODEL_SPECS[model_name]["checkpoint"]),
        "config_sha256": _sha256(MODEL_SPECS[model_name]["config"]),
        "checkpoint_sha256": _sha256(MODEL_SPECS[model_name]["checkpoint"]),
        "ranking_status": "diagnostic_identity_not_formal_taylor",
        "attention_instances": [value.__dict__ for value in attention],
        "ffn_instances": [value.__dict__ for value in ffn],
        "projection_free_attention": [
            dict(value) for value in components.projection_free_attention
        ],
        "precision_units": [value.to_dict() for value in components.precision_units],
        "domains": [value.to_dict() for value in domains],
    }
    _write_json(output_root / "domain_manifests" / f"{model_name}_transformer_domains_diagnostic.json", manifest)

    attention_domains = [value for value in domains if value.domain_type == "attention_dh"]
    ffn_domains = [value for value in domains if value.domain_type == "ffn_hidden"]
    widths = {value.domain_id: int(value.original_width) for value in domains}
    selected: list[str] = []
    if mode in {"attention", "mixed"}:
        if not attention_domains:
            raise RuntimeError(f"attention_domain_missing_for_mode:{model_name}:{mode}")
        domain = attention_domains[0]
        widths[domain.domain_id] = _next_lower(domain)
        selected.append(domain.domain_id)
    if mode in {"ffn", "mixed"}:
        if not ffn_domains:
            raise RuntimeError(f"ffn_domain_missing_for_mode:{model_name}:{mode}")
        domain = ffn_domains[0]
        widths[domain.domain_id] = _next_lower(domain)
        selected.append(domain.domain_id)

    if mode == "inventory":
        result = {
            "schema_version": "heal-transformer-physical-smoke-v1",
            "model": model_name,
            "mode": mode,
            "attention_domain_count": len(attention_domains),
            "ffn_domain_count": len(ffn_domains),
            "passed": True,
            "note": "no trainable Transformer width domain was materialized",
        }
    else:
        report = materialize_transformer_widths(
            model,
            domains,
            widths,
            model_name=model_name,
        )
        with torch.inference_mode():
            outputs = adapter.forward_for_task(model, batch)
        tensors = list(_iter_tensors(outputs))
        finite = bool(tensors) and all(
            bool(torch.isfinite(value).all().item()) if value.is_floating_point() else True
            for value in tensors
        )
        result = {
            "schema_version": "heal-transformer-physical-smoke-v1",
            "model": model_name,
            "mode": mode,
            "selected_domains": selected,
            "diagnostic_identity_ranking": True,
            "formal_ranking_eligible": False,
            "report": report.to_dict(),
            "forward": {
                "finite": finite,
                "outputs": [
                    {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
                    for value in tensors
                ],
            },
            "passed": bool(report.passed and finite),
        }
        if not result["passed"]:
            raise RuntimeError(f"real_transformer_physical_smoke_failed:{model_name}:{mode}:{result}")
    _write_json(output_root / "structures" / f"{model_name}_{mode}_physical_smoke.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=("inventory", "attention", "ffn", "mixed"), required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    result = run(args.model, args.output_root.resolve(), device, args.mode)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
