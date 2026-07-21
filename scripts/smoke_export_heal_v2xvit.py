#!/usr/bin/env python3
"""Strict-load, parity-check, and ONNX-smoke the isolated HEAL V2X-ViT adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch


REPO = Path(__file__).resolve().parents[1]
PROJECT_PARENT = REPO.parent
for path in (str(PROJECT_PARENT), str(REPO)):
    if path not in sys.path:
        sys.path.insert(0, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parity(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    difference = (reference.float() - actual.float()).abs()
    return {
        "reference_shape": list(reference.shape),
        "actual_shape": list(actual.shape),
        "max_abs": float(difference.max().item()),
        "mean_abs": float(difference.mean().item()),
        "allclose_atol_1e-5_rtol_1e-5": bool(
            torch.allclose(reference.float(), actual.float(), atol=1.0e-5, rtol=1.0e-5)
        ),
        "allclose_atol_5e-4_rtol_1e-4": bool(
            torch.allclose(reference.float(), actual.float(), atol=5.0e-4, rtol=1.0e-4)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--heal-root", type=Path, default=Path("/home/lixingfeng/UniAD_examine/HEAL"))
    parser.add_argument("--device", default="cuda:6")
    fixed_k_source = parser.add_mutually_exclusive_group(required=True)
    fixed_k_source.add_argument("--fixed-k", type=int)
    fixed_k_source.add_argument("--fixed-k-manifest", type=Path)
    parser.add_argument("--max-agents", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--skip-onnx", action="store_true")
    args = parser.parse_args()

    from search.model_family import (
        build_model_family_search_readiness,
        build_v2xvit_onnx_mapping,
        load_heal_model_family,
    )
    from quantization.export.signal_maxk import capture_weighted_module_calls
    from search.model_family.export import (
        HealV2XViTExportPolicy,
        build_heal_v2xvit_export_module,
        prepare_v2xvit_fixed_k_inputs,
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    report_path = args.output_dir / "export_report.json"
    onnx_path = args.output_dir / "heal_lidar_v2xvit_fixedk.onnx"
    bundle = load_heal_model_family(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        heal_root=args.heal_root,
        device=args.device,
        family_id="heal_lidar_v2xvit",
        forward_smoke=True,
    )
    if args.fixed_k_manifest is not None:
        policy = HealV2XViTExportPolicy.from_frozen_train_manifest(args.fixed_k_manifest)
        if int(args.max_agents) != policy.max_agents:
            raise RuntimeError(
                f"v2xvit_manifest_max_agents_mismatch:{args.max_agents}!={policy.max_agents}"
            )
    else:
        policy = HealV2XViTExportPolicy(fixed_k=args.fixed_k, max_agents=args.max_agents)
    wrapper = build_heal_v2xvit_export_module(bundle.model, policy=policy).eval()
    prepared = prepare_v2xvit_fixed_k_inputs(bundle.example_batch, policy=policy)
    input_names = tuple(prepared)
    inputs = tuple(prepared[name] for name in input_names)

    with torch.no_grad():
        actual = wrapper(*inputs)
    reference = tuple(bundle.smoke_outputs[name] for name in policy.output_names)
    parity = {
        name: _parity(expected, observed)
        for name, expected, observed in zip(policy.output_names, reference, actual)
    }
    parity_gate = all(row["allclose_atol_5e-4_rtol_1e-4"] for row in parity.values())

    payload: dict[str, Any] = {
        "schema_version": "heal-v2xvit-onnx-smoke-v1",
        "environment": {
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
            "conda_prefix": os.environ.get("CONDA_PREFIX", ""),
            "python": sys.executable,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": args.device,
        },
        "provenance": {
            "config": str(bundle.config_path),
            "config_sha256": bundle.config_hash,
            "checkpoint": str(bundle.checkpoint_path),
            "checkpoint_sha256": bundle.checkpoint_hash,
            "strict_state_dict_load": True,
            "model_family_audit_hash": bundle.audit.to_dict()["audit_hash"],
            "fixed_k_manifest": str(args.fixed_k_manifest.resolve()) if args.fixed_k_manifest else None,
            "fixed_k_manifest_hash": policy.calibration_manifest_hash,
        },
        "input_contract": {
            "fixed_k": policy.fixed_k,
            "max_agents": policy.max_agents,
            "input_names": list(input_names),
            "input_shapes": {name: list(prepared[name].shape) for name in input_names},
            "output_names": list(policy.output_names),
        },
        "pytorch_wrapper_parity": {
            "passed": parity_gate,
            "tolerance": {"atol": 5.0e-4, "rtol": 1.0e-4},
            "reason": "preserves prewarp plus identity STTF/ROI GridSample operations",
            "outputs": parity,
        },
        "onnx_export": {"requested": not args.skip_onnx, "passed": None},
    }
    if not args.skip_onnx:
        try:
            with capture_weighted_module_calls(wrapper) as module_calls:
                torch.onnx.export(
                    wrapper,
                    inputs,
                    str(onnx_path),
                    export_params=True,
                    opset_version=args.opset,
                    do_constant_folding=True,
                    input_names=list(input_names),
                    output_names=list(policy.output_names),
                    custom_opsets={"trt": 1},
                )
            import onnx

            graph = onnx.load(str(onnx_path), load_external_data=False)
            onnx.checker.check_model(graph)
            op_counts: dict[str, int] = {}
            for node in graph.graph.node:
                key = f"{node.domain or 'onnx'}::{node.op_type}"
                op_counts[key] = op_counts.get(key, 0) + 1
            payload["onnx_export"] = {
                "requested": True,
                "passed": True,
                "path": str(onnx_path.resolve()),
                "size_bytes": onnx_path.stat().st_size,
                "sha256": _sha256(onnx_path),
                "checker_passed": True,
                "opset": args.opset,
                "node_count": len(graph.graph.node),
                "initializer_count": len(graph.graph.initializer),
                "op_counts": dict(sorted(op_counts.items())),
                "contains_linalg_inverse": any(
                    node.op_type in {"Inverse", "LinalgInv"} for node in graph.graph.node
                ),
                "contains_scatter_plugin": any(
                    node.domain == "trt" and node.op_type == "PointPillarScatterTRT"
                    for node in graph.graph.node
                ),
            }
            canonical_mapping = build_v2xvit_onnx_mapping(
                onnx_path,
                bundle.audit,
                module_calls,
            ).to_dict()
            mapping_path = args.output_dir / "canonical_weight_mapping.json"
            mapping_path.write_text(
                json.dumps(canonical_mapping, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            payload["canonical_weight_mapping"] = {
                "path": str(mapping_path.resolve()),
                "sha256": _sha256(mapping_path),
                "mapping_hash": canonical_mapping["mapping_hash"],
                "metadata": canonical_mapping["metadata"],
                "unresolved": canonical_mapping["unresolved"],
            }
        except Exception as error:  # preserve the exact first unsupported operator
            payload["onnx_export"] = {
                "requested": True,
                "passed": False,
                "error_type": type(error).__name__,
                "error": str(error),
                "partial_onnx_exists": onnx_path.is_file(),
            }

    payload["search_readiness"] = build_model_family_search_readiness(
        bundle.audit,
        {
            "strict_state_dict_load": True,
            "onnx_export": bool(payload["onnx_export"].get("passed")),
            "onnx_checker": bool(payload["onnx_export"].get("checker_passed")),
            "canonical_weight_mapping": bool(
                payload.get("canonical_weight_mapping", {})
                .get("metadata", {})
                .get("realized_graph_mapping_complete", False)
            ),
            "tensor_parity": parity_gate,
        },
    ).to_dict()

    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if parity_gate and (args.skip_onnx or payload["onnx_export"]["passed"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
