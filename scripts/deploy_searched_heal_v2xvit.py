#!/usr/bin/env python3
"""Deploy and evaluate one immutable, genuinely searched HEAL V2X-ViT candidate."""

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
for value in (REPO.parent, REPO):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite_v2xvit_deployment_artifact:{path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _state_parity(reference: dict[str, torch.Tensor], actual: dict[str, torch.Tensor]) -> dict[str, Any]:
    keys_equal = set(reference) == set(actual)
    rows = []
    for name in sorted(set(reference) | set(actual)):
        left = reference.get(name)
        right = actual.get(name)
        if left is None or right is None:
            rows.append({"name": name, "equal": False, "reason": "missing"})
            continue
        equal = left.shape == right.shape and left.dtype == right.dtype and torch.equal(left, right)
        rows.append(
            {
                "name": name,
                "shape": list(left.shape),
                "dtype": str(left.dtype),
                "reference_hash": _tensor_hash(left),
                "physical_hash": _tensor_hash(right),
                "equal": bool(equal),
            }
        )
    return {
        "keys_equal": keys_equal,
        "tensor_count": len(rows),
        "all_tensors_exact": keys_equal and all(bool(row["equal"]) for row in rows),
        "tensors": rows,
    }


def _output_parity(reference: dict[str, torch.Tensor], actual: tuple[torch.Tensor, ...], names: tuple[str, ...]) -> dict[str, Any]:
    result = {}
    for name, observed in zip(names, actual):
        expected = reference[name]
        difference = (expected.float() - observed.float()).abs()
        result[name] = {
            "shape_equal": tuple(expected.shape) == tuple(observed.shape),
            "max_abs": float(difference.max().item()),
            "mean_abs": float(difference.mean().item()),
            "allclose": bool(torch.allclose(expected.float(), observed.float(), atol=5.0e-4, rtol=1.0e-4)),
        }
    return {"passed": all(row["allclose"] for row in result.values()), "outputs": result}


def _evaluation_manifest(bundle: Any, output: Path, *, warmup: int, frames: int) -> Path:
    from opencood.hypes_yaml import yaml_utils
    from search.model_family.deployment import file_sha256
    from quantization.types import stable_json_hash

    hypes = yaml_utils.load_yaml(str(bundle.config_path))
    hypes = bundle.adapter._absolutize_dataset_paths(hypes)
    split_path = Path(str(hypes["validate_dir"]))
    ids = [str(value) for value in json.loads(split_path.read_text(encoding="utf-8"))]
    required = int(warmup) + int(frames)
    if len(ids) < required:
        raise RuntimeError(f"v2xvit_validation_split_too_short:{len(ids)}<{required}")
    payload = {
        "schema_version": "heal-v2xvit-fixed-evaluation-manifest-v1",
        "source_validation_split": str(split_path.resolve()),
        "source_validation_split_sha256": file_sha256(split_path),
        "warmup_frame_ids": ids[: int(warmup)],
        "evaluation_frame_ids": ids[int(warmup) : required],
        "reset_after_warmup": True,
        "selection": "deterministic_validation_prefix_no_cherry_pick",
    }
    payload["manifest_hash"] = stable_json_hash(payload)
    path = output / "evaluation_manifest.json"
    _write_json(path, payload)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--search-artifact", type=Path, required=True)
    parser.add_argument("--search-space", type=Path, required=True)
    parser.add_argument("--search-target", type=float, default=0.25)
    parser.add_argument("--candidate-hash", default="")
    parser.add_argument("--heal-root", type=Path, default=Path("../../HEAL"))
    parser.add_argument("--tensorrt-root", type=Path, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-gpu", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--eval-frames", type=int, default=20)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--latency-rounds", type=int, default=1)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args(argv)

    if os.environ.get("CONDA_DEFAULT_ENV") != "univ2x-opt":
        raise RuntimeError(f"v2xvit_deployment_requires_univ2x_opt:{os.environ.get('CONDA_DEFAULT_ENV', '')}")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("v2xvit_deployment_requires_cuda")
    torch.cuda.set_device(device)

    from quantization.config import QDQConfig, TensorRTBuildConfig
    from quantization.precision.qdq_inserter import insert_explicit_qdq
    from quantization.export.signal_maxk import capture_weighted_module_calls
    from search.model_family import build_v2xvit_onnx_mapping, load_heal_model_family
    from search.model_family.calibration_manifest import load_v2xvit_train_manifest
    from search.model_family.deployment import (
        build_physical_structure_snapshot_v2,
        build_v2xvit_precision_mapping,
        canonicalize_v2xvit_onnx,
        collect_v2xvit_train200_entropy_scales,
        file_sha256,
        load_searched_candidate_profile,
        load_v2xvit_pruning_domains,
    )
    from search.model_family.evaluation import evaluate_v2xvit_engine_modelopt
    from search.model_family.export import (
        HealV2XViTExportPolicy,
        build_heal_v2xvit_export_module,
        prepare_v2xvit_fixed_k_inputs,
    )
    from search.model_family.search_smoke import load_v2xvit_manifest_batches
    from search.stage2.trt_modelopt import build_engine_modelopt

    from search.candidate import CandidatePhenotype, PrecisionDecision
    from search.model_family.pruning import materialize_v2xvit_ffn_pruning

    for required in (
        args.config,
        args.checkpoint,
        args.manifest,
        args.search_artifact,
        args.search_space,
        args.plugin,
    ):
        if not required.is_file():
            raise RuntimeError(f"v2xvit_deployment_input_missing:{required}")
    trtexec = args.tensorrt_root / "bin/trtexec"
    if not trtexec.is_file():
        trtexec = args.tensorrt_root / "targets/x86_64-linux-gnu/bin/trtexec"
    if not trtexec.is_file():
        raise RuntimeError(f"v2xvit_trtexec_missing:{args.tensorrt_root}")

    candidate = load_searched_candidate_profile(
        args.search_artifact,
        target=args.search_target,
        candidate_hash=str(args.candidate_hash),
    )
    manifest = load_v2xvit_train_manifest(args.manifest)
    bundle = load_heal_model_family(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        heal_root=args.heal_root,
        device=str(device),
        family_id="heal_lidar_v2xvit",
        forward_smoke=False,
    )
    domains = load_v2xvit_pruning_domains(args.search_space)
    widths = candidate["domain_width_profile"]
    if set(widths) != {row.domain_id for row in domains}:
        raise RuntimeError("v2xvit_candidate_domain_set_mismatch")
    pruned_units = sorted(
        unit_id
        for domain in domains
        for unit_id in domain.pruned_unit_ids_for_width(int(widths[domain.domain_id]))
    )
    phenotype = CandidatePhenotype(
        pruned_unit_ids=pruned_units,
        precision_profile={
            name: PrecisionDecision(value, value)
            for name, value in candidate["module_precision_profile"].items()
        },
        pruning_policy_version="v2xvit-ffn-legal-domain-width-fixed-ranking-v1",
        precision_policy_version="heal-v2xvit-explicit-qdq-strong-type-v1",
        metadata={"domain_width_profile": widths},
    )
    physical = materialize_v2xvit_ffn_pruning(bundle.model, phenotype, domains)
    physical_snapshot = build_physical_structure_snapshot_v2(physical.model)
    state = {name: value.detach().cpu() for name, value in physical.model.state_dict().items()}
    physical_checkpoint = output / "physical_searched_checkpoint.pth"
    torch.save(
        {
            "model": state,
            "candidate_identity": candidate["candidate_identity"],
            "source_checkpoint_sha256": bundle.checkpoint_hash,
        },
        physical_checkpoint,
    )
    reload_physical = materialize_v2xvit_ffn_pruning(bundle.model, phenotype, domains)
    saved = torch.load(physical_checkpoint, map_location=device)
    reload_physical.model.load_state_dict(saved["model"], strict=True)
    reload_physical.model.eval()
    state_proof = _state_parity(
        state,
        {name: value.detach().cpu() for name, value in reload_physical.model.state_dict().items()},
    )
    if not state_proof["all_tensors_exact"]:
        raise RuntimeError("v2xvit_physical_checkpoint_strict_reload_not_exact")
    _write_json(output / "searched_candidate.json", candidate)
    _write_json(output / "physical_structure_snapshot_v2.json", physical_snapshot)
    _write_json(
        output / "physical_pruning_snapshot.json",
        {**physical.snapshot, "snapshot_hash": physical.snapshot_hash},
    )
    _write_json(output / "physical_state_exact_parity.json", state_proof)

    # Reuse the strictly loaded provider context while replacing only its
    # model object with the independently materialized and strictly reloaded
    # physical candidate.
    bundle.model = reload_physical.model
    reload_bundle = bundle

    batches, evidence = load_v2xvit_manifest_batches(
        reload_bundle, manifest, sample_count=1, device=device
    )
    real_batch = batches[0]
    policy = HealV2XViTExportPolicy.from_frozen_train_manifest(args.manifest)
    wrapper = build_heal_v2xvit_export_module(reload_bundle.model, policy=policy).eval()
    prepared = prepare_v2xvit_fixed_k_inputs(real_batch["ego"], policy=policy)
    input_names = tuple(prepared)
    inputs = tuple(prepared[name] for name in input_names)
    with torch.inference_mode():
        reference = reload_bundle.adapter.forward_for_task(reload_bundle.model, real_batch)
        wrapper_outputs = wrapper(*inputs)
    parity = _output_parity(reference, wrapper_outputs, policy.output_names)
    if not parity["passed"]:
        raise RuntimeError(f"v2xvit_real_wrapper_parity_failed:{parity}")
    _write_json(output / "real_wrapper_parity.json", {"sample": evidence[0], **parity})

    base_onnx = output / "base_fixedk27904.onnx"
    with capture_weighted_module_calls(wrapper) as module_calls:
        torch.onnx.export(
            wrapper,
            inputs,
            str(base_onnx),
            export_params=True,
            opset_version=int(args.opset),
            do_constant_folding=True,
            input_names=list(input_names),
            output_names=list(policy.output_names),
            custom_opsets={"trt": 1},
        )
    import onnx

    onnx.checker.check_model(onnx.load(str(base_onnx), load_external_data=False))
    family_mapping = build_v2xvit_onnx_mapping(base_onnx, reload_bundle.audit, module_calls)
    if family_mapping.unresolved:
        raise RuntimeError(f"v2xvit_export_mapping_unresolved:{family_mapping.unresolved}")
    _write_json(output / "model_family_onnx_mapping.json", family_mapping.to_dict())
    origin_map = canonicalize_v2xvit_onnx(base_onnx, module_calls)
    mapping = build_v2xvit_precision_mapping(
        origin_map,
        candidate["module_precision_profile"],
        canonical_onnx_path=base_onnx,
        profile_id=(
            f"{candidate['search_algorithm']}_bops_{args.search_target:.3f}_"
            f"{candidate['candidate_identity'][:12]}"
        ),
    )
    _write_json(output / "canonical_origin_map.json", origin_map.to_dict())
    _write_json(output / "canonical_precision_mapping.json", mapping.to_dict())

    scales, calibration_metadata = collect_v2xvit_train200_entropy_scales(
        bundle=reload_bundle,
        manifest=manifest,
        mapping=mapping,
        canonical_onnx_path=base_onnx,
        device=device,
    )
    _write_json(
        output / "calibration_scales.json",
        {"metadata": calibration_metadata, "scales": scales},
    )
    qdq_config = QDQConfig(
        allowed_precisions=("fp32", "fp16", "int8"),
        insert_activation_input_qdq=True,
        insert_weight_qdq=True,
        insert_activation_output_qdq=False,
        weight_granularity="per_channel",
        merge_policy="fp16_merge",
        explicit_fp16_compute_casts=True,
        explicit_fp32_compute_casts=True,
        policy_version="heal-v2xvit-explicit-qdq-strongly-typed-v1",
    )
    qdq_path = output / "searched_candidate_explicit_qdq.onnx"
    qdq = insert_explicit_qdq(
        base_onnx,
        qdq_path,
        mapping,
        scales=scales,
        config=qdq_config,
        calibration_metadata=calibration_metadata,
    )
    _write_json(output / "qdq_insertion_report.json", qdq.to_dict())

    engine_path = output / "searched_candidate_strongly_typed.plan"
    build_config = TensorRTBuildConfig(
        trtexec_path=trtexec,
        plugin_path=args.plugin,
        workspace_mib=4096,
        timeout_seconds=3600,
        no_tf32=True,
        skip_inference=True,
        export_layer_info=True,
        strongly_typed=True,
        enable_fp16=False,
        enable_int8=False,
        policy_version="heal-v2xvit-strongly-typed-explicit-qdq-v1",
    )
    build = build_engine_modelopt(
        qdq_onnx=qdq_path,
        engine_path=engine_path,
        precision_mapping=mapping,
        build_config=build_config,
        physical_snapshot=physical_snapshot,
        output_dir=output / "engine_build",
        tensorrt_root=args.tensorrt_root,
        conda_env="modelopt",
        gpu_id=int(args.physical_gpu),
    )
    _write_json(output / "engine_acceptance.json", build)
    if build.get("status") != "ok":
        raise RuntimeError(f"v2xvit_engine_acceptance_failed:{build.get('status')}:{build.get('failure_reason', '')}")
    if not engine_path.is_file() or engine_path.stat().st_size <= 0:
        raise RuntimeError("v2xvit_engine_missing_after_success")

    eval_manifest = _evaluation_manifest(
        reload_bundle,
        output,
        warmup=int(args.warmup_frames),
        frames=int(args.eval_frames),
    )
    evaluation = evaluate_v2xvit_engine_modelopt(
        engine_path=engine_path,
        model_config=args.config,
        heal_root=args.heal_root,
        output_dir=output / "evaluation",
        tensorrt_root=args.tensorrt_root,
        plugin_path=args.plugin,
        eval_manifest_path=eval_manifest,
        physical_gpu_id=int(args.physical_gpu),
        fixed_k=int(policy.fixed_k),
        max_agents=int(policy.max_agents),
        num_frames=int(args.eval_frames),
        warmup_frames=int(args.warmup_frames),
        latency_rounds=int(args.latency_rounds),
        dataloader_num_workers=8,
    )
    _write_json(output / "evaluation_acceptance.json", evaluation)
    if evaluation.get("status") != "ok":
        raise RuntimeError(f"v2xvit_evaluation_failed:{evaluation.get('failure_reason', '')}")

    int8_modules = sorted(
        name for name, precision in candidate["module_precision_profile"].items() if precision == "INT8"
    )
    acceptance = {
        "schema_version": "heal-v2xvit-searched-subnet-deployment-acceptance-v1",
        "passed": True,
        "search_algorithm": candidate["search_algorithm"],
        "candidate_identity": candidate["candidate_identity"],
        "search_artifact_sha256": candidate["source_artifact_sha256"],
        "bops_target": candidate["target"],
        "actual_bops_retention": candidate["candidate_metrics"].get("R_bops_vs_fp32"),
        "all_keep_structure": not pruned_units,
        "pruned_unit_count": len(pruned_units),
        "parameter_count_before": physical.parameter_count_before,
        "parameter_count_after": physical.parameter_count_after,
        "parameter_reduction": physical.parameter_count_before - physical.parameter_count_after,
        "physical_checkpoint_exact": state_proof["all_tensors_exact"],
        "fixed_k": policy.fixed_k,
        "train200_entropy_calibration": calibration_metadata["frame_count"] == 200,
        "train200_entropy_calibration_status": (
            "passed"
            if int8_modules and calibration_metadata["frame_count"] == 200
            else "not_applicable_no_int8"
            if not int8_modules
            else "failed"
        ),
        "activation_output_boundary_policy": "next_weighted_input_owns_requantization_after_fp16_output",
        "per_channel_weight_qdq": bool(int8_modules),
        "per_channel_weight_qdq_status": (
            "passed" if int8_modules else "not_applicable_no_int8"
        ),
        "explicit_qdq_layer_count": qdq.inserted_layer_count,
        "requested_int8_modules": int8_modules,
        "strongly_typed_engine": True,
        "engine_sha256": file_sha256(engine_path),
        "precision_realization_passed": build["precision_realization_validation"]["passed"],
        "realized_int8_weighted_call_count": build["precision_realization_validation"][
            "realized_int8_count"
        ],
        "realized_fp16_weighted_call_count": build["precision_realization_validation"][
            "realized_fp16_count"
        ],
        "unresolved_engine_weighted_call_count": build["precision_realization_validation"][
            "unresolved_layer_count"
        ],
        "structure_validation_passed": build["engine_structure_validation"]["passed"],
        "physical_snapshot_hash": physical.snapshot_hash,
        "evaluation_frames": evaluation["num_evaluated_frames"],
        "evaluation_skipped": evaluation["num_skipped_frames"],
        "AP@0.3": evaluation["AP@0.3"],
        "AP@0.5": evaluation["AP@0.5"],
        "AP@0.7": evaluation["AP@0.7"],
        "mAP": evaluation["mAP"],
        "forward_p50_ms": evaluation.get("forward_p50_ms"),
    }
    _write_json(output / "acceptance.json", acceptance)
    print(json.dumps(acceptance, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
