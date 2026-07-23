#!/usr/bin/env python3
"""Bounded physical CNN+Transformer ONNX/TensorRT Stage-2 smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn

from scripts.audit_heal_transformer_search_models import MODEL_SPECS, _load, _sha256
from scripts.smoke_transformer_unified_search import (
    CNN_ROOT_TYPES,
    _choose_cnn_domain,
    _choose_transformer_domains,
    _merge_slices,
    _multi_agent_validation_batch,
)
from search.adapters.transformer_models import build_transformer_search_components
from search.model_family.deployment import build_physical_structure_snapshot_v2
from search.proxy.fisher_proxy import (
    collect_task_loss_fisher_statistics,
)
from search.proxy.parameter_slice_resolver import build_unit_parameter_slices
from search.proxy.runtime_shape_profiler import profile_runtime_layer_shapes
from search.proxy.transformer_parameter_slices import (
    build_transformer_unit_parameter_slices,
)
from search.pruning_space.domain_importance import (
    score_atomic_units_for_fixed_ranking,
)
from search.pruning_space.local_domains import build_local_pruning_domains
from search.pruning_space.transformer_domains import (
    fixed_transformer_rankings_from_unit_scores,
)
from search.pruning_space.unified_physical_pruner import (
    materialize_unified_widths,
)
from search.stage2.transformer_precision_export import (
    audit_onnx_attention_fp32_contract,
    audit_trt_attention_fp32_contract,
    build_transformer_precision_mapping,
)
from tracer.api import trace_model
from tracer.config import TraceConfig


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite_stage2_artifact:{path}")
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_fixed_k(
    observed_voxel_count: int,
    requested_fixed_k: int | None,
    contract_path: Path | None,
) -> tuple[int, dict[str, Any]]:
    fixed_k = int(requested_fixed_k or observed_voxel_count)
    if fixed_k < int(observed_voxel_count):
        raise RuntimeError(
            f"stage2_fixed_k_below_observed_voxel_count:{fixed_k}:"
            f"{observed_voxel_count}"
        )
    provenance: dict[str, Any] = {
        "observed_voxel_count": int(observed_voxel_count),
        "fixed_k": fixed_k,
        "source": "explicit_cli" if requested_fixed_k is not None else "observed_smoke_sample",
        "contract_path": "",
        "contract_sha256": "",
    }
    if contract_path is not None:
        contract = contract_path.expanduser().resolve()
        if not contract.is_file():
            raise RuntimeError(f"stage2_fixed_k_contract_missing:{contract}")
        payload = json.loads(contract.read_text(encoding="utf-8"))
        if int(payload.get("fixed_k", -1)) != fixed_k:
            raise RuntimeError(
                f"stage2_fixed_k_contract_mismatch:{payload.get('fixed_k')}:{fixed_k}"
            )
        provenance.update(
            {
                "source": "audited_fixed_k_contract",
                "contract_path": str(contract),
                "contract_sha256": _file_sha256(contract),
                "contract_hash": str(payload.get("contract_hash", "")),
                "overflow_count": int(payload.get("overflow_count", -1)),
            }
        )
    return fixed_k, provenance


def _collect_int8_smoke_calibration(
    model: nn.Module,
    adapter: Any,
    hypes: Mapping[str, Any],
    *,
    model_name: str,
    module_path: str,
    physical_structure_hash: str,
    device: torch.device,
    frame_count: int,
    seed: int = 20260723,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Collect bounded train-split scales on the exact physical candidate."""

    import random

    import numpy as np
    from opencood.data_utils.datasets import build_dataset

    from quantization.config import CalibrationConfig
    from quantization.precision.calibration import collect_calibration_scales
    from quantization.types import stable_json_hash
    from search.integration.data_provider import move_batch_to_device

    if int(frame_count) <= 0:
        raise RuntimeError("stage2_int8_calibration_frame_count_must_be_positive")
    dataset_hypes = adapter._absolutize_dataset_paths(dict(hypes))
    dataset = build_dataset(dataset_hypes, visualize=False, train=True)
    split_path = Path(str(dataset_hypes["root_dir"])).resolve()
    if not split_path.is_file():
        raise RuntimeError(f"stage2_int8_train_split_missing:{split_path}")
    split_ids = json.loads(split_path.read_text(encoding="utf-8"))
    batches = []
    selected_indices = []
    selected_frame_ids = []
    for index in range(len(dataset)):
        sample_seed = int(seed) + int(index)
        random.seed(sample_seed)
        np.random.seed(sample_seed % (2**32))
        torch.manual_seed(sample_seed)
        item = dataset[index]
        batch = dataset.collate_batch_train([item])
        if batch is None:
            continue
        batches.append(move_batch_to_device(batch, device))
        selected_indices.append(int(index))
        selected_frame_ids.append(
            str(split_ids[index]) if index < len(split_ids) else f"index:{index}"
        )
        if len(batches) == int(frame_count):
            break
    if len(batches) != int(frame_count):
        raise RuntimeError(
            f"stage2_int8_train_calibration_incomplete:{len(batches)}:{frame_count}"
        )
    result = collect_calibration_scales(
        model,
        batches,
        module_paths=(str(module_path),),
        forward_fn=adapter.forward_for_task,
        config=CalibrationConfig(
            split="train",
            frame_count=int(frame_count),
            require_observed_scales=True,
            activation_granularity="per_tensor",
            weight_granularity="per_tensor",
            schema_version="transformer-physical-smoke-calibration-v1",
        ),
    )
    scales = result.scales()
    manifest = {
        "schema_version": "transformer-physical-int8-smoke-manifest-v1",
        "model": str(model_name),
        "split": "train",
        "frame_count": int(frame_count),
        "frame_indices": selected_indices,
        "frame_ids": selected_frame_ids,
        "seed": int(seed),
        "dataset_manifest_path": str(split_path),
        "dataset_manifest_sha256": _file_sha256(split_path),
        "model_config_path": str(MODEL_SPECS[model_name]["config"]),
        "model_config_sha256": _sha256(MODEL_SPECS[model_name]["config"]),
        "checkpoint_path": str(MODEL_SPECS[model_name]["checkpoint"]),
        "checkpoint_sha256": _sha256(MODEL_SPECS[model_name]["checkpoint"]),
        "physical_structure_hash": str(physical_structure_hash),
        "module_paths": [str(module_path)],
        "activation_granularity": "per_tensor",
        "weight_granularity": "per_tensor",
        "smoothquant_applied": False,
        "smoothquant_reason": "bounded_ffn_int8_engine_smoke_not_qkv_projection",
    }
    manifest["manifest_hash"] = stable_json_hash(manifest)
    scale_payload = {
        "schema_version": "transformer-physical-int8-smoke-scales-v1",
        "calibration_manifest_hash": manifest["manifest_hash"],
        "scales": scales,
        "scale_hash": stable_json_hash(scales),
    }
    del batches
    torch.cuda.empty_cache()
    return scales, manifest, {**result.to_dict(), **scale_payload}


def _to_dict(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    return value


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
    legal = tuple(int(value) for value in domain.legal_widths)
    position = legal.index(int(domain.original_width))
    if position <= 0:
        raise RuntimeError(f"stage2_domain_has_no_lower_width:{domain.domain_id}")
    return legal[position - 1]


def _suffix_match(realized_path: str, requested_path: str) -> bool:
    return str(realized_path) == str(requested_path) or str(realized_path).endswith(
        f".{requested_path}"
    )


def _parity(
    reference: Mapping[str, torch.Tensor],
    actual: Sequence[torch.Tensor],
    names: Sequence[str],
) -> dict[str, Any]:
    rows = {}
    for name, observed in zip(names, actual):
        expected = reference[str(name)]
        delta = (expected.float() - observed.float()).abs()
        rows[str(name)] = {
            "expected_shape": list(expected.shape),
            "actual_shape": list(observed.shape),
            "max_abs": float(delta.max().item()),
            "mean_abs": float(delta.mean().item()),
            "finite": bool(torch.isfinite(observed).all().item()),
            "allclose": bool(
                expected.shape == observed.shape
                and torch.allclose(
                    expected.float(), observed.float(), atol=5.0e-3, rtol=1.0e-4
                )
            ),
        }
    return {
        "passed": len(rows) == len(tuple(names))
        and all(bool(row["allclose"] and row["finite"]) for row in rows.values()),
        "outputs": rows,
        "atol": 5.0e-3,
        "rtol": 1.0e-4,
    }


def _ranked_physical_candidate(
    model_name: str,
    device: torch.device,
) -> tuple[nn.Module, Any, Mapping[str, Any], Any, dict[str, Any]]:
    model, adapter, hypes, initial_batch = _load(model_name, device)
    batch, dataset_index, agents = _multi_agent_validation_batch(
        adapter, hypes, device
    )
    trace = trace_model(
        model,
        batch,
        config=TraceConfig(fail_on_fx_trace_error=False),
        forward_fn=adapter.forward_for_task,
    )
    runtime = profile_runtime_layer_shapes(
        model, batch, forward_fn=adapter.forward_for_task
    )
    active_paths = sorted({row.module_path for row in runtime.shapes})
    modules = dict(model.named_modules())
    cnn_units = [
        unit
        for unit in trace.atomic_prune_units
        if not bool(unit.protected)
        and isinstance(modules.get(unit.root_module_path), CNN_ROOT_TYPES)
        and unit.root_axis in {"out", "channel"}
        and bool(unit.root_indices)
    ]
    preliminary = build_local_pruning_domains(
        cnn_units,
        ranking_method="diagnostic_placeholder_before_common_task_loss_taylor",
        minimum_retained_ratio=0.10,
        dense_alignment=4,
    )
    selected_cnn_diagnostic = _choose_cnn_domain(preliminary)
    selected_ids = set(selected_cnn_diagnostic.ordered_unit_ids)
    selected_cnn_units = [
        unit for unit in cnn_units if unit.stable_id in selected_ids
    ]
    diagnostic_components = build_transformer_search_components(
        model,
        hypes,
        allow_identity_ranking=True,
        active_module_paths=active_paths,
    )
    cnn_slices = build_unit_parameter_slices(model, selected_cnn_units)
    transformer_slices = build_transformer_unit_parameter_slices(
        model, diagnostic_components.transformer_domains
    )
    all_slices = _merge_slices(cnn_slices, transformer_slices)
    calibration_hash = hashlib.sha256(
        json.dumps(
            {
                "model": model_name,
                "config": _sha256(MODEL_SPECS[model_name]["config"]),
                "checkpoint": _sha256(MODEL_SPECS[model_name]["checkpoint"]),
                "validation_dataset_index": dataset_index,
                "agents": agents,
                "sample_count": 1,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    fisher, fisher_report = collect_task_loss_fisher_statistics(
        model,
        (batch,),
        forward_fn=adapter.forward_for_task,
        loss_fn=adapter.compute_task_loss,
        calibration_manifest_hash=calibration_hash,
    )
    scores, ranking_report = score_atomic_units_for_fixed_ranking(
        model, fisher, all_slices, strict=False
    )
    formal_cnn = build_local_pruning_domains(
        selected_cnn_units,
        importance_scores=scores,
        ranking_method="raw_common_task_loss_first_plus_second_order_taylor",
        minimum_retained_ratio=0.10,
        dense_alignment=4,
    )
    if len(formal_cnn) != 1:
        raise RuntimeError(f"stage2_formal_cnn_domain_count:{len(formal_cnn)}")
    attention_rankings, ffn_rankings, transformer_ranking = (
        fixed_transformer_rankings_from_unit_scores(
            diagnostic_components.attention_instances,
            diagnostic_components.ffn_instances,
            scores,
        )
    )
    formal_components = build_transformer_search_components(
        model,
        hypes,
        attention_rankings=attention_rankings,
        ffn_rankings=ffn_rankings,
        active_module_paths=active_paths,
    )
    selected_transformer = _choose_transformer_domains(
        formal_components.transformer_domains
    )
    if len(selected_transformer) != 2:
        raise RuntimeError(
            f"stage2_attention_ffn_domain_count:{len(selected_transformer)}"
        )
    selected_domains = (formal_cnn[0], *selected_transformer)
    widths = {
        domain.domain_id: _next_lower(domain) for domain in selected_domains
    }
    first = materialize_unified_widths(
        model,
        selected_cnn_units,
        selected_domains,
        widths,
        model_name=model_name,
    )
    if not first.report.passed:
        raise RuntimeError(f"stage2_unified_physical_failed:{first.report.issues}")
    state = {
        name: value.detach().cpu() for name, value in first.model.state_dict().items()
    }
    replay = materialize_unified_widths(
        model,
        selected_cnn_units,
        selected_domains,
        widths,
        model_name=model_name,
    )
    replay.model.load_state_dict(state, strict=True)
    replay.model.eval()
    with torch.inference_mode():
        outputs = adapter.forward_for_task(replay.model, batch)
    tensors = list(_iter_tensors(outputs))
    finite = bool(tensors) and all(
        not value.is_floating_point()
        or bool(torch.isfinite(value).all().item())
        for value in tensors
    )
    if not finite:
        raise RuntimeError("stage2_physical_forward_nonfinite")
    evidence = {
        "schema_version": "transformer-unified-physical-stage2-candidate-v1",
        "model": model_name,
        "validation_dataset_index": dataset_index,
        "agent_count": agents,
        "calibration_manifest_hash": calibration_hash,
        "trace_backend": trace.config["realized_backend"],
        "trace_hash": trace.trace_hash,
        "fisher": fisher_report,
        "ranking": ranking_report,
        "transformer_ranking": transformer_ranking,
        "selected_domains": [domain.to_dict() for domain in selected_domains],
        "requested_widths": widths,
        "physical_report": replay.report.to_dict(),
        "cnn_plan": _to_dict(replay.cnn_plan),
        "cnn_ledger": _to_dict(replay.cnn_ledger),
        "cnn_validation": _to_dict(replay.cnn_validation),
        "strict_physical_state_reload": True,
        "finite_forward": finite,
    }
    return replay.model, adapter, hypes, batch, evidence


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type != "cuda" or device.index not in {None, 0}:
        raise RuntimeError("stage2_transformer_smoke_requires_visible_cuda0")
    torch.cuda.set_device(device)
    model, adapter, hypes, batch, physical = _ranked_physical_candidate(
        args.model, device
    )
    _write(output / "physical_candidate.json", physical)
    checkpoint = output / "physical_candidate_state_dict.pth"
    torch.save(
        {
            "model": {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            },
            "physical_structure_hash": physical["physical_report"][
                "structure_hash"
            ],
        },
        checkpoint,
    )
    selected_domains = physical["selected_domains"]
    attention_domain = next(
        row for row in selected_domains if row["domain_type"] == "attention_dh"
    )
    ffn_domain = next(
        row for row in selected_domains if row["domain_type"] == "ffn_hidden"
    )
    cnn_domain = next(
        row
        for row in selected_domains
        if row["domain_type"] in {"cnn_channel", "grouped_conv_channel"}
    )
    int8_requested_paths: set[str] = set()
    calibration_scales: dict[str, Any] = {}
    calibration_manifest: dict[str, Any] = {}
    if args.int8_role != "none":
        requested_member_role = {"ffn1": "first", "ffn2": "second"}[
            args.int8_role
        ]
        int8_requested_paths = {
            str(member["module_path"])
            for member in ffn_domain["dependency_members"]
            if str(member["role"]) == requested_member_role
        }
        if len(int8_requested_paths) != 1:
            raise RuntimeError(
                f"stage2_int8_role_path_count:{args.int8_role}:"
                f"{sorted(int8_requested_paths)}"
            )
        calibration_scales, calibration_manifest, calibration_report = (
            _collect_int8_smoke_calibration(
                model,
                adapter,
                hypes,
                model_name=args.model,
                module_path=next(iter(int8_requested_paths)),
                physical_structure_hash=physical["physical_report"][
                    "structure_hash"
                ],
                device=device,
                frame_count=int(args.calibration_frames),
            )
        )
        _write(output / "calibration_manifest.json", calibration_manifest)
        _write(output / "calibration_scales.json", calibration_report)
    observed_voxel_count = int(
        batch["ego"]["inputs_m1"]["voxel_features"].shape[0]
    )
    fixed_k, fixed_k_provenance = _resolve_fixed_k(
        observed_voxel_count,
        args.fixed_k,
        args.fixed_k_contract,
    )
    if args.model == "v2xvit":
        from search.model_family.export.heal_v2xvit import (
            HealV2XViTExportPolicy,
            build_heal_v2xvit_export_module,
            prepare_v2xvit_fixed_k_inputs,
        )

        policy = HealV2XViTExportPolicy(fixed_k=fixed_k, max_agents=2)
        wrapper = build_heal_v2xvit_export_module(model, policy=policy).eval()
        prepared = prepare_v2xvit_fixed_k_inputs(batch["ego"], policy=policy)
    elif args.model == "cobevt":
        from search.model_family.export.heal_lidar_baselines import (
            HealLidarBaselineExportPolicy,
            build_heal_lidar_baseline_export_module,
            prepare_heal_lidar_baseline_inputs,
        )

        policy = HealLidarBaselineExportPolicy(fixed_k=fixed_k, max_agents=2)
        wrapper = build_heal_lidar_baseline_export_module(model, policy=policy).eval()
        prepared = prepare_heal_lidar_baseline_inputs(batch["ego"], policy=policy)
    else:
        raise RuntimeError(f"stage2_engine_model_not_supported:{args.model}")

    input_names = tuple(prepared)
    output_names = tuple(policy.output_names)
    inputs = tuple(prepared[name] for name in input_names)
    with torch.inference_mode():
        reference = adapter.forward_for_task(model, batch)
        wrapper_outputs = wrapper(*inputs)
    parity = _parity(reference, wrapper_outputs, output_names)
    _write(output / "wrapper_parity.json", parity)
    if not parity["passed"]:
        raise RuntimeError(f"stage2_wrapper_parity_failed:{parity}")

    from quantization.export.origin_mapping import (
        apply_canonical_node_names,
        build_onnx_origin_map,
    )
    from quantization.export.signal_maxk import capture_weighted_module_calls

    onnx_path = output / "physical_fp32.onnx"
    with capture_weighted_module_calls(wrapper) as module_calls:
        torch.onnx.export(
            wrapper,
            inputs,
            str(onnx_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=list(input_names),
            output_names=list(output_names),
            custom_opsets={"trt": 1},
        )
    import onnx

    graph = onnx.load(str(onnx_path), load_external_data=False)
    onnx.checker.check_model(graph)
    inferred = onnx.shape_inference.infer_shapes(graph, strict_mode=False)
    inferred_path = output / "physical_fp32_inferred.onnx"
    onnx.save(inferred, str(inferred_path))
    origin_map = build_onnx_origin_map(onnx_path, module_calls)
    apply_canonical_node_names(
        onnx_path, origin_map, output_path=onnx_path, allow_custom_ops=True
    )
    onnx.checker.check_model(onnx.load(str(onnx_path), load_external_data=False))

    fp16_requested_paths = {
        str(cnn_domain["module_path"]),
        *(
            str(member["module_path"])
            for member in ffn_domain["dependency_members"]
        ),
    }
    profile = {
        str(row.module_path): (
            "int8"
            if any(
                _suffix_match(str(row.module_path), path)
                for path in int8_requested_paths
            )
            else "fp16"
            if any(
                _suffix_match(str(row.module_path), path)
                for path in fp16_requested_paths
            )
            else "fp32"
        )
        for row in origin_map.entries
    }
    if not any(value == "fp16" for value in profile.values()):
        raise RuntimeError(
            f"stage2_fp16_modules_not_realized:{sorted(fp16_requested_paths)}"
        )
    requested_int8_calls = sum(value == "int8" for value in profile.values())
    if args.int8_role != "none" and requested_int8_calls <= 0:
        raise RuntimeError(
            f"stage2_int8_modules_not_realized:{sorted(int8_requested_paths)}"
        )
    mapping = build_transformer_precision_mapping(
        origin_map,
        profile,
        profile_id=f"{args.model}_physical_dh_dff_cnn_mixed_w16_smoke",
    )
    qkv_paths = {
        str(member["module_path"])
        for member in attention_domain["dependency_members"]
        if member["role"] in {"q", "k", "v"}
    }
    qkv_nodes = sorted(
        {
            str(row.canonical_node_name)
            for row in origin_map.entries
            if any(
                _suffix_match(str(row.module_path), path) for path in qkv_paths
            )
        }
    )
    if not qkv_nodes:
        raise RuntimeError(f"stage2_qkv_origin_mapping_missing:{sorted(qkv_paths)}")
    onnx_attention = audit_onnx_attention_fp32_contract(
        onnx_path, qkv_canonical_node_names=qkv_nodes
    )
    _write(output / "onnx_attention_fp32_audit.json", onnx_attention)
    if not onnx_attention["passed"]:
        raise RuntimeError(f"stage2_onnx_qk_fp32_contract_failed:{onnx_attention}")
    _write(output / "canonical_origin_map.json", origin_map.to_dict())
    _write(output / "canonical_precision_mapping.json", mapping.to_dict())

    from quantization.config import QDQConfig
    from quantization.precision.qdq_inserter import insert_explicit_qdq

    qdq_path = output / "physical_mixed_fp16_qdq.onnx"
    qdq = insert_explicit_qdq(
        onnx_path,
        qdq_path,
        mapping,
        scales=calibration_scales,
        config=QDQConfig(
            allowed_precisions=("fp32", "fp16", "int8"),
            require_calibration_scales=bool(requested_int8_calls),
            insert_activation_input_qdq=True,
            insert_weight_qdq=True,
            insert_activation_output_qdq=False,
            merge_policy="fp16_merge",
            explicit_fp16_compute_casts=True,
            explicit_fp32_compute_casts=True,
            policy_version="transformer-mixed-w8a8-fp16-qk-fp32-strongly-typed-v1",
        ),
        calibration_metadata={
            "schema_version": (
                "transformer-physical-int8-smoke-calibration-v1"
                if requested_int8_calls
                else "no-int8-calibration-required-v1"
            ),
            "physical_structure_hash": physical["physical_report"][
                "structure_hash"
            ],
            "calibration_manifest_hash": calibration_manifest.get(
                "manifest_hash", ""
            ),
        },
    )
    if int(qdq.inserted_layer_count) != int(requested_int8_calls):
        raise RuntimeError(
            f"stage2_qdq_int8_count_mismatch:{qdq.inserted_layer_count}:"
            f"{requested_int8_calls}"
        )
    _write(output / "qdq_insertion_report.json", qdq.to_dict())
    onnx.checker.check_model(onnx.load(str(qdq_path), load_external_data=False))
    qdq_attention = audit_onnx_attention_fp32_contract(
        qdq_path, qkv_canonical_node_names=qkv_nodes
    )
    _write(output / "qdq_attention_fp32_audit.json", qdq_attention)
    if not qdq_attention["passed"]:
        raise RuntimeError(f"stage2_qdq_qk_fp32_contract_failed:{qdq_attention}")

    engine_result: dict[str, Any] = {"requested": bool(args.build_engine)}
    if args.build_engine:
        if args.plugin is None or not args.plugin.is_file():
            raise RuntimeError(f"stage2_plugin_missing:{args.plugin}")
        from quantization.config import TensorRTBuildConfig
        from search.stage2.trt_modelopt import build_engine_modelopt

        trtexec = args.tensorrt_root / "bin/trtexec"
        if not trtexec.is_file():
            trtexec = (
                args.tensorrt_root
                / "targets/x86_64-linux-gnu/bin/trtexec"
            )
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
            policy_version="transformer-mixed-w8a8-fp16-qk-fp32-strongly-typed-v1",
        )
        snapshot = build_physical_structure_snapshot_v2(
            model, model_family=f"heal_lidar_{args.model}"
        )
        engine_path = output / "candidate.plan"
        build = build_engine_modelopt(
            qdq_onnx=qdq_path,
            engine_path=engine_path,
            precision_mapping=mapping,
            build_config=build_config,
            physical_snapshot=snapshot,
            output_dir=output / "engine_build",
            tensorrt_root=args.tensorrt_root,
            conda_env="modelopt",
            gpu_id=0,
        )
        _write(output / "engine_build_acceptance.json", build)
        if build.get("status") != "ok":
            raise RuntimeError(
                f"stage2_engine_build_failed:{build.get('status')}:"
                f"{build.get('failure_reason', '')}"
            )
        precision_realization = dict(
            build.get("precision_realization_validation", {}) or {}
        )
        if (
            int(precision_realization.get("requested_int8_count", -1))
            != int(requested_int8_calls)
            or int(precision_realization.get("realized_int8_count", -1))
            != int(requested_int8_calls)
        ):
            raise RuntimeError(
                "stage2_requested_realized_int8_conflict:"
                f"{precision_realization}:expected={requested_int8_calls}"
            )
        trt_attention = audit_trt_attention_fp32_contract(
            output / "engine_build/engine_layer_info.json", qdq_attention
        )
        _write(output / "trt_attention_fp32_audit.json", trt_attention)
        if not trt_attention["passed"]:
            raise RuntimeError(
                f"stage2_trt_qk_fp32_contract_failed:{trt_attention}"
            )
        engine_result = {
            "requested": True,
            "passed": True,
            "engine_path": str(engine_path),
            "engine_sha256": _file_sha256(engine_path),
            "size_bytes": engine_path.stat().st_size,
            "trt_attention_fp32": trt_attention,
            "precision_realization": precision_realization,
        }

    acceptance = {
        "schema_version": "transformer-unified-stage2-smoke-v1",
        "model": args.model,
        "passed": True,
        "strict_checkpoint_load": True,
        "strict_physical_state_reload": True,
        "physical_widths_requested_realized_exact": True,
        "mask_only": False,
        "hidden_padding": False,
        "wrapper_parity": True,
        "onnx_checker": True,
        "onnx_shape_inference": True,
        "qk_onnx_fp32": True,
        "softmax_compute_onnx_fp32": True,
        "mixed_fp16_weighted_unit_count": sum(
            value == "fp16" for value in profile.values()
        ),
        "requested_int8_weighted_call_count": int(requested_int8_calls),
        "int8_role": str(args.int8_role),
        "int8_module_paths": sorted(int8_requested_paths),
        "w8a8_weight_activation_bound": bool(requested_int8_calls),
        "calibration_manifest_hash": calibration_manifest.get("manifest_hash", ""),
        "fixed_k_provenance": fixed_k_provenance,
        "engine": engine_result,
        "full1789_executed": False,
        "formal_full_search_executed": False,
    }
    _write(output / "acceptance.json", acceptance)
    print(json.dumps(acceptance, indent=2, sort_keys=True), flush=True)
    return acceptance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("v2xvit", "cobevt"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fixed-k", type=int)
    parser.add_argument("--fixed-k-contract", type=Path)
    parser.add_argument(
        "--int8-role", choices=("none", "ffn1", "ffn2"), default="none"
    )
    parser.add_argument("--calibration-frames", type=int, default=4)
    parser.add_argument(
        "--tensorrt-root",
        type=Path,
        default=Path("/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118"),
    )
    parser.add_argument("--plugin", type=Path)
    parser.add_argument("--build-engine", action="store_true")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
