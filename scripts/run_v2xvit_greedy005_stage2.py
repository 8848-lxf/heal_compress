#!/usr/bin/env python3
"""Materialize and export the V2X-ViT R_BOPS=0.05 Greedy Stage-2 pool.

This script deliberately consumes the frozen Stage-1 genotype artifacts.  It
does not run search, repair, budget projection, or GA.  Every candidate is
materialized with all 20 CNN, 12 attention and 3 FFN domains before any export
is attempted; requested/realized conflicts fail closed.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

import torch
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_heal_transformer_search_models import MODEL_SPECS, _load, _sha256
from scripts.smoke_transformer_unified_search import _multi_agent_validation_batch
from scripts.analyze_v2xvit_greedy005_bops_floor import _build_full_space
from scripts.run_v2xvit_greedy005_full import _formal_space
from search.candidate import CandidateGenotype
from search.canonicalization import canonicalize_candidate, repair_genotype
from search.hashing import candidate_hash as canonical_candidate_hash
from search.model_family.deployment import build_physical_structure_snapshot_v2
from search.pruning_space.local_domains import LocalPruningDomain
from search.pruning_space.unified_physical_pruner import materialize_unified_widths


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensors(value: Any):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _tensors(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _tensors(child)


def _finite_forward(model: Any, adapter: Any, batch: Any) -> tuple[bool, dict[str, Any]]:
    model.eval()
    with torch.inference_mode():
        output = adapter.forward_for_task(model, batch)
    tensors = list(_tensors(output))
    finite = bool(tensors) and all(
        (not tensor.is_floating_point()) or bool(torch.isfinite(tensor).all().item())
        for tensor in tensors
    )
    return finite, {"tensor_count": len(tensors), "finite": finite}


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _frozen_domain(payload: Mapping[str, Any]) -> LocalPruningDomain:
    """Restore one immutable Stage-1 domain without recomputing its ranking."""
    value = dict(payload)
    value.pop("width_semantics", None)
    for key in ("ordered_unit_ids", "legal_widths", "precision_units", "dependency_members"):
        value[key] = tuple(value.get(key, ()))
    value["unit_root_indices"] = {
        str(key): tuple(int(item) for item in items)
        for key, items in dict(value.get("unit_root_indices", {})).items()
    }
    value["width_to_pruned_unit_ids"] = {
        int(key): tuple(str(item) for item in items)
        for key, items in dict(value.get("width_to_pruned_unit_ids", {})).items()
    }
    value["ordered_unit_ids_by_group"] = {
        int(key): tuple(str(item) for item in items)
        for key, items in dict(value.get("ordered_unit_ids_by_group", {})).items()
    }
    value["group_local_indices"] = {
        int(key): {str(unit): int(index) for unit, index in dict(items).items()}
        for key, items in dict(value.get("group_local_indices", {})).items()
    }
    for field in ("group_keep_maps", "group_prune_maps"):
        value[field] = {
            int(width): {int(group): [int(item) for item in items] for group, items in dict(mapping).items()}
            for width, mapping in dict(value.get(field, {})).items()
        }
    return LocalPruningDomain(**value)


def _profile_for_origin(phenotype: Any, origin_map: Any) -> dict[str, str]:
    """Map candidate module precision to every realized ONNX call."""
    requested = phenotype.realized_precision_profile
    profile: dict[str, str] = {}
    unknown: list[str] = []
    for entry in origin_map.entries:
        path = str(entry.module_path)
        matches = [value for key, value in requested.items() if path == key or path.endswith(f".{key}")]
        if len(set(matches)) > 1:
            raise RuntimeError(f"precision_profile_ambiguous:{path}:{matches}")
        profile[path] = (matches[0] if matches else "FP32").lower()
        if not matches:
            unknown.append(path)
    return profile


def _force_attention_fp32_contract(path: Path, attention_audit: Mapping[str, Any]) -> dict[str, Any]:
    """Insert explicit FLOAT casts at QK/Softmax boundaries after Q/DQ.

    INT8/FP16 projection outputs are allowed, but the contract requires the
    QK operands, QK result and Softmax input/output to be FLOAT.  The generic
    Q/DQ inserter intentionally leaves this graph-level protected boundary to
    the Transformer exporter, so this explicit, auditable pass closes it.
    """
    import onnx
    from onnx import helper
    graph = onnx.load(str(path), load_external_data=False).graph
    target_names = {
        str(row["node_name"]): "qk" for row in attention_audit.get("qk_nodes", ())
    }
    target_names.update({str(row["node_name"]): "softmax" for row in attention_audit.get("softmax_nodes", ())})
    softmax_output_types = {
        str(row["node_name"]): int((row.get("output_element_types") or [1])[0])
        for row in attention_audit.get("softmax_nodes", ())
    }
    nodes = list(graph.node)
    by_name = {str(node.name): node for node in nodes}
    consumers: dict[str, list[Any]] = {}
    for node in nodes:
        for input_name in node.input:
            consumers.setdefault(str(input_name), []).append(node)
    softmax_names = {
        str(row["node_name"]) for row in attention_audit.get("softmax_nodes", ())
    }
    # Promote the complete logits path, including scale/mask elementwise
    # operators.  TensorRT strongly typed mode rejects a FLOAT QK result fed to
    # a HALF scale/mask peer even if QK and Softmax themselves are protected.
    for qk_row in attention_audit.get("qk_nodes", ()):
        qk = by_name.get(str(qk_row["node_name"]))
        if qk is None:
            continue
        queue = [(str(output), 0) for output in qk.output]
        seen: set[str] = set()
        while queue:
            tensor, depth = queue.pop(0)
            if tensor in seen or depth > 96:
                continue
            seen.add(tensor)
            for consumer in consumers.get(tensor, ()):
                name = str(consumer.name)
                if name in softmax_names:
                    continue
                target_names.setdefault(name, "attention_logits")
                queue.extend((str(output), depth + 1) for output in consumer.output)
    output_insertions: dict[int, list[Any]] = {}
    input_insertions: dict[int, list[Any]] = {}
    inserted = 0
    for index, node in enumerate(nodes):
        role = target_names.get(str(node.name))
        if role is None:
            continue
        for input_index, source in enumerate(list(node.input)):
            if str(node.op_type) == "Where" and input_index == 0:
                # The condition is a protected boolean control tensor; only
                # the two selected data branches are promoted to FLOAT.
                continue
            cast_out = f"{source}__{role}_fp32_input"
            cast_name = f"{node.name}__{role}_fp32_input_cast_{input_index}"
            input_insertions.setdefault(index, []).append(helper.make_node("Cast", [source], [cast_out], name=cast_name, to=1))
            node.input[input_index] = cast_out
            inserted += 1
        for output_index, source in enumerate(list(node.output)):
            raw = f"{source}__{role}_raw"
            node.output[output_index] = raw
            cast_name = f"{node.name}__{role}_fp32_output_cast_{output_index}"
            output_type = softmax_output_types.get(str(node.name), 1) if role == "softmax" else 1
            output_insertions.setdefault(index, []).append(helper.make_node("Cast", [raw], [source], name=cast_name, to=output_type))
            inserted += 1
    rebuilt: list[Any] = []
    for index, node in enumerate(nodes):
        rebuilt.extend(input_insertions.get(index, ()))
        rebuilt.append(node)
        rebuilt.extend(output_insertions.get(index, ()))
    del graph.node[:]
    graph.node.extend(rebuilt)
    model = onnx.load(str(path), load_external_data=False)
    model.graph.CopyFrom(graph)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return {
        "inserted_cast_count": inserted,
        "roles": {
            role: sum(value == role for value in target_names.values())
            for role in ("qk", "attention_logits", "softmax")
        },
    }


def _export_candidate(
    candidate_dir: Path,
    model: Any,
    adapter: Any,
    batch: Any,
    hypes: Mapping[str, Any],
    phenotype: Any,
    candidate_hash: str,
    physical_structure_hash: str,
    *,
    build_engine: bool,
    tensorrt_root: Path,
    plugin: Path | None,
    calibration_frames: int,
    qkv_paths: Sequence[str],
    fixed_k_override: int | None,
) -> dict[str, Any]:
    """Export one physical candidate; every failure is returned and persisted."""
    result: dict[str, Any] = {"candidate_hash": candidate_hash, "onnx": {"attempted": False}, "engine": {"attempted": False}}
    try:
        from search.model_family.export.heal_v2xvit import (
            HealV2XViTExportPolicy,
            build_heal_v2xvit_export_module,
            prepare_v2xvit_fixed_k_inputs,
        )
        observed_fixed_k = int(batch["ego"]["inputs_m1"]["voxel_features"].shape[0])
        fixed_k = int(fixed_k_override or observed_fixed_k)
        if fixed_k < observed_fixed_k:
            raise RuntimeError(f"fixed_k_below_observed:{fixed_k}:{observed_fixed_k}")
        policy = HealV2XViTExportPolicy(fixed_k=fixed_k, max_agents=2)
        wrapper = build_heal_v2xvit_export_module(model, policy=policy).eval()
        prepared = prepare_v2xvit_fixed_k_inputs(batch["ego"], policy=policy)
        inputs = tuple(prepared[name] for name in prepared)
        with torch.inference_mode():
            wrapped = wrapper(*inputs)
        result["wrapper_output_count"] = len(tuple(_tensors(wrapped)))
        from quantization.export.origin_mapping import apply_canonical_node_names, build_onnx_origin_map
        from quantization.export.signal_maxk import capture_weighted_module_calls
        onnx_path = candidate_dir / "physical_fp32.onnx"
        with capture_weighted_module_calls(wrapper) as module_calls:
            torch.onnx.export(
                wrapper, inputs, str(onnx_path), export_params=True, opset_version=17,
                do_constant_folding=True, input_names=list(prepared),
                output_names=list(policy.output_names), custom_opsets={"trt": 1},
            )
        import onnx
        graph = onnx.load(str(onnx_path), load_external_data=False)
        onnx.checker.check_model(graph)
        inferred = onnx.shape_inference.infer_shapes(graph, strict_mode=False)
        inferred_path = candidate_dir / "physical_fp32_inferred.onnx"
        onnx.save(inferred, str(inferred_path))
        origin_map = build_onnx_origin_map(onnx_path, module_calls)
        apply_canonical_node_names(onnx_path, origin_map, output_path=onnx_path, allow_custom_ops=True)
        onnx.checker.check_model(onnx.load(str(onnx_path), load_external_data=False))
        profile = _profile_for_origin(phenotype, origin_map)
        from search.stage2.transformer_precision_export import build_transformer_precision_mapping, audit_onnx_attention_fp32_contract, audit_trt_attention_fp32_contract
        mapping = build_transformer_precision_mapping(origin_map, profile, profile_id=f"v2xvit_greedy005_{candidate_hash[:12]}")
        qkv_nodes = [str(entry.canonical_node_name) for entry in origin_map.entries if any(str(entry.module_path) == path or str(entry.module_path).endswith(f".{path}") for path in qkv_paths)]
        if not qkv_nodes:
            raise RuntimeError("onnx_qkv_origin_mapping_missing")
        qk_audit = audit_onnx_attention_fp32_contract(onnx_path, qkv_canonical_node_names=qkv_nodes)
        _write(candidate_dir / "canonical_origin_map.json", origin_map.to_dict())
        _write(candidate_dir / "canonical_precision_mapping.json", mapping.to_dict())
        _write(candidate_dir / "onnx_attention_fp32_audit.json", qk_audit)
        if not qk_audit.get("passed"):
            raise RuntimeError("onnx_qk_softmax_fp32_contract_failed")
        result["onnx"] = {"attempted": True, "passed": True, "path": str(onnx_path), "sha256": _sha256_file(onnx_path), "origin_entries": len(origin_map.entries), "profile_counts": {state: sum(value == state for value in profile.values()) for state in ("fp32", "fp16", "int8")}, "qk_audit": qk_audit}
        # QDQ/engine is attempted only when explicitly requested.  Calibration
        # is collected on this exact physical structure; no historical scale is
        # ever reused.
        if build_engine:
            from opencood.data_utils.datasets import build_dataset
            from quantization.config import CalibrationConfig, QDQConfig, TensorRTBuildConfig
            from quantization.precision.calibration import collect_calibration_scales
            from quantization.precision.qdq_inserter import insert_explicit_qdq
            from quantization.types import stable_json_hash
            from search.integration.data_provider import move_batch_to_device
            dataset_hypes = adapter._absolutize_dataset_paths(dict(hypes))
            dataset = build_dataset(dataset_hypes, visualize=False, train=True)
            split_path = Path(str(dataset_hypes["root_dir"])).resolve()
            split_ids = json.loads(split_path.read_text(encoding="utf-8"))
            batches = []
            frame_ids = []
            for index in range(len(dataset)):
                item = dataset[index]
                train_batch = dataset.collate_batch_train([item])
                if train_batch is None:
                    continue
                batches.append(move_batch_to_device(train_batch, next(model.parameters()).device))
                frame_ids.append(str(split_ids[index]) if index < len(split_ids) else f"index:{index}")
                if len(batches) >= int(calibration_frames):
                    break
            if len(batches) != int(calibration_frames):
                raise RuntimeError(f"structure_specific_calibration_incomplete:{len(batches)}:{calibration_frames}")
            int8_paths = sorted(path for path, value in profile.items() if value == "int8")
            scales_result = collect_calibration_scales(
                model, batches, module_paths=int8_paths,
                forward_fn=adapter.forward_for_task,
                config=CalibrationConfig(split="train", frame_count=len(batches), require_observed_scales=True, schema_version="v2xvit-greedy005-physical-calibration-v1"),
            )
            scales = scales_result.scales()
            calibration_manifest = {
                "schema_version": "v2xvit-greedy005-physical-calibration-v1",
                "split": "train", "frame_count": len(batches), "frame_ids": frame_ids,
                "module_paths": int8_paths,
                "dataset_manifest_path": str(split_path),
                "dataset_manifest_sha256": _sha256_file(split_path),
                "checkpoint_sha256": _sha256(MODEL_SPECS["v2xvit"]["checkpoint"]),
                "physical_structure_hash": physical_structure_hash,
                "scale_hash": stable_json_hash(scales),
            }
            calibration_manifest["manifest_hash"] = stable_json_hash(calibration_manifest)
            _write(candidate_dir / "calibration_manifest.json", calibration_manifest)
            _write(candidate_dir / "calibration_scales.json", {"scales": scales, "scale_hash": stable_json_hash(scales), "calibration_manifest_hash": calibration_manifest["manifest_hash"]})
            qdq_path = candidate_dir / "physical_mixed_qdq.onnx"
            qdq = insert_explicit_qdq(
                onnx_path, qdq_path, mapping, scales=scales,
                config=QDQConfig(allowed_precisions=("fp32", "fp16", "int8"), require_calibration_scales=True, insert_activation_input_qdq=True, insert_weight_qdq=True, insert_activation_output_qdq=False, merge_policy="fp16_merge", explicit_fp16_compute_casts=True, explicit_fp32_compute_casts=True, policy_version="v2xvit-greedy005-w8a8-qk-fp32-v1"),
                calibration_metadata={"calibration_manifest_hash": calibration_manifest["manifest_hash"], "physical_structure_hash": physical_structure_hash},
            )
            _write(candidate_dir / "qdq_insertion_report.json", qdq.to_dict())
            import onnx
            onnx.checker.check_model(onnx.load(str(qdq_path), load_external_data=False))
            qdq_pre_cast_audit = audit_onnx_attention_fp32_contract(qdq_path, qkv_canonical_node_names=qkv_nodes)
            cast_report = _force_attention_fp32_contract(qdq_path, qdq_pre_cast_audit)
            _write(candidate_dir / "qk_fp32_cast_report.json", {"pre_cast_audit": qdq_pre_cast_audit, **cast_report})
            qdq_audit = audit_onnx_attention_fp32_contract(qdq_path, qkv_canonical_node_names=qkv_nodes)
            _write(candidate_dir / "qdq_attention_fp32_audit.json", qdq_audit)
            if not qdq_audit.get("passed"):
                raise RuntimeError("qdq_qk_softmax_fp32_contract_failed")
            if plugin is None or not plugin.is_file():
                raise RuntimeError(f"trt_plugin_missing:{plugin}")
            from search.stage2.trt_modelopt import build_engine_modelopt
            trtexec = tensorrt_root / "bin/trtexec"
            if not trtexec.is_file():
                trtexec = tensorrt_root / "targets/x86_64-linux-gnu/bin/trtexec"
            build = build_engine_modelopt(
                qdq_onnx=qdq_path, engine_path=candidate_dir / "candidate.plan",
                precision_mapping=mapping,
                build_config=TensorRTBuildConfig(trtexec_path=trtexec, plugin_path=plugin, workspace_mib=4096, timeout_seconds=3600, no_tf32=True, skip_inference=True, export_layer_info=True, strongly_typed=True, enable_fp16=False, enable_int8=False, policy_version="v2xvit-greedy005-w8a8-qk-fp32-v1"),
                physical_snapshot=build_physical_structure_snapshot_v2(model, model_family="heal_lidar_v2xvit"),
                output_dir=candidate_dir / "engine_build", tensorrt_root=tensorrt_root,
                conda_env="modelopt", gpu_id=0,
            )
            _write(candidate_dir / "engine_build_acceptance.json", build)
            if build.get("status") != "ok":
                raise RuntimeError(f"trt_build_failed:{build.get('status')}:{build.get('failure_reason','')}")
            trt_attention = audit_trt_attention_fp32_contract(
                candidate_dir / "engine_build" / "engine_layer_info.json", qdq_audit
            )
            _write(candidate_dir / "trt_attention_fp32_audit.json", trt_attention)
            if not trt_attention.get("passed"):
                raise RuntimeError("trt_qk_softmax_fp32_contract_failed")
            result["engine"] = {"attempted": True, "passed": True, "build": build, "trt_attention": trt_attention}
            del batches
            torch.cuda.empty_cache()
    except Exception as exc:
        result.setdefault("onnx", {}).setdefault("attempted", True)
        result["passed"] = False
        result["failure"] = f"{type(exc).__name__}:{exc}"
        _write(candidate_dir / "stage2_failure.json", result)
        return result
    result["passed"] = True
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.output_root.resolve()
    stage2_root = root / "structures" / str(args.stage2_dir_name)
    stage2_root.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    if device.type != "cuda" or device.index not in {None, 0}:
        raise RuntimeError("stage2_requires_visible_cuda0")
    torch.cuda.set_device(device)
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    torch.cuda.manual_seed_all(int(args.seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model, adapter, hypes, _ = _load("v2xvit", device)
    batch, dataset_index, agent_count = _multi_agent_validation_batch(adapter, hypes, device)
    identity = _build_full_space(model, adapter, hypes, batch)
    manifest = json.loads((root / "greedy" / "v2xvit_greedy_search_manifest.json").read_text(encoding="utf-8"))
    formal = _formal_space(model, adapter, hypes, batch, identity, str(manifest["calibration_manifest_hash"]))
    frozen_ranking = json.loads(
        (root / "rankings" / "v2xvit_fixed_rankings.json").read_text(encoding="utf-8")
    )
    frozen_domains = tuple(_frozen_domain(row) for row in frozen_ranking["domains"])
    formal["space"] = replace(formal["space"], pruning_domains=frozen_domains)
    rows = _load_rows(root / "greedy" / "v2xvit_stage2_top5_selection.csv")
    if args.max_candidates is not None:
        rows = rows[: int(args.max_candidates)]
    reports: list[dict[str, Any]] = []
    qkv_paths = tuple(
        path
        for spec in formal["components"].attention_instances
        for path in (spec.q_projection_paths + spec.k_projection_paths)
    )
    for row in rows:
        candidate_hash = str(row["candidate_hash"])
        candidate_dir = stage2_root / candidate_hash
        candidate_dir.mkdir(parents=True, exist_ok=False)
        genotype = CandidateGenotype.from_dict(json.loads(row["genotype_json"]))
        candidate = repair_genotype(genotype, formal["space"])
        phenotype = canonicalize_candidate(candidate, formal["space"])
        realized_candidate_hash = canonical_candidate_hash(phenotype, formal["space"])
        hash_replay_exact = realized_candidate_hash == candidate_hash
        if row["candidate_id"] == "stage2_01":
            frozen_winner = json.loads(
                (root / "greedy" / "v2xvit_greedy_winner_config.json").read_text(encoding="utf-8")
            )["phenotype"]
            frozen_metadata = frozen_winner["metadata"]
            if (
                phenotype.pruned_unit_ids != frozen_winner["pruned_unit_ids"]
                or phenotype.realized_precision_profile != frozen_winner["realized_precision_profile"]
                or phenotype.metadata.get("domain_width_expansion_hash")
                != frozen_metadata.get("domain_width_expansion_hash")
            ):
                raise RuntimeError("stage2_frozen_stage1_phenotype_mismatch")
        physical = materialize_unified_widths(
            model, identity["cnn_units"], formal["space"].pruning_domains,
            candidate.pruning_width_genes, model_name="lidar_v2xvit",
        )
        report = physical.report.to_dict()
        if not physical.report.passed:
            raise RuntimeError(f"physical_materialization_failed:{candidate_hash}:{physical.report.issues}")
        state_path = candidate_dir / "physical_state_dict.pth"
        torch.save({"model": {name: value.detach().cpu() for name, value in physical.model.state_dict().items()}, "structure_hash": physical.report.structure_hash}, state_path)
        replay = materialize_unified_widths(
            model, identity["cnn_units"], formal["space"].pruning_domains,
            candidate.pruning_width_genes, model_name="lidar_v2xvit",
        )
        replay.model.load_state_dict(torch.load(state_path, map_location="cpu")["model"], strict=True)
        finite, forward_report = _finite_forward(replay.model, adapter, batch)
        if not finite:
            raise RuntimeError(f"physical_forward_nonfinite:{candidate_hash}")
        requested_realized = {"requested_widths": report["requested_widths"], "realized_widths": report["realized_widths"], "exact": report["requested_widths"] == report["realized_widths"], "structure_hash": report["structure_hash"], "state_dict_shape_hash": report["state_dict_shape_hash"], "mask_only": report["mask_only"], "hidden_padding": report["hidden_padding"]}
        _write(candidate_dir / "physical_candidate.json", {"candidate_id": row["candidate_id"], "candidate_hash": candidate_hash, "candidate_hash_replay": realized_candidate_hash, "candidate_hash_replay_exact": hash_replay_exact, "candidate_hash_replay_note": "stage1 trace snapshot hash is not serialized in the Greedy manifest" if not hash_replay_exact else "", "genotype": candidate.to_dict(), "phenotype": phenotype.to_dict(), "physical_report": report, "forward": forward_report, "dataset_index": dataset_index, "agent_count": agent_count, "calibration_manifest_hash": manifest["calibration_manifest_hash"]})
        _write(candidate_dir / "requested_vs_realized.json", requested_realized)
        export = {"passed": False, "skipped": True, "reason": "not_top1"}
        if row["candidate_id"] == "stage2_01" or args.export_all:
            export = _export_candidate(candidate_dir, replay.model, adapter, batch, hypes, phenotype, candidate_hash, report["structure_hash"], build_engine=bool(args.build_engine and row["candidate_id"] == "stage2_01"), tensorrt_root=args.tensorrt_root, plugin=args.plugin, calibration_frames=args.calibration_frames, qkv_paths=qkv_paths, fixed_k_override=args.fixed_k)
        reports.append({"candidate_id": row["candidate_id"], "candidate_hash": candidate_hash, "bops_retention": float(row["bops_retention"]), "physical_passed": True, "finite_forward": finite, "requested_realized_exact": requested_realized["exact"], "structure_hash": report["structure_hash"], "parameter_count": report["physical_parameter_count"], "export": export})
        del physical, replay
        torch.cuda.empty_cache()
    summary = {"schema_version": "v2xvit-greedy005-stage2-v1", "model": "lidar_v2xvit", "candidate_count": len(reports), "reports": reports, "all_physical_passed": all(row["physical_passed"] for row in reports), "all_requested_realized_exact": all(row["requested_realized_exact"] for row in reports), "engine_success_count": sum(bool(row["export"].get("engine", {}).get("passed")) for row in reports), "full1789_executed": False, "formal_ga_search_executed": False, "six_budget_search_executed": False}
    _write(root / "reports" / f"{args.stage2_dir_name}_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--stage2-dir-name", default="v2xvit_greedy005_stage2")
    parser.add_argument("--export-all", action="store_true")
    parser.add_argument("--build-engine", action="store_true")
    parser.add_argument("--calibration-frames", type=int, default=4)
    parser.add_argument("--fixed-k", type=int)
    parser.add_argument("--tensorrt-root", type=Path, default=Path("/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118"))
    parser.add_argument("--plugin", type=Path)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
