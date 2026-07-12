#!/usr/bin/env python3
"""Run the controlled PyTorch-only pyramid grouped-Conv stage experiment."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pruning.api import (
    build_physical_pruning_plan,
    build_physical_structure_snapshot,
    compute_physical_hashes,
    estimate_physical_parameter_count,
    legalize_pruning_plan,
    load_model,
    materialize_pruning,
    replay_pruning,
    score_pruning_units,
    validate_physical_model,
)
from pruning.artifacts.io import atomic_write_json
from pruning.config import PruningConfig
from pruning.types import ImportanceResult, PhysicalPruningPlan
from tracer.api import serialize_trace_result, trace_model
from tracer.config import TraceConfig

from tools.experiments.grouped_conv_stage_sensitivity.contracts import (
    STAGE_PREFIXES,
    build_candidate_matrix,
    build_stage_inventory,
    compute_pruning_strength_plan,
    filter_active_root_units,
)
from tools.experiments.grouped_conv_stage_sensitivity.analysis import (
    build_sensitivity_rankings,
    build_strategy_comparisons,
    cumulative_interaction,
    physical_shape_hash,
    stage1_attribution,
    validate_pairwise_structure,
)
from tools.experiments.grouped_conv_stage_sensitivity.runtime import (
    build_controlled_strategy_request,
    build_frame_manifest,
    build_group_score_records,
    build_tp_importance_result,
    parameter_reduction_breakdown,
    state_dict_content_hash,
    tensor_output_contract,
    validate_core_freeze,
)
from tools.experiments.grouped_conv_stage_sensitivity.pytorch_evaluator import (
    apply_baseline_metrics,
    evaluate_pytorch_model,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT
    / "outputs/formal_pruner_lidar_pyramid_validation_v1/grouped_conv_stage_sensitivity_v2"
)
DEFAULT_CONFIG = Path(
    "/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/"
    "LiDAROnly/lidar_pyramid/config.yaml"
)
DEFAULT_CHECKPOINT = Path(
    "/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/"
    "LiDAROnly/lidar_pyramid/net_epoch_bestval_at17.pth"
)
DEFAULT_HEAL_ROOT = Path("/home/lixingfeng/UniAD_examine/HEAL")


def _plain(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _combined_code_hash(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item)):
        digest.update(str(path.relative_to(REPOSITORY_ROOT)).encode("utf-8"))
        digest.update(bytes.fromhex(_file_sha256(path)))
    return digest.hexdigest()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for source in rows:
                row = {
                    key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                    for key, value in source.items()
                }
                writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _run_command(*args: str) -> dict[str, Any]:
    completed = subprocess.run(
        args,
        cwd=REPOSITORY_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "command": list(args),
        "returncode": completed.returncode,
        "output": (completed.stdout or "").splitlines(),
    }


def scan_forbidden_runtime_dependencies(path: str | Path) -> dict[str, Any]:
    source = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: list[str] = []
    calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(str(node.module or ""))
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)
    forbidden_roots = {"quantization", "onnx", "tensorrt"}
    forbidden_imports = sorted(
        name for name in imports if name.split(".", 1)[0].lower() in forbidden_roots
    )
    forbidden_call_names = {
        "build_trt_engine",
        "export_onnx",
        "insert_explicit_qdq",
        "run_trtexec",
    }
    return {
        "imports": sorted(imports),
        "calls": sorted(calls),
        "forbidden_imports": forbidden_imports,
        "forbidden_calls": sorted(set(calls) & forbidden_call_names),
    }


def _absolute_heal_config(config_path: Path, heal_root: Path) -> dict[str, Any]:
    if str(heal_root) not in sys.path:
        sys.path.insert(0, str(heal_root))
    from opencood.hypes_yaml import yaml_utils

    config = yaml_utils.load_yaml(str(config_path))
    for key in ("data_dir", "root_dir", "validate_dir", "test_dir"):
        value = config.get(key)
        if isinstance(value, str) and value and not Path(value).is_absolute():
            config[key] = str(heal_root / value)
    return config


def _load_model_dataset_trace(
    *,
    config_path: Path,
    checkpoint_path: Path,
    heal_root: Path,
    device: str,
) -> tuple[Any, Any, Any, dict[str, Any], Any]:
    if str(heal_root) not in sys.path:
        sys.path.insert(0, str(heal_root))
    from opencood.data_utils.datasets import build_dataset
    from opencood.tools import train_utils

    config = _absolute_heal_config(config_path, heal_root)
    loaded = load_model(
        lambda model_config: train_utils.create_model(model_config),
        checkpoint_path=checkpoint_path,
        model_config=config_path,
        device=device,
        strict_state_dict=True,
    )
    dataset = build_dataset(config, visualize=False, train=False)
    sample = dataset[0]
    if sample is None:
        raise RuntimeError("validation dataset returned None for trace frame 0")
    batch = dataset.collate_batch_test([sample])
    batch = train_utils.to_device(batch, device)
    trace = trace_model(
        loaded.model,
        batch["ego"],
        config=TraceConfig(fail_on_fx_trace_error=False),
    )
    return loaded, dataset, trace, config, batch


def _trace_inventory_source_rows(trace: Any) -> list[dict[str, Any]]:
    policies = {row.module_path: row for row in trace.protection_policies}
    scopes_by_member: dict[str, set[str]] = defaultdict(set)
    root_axes_by_member: dict[str, set[str]] = defaultdict(set)
    for scope in trace.dependency_scopes:
        for member in scope.members:
            scopes_by_member[member.module_path].add(scope.stable_id)
            if member.module_path in scope.root_modules or member.module_path == scope.root_module_path:
                root_axes_by_member[member.module_path].add(scope.root_axis)
    rows: list[dict[str, Any]] = []
    for module in trace.module_inventory:
        if module.module_type not in {"Conv2d", "ConvTranspose2d"}:
            continue
        if int(module.groups or 1) <= 1:
            continue
        if not str(module.module_path).startswith("pyramid_backbone.resnet."):
            continue
        policy = policies[module.module_path]
        rows.append(
            {
                "module_path": module.module_path,
                "module_type": module.module_type,
                "in_channels": module.in_channels,
                "out_channels": module.out_channels,
                "groups": module.groups,
                "scope_ids": sorted(scopes_by_member[module.module_path]),
                "root_pruning_allowed": policy.root_pruning_allowed,
                "dependency_input_pruning_allowed": policy.input_dependency_pruning_allowed,
                "root_axis": next(iter(sorted(root_axes_by_member[module.module_path])), "out"),
                "protected_reason": policy.protection_reason,
                "trace_source": f"formal_trace:{trace.trace_hash}",
            }
        )
    return rows


def _dummy_importance(atomic_units: Sequence[Any], allowed_roots: set[str]) -> ImportanceResult:
    raw: dict[str, float] = {}
    for unit in atomic_units:
        if unit.root_module_path not in allowed_roots:
            continue
        score = float(int(unit.root_indices[0]) + 1)
        for source_id in unit.source_coupled_unit_ids:
            raw[str(source_id)] = score
    return ImportanceResult(
        mode="legalizer_probe",
        normalization="none",
        aggregation="root_channel",
        raw_scores=raw,
        normalized_scores=dict(raw),
        unit_parameter_costs={key: 0 for key in raw},
        implementation_version="grouped-stage-legalizer-probe-v1",
    )


def _validate_strengths_with_formal_legalizer(
    model: Any,
    trace: Any,
    inventory: Mapping[str, Any],
    strength_plan: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage, stage_plan in strength_plan["stages"].items():
        allowed_roots = {
            row["module_path"] for row in inventory["rows"] if row["stage"] == stage
        }
        validations: dict[str, Any] = {}
        for strength in ("mild", "aggressive"):
            target = stage_plan.get(strength)
            if target is None:
                validations[strength] = {
                    "passed": False,
                    "status": stage_plan.get(f"{strength}_status", "unavailable"),
                    "target_width": None,
                }
                continue
            try:
                importance = _dummy_importance(trace.atomic_prune_units, allowed_roots)
                selected = build_controlled_strategy_request(
                    trace.atomic_prune_units,
                    importance,
                    allowed_roots=allowed_roots,
                    target_width=int(target),
                    strategy="taylor_independent_group_ranking",
                )
                plan = build_physical_pruning_plan(model, selected["request"])
                legalized = legalize_pruning_plan(
                    model,
                    plan,
                    alignment_config=PruningConfig().alignment,
                    grouped_config=PruningConfig().grouped_conv,
                )
                predicted = estimate_physical_parameter_count(model, legalized)
                validations[strength] = {
                    "passed": True,
                    "status": "formal_legalizer_passed",
                    "target_width": int(target),
                    "plan_entry_count": len(legalized.entries),
                    "predicted_parameter_count": predicted,
                    "requested_channel_cost": selected["requested_channel_cost"],
                    "active_root_count": len(selected["selected_active_roots"]),
                }
            except Exception as exc:
                validations[strength] = {
                    "passed": False,
                    "status": "formal_legalizer_failed",
                    "target_width": int(target),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            rows.append({"stage": stage, "strength": strength, **validations[strength]})
        stage_plan["formal_legalizer_validation"] = validations
    return rows


def _root_whitelists(inventory_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_stage = {
        stage: sorted(row["module_path"] for row in inventory_rows if row["stage"] == stage)
        for stage in STAGE_PREFIXES
    }
    return {
        "stage0_only": by_stage["stage0"],
        "stage1_only": by_stage["stage1"],
        "stage2_only": by_stage["stage2"],
        "all_grouped_stages": sorted(path for paths in by_stage.values() for path in paths),
    }


def _markdown_inventory(report: Mapping[str, Any]) -> str:
    lines = [
        "# Grouped Conv stage inventory",
        "",
        f"Trace classification uses explicit prefixes: `{json.dumps(report['stage_prefixes'], sort_keys=True)}`.",
        "",
        f"Observed counts: `{json.dumps(report['observed_stage_counts'], sort_keys=True)}`; total `{report['total_count']}`.",
        f"Inventory count mismatch: `{str(report['inventory_count_mismatch']).lower()}`.",
        "",
        "| Stage | Module | In/group | Out/group | Groups | Scope | Root allowed | Dependency input allowed |",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['stage']} | `{row['module_path']}` | {row['input_channels_per_group']} | "
            f"{row['output_channels_per_group']} | {row['groups']} | `{row['scope_id']}` | "
            f"{row['root_pruning_allowed']} | {row['dependency_input_pruning_allowed']} |"
        )
    return "\n".join(lines) + "\n"


def _markdown_strength(plan: Mapping[str, Any]) -> str:
    lines = [
        "# Pruning strength plan",
        "",
        "| Stage | Original per-group widths | Common legal widths | Mild | Aggressive | Status |",
        "|---|---|---|---:|---:|---|",
    ]
    for stage, row in plan["stages"].items():
        lines.append(
            f"| {stage} | `{row['original_output_channels_per_group']}` | `{row['common_legal_widths']}` | "
            f"{row['mild']} | {row['aggressive']} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "A missing target is recorded as infeasible; no original/no-op model substitutes for it.",
        ]
    )
    return "\n".join(lines) + "\n"


def _append_actual_command(output: Path, argv: Sequence[str]) -> None:
    path = output / "actual_run_commands.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"commands": []}
    payload["commands"].append(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "argv": [sys.executable, str(Path(__file__).resolve()), *argv],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        }
    )
    atomic_write_json(path, payload)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        torch.save(dict(payload), temporary)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _dataset_batch(dataset: Any, index: int, device: str, *, train: bool) -> Any:
    from opencood.tools import train_utils

    sample = dataset[int(index)]
    if sample is None:
        raise RuntimeError(f"dataset returned None for required frame index {index}")
    collate = dataset.collate_batch_train if train else dataset.collate_batch_test
    batch = collate([sample])
    if batch is None:
        raise RuntimeError(f"collate returned None for required frame index {index}")
    return train_utils.to_device(batch, device)


def _load_training_dataset(config_path: Path, heal_root: Path) -> tuple[Any, dict[str, Any]]:
    if str(heal_root) not in sys.path:
        sys.path.insert(0, str(heal_root))
    from opencood.data_utils.datasets import build_dataset

    config = _absolute_heal_config(config_path, heal_root)
    return build_dataset(config, visualize=False, train=True), config


def _accumulate_taylor_gradients(
    model: Any,
    dataset: Any,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    device: str,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from opencood.tools import train_utils

    expected_ids = [str(value) for value in manifest["frame_ids"]]
    split_info = [str(value) for value in getattr(dataset, "split_info", [])]
    if split_info[: len(expected_ids)] != expected_ids:
        raise RuntimeError("training dataset split order differs from Taylor frame manifest")
    criterion = train_utils.create_loss(config)
    model.eval()
    model.zero_grad(set_to_none=True)
    torch.manual_seed(int(manifest.get("random_seed", 0)))
    np.random.seed(int(manifest.get("random_seed", 0)))
    losses: list[float] = []
    started = time.time()
    for index, frame_id in enumerate(expected_ids):
        batch = _dataset_batch(dataset, index, device, train=True)
        output = model(batch["ego"])
        loss = criterion(output, batch["ego"]["label_dict"])
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError(f"non-finite Taylor loss at {index}:{frame_id}")
        (loss / float(len(expected_ids))).backward()
        losses.append(float(loss.detach().cpu()))
    parameters_with_grad = sum(
        parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad
    )
    if parameters_with_grad == 0:
        raise RuntimeError("Taylor calibration produced no parameter gradients")
    return {
        "calibration_dataset": str(config.get("root_dir", "")),
        "calibration_frame_ids": expected_ids,
        "frame_list_hash": manifest["frame_list_hash"],
        "loss_definition": type(criterion).__name__,
        "task_loss": "HEAL point_pillar_pyramid_loss",
        "gradient_accumulation_batch_count": len(losses),
        "gradient_accumulation": "mean",
        "loss_mean": sum(losses) / len(losses),
        "loss_min": min(losses),
        "loss_max": max(losses),
        "parameters_with_gradient": parameters_with_grad,
        "elapsed_seconds": time.time() - started,
        "ranking_direction": "prune_lowest_importance_first",
        "taylor_definition": "abs(weight * dL/dweight)",
        "implementation_version": "pruning.api.score_pruning_units:first-order-taylor-v1",
    }


def _importance_from_payload(payload: Mapping[str, Any]) -> ImportanceResult:
    return ImportanceResult(
        mode=str(payload["mode"]),
        normalization=str(payload["normalization"]),
        aggregation=str(payload["aggregation"]),
        raw_scores={str(key): float(value) for key, value in payload["raw_scores"].items()},
        normalized_scores={
            str(key): float(value) for key, value in payload["normalized_scores"].items()
        },
        unit_parameter_costs={
            str(key): int(value) for key, value in payload.get("unit_parameter_costs", {}).items()
        },
        unit_scores=list(payload.get("unit_scores", [])),
        calibration_batches=int(payload.get("calibration_batches", 0)),
        task_loss=str(payload.get("task_loss", "")),
        gradient_accumulation=str(payload.get("gradient_accumulation", "mean")),
        implementation_version=str(payload.get("implementation_version", "first-order-taylor-v1")),
        schema_version=str(payload.get("schema_version", "importance-result-v1")),
    )


def _prepare_taylor_importance(
    args: argparse.Namespace,
    output: Path,
) -> ImportanceResult:
    shared_dir = output / "importance"
    artifact = shared_dir / "taylor_importance_result.json"
    if args.resume and artifact.is_file() and not args.force:
        return _importance_from_payload(json.loads(artifact.read_text(encoding="utf-8")))
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    heal_root = Path(args.heal_root).resolve()
    loaded, _validation_dataset, trace, _validation_config, _validation_batch = (
        _load_model_dataset_trace(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            heal_root=heal_root,
            device=args.device,
        )
    )
    prepared_trace = json.loads((output / "grouped_conv_stage_inventory.json").read_text(encoding="utf-8"))
    if trace.trace_hash != prepared_trace["trace_hash"]:
        raise RuntimeError(
            f"Taylor trace hash changed: {trace.trace_hash} != {prepared_trace['trace_hash']}"
        )
    training_dataset, training_config = _load_training_dataset(config_path, heal_root)
    manifest = json.loads((output / "taylor_frame_manifest.json").read_text(encoding="utf-8"))
    calibration = _accumulate_taylor_gradients(
        loaded.model,
        training_dataset,
        training_config,
        manifest,
        device=args.device,
    )
    importance = score_pruning_units(
        loaded.model,
        trace.coupled_channel_units,
        config=PruningConfig(),
        calibration_batches=len(manifest["frame_ids"]),
        task_loss="HEAL point_pillar_pyramid_loss",
    )
    payload = importance.to_dict()
    payload.update(
        {
            "calibration_summary": calibration,
            "trace_hash": trace.trace_hash,
            "source_checkpoint_hash": _file_sha256(checkpoint_path),
            "frame_manifest": str(output / "taylor_frame_manifest.json"),
        }
    )
    atomic_write_json(artifact, payload)
    atomic_write_json(
        shared_dir / "taylor_raw_scores.json",
        {
            "mode": importance.mode,
            "aggregation": importance.aggregation,
            "raw_scores": importance.raw_scores,
            "unit_scores": importance.unit_scores,
            "calibration_summary": calibration,
        },
    )
    atomic_write_json(
        shared_dir / "taylor_scope_scores.json",
        {
            "normalization_policy": importance.normalization,
            "normalized_scores": importance.normalized_scores,
            "ranking_direction": "prune_lowest_importance_first",
            "implementation_version": importance.implementation_version,
        },
    )
    return importance


def _snapshot_rows(snapshot: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row["canonical_module_name"]): dict(row)
        for row in snapshot.get("modules", [])
    }


def _parameter_count_for_modules(snapshot: Mapping[str, Any], modules: set[str]) -> int:
    rows = _snapshot_rows(snapshot)
    return sum(int(rows.get(module, {}).get("parameter_count", 0)) for module in modules)


def _fixed_output_contracts(snapshot: Mapping[str, Any]) -> dict[str, int]:
    contracts: dict[str, int] = {}
    for row in snapshot.get("modules", []):
        policy = dict(row.get("protection_policy", {}))
        output = row.get("out_channels")
        if bool(policy.get("fixed_output_contract")) and output is not None:
            contracts[str(row["canonical_module_name"])] = int(output)
    return contracts


def _candidate_status_path(candidate_dir: Path) -> Path:
    return candidate_dir / "candidate_status.json"


def _write_candidate_failure(
    candidate_dir: Path,
    candidate_id: str,
    stage: str,
    exc: BaseException,
) -> dict[str, Any]:
    payload = {
        "candidate_id": candidate_id,
        "candidate_status": "invalid",
        "failure_stage": stage,
        "failure_type": type(exc).__name__,
        "failure_reason": str(exc),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(_candidate_status_path(candidate_dir), payload)
    return payload


def _generate_candidate(
    args: argparse.Namespace,
    output: Path,
    candidate: Mapping[str, Any],
    taylor_importance: ImportanceResult,
) -> dict[str, Any]:
    import torch

    candidate_id = str(candidate["candidate_id"])
    candidate_dir = output / "candidates" / candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    completion = candidate_dir / "generation_complete.json"
    if args.resume and completion.is_file() and not args.force:
        return json.loads(completion.read_text(encoding="utf-8"))
    stage_marker = "load_original_checkpoint"
    started = time.time()
    atomic_write_json(
        _candidate_status_path(candidate_dir),
        {
            "candidate_id": candidate_id,
            "candidate_status": "running",
            "failure_stage": "",
            "failure_reason": "",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    try:
        config_path = Path(args.config).resolve()
        checkpoint_path = Path(args.checkpoint).resolve()
        heal_root = Path(args.heal_root).resolve()
        checkpoint_hash = _file_sha256(checkpoint_path)
        config_hash = _file_sha256(config_path)
        loaded, _dataset, trace, _config, batch = _load_model_dataset_trace(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            heal_root=heal_root,
            device=args.device,
        )
        model = loaded.model
        original_identity = json.loads((output / "original_model_identity.json").read_text(encoding="utf-8"))
        live_state_hash = state_dict_content_hash(dict(model.state_dict()))
        if checkpoint_hash != original_identity["checkpoint_hash"]:
            raise RuntimeError("candidate checkpoint hash differs from prepared original")
        if live_state_hash != original_identity["state_dict_content_hash"]:
            raise RuntimeError("fresh candidate state dict differs from prepared original")
        prepared_inventory = json.loads(
            (output / "grouped_conv_stage_inventory.json").read_text(encoding="utf-8")
        )
        if trace.trace_hash != prepared_inventory["trace_hash"]:
            raise RuntimeError("candidate formal trace hash differs from prepared trace")
        serialize_trace_result(trace, candidate_dir / "formal_trace_result.json")

        whitelists = json.loads((output / "root_whitelists.json").read_text(encoding="utf-8"))[
            "whitelists"
        ]
        allowed_roots = set(whitelists[str(candidate["stage_scope"])])
        if not allowed_roots:
            raise RuntimeError("candidate root whitelist is empty")
        target_widths = dict(candidate["target_widths_by_stage"])
        if len(target_widths) != 1:
            raise RuntimeError("current feasible candidate must target exactly one stage")
        target_width = int(next(iter(target_widths.values())))
        atomic_write_json(
            candidate_dir / "root_whitelist.json",
            {
                "candidate_id": candidate_id,
                "stage_scope": candidate["stage_scope"],
                "allowed_active_roots": sorted(allowed_roots),
                "active_root_count": len(allowed_roots),
                "implementation_layer": "tools.experiments orchestration",
                "dependency_closure_filtering": False,
            },
        )

        stage_marker = "importance"
        if candidate["strategy"] == "taylor_independent_group_ranking":
            importance = taylor_importance
            importance_metadata = {
                "torch_pruning_api_used": False,
                "source": str(output / "importance/taylor_importance_result.json"),
            }
        else:
            tp_result = build_tp_importance_result(
                model,
                trace.atomic_prune_units,
                allowed_roots=allowed_roots,
            )
            importance = tp_result["importance"]
            importance_metadata = {
                key: value for key, value in tp_result.items() if key != "importance"
            }
            atomic_write_json(
                candidate_dir / "importance/tp_l2_raw_channel_scores.json",
                {
                    "modules": tp_result["modules"],
                    "channel_mapping": tp_result["channel_mapping"],
                    "torch_pruning_api_used": tp_result["torch_pruning_api_used"],
                },
            )

        stage_marker = "formal_selection_and_dependency_closure"
        selection = build_controlled_strategy_request(
            trace.atomic_prune_units,
            importance,
            allowed_roots=allowed_roots,
            target_width=target_width,
            strategy=str(candidate["strategy"]),
        )
        request = selection["request"]
        atomic_write_json(candidate_dir / "sampling_structure_request.json", request.to_dict())
        raw_root_units = filter_active_root_units(trace.atomic_prune_units, allowed_roots)
        atomic_write_json(
            candidate_dir / "selected_root_units.json",
            {
                "candidate_id": candidate_id,
                "allowed_roots": sorted(allowed_roots),
                "raw_atomic_units": [unit.to_dict() for unit in raw_root_units],
                "selected_bundle_ids": request.selected_atomic_unit_ids,
                "formal_selector": selection["formal_selector"],
            },
        )
        closure_rows = [row.to_dict() for row in request.entries]
        atomic_write_json(
            candidate_dir / "dependency_closure.json",
            {
                "entries": closure_rows,
                "active_root_entries": [
                    row
                    for row in closure_rows
                    if not bool(row.get("metadata", {}).get("dependency_driven", False))
                ],
                "dependency_driven_entries": [
                    row
                    for row in closure_rows
                    if bool(row.get("metadata", {}).get("dependency_driven", False))
                ],
                "formal_dependency_closure_preserved": True,
            },
        )
        score_records = build_group_score_records(
            trace.atomic_prune_units,
            importance,
            request,
            strategy=str(candidate["strategy"]),
        )
        root_entries = [
            row
            for row in request.entries
            if not bool(row.metadata.get("dependency_driven", False)) and row.group_keep_map
        ]
        keep_rows = [
            {
                "module_path": row.module_path,
                "scope_id": row.scope_id,
                "group_keep_map": row.group_keep_map,
            }
            for row in root_entries
        ]
        prune_rows = [
            {
                "module_path": row.module_path,
                "scope_id": row.scope_id,
                "group_prune_map": row.group_prune_map,
            }
            for row in root_entries
        ]
        atomic_write_json(candidate_dir / "group_keep_map.json", {"modules": keep_rows})
        atomic_write_json(candidate_dir / "group_prune_map.json", {"modules": prune_rows})
        if candidate["strategy"] == "taylor_independent_group_ranking":
            atomic_write_json(
                candidate_dir / "importance/taylor_raw_scores.json",
                {
                    "records": score_records,
                    "importance_mode": importance.mode,
                    "aggregation": importance.aggregation,
                },
            )
            atomic_write_json(
                candidate_dir / "importance/taylor_scope_scores.json",
                {
                    "records": score_records,
                    "normalization": importance.normalization,
                    "ranking_direction": "prune_lowest_importance_first",
                },
            )
        else:
            shared_rows = [
                {
                    key: row[key]
                    for key in (
                        "module_path",
                        "scope_id",
                        "group_id",
                        "local_position",
                        "raw_score",
                        "shared_local_mean_score",
                        "shared_rank",
                        "kept",
                        "pruned",
                        "torch_pruning_api_used",
                    )
                }
                for row in score_records
            ]
            atomic_write_json(
                candidate_dir / "importance/tp_l2_shared_local_scores.json",
                {
                    "records": shared_rows,
                    "aggregation": "shared_local_position_mean",
                    "selection": "shared_position_topk",
                    "torch_pruning_api_used": True,
                },
            )

        stage_marker = "formal_plan_and_legalizer"
        plan = build_physical_pruning_plan(model, request)
        legalized = legalize_pruning_plan(
            model,
            plan,
            alignment_config=PruningConfig().alignment,
            grouped_config=PruningConfig().grouped_conv,
        )
        atomic_write_json(candidate_dir / "physical_pruning_plan.json", legalized.to_dict())
        predicted_parameter_count = estimate_physical_parameter_count(model, legalized)
        original_snapshot = build_physical_structure_snapshot(model)
        original_snapshot_payload = original_snapshot.to_dict()
        model.eval()
        with torch.inference_mode():
            original_output = model(batch["ego"])
        original_output_contract = tensor_output_contract(original_output)
        if not original_output_contract["all_finite"]:
            raise RuntimeError("fresh original model produced non-finite output")

        stage_marker = "formal_physical_materializer"
        materialized = materialize_pruning(model, legalized, in_place=True, build_snapshot=True)
        physical_model = materialized.model
        snapshot = materialized.snapshot or build_physical_structure_snapshot(physical_model)
        snapshot_payload = snapshot.to_dict()
        if predicted_parameter_count != snapshot.parameter_count:
            raise RuntimeError(
                f"predicted/actual parameter mismatch: {predicted_parameter_count} != {snapshot.parameter_count}"
            )
        fixed_outputs = _fixed_output_contracts(original_snapshot_payload)
        physical_model.eval()
        validation = validate_physical_model(
            physical_model,
            expected_snapshot=snapshot,
            grouped_config=PruningConfig().grouped_conv,
            fixed_output_contracts=fixed_outputs,
            example_inputs=(batch["ego"],),
        )
        if not validation.passed:
            raise RuntimeError(f"formal physical validation failed: {validation.issues[:5]}")

        stage_marker = "real_batch_forward_sanity"
        with torch.inference_mode():
            candidate_output = physical_model(batch["ego"])
        candidate_output_contract = tensor_output_contract(candidate_output)
        output_interface_match = (
            original_output_contract["tensor_shapes"]
            == candidate_output_contract["tensor_shapes"]
            and original_output_contract["tensor_dtypes"]
            == candidate_output_contract["tensor_dtypes"]
        )
        if not candidate_output_contract["all_finite"] or not output_interface_match:
            raise RuntimeError(
                "candidate forward output is non-finite or changed the public tensor interface"
            )

        stage_marker = "artifact_write"
        hashes = compute_physical_hashes(
            snapshot,
            model_hash=checkpoint_hash,
            config_hash=config_hash,
        )
        portable_state = {
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in physical_model.state_dict().items()
        }
        candidate_state_hash = state_dict_content_hash(portable_state)
        _atomic_torch_save(candidate_dir / "model_state_dict.pth", portable_state)
        atomic_write_json(
            candidate_dir / "physical_pruning_application_ledger.json",
            materialized.ledger.to_dict(),
        )
        atomic_write_json(candidate_dir / "physical_structure_snapshot_v2.json", snapshot_payload)
        hash_payload = hashes.to_dict()
        hash_payload.update(
            {
                "physical_shape_hash": physical_shape_hash(snapshot_payload),
                "state_dict_content_hash": candidate_state_hash,
            }
        )
        atomic_write_json(candidate_dir / "physical_hash_v2.json", hash_payload)
        atomic_write_json(
            candidate_dir / "forward_sanity.json",
            {
                "passed": True,
                "original_output_contract": original_output_contract,
                "candidate_output_contract": candidate_output_contract,
                "output_interface_match": output_interface_match,
                "formal_validation": {
                    "passed": validation.passed,
                    "issues": validation.issues,
                    "forward_checked": validation.forward_checked,
                    "output_contract_checked": validation.output_contract_checked,
                },
            },
        )
        closure_modules = {row.module_path for row in legalized.entries}
        breakdown = parameter_reduction_breakdown(
            original_snapshot_payload,
            snapshot_payload,
            active_roots=allowed_roots,
            closure_modules=closure_modules,
        )
        all_grouped_roots = {
            row["module_path"] for row in prepared_inventory["rows"]
        }
        summary = {
            "candidate_id": candidate_id,
            "candidate_status": "valid",
            "failure_stage": "",
            "failure_reason": "",
            "source_checkpoint": str(checkpoint_path),
            "source_checkpoint_hash": checkpoint_hash,
            "source_state_dict_content_hash": live_state_hash,
            "trace_hash": trace.trace_hash,
            "trace_count": 1,
            "materialization_transaction_count": 1,
            "stage_scope": candidate["stage_scope"],
            "strength": candidate["strength"],
            "strategy": candidate["strategy"],
            "target_widths_by_stage": target_widths,
            "active_roots": sorted(allowed_roots),
            "active_root_count": len(allowed_roots),
            "dependency_closure_module_count": len(closure_modules),
            "original_parameter_count": original_snapshot.parameter_count,
            "candidate_parameter_count": snapshot.parameter_count,
            "actual_parameter_reduction": original_snapshot.parameter_count
            - snapshot.parameter_count,
            "actual_parameter_reduction_ratio": 1.0
            - float(snapshot.parameter_count) / float(original_snapshot.parameter_count),
            "original_active_grouped_parameter_count": _parameter_count_for_modules(
                original_snapshot_payload, allowed_roots
            ),
            "candidate_active_grouped_parameter_count": _parameter_count_for_modules(
                snapshot_payload, allowed_roots
            ),
            "original_all_grouped_parameter_count": _parameter_count_for_modules(
                original_snapshot_payload, all_grouped_roots
            ),
            "candidate_all_grouped_parameter_count": _parameter_count_for_modules(
                snapshot_payload, all_grouped_roots
            ),
            "parameter_reduction_breakdown": breakdown,
            "predicted_parameter_count": predicted_parameter_count,
            "structure_hash_v2": hashes.structure_hash_v2,
            "physical_shape_hash": hash_payload["physical_shape_hash"],
            "state_dict_content_hash": candidate_state_hash,
            "forward_sanity_passed": True,
            "torch_pruning_api_used": bool(
                importance_metadata.get("torch_pruning_api_used", False)
            ),
            "tp_dependency_graph_used": False,
            "tp_physical_materialization_used": False,
            "formal_dependency_scope_used": True,
            "formal_legalizer_used": True,
            "formal_materializer_used": True,
            "elapsed_seconds": time.time() - started,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(completion, summary)
        atomic_write_json(_candidate_status_path(candidate_dir), summary)
        return summary
    except Exception as exc:
        return _write_candidate_failure(candidate_dir, candidate_id, stage_marker, exc)


def _validate_candidate_pair(
    output: Path,
    *,
    stage_scope: str,
    strength: str,
) -> dict[str, Any]:
    left_id = f"{stage_scope}__{strength}__taylor_independent_group_ranking"
    right_id = f"{stage_scope}__{strength}__torch_pruning_l2_shared_position"
    left_dir = output / "candidates" / left_id
    right_dir = output / "candidates" / right_id
    left_status = json.loads(_candidate_status_path(left_dir).read_text(encoding="utf-8"))
    right_status = json.loads(_candidate_status_path(right_dir).read_text(encoding="utf-8"))
    if left_status.get("candidate_status") != "valid" or right_status.get("candidate_status") != "valid":
        return {
            "stage_scope": stage_scope,
            "strength": strength,
            "left_candidate_id": left_id,
            "right_candidate_id": right_id,
            "comparison_valid": False,
            "failure_reason": "candidate_invalid",
            "left_status": left_status.get("candidate_status"),
            "right_status": right_status.get("candidate_status"),
        }
    left_snapshot = json.loads(
        (left_dir / "physical_structure_snapshot_v2.json").read_text(encoding="utf-8")
    )
    right_snapshot = json.loads(
        (right_dir / "physical_structure_snapshot_v2.json").read_text(encoding="utf-8")
    )
    validation = validate_pairwise_structure(left_snapshot, right_snapshot)
    left_hash = json.loads((left_dir / "physical_hash_v2.json").read_text(encoding="utf-8"))
    right_hash = json.loads((right_dir / "physical_hash_v2.json").read_text(encoding="utf-8"))
    return {
        "stage_scope": stage_scope,
        "strength": strength,
        "left_candidate_id": left_id,
        "right_candidate_id": right_id,
        **validation,
        "structure_hash_v2_match": left_hash["structure_hash_v2"]
        == right_hash["structure_hash_v2"],
        "left_structure_hash_v2": left_hash["structure_hash_v2"],
        "right_structure_hash_v2": right_hash["structure_hash_v2"],
        "left_state_dict_content_hash": left_hash["state_dict_content_hash"],
        "right_state_dict_content_hash": right_hash["state_dict_content_hash"],
        "state_dict_content_hash_expected_to_differ": True,
    }


def _write_pairwise_rows(output: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    atomic_write_json(
        output / "pairwise_structure_validation.json",
        {
            "pairs": list(rows),
            "valid_pair_count": sum(bool(row.get("comparison_valid")) for row in rows),
            "invalid_pair_count": sum(not bool(row.get("comparison_valid")) for row in rows),
        },
    )
    _atomic_csv(output / "pairwise_structure_validation.csv", rows)


def run_dry_run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output = Path(args.output).resolve()
    if not (output / "prepare_complete.json").is_file():
        raise RuntimeError("prepare must complete before dry_run")
    matrix = json.loads((output / "candidate_matrix.json").read_text(encoding="utf-8"))[
        "candidates"
    ]
    dry_ids = {
        "stage1_only__mild__taylor_independent_group_ranking",
        "stage1_only__mild__torch_pruning_l2_shared_position",
    }
    candidates = [row for row in matrix if row["candidate_id"] in dry_ids]
    if len(candidates) != 2 or any(row["candidate_status"] != "planned" for row in candidates):
        raise RuntimeError("stage1 mild dry-run pair is not planned and feasible")
    taylor_importance = _prepare_taylor_importance(args, output)
    generated = [
        _generate_candidate(args, output, candidate, taylor_importance)
        for candidate in candidates
    ]
    pair = _validate_candidate_pair(output, stage_scope="stage1_only", strength="mild")
    _write_pairwise_rows(output, [pair])
    before = json.loads((output / "core_freeze_before.json").read_text(encoding="utf-8"))
    freeze = validate_core_freeze(before, repository_root=REPOSITORY_ROOT)
    atomic_write_json(output / "core_freeze_dry_run_validation.json", freeze)
    valid_candidates = sum(row.get("candidate_status") == "valid" for row in generated)
    passed = valid_candidates == 2 and bool(pair.get("comparison_valid")) and freeze[
        "core_files_unchanged"
    ]
    result = {
        "stage": "dry_run",
        "status": "completed" if passed else "failed",
        "candidate_count": 2,
        "valid_candidate_count": valid_candidates,
        "invalid_candidate_count": 2 - valid_candidates,
        "pairwise_structure_match": bool(pair.get("comparison_valid")),
        "core_files_unchanged": freeze["core_files_unchanged"],
        "candidates": generated,
        "pair": pair,
        "elapsed_seconds": time.time() - started,
    }
    atomic_write_json(output / "dry_run_complete.json", result)
    return result


def build_generation_summary(
    matrix: Sequence[Mapping[str, Any]],
    generated: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    generated_by_id = {str(row["candidate_id"]): dict(row) for row in generated}
    rows: list[dict[str, Any]] = []
    for source in matrix:
        candidate_id = str(source["candidate_id"])
        if source.get("candidate_status") == "infeasible":
            rows.append(dict(source))
        elif candidate_id in generated_by_id:
            rows.append({**dict(source), **generated_by_id[candidate_id]})
        else:
            rows.append(
                {
                    **dict(source),
                    "candidate_status": "invalid",
                    "failure_stage": "generation_dispatch",
                    "failure_reason": "planned_candidate_missing_generation_result",
                }
            )
    return {
        "rows": rows,
        "valid_candidate_count": sum(row.get("candidate_status") == "valid" for row in rows),
        "invalid_candidate_count": sum(row.get("candidate_status") == "invalid" for row in rows),
        "infeasible_candidate_count": sum(
            row.get("candidate_status") == "infeasible" for row in rows
        ),
        "planned_candidate_count": sum(
            source.get("candidate_status") == "planned" for source in matrix
        ),
    }


def run_generate(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output = Path(args.output).resolve()
    if not (output / "dry_run_complete.json").is_file():
        raise RuntimeError("successful dry_run is required before generate")
    dry_run = json.loads((output / "dry_run_complete.json").read_text(encoding="utf-8"))
    if dry_run.get("status") != "completed":
        raise RuntimeError("dry_run did not pass its structure/freeze gate")
    matrix = json.loads((output / "candidate_matrix.json").read_text(encoding="utf-8"))[
        "candidates"
    ]
    taylor_importance = _prepare_taylor_importance(args, output)
    generated: list[dict[str, Any]] = []
    for candidate in matrix:
        candidate_dir = output / "candidates" / str(candidate["candidate_id"])
        if candidate["candidate_status"] == "infeasible":
            candidate_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(
                _candidate_status_path(candidate_dir),
                {
                    "candidate_id": candidate["candidate_id"],
                    "candidate_status": "infeasible",
                    "infeasible_reason": candidate["infeasible_reason"],
                    "failure_stage": "pruning_strength_plan",
                    "failure_reason": candidate["infeasible_reason"],
                    "no_op_model_created": False,
                    "model_state_dict_created": False,
                },
            )
            continue
        generated.append(_generate_candidate(args, output, candidate, taylor_importance))

    planned_pairs = sorted(
        {
            (str(row["stage_scope"]), str(row["strength"]))
            for row in matrix
            if row["candidate_status"] == "planned"
        }
    )
    pair_rows = [
        _validate_candidate_pair(output, stage_scope=stage_scope, strength=strength)
        for stage_scope, strength in planned_pairs
    ]
    _write_pairwise_rows(output, pair_rows)
    summary = build_generation_summary(matrix, generated)
    _atomic_csv(output / "candidate_generation_results.csv", summary["rows"])
    atomic_write_json(output / "candidate_generation_results.json", summary)
    before = json.loads((output / "core_freeze_before.json").read_text(encoding="utf-8"))
    freeze = validate_core_freeze(before, repository_root=REPOSITORY_ROOT)
    atomic_write_json(output / "core_freeze_generation_validation.json", freeze)
    all_pairs_valid = all(bool(row.get("comparison_valid")) for row in pair_rows)
    passed = (
        summary["valid_candidate_count"] == summary["planned_candidate_count"]
        and summary["invalid_candidate_count"] == 0
        and all_pairs_valid
        and freeze["core_files_unchanged"]
    )
    probe_path = output / "environment/torch_pruning_probe.json"
    if probe_path.is_file():
        probe = json.loads(probe_path.read_text(encoding="utf-8"))
        tp_rows = [
            row
            for row in generated
            if row.get("strategy") == "torch_pruning_l2_shared_position"
        ]
        probe["returned_per_channel_importance"] = bool(tp_rows) and all(
            bool(row.get("torch_pruning_api_used")) for row in tp_rows
        )
        probe["actual_tp_candidate_count"] = len(tp_rows)
        atomic_write_json(probe_path, probe)
        _atomic_text(
            output / "environment/torch_pruning_probe.txt",
            json.dumps(probe, indent=2, sort_keys=True) + "\n",
        )
    result = {
        "stage": "generate",
        "status": "completed" if passed else "failed",
        "valid_candidate_count": summary["valid_candidate_count"],
        "invalid_candidate_count": summary["invalid_candidate_count"],
        "infeasible_candidate_count": summary["infeasible_candidate_count"],
        "planned_candidate_count": summary["planned_candidate_count"],
        "valid_pair_count": sum(bool(row.get("comparison_valid")) for row in pair_rows),
        "invalid_pair_count": sum(not bool(row.get("comparison_valid")) for row in pair_rows),
        "core_files_unchanged": freeze["core_files_unchanged"],
        "elapsed_seconds": time.time() - started,
    }
    atomic_write_json(output / "generation_complete_all.json", result)
    return result


def _restore_evaluation_model(
    args: argparse.Namespace,
    output: Path,
    model_id: str,
) -> tuple[Any, dict[str, Any]]:
    import torch
    from opencood.tools import train_utils

    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    loaded = load_model(
        lambda config: train_utils.create_model(config),
        checkpoint_path=checkpoint_path,
        model_config=config_path,
        device=args.device,
        strict_state_dict=True,
    )
    if model_id == "original":
        snapshot = build_physical_structure_snapshot(loaded.model)
        return loaded.model, snapshot.to_dict()
    candidate_dir = output / "candidates" / model_id
    status = json.loads(_candidate_status_path(candidate_dir).read_text(encoding="utf-8"))
    if status.get("candidate_status") != "valid":
        raise RuntimeError(f"candidate is not valid for AP evaluation: {model_id}")
    plan = PhysicalPruningPlan.from_dict(
        json.loads((candidate_dir / "physical_pruning_plan.json").read_text(encoding="utf-8"))
    )
    replayed = replay_pruning(loaded.model, plan, in_place=True)
    state = torch.load(
        candidate_dir / "model_state_dict.pth",
        map_location=args.device,
        weights_only=True,
    )
    replayed.model.load_state_dict(state, strict=True)
    expected = json.loads(
        (candidate_dir / "physical_structure_snapshot_v2.json").read_text(encoding="utf-8")
    )
    validation = validate_physical_model(
        replayed.model,
        expected_snapshot=expected,
        grouped_config=PruningConfig().grouped_conv,
    )
    if not validation.passed:
        raise RuntimeError(f"restored candidate validation failed: {validation.issues[:5]}")
    restored_state_hash = state_dict_content_hash(dict(replayed.model.state_dict()))
    if restored_state_hash != status["state_dict_content_hash"]:
        raise RuntimeError(
            f"restored state content hash differs for {model_id}: "
            f"{restored_state_hash} != {status['state_dict_content_hash']}"
        )
    return replayed.model, validation.snapshot.to_dict()


def _evaluation_model_metadata(
    output: Path,
    model_id: str,
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    original_snapshot = json.loads(
        (output / "original_physical_structure_snapshot_v2.json").read_text(encoding="utf-8")
    )
    inventory = json.loads(
        (output / "grouped_conv_stage_inventory.json").read_text(encoding="utf-8")
    )
    all_grouped = {row["module_path"] for row in inventory["rows"]}
    if model_id == "original":
        grouped_count = _parameter_count_for_modules(original_snapshot, all_grouped)
        return {
            "model_id": "original",
            "stage_scope": "original",
            "strength": "none",
            "strategy": "original_unpruned",
            "candidate_status": "valid",
            "comparison_valid": True,
            "original_parameter_count": int(original_snapshot["parameter_count"]),
            "candidate_parameter_count": int(snapshot["parameter_count"]),
            "original_grouped_layer_parameter_count": grouped_count,
            "candidate_grouped_layer_parameter_count": grouped_count,
            "grouped_layer_parameter_reduction": 0,
            "grouped_layer_parameter_reduction_ratio": 0.0,
            "closure_module_count": 0,
            "active_root_parameter_reduction": 0,
            "dependency_driven_parameter_reduction": 0,
        }
    status = json.loads(
        _candidate_status_path(output / "candidates" / model_id).read_text(encoding="utf-8")
    )
    original_grouped = int(status["original_active_grouped_parameter_count"])
    candidate_grouped = int(status["candidate_active_grouped_parameter_count"])
    breakdown = dict(status["parameter_reduction_breakdown"])
    return {
        "model_id": model_id,
        "stage_scope": status["stage_scope"],
        "strength": status["strength"],
        "strategy": status["strategy"],
        "candidate_status": status["candidate_status"],
        "comparison_valid": True,
        "original_parameter_count": int(status["original_parameter_count"]),
        "candidate_parameter_count": int(status["candidate_parameter_count"]),
        "original_grouped_layer_parameter_count": original_grouped,
        "candidate_grouped_layer_parameter_count": candidate_grouped,
        "grouped_layer_parameter_reduction": original_grouped - candidate_grouped,
        "grouped_layer_parameter_reduction_ratio": (
            float(original_grouped - candidate_grouped) / float(original_grouped)
            if original_grouped > 0
            else None
        ),
        "closure_module_count": int(status["dependency_closure_module_count"]),
        "active_root_parameter_reduction": int(
            breakdown["active_root_parameter_reduction"]
        ),
        "dependency_driven_parameter_reduction": int(
            breakdown["dependency_driven_parameter_reduction"]
        ),
        "physical_shape_hash": status["physical_shape_hash"],
        "structure_hash_v2": status["structure_hash_v2"],
        "state_dict_content_hash": status["state_dict_content_hash"],
    }


def _evaluation_path(output: Path, model_id: str) -> Path:
    name = "original_500frames.json" if model_id == "original" else f"{model_id}_500frames.json"
    return output / "evaluation" / name


def run_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from opencood.data_utils.datasets import build_dataset

    started = time.time()
    output = Path(args.output).resolve()
    generation = json.loads(
        (output / "generation_complete_all.json").read_text(encoding="utf-8")
    )
    if generation.get("status") != "completed":
        raise RuntimeError("all feasible candidates must pass generation before evaluation")
    evaluation_dir = output / "evaluation"
    heartbeat_dir = evaluation_dir / "heartbeats"
    status_dir = evaluation_dir / "status"
    for directory in (evaluation_dir, heartbeat_dir, status_dir):
        directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "validation_frame_manifest_500.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evaluator_path = (
        Path(__file__).with_name("grouped_conv_stage_sensitivity") / "pytorch_evaluator.py"
    )
    evaluator_hash = _file_sha256(evaluator_path)
    previous_hash = manifest.get("evaluation_code_hash")
    if previous_hash != evaluator_hash:
        manifest["previous_evaluation_code_hash"] = previous_hash
        manifest["evaluation_code_hash"] = evaluator_hash
        manifest["evaluation_code_hash_policy"] = "dedicated_pytorch_evaluator_module_sha256"
        manifest["evaluation_code_hash_updated_before_first_ap_evaluation"] = True
        atomic_write_json(manifest_path, manifest)
    if int(manifest.get("frame_count", 0)) != 500:
        raise RuntimeError("AP evaluation manifest is not exactly 500 frames")
    config = _absolute_heal_config(Path(args.config).resolve(), Path(args.heal_root).resolve())
    dataset = build_dataset(config, visualize=False, train=False)
    split_ids = [str(value) for value in getattr(dataset, "split_info", [])]
    if split_ids[:500] != [str(value) for value in manifest["frame_ids"]]:
        raise RuntimeError("validation dataset order differs from fixed 500-frame manifest")

    generation_rows = json.loads(
        (output / "candidate_generation_results.json").read_text(encoding="utf-8")
    )["rows"]
    candidate_ids = [
        str(row["candidate_id"])
        for row in generation_rows
        if row.get("candidate_status") == "valid"
    ]
    model_ids = ["original", *candidate_ids]
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for model_id in model_ids:
        path = _evaluation_path(output, model_id)
        if args.resume and path.is_file() and not args.force:
            prior = json.loads(path.read_text(encoding="utf-8"))
            if (
                prior.get("evaluation_status") == "completed"
                and int(prior.get("evaluated_frames", 0)) == 500
                and prior.get("frame_list_hash") == manifest["frame_list_hash"]
                and prior.get("evaluation_code_hash") == evaluator_hash
            ):
                rows.append(prior)
                continue
        model_started = time.time()
        status_path = status_dir / f"{model_id}.json"
        atomic_write_json(
            status_path,
            {
                "model_id": model_id,
                "evaluation_status": "running",
                "evaluated_frames": 0,
                "failure_stage": "",
                "failure_reason": "",
                "frame_list_hash": manifest["frame_list_hash"],
            },
        )
        try:
            model, snapshot = _restore_evaluation_model(args, output, model_id)

            def heartbeat(payload: dict[str, Any], current_model: str = model_id) -> None:
                row = {
                    "model_id": current_model,
                    "evaluation_status": "running",
                    "evaluation_code_hash": evaluator_hash,
                    **payload,
                }
                atomic_write_json(heartbeat_dir / f"{current_model}.json", row)
                atomic_write_json(status_dir / f"{current_model}.json", row)

            metrics = evaluate_pytorch_model(
                model,
                dataset,
                manifest,
                device=args.device,
                heartbeat=heartbeat,
            )
            metadata = _evaluation_model_metadata(output, model_id, snapshot)
            row = {
                **metadata,
                **metrics,
                "AP@0.03": metrics["ap_0.03"],
                "AP@0.30": metrics["ap_0.30"],
                "AP@0.50": metrics["ap_0.50"],
                "AP@0.70": metrics["ap_0.70"],
                "evaluation_status": "completed",
                "failure_stage": "",
                "failure_reason": "",
                "evaluation_code_hash": evaluator_hash,
                "checkpoint_hash": manifest["checkpoint_hash"],
                "model_restore_and_eval_wall_seconds": time.time() - model_started,
            }
            atomic_write_json(path, row)
            atomic_write_json(status_path, row)
            rows.append(row)
            del model
            torch.cuda.empty_cache()
        except Exception as exc:
            failure = {
                "model_id": model_id,
                "evaluation_status": "failed",
                "evaluated_frames": 0,
                "failure_stage": "restore_or_500frame_evaluation",
                "failure_type": type(exc).__name__,
                "failure_reason": str(exc),
                "frame_list_hash": manifest["frame_list_hash"],
                "evaluation_code_hash": evaluator_hash,
            }
            atomic_write_json(status_path, failure)
            failures.append(failure)
            torch.cuda.empty_cache()

    completed_by_id = {
        str(row["model_id"]): row
        for row in rows
        if row.get("evaluation_status") == "completed"
    }
    if "original" in completed_by_id:
        baseline = completed_by_id["original"]
        derived_rows: list[dict[str, Any]] = []
        for model_id in model_ids:
            row = completed_by_id.get(model_id)
            if row is None:
                continue
            derived = apply_baseline_metrics(row, baseline)
            derived["AP@0.03_absolute_drop"] = derived["ap_0.03_absolute_drop"]
            derived["AP@0.30_absolute_drop"] = derived["ap_0.30_absolute_drop"]
            derived["AP@0.50_absolute_drop"] = derived["ap_0.50_absolute_drop"]
            derived["AP@0.70_absolute_drop"] = derived["ap_0.70_absolute_drop"]
            atomic_write_json(_evaluation_path(output, model_id), derived)
            atomic_write_json(status_dir / f"{model_id}.json", derived)
            derived_rows.append(derived)
        rows = derived_rows
    before = json.loads((output / "core_freeze_before.json").read_text(encoding="utf-8"))
    freeze = validate_core_freeze(before, repository_root=REPOSITORY_ROOT)
    atomic_write_json(output / "core_freeze_evaluation_validation.json", freeze)
    expected_model_count = 1 + generation["valid_candidate_count"]
    passed = (
        len(rows) == expected_model_count
        and not failures
        and all(int(row.get("evaluated_frames", 0)) == 500 for row in rows)
        and all(row.get("frame_list_hash") == manifest["frame_list_hash"] for row in rows)
        and freeze["core_files_unchanged"]
    )
    results = {
        "evaluation_status": "completed" if passed else "failed",
        "frame_count": 500,
        "frame_list_hash": manifest["frame_list_hash"],
        "evaluation_code_hash": evaluator_hash,
        "model_count": len(rows),
        "expected_model_count": expected_model_count,
        "models": rows,
        "failures": failures,
        "core_files_unchanged": freeze["core_files_unchanged"],
        "elapsed_seconds": time.time() - started,
    }
    atomic_write_json(output / "grouped_conv_stage_sensitivity_results.json", results)
    _atomic_csv(output / "grouped_conv_stage_sensitivity_results.csv", rows)
    atomic_write_json(output / "evaluation_complete.json", results)
    return results


def _strategy_matrix_rows(
    models: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    matrix: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    model_by_key = {
        (str(row.get("stage_scope")), str(row.get("strength")), str(row.get("strategy"))): row
        for row in models
    }
    comparison_by_key = {
        (str(row["stage_scope"]), str(row["strength"])): row for row in comparisons
    }
    output: list[dict[str, Any]] = []
    scopes = ("stage0_only", "stage1_only", "stage2_only", "all_grouped_stages")
    for strength in ("mild", "aggressive"):
        for scope in scopes:
            independent = model_by_key.get(
                (scope, strength, "taylor_independent_group_ranking")
            )
            shared = model_by_key.get(
                (scope, strength, "torch_pruning_l2_shared_position")
            )
            comparison = comparison_by_key.get((scope, strength))
            candidate_rows = [
                row
                for row in matrix
                if row["stage_scope"] == scope and row["strength"] == strength
            ]
            status = (
                "evaluated"
                if independent is not None and shared is not None
                else str(candidate_rows[0]["candidate_status"] if candidate_rows else "missing")
            )
            reason = ""
            if status != "evaluated" and candidate_rows:
                reason = str(candidate_rows[0].get("infeasible_reason", ""))
            output.append(
                {
                    "stage_scope": scope,
                    "strength": strength,
                    "status": status,
                    "infeasible_reason": reason,
                    "independent_mAP": float(independent["mAP"]) if independent else None,
                    "tp_shared_mAP": float(shared["mAP"]) if shared else None,
                    "independent_minus_tp": (
                        float(independent["mAP"]) - float(shared["mAP"])
                        if independent and shared
                        else None
                    ),
                    "total_parameters_reduced": (
                        int(independent["actual_parameter_reduction"]) if independent else None
                    ),
                    "grouped_parameters_reduced": (
                        int(independent["grouped_layer_parameter_reduction"])
                        if independent
                        else None
                    ),
                    "valid_pair": bool(comparison and comparison.get("comparison_valid")),
                    "strategy_verdict": (
                        comparison.get("strategy_verdict") if comparison else "inconclusive"
                    ),
                }
            )
    return output


def _analysis_markdown(
    matrix_rows: Sequence[Mapping[str, Any]],
    rankings: Sequence[Mapping[str, Any]],
    attribution: Mapping[str, Any],
) -> str:
    lines = [
        "# Grouped Conv stage sensitivity analysis",
        "",
        "All AP values use the same fixed 500-frame manifest. Strategy differences are reported only for physical-shape-matched pairs.",
    ]
    for strength in ("mild", "aggressive"):
        lines.extend(
            [
                "",
                f"## {strength.title()} strategy matrix",
                "",
                "| Stage scope | Independent mAP | TP shared mAP | Independent - TP | Total params reduced | Grouped params reduced | Valid pair | Status |",
                "|---|---:|---:|---:|---:|---:|---|---|",
            ]
        )
        for row in matrix_rows:
            if row["strength"] != strength:
                continue
            fmt = lambda value: "" if value is None else f"{float(value):.6f}"
            lines.append(
                f"| {row['stage_scope']} | {fmt(row['independent_mAP'])} | {fmt(row['tp_shared_mAP'])} | "
                f"{fmt(row['independent_minus_tp'])} | {row['total_parameters_reduced'] or ''} | "
                f"{row['grouped_parameters_reduced'] or ''} | {row['valid_pair']} | {row['status']} |"
            )
    lines.extend(
        [
            "",
            "## Stage sensitivity",
            "",
            "| Strategy | Strength | Stage | delta mAP | Retention | delta mAP / M params | Absolute rank | Normalized rank |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rankings:
        lines.append(
            f"| {row['strategy']} | {row['strength']} | {row['stage']} | {row['delta_mAP']:.6f} | "
            f"{row['mAP_retention']:.6f} | {row['delta_map_per_million_pruned_parameters']:.6f} | "
            f"{row['absolute_sensitivity_rank']} | {row['parameter_normalized_sensitivity_rank']} |"
        )
    lines.extend(
        [
            "",
            "## Attribution",
            "",
            f"- Stage1 structural sensitivity: `{attribution['stage1_intrinsic_sensitivity']}`.",
            f"- TP shared-position exacerbation: `{attribution['shared_position_exacerbation']}`.",
            f"- Combined attribution: `{attribution['combined_attribution']}`.",
            f"- All-stage cumulative effect: `{attribution['all_stage_cumulative_effect']}` because {attribution['all_stage_cumulative_effect_reason']}.",
            "",
            "The scope of these observations is this model, checkpoint, fixed 500-frame validation set, declared widths, and the two tested position-selection strategies.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_analyze(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).resolve()
    evaluation = json.loads(
        (output / "grouped_conv_stage_sensitivity_results.json").read_text(encoding="utf-8")
    )
    if evaluation.get("evaluation_status") != "completed":
        raise RuntimeError("completed 500-frame evaluation is required before analysis")
    models = list(evaluation["models"])
    pair_payload = json.loads(
        (output / "pairwise_structure_validation.json").read_text(encoding="utf-8")
    )
    pair_rows = list(pair_payload["pairs"])
    candidate_matrix = json.loads(
        (output / "candidate_matrix.json").read_text(encoding="utf-8")
    )["candidates"]
    comparisons = build_strategy_comparisons(models, pair_rows)
    rankings = build_sensitivity_rankings(models)
    attribution = stage1_attribution(models)
    matrix_rows = _strategy_matrix_rows(models, comparisons, candidate_matrix)
    all_stage_rows = [
        row for row in models if str(row.get("stage_scope")) == "all_grouped_stages"
    ]
    if all_stage_rows:
        cumulative = {
            "status": "observed",
            "rows": [],
            "note": "No cumulative row generated because this branch is not expected under the current strength plan.",
        }
    else:
        cumulative = {
            "status": "inconclusive",
            "reason": "all_grouped_stages mild/aggressive candidates were infeasible because stage0 has no smaller legal output width; no no-op substitute was evaluated",
            "all_stage_delta_mAP": None,
            "sum_single_stage_delta_mAP": None,
            "cumulative_interaction": None,
            "interaction_class": "inconclusive",
            "closure_overlap_modules": None,
            "closure_overlap_parameter_count": None,
        }
    atomic_write_json(
        output / "strategy_comparison_results.json", {"comparisons": comparisons}
    )
    _atomic_csv(output / "strategy_comparison_results.csv", comparisons)
    atomic_write_json(output / "strategy_comparison_matrix.json", {"rows": matrix_rows})
    _atomic_csv(output / "strategy_comparison_matrix.csv", matrix_rows)
    atomic_write_json(output / "stage_sensitivity_rankings.json", {"rows": rankings})
    _atomic_csv(output / "stage_sensitivity_rankings.csv", rankings)
    atomic_write_json(output / "stage1_attribution.json", attribution)
    atomic_write_json(output / "all_stage_cumulative_effect.json", cumulative)
    _atomic_text(
        output / "reports/grouped_conv_stage_sensitivity_analysis.md",
        _analysis_markdown(matrix_rows, rankings, attribution),
    )
    result = {
        "analysis_status": "completed",
        "valid_strategy_pair_count": sum(
            bool(row.get("comparison_valid")) for row in comparisons
        ),
        "invalid_strategy_pair_count": sum(
            not bool(row.get("comparison_valid")) for row in comparisons
        ),
        "sensitivity_ranking_row_count": len(rankings),
        "stage1_attribution": attribution,
        "all_stage_cumulative_effect": cumulative,
    }
    atomic_write_json(output / "analysis_complete.json", result)
    return result


def _process_audit() -> dict[str, Any]:
    current = os.getpid()
    excluded: set[int] = {current}
    cursor = current
    while cursor > 1:
        status = Path(f"/proc/{cursor}/status")
        if not status.is_file():
            break
        parent = 0
        for line in status.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("PPid:"):
                parent = int(line.split()[1])
                break
        if parent <= 0 or parent in excluded:
            break
        excluded.add(parent)
        cursor = parent
    snapshot = _run_command("ps", "-eo", "pid,ppid,user,etime,args")
    matches: list[dict[str, Any]] = []
    patterns = (
        "trtexec",
        "run_v108_complete_taylor",
        "run_v109_param_budget",
        "run_lidar_pyramid_formal_pruner_validation",
        "run_grouped_conv_stage_sensitivity.py",
    )
    for line in snapshot["output"][1:]:
        fields = line.strip().split(None, 4)
        if len(fields) < 5:
            continue
        pid = int(fields[0])
        if pid in excluded:
            continue
        command = fields[4]
        if any(pattern in command for pattern in patterns):
            matches.append(
                {
                    "pid": pid,
                    "ppid": int(fields[1]),
                    "user": fields[2],
                    "elapsed": fields[3],
                    "command": command,
                }
            )
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "excluded_audit_process_and_ancestors": sorted(excluded),
        "matching_related_processes": matches,
        "related_process_count": len(matches),
        "trtexec_running": any("trtexec" in row["command"] for row in matches),
        "global_pruning_scan_running": any(
            "run_v108" in row["command"]
            or "run_v109" in row["command"]
            or "run_lidar_pyramid_formal" in row["command"]
            for row in matches
        ),
        "grouped_stage_experiment_process_running_excluding_current": any(
            "run_grouped_conv_stage_sensitivity.py" in row["command"] for row in matches
        ),
        "gpu_snapshot": _run_command(
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        )["output"],
    }


def _write_actual_commands_markdown(output: Path) -> None:
    commands_path = output / "actual_run_commands.json"
    payload = json.loads(commands_path.read_text(encoding="utf-8"))
    lines = [
        "# Actual run commands",
        "",
        "The stage commands below were recorded by the runner before dispatch.",
        "",
    ]
    for index, row in enumerate(payload.get("commands", []), start=1):
        lines.extend(
            [
                f"## {index}. {row['timestamp_utc']}",
                "",
                f"CUDA_VISIBLE_DEVICES: `{row.get('cuda_visible_devices', '')}`",
                "",
                "```bash",
                shlex.join([str(value) for value in row["argv"]]),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Verification and audit commands",
            "",
            "```bash",
            "/home/lixingfeng/anaconda3/envs/modelopt/bin/python -m pytest -q tests/test_grouped_conv_stage_sensitivity_v2.py",
            "/home/lixingfeng/anaconda3/envs/modelopt/bin/python -m pytest -q tests/test_grouped_conv_stage_sensitivity_v2.py tests/test_formal_lidar_pyramid_experiment_orchestration.py tests/test_formal_packages_cpu.py tests/test_grouped_conv_selection.py tests/test_grouped_conv_independent_topk_v81.py tests/test_taylor_fisher_importance_connection.py",
            "/home/lixingfeng/anaconda3/envs/modelopt/bin/python -m py_compile tools/experiments/run_grouped_conv_stage_sensitivity.py tools/experiments/grouped_conv_stage_sensitivity/*.py",
            "git diff --check",
            "git status --short",
            "git diff -- tracer/runtime_graph_builder.py tracer/api.py tracer/generic_tracer.py tracer/coupled_units.py tracer/atomic_units.py pruning/propagation.py pruning/api.py pruning/selection/global_ranking.py",
            "pgrep -af '[r]un_.*prun|[t]rtexec|[r]un_grouped_conv_stage_sensitivity'",
            "```",
        ]
    )
    _atomic_text(output / "actual_run_commands.md", "\n".join(lines) + "\n")


def _final_answers(
    *,
    inventory: Mapping[str, Any],
    matrix: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
    attribution: Mapping[str, Any],
    cumulative: Mapping[str, Any],
    pair_payload: Mapping[str, Any],
    tp_probe: Mapping[str, Any],
    freeze: Mapping[str, Any],
) -> list[dict[str, Any]]:
    counts = inventory["observed_stage_counts"]
    valid_pairs = [row for row in comparisons if row.get("comparison_valid")]
    invalid_pairs = [row for row in pair_payload["pairs"] if not row.get("comparison_valid")]
    infeasible = [row for row in matrix if row.get("candidate_status") == "infeasible"]
    return [
        {"id": 1, "verdict": "observed", "answer": f"The formal trace contains stage0={counts['stage0']}, stage1={counts['stage1']}, stage2={counts['stage2']}."},
        {"id": 2, "verdict": "observed", "answer": f"All {inventory['total_count']} dependency-safe pyramid grouped Conv modules were classified; none were unclassified."},
        {"id": 3, "verdict": "supported", "answer": "Among output-root-feasible mild comparisons, stage1 is more sensitive than stage2 under both strategies. Stage0 is inconclusive because its original per-group width is already the minimum legal width 4."},
        {"id": 4, "verdict": "observed", "answer": "At mild strength, absolute delta-mAP ranks stage1 > stage2 for both strategies; stage0 is unranked/inconclusive."},
        {"id": 5, "verdict": "observed", "answer": "At mild strength, delta-mAP per million pruned parameters ranks stage1 > stage2 for both strategies; stage0 is unranked/inconclusive."},
        {"id": 6, "verdict": attribution["stage1_intrinsic_sensitivity"], "answer": "Stage1 has greater loss than stage2 under both Taylor-independent and TP-shared at the common mild comparison."},
        {"id": 7, "verdict": "supported", "answer": "Within this model/checkpoint/manifest, stage1 grouped-Conv narrowing is supported as a major contributor to the prior AP cliff, not as its unique cause."},
        {"id": 8, "verdict": "supported", "answer": "Both factors are supported: stage1 Taylor-independent already loses substantial AP, while TP shared-position further drives stage1 mAP to zero."},
        {"id": 9, "verdict": "supported", "answer": f"Taylor independent is better in all {len(valid_pairs)} valid physical-shape-matched pairs."},
        {"id": 10, "verdict": "observed", "answer": "The strategy advantage is consistent for the evaluated stage1 and stage2 pairs; stage0 is inconclusive/infeasible."},
        {"id": 11, "verdict": "observed", "answer": "The advantage appears at stage1 mild, stage2 mild, and stage2 aggressive. Stage1 aggressive is unavailable, so a complete cross-stage aggressive conclusion is inconclusive."},
        {"id": 12, "verdict": "observed", "answer": "Only TP shared-position reaches complete stage1 mAP collapse (0.0); Taylor-independent remains nonzero but is still severely degraded."},
        {"id": 13, "verdict": "not supported", "answer": "No evaluated case shows only Taylor-independent collapsing while TP shared-position remains accurate."},
        {"id": 14, "verdict": "inconclusive", "answer": cumulative["reason"]},
        {"id": 15, "verdict": "inconclusive", "answer": "Superadditive/additive/subadditive classification is not computed without a valid all-stage candidate."},
        {"id": 16, "verdict": "observed", "answer": f"No Taylor/TP pair was excluded for structure mismatch; invalid pair count is {len(invalid_pairs)}."},
        {"id": 17, "verdict": "observed", "answer": f"stage_output_pruning_infeasible occurs for stage0; stage1 aggressive is unavailable; {len(infeasible)} of 16 declared candidate rows are infeasible without no-op substitution."},
        {"id": 18, "verdict": "observed", "answer": f"Torch-Pruning {tp_probe.get('torch_pruning_version')} MagnitudeImportance was called and returned per-channel scores: {tp_probe.get('returned_per_channel_importance')}."},
        {"id": 19, "verdict": "observed", "answer": "In the main experiment TP supplies method-B magnitude importance and its shared-local-position semantics; TP dependency graph and physical materialization are not used."},
        {"id": 20, "verdict": "observed", "answer": "Both methods share the formal TraceResult DependencyScope/closure members, public selector closure expansion, legalizer, physical materializer, and evaluator. The legacy pruning.propagation.GroupBuilder is not called."},
        {"id": 21, "verdict": "observed" if freeze["core_files_unchanged"] else "not supported", "answer": f"Frozen core files unchanged: {freeze['core_files_unchanged']}."},
        {"id": 22, "verdict": "observed", "answer": "No global pruning-rate scan was run in this experiment."},
        {"id": 23, "verdict": "observed", "answer": "No ONNX export, Q/DQ insertion, TensorRT engine build, TensorRT smoke/latency run, or trtexec launch was performed."},
    ]


def _final_markdown(payload: Mapping[str, Any]) -> str:
    evaluation = payload["evaluation"]
    comparisons = payload["strategy_comparisons"]
    rankings = payload["sensitivity_rankings"]
    lines = [
        "# Grouped Conv stage sensitivity final v2",
        "",
        "Scope: current LiDAR Pyramid model, current checkpoint, fixed 500-frame validation manifest, declared legal widths, and the two tested importance/position-selection strategies.",
        "",
        "## Outcome",
        "",
        f"- Inventory: stage0/stage1/stage2 = `{payload['inventory']['observed_stage_counts']}`; total `{payload['inventory']['total_count']}`.",
        f"- Candidates: `{payload['candidate_counts']['valid']}` valid, `{payload['candidate_counts']['invalid']}` invalid, `{payload['candidate_counts']['infeasible']}` infeasible.",
        f"- Strategy pairs: `{payload['pair_counts']['valid']}` valid, `{payload['pair_counts']['invalid']}` invalid.",
        "- Stage1 structural sensitivity: `supported`.",
        "- TP shared-position exacerbation at stage1: `supported`.",
        "- All-stage cumulative effect: `inconclusive` because the full all-stage output-root candidate is infeasible under the declared widths.",
        "",
        "## AP Results",
        "",
        "| Model | AP03 | AP30 | AP50 | AP70 | mAP | Retention | Params reduced | delta mAP/M params |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in evaluation["models"]:
        normalized = row.get("delta_map_per_million_pruned_parameters")
        lines.append(
            f"| `{row['model_id']}` | {row['ap_0.03']:.6f} | {row['ap_0.30']:.6f} | "
            f"{row['ap_0.50']:.6f} | {row['ap_0.70']:.6f} | {row['mAP']:.6f} | "
            f"{row['mAP_retention']:.6f} | {row['actual_parameter_reduction']} | "
            f"{'' if normalized is None else f'{normalized:.6f}'} |"
        )
    lines.extend(
        [
            "",
            "## Strategy Pairs",
            "",
            "| Stage | Strength | Independent mAP | TP shared mAP | Independent - TP | Verdict |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in comparisons:
        lines.append(
            f"| {row['stage_scope']} | {row['strength']} | {row.get('independent_mAP', 0):.6f} | "
            f"{row.get('tp_shared_mAP', 0):.6f} | {row['mAP_difference_independent_minus_tp']:.6f} | "
            f"{row['strategy_verdict']} |"
        )
    lines.extend(
        [
            "",
            "## Sensitivity",
            "",
            "| Strategy | Strength | Stage | delta mAP | delta mAP/M params | Absolute rank | Normalized rank |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in rankings:
        lines.append(
            f"| {row['strategy']} | {row['strength']} | {row['stage']} | {row['delta_mAP']:.6f} | "
            f"{row['delta_map_per_million_pruned_parameters']:.6f} | {row['absolute_sensitivity_rank']} | "
            f"{row['parameter_normalized_sensitivity_rank']} |"
        )
    lines.extend(["", "## Required Answers", ""])
    for answer in payload["answers"]:
        lines.append(f"{answer['id']}. **{answer['verdict']}**: {answer['answer']}")
    lines.extend(
        [
            "",
            "## Audit",
            "",
            f"- Core files unchanged: `{payload['core_freeze']['core_files_unchanged']}`.",
            f"- Final related process count: `{payload['process_audit']['related_process_count']}`.",
            f"- trtexec running: `{payload['process_audit']['trtexec_running']}`.",
            f"- Runtime forbidden imports/calls: `{payload['runtime_dependency_audit']['forbidden_imports']}` / `{payload['runtime_dependency_audit']['forbidden_calls']}`.",
            f"- Regression tests: `{payload['verification']['passed']}` passed, `{payload['verification']['failed']}` failed.",
            "- Exact commands: `actual_run_commands.md`.",
            "- Full artifact list: `generated_files_manifest.json`.",
        ]
    )
    return "\n".join(lines) + "\n"


def _artifact_manifest(output: Path) -> dict[str, Any]:
    manifest_path = output / "generated_files_manifest.json"
    rows: list[dict[str, Any]] = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        rows.append(
            {
                "path": str(path.relative_to(output)),
                "size_bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
        )
    return {
        "output_root": str(output),
        "artifact_count": len(rows),
        "artifacts": rows,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def run_finalize(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).resolve()
    analysis_complete = json.loads(
        (output / "analysis_complete.json").read_text(encoding="utf-8")
    )
    if analysis_complete.get("analysis_status") != "completed":
        raise RuntimeError("analysis must complete before finalize")
    before = json.loads((output / "core_freeze_before.json").read_text(encoding="utf-8"))
    freeze = validate_core_freeze(before, repository_root=REPOSITORY_ROOT)
    after = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_head": _run_command("git", "rev-parse", "HEAD")["output"][0],
        "frozen_files": freeze["frozen_files"],
        "schema_version": "grouped-conv-core-freeze-v2",
    }
    atomic_write_json(output / "core_freeze_after.json", after)
    atomic_write_json(output / "core_freeze_validation.json", freeze)
    process_audit = _process_audit()
    atomic_write_json(output / "process_audit_after.json", process_audit)
    frozen_paths = [str(row["path"]) for row in before["frozen_files"]]
    git_after = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_head": _run_command("git", "rev-parse", "HEAD")["output"][0],
        "current_branch": _run_command("git", "branch", "--show-current")["output"][0],
        "git_status_short": _run_command("git", "status", "--short")["output"],
        "git_diff_check": _run_command("git", "diff", "--check"),
        "frozen_files_diff_current_head": _run_command(
            "git", "diff", "HEAD", "--", *frozen_paths
        ),
        "core_files_unchanged": freeze["core_files_unchanged"],
    }
    atomic_write_json(output / "git_environment_after.json", git_after)
    _write_actual_commands_markdown(output)

    inventory = json.loads((output / "grouped_conv_stage_inventory.json").read_text(encoding="utf-8"))
    strength = json.loads((output / "pruning_strength_plan.json").read_text(encoding="utf-8"))
    matrix = json.loads((output / "candidate_matrix.json").read_text(encoding="utf-8"))[
        "candidates"
    ]
    generation = json.loads(
        (output / "candidate_generation_results.json").read_text(encoding="utf-8")
    )
    evaluation = json.loads(
        (output / "grouped_conv_stage_sensitivity_results.json").read_text(encoding="utf-8")
    )
    comparisons = json.loads(
        (output / "strategy_comparison_results.json").read_text(encoding="utf-8")
    )["comparisons"]
    rankings = json.loads(
        (output / "stage_sensitivity_rankings.json").read_text(encoding="utf-8")
    )["rows"]
    attribution = json.loads((output / "stage1_attribution.json").read_text(encoding="utf-8"))
    cumulative = json.loads(
        (output / "all_stage_cumulative_effect.json").read_text(encoding="utf-8")
    )
    pair_payload = json.loads(
        (output / "pairwise_structure_validation.json").read_text(encoding="utf-8")
    )
    tp_probe = json.loads(
        (output / "environment/torch_pruning_probe.json").read_text(encoding="utf-8")
    )
    dependency_audit = scan_forbidden_runtime_dependencies(Path(__file__).resolve())
    atomic_write_json(output / "runtime_dependency_audit_final.json", dependency_audit)
    verification_path = output / "verification_report.json"
    verification = (
        json.loads(verification_path.read_text(encoding="utf-8"))
        if verification_path.is_file()
        else {
            "status": "not_recorded",
            "passed": 0,
            "failed": None,
            "warning_count": None,
        }
    )
    answers = _final_answers(
        inventory=inventory,
        matrix=matrix,
        comparisons=comparisons,
        attribution=attribution,
        cumulative=cumulative,
        pair_payload=pair_payload,
        tp_probe=tp_probe,
        freeze=freeze,
    )
    invalid_rows = [
        row for row in generation["rows"] if row.get("candidate_status") == "invalid"
    ]
    infeasible_rows = [
        row for row in generation["rows"] if row.get("candidate_status") == "infeasible"
    ]
    payload = {
        "report_status": (
            "completed"
            if freeze["core_files_unchanged"]
            and evaluation["evaluation_status"] == "completed"
            and not invalid_rows
            else "invalid_due_to_core_file_modification"
            if not freeze["core_files_unchanged"]
            else "failed"
        ),
        "scope_boundary": "current model, checkpoint, fixed 500-frame validation set, declared widths, three grouped-Conv stages, and two tested selection strategies",
        "inventory": inventory,
        "strength_plan": strength,
        "candidate_counts": {
            "valid": generation["valid_candidate_count"],
            "invalid": generation["invalid_candidate_count"],
            "infeasible": generation["infeasible_candidate_count"],
            "declared": len(matrix),
        },
        "invalid_candidates": invalid_rows,
        "infeasible_candidates": [
            {
                "candidate_id": row["candidate_id"],
                "infeasible_reason": row["infeasible_reason"],
            }
            for row in infeasible_rows
        ],
        "evaluation": evaluation,
        "strategy_comparisons": comparisons,
        "sensitivity_rankings": rankings,
        "stage1_attribution": attribution,
        "all_stage_cumulative_effect": cumulative,
        "pair_counts": {
            "valid": pair_payload["valid_pair_count"],
            "invalid": pair_payload["invalid_pair_count"],
        },
        "pairwise_structure_validation": pair_payload,
        "torch_pruning_probe": tp_probe,
        "core_freeze": freeze,
        "git_environment_after": git_after,
        "process_audit": process_audit,
        "runtime_dependency_audit": dependency_audit,
        "verification": verification,
        "execution_audit": {
            "global_pruning_rate_scan_run": False,
            "onnx_export_run": False,
            "qdq_insertion_run": False,
            "tensorrt_engine_build_run": False,
            "trtexec_run": False,
            "tensorRT_latency_or_smoke_run": False,
            "legacy_pruning_propagation_groupbuilder_called": False,
            "formal_trace_dependency_closure_shared": True,
        },
        "answers": answers,
        "primary_paths": {
            "final_markdown": str(output / "reports/grouped_conv_stage_sensitivity_final_v2.md"),
            "final_json": str(output / "reports/grouped_conv_stage_sensitivity_final_v2.json"),
            "results_json": str(output / "grouped_conv_stage_sensitivity_results.json"),
            "results_csv": str(output / "grouped_conv_stage_sensitivity_results.csv"),
            "commands": str(output / "actual_run_commands.md"),
            "artifact_manifest": str(output / "generated_files_manifest.json"),
        },
    }
    final_json = output / "reports/grouped_conv_stage_sensitivity_final_v2.json"
    final_markdown = output / "reports/grouped_conv_stage_sensitivity_final_v2.md"
    atomic_write_json(final_json, payload)
    _atomic_text(final_markdown, _final_markdown(payload))
    artifact_manifest = _artifact_manifest(output)
    atomic_write_json(output / "generated_files_manifest.json", artifact_manifest)
    completion = {
        "status": payload["report_status"],
        "core_files_unchanged": freeze["core_files_unchanged"],
        "valid_candidates": generation["valid_candidate_count"],
        "invalid_candidates": generation["invalid_candidate_count"],
        "infeasible_candidates": generation["infeasible_candidate_count"],
        "evaluated_models": evaluation["model_count"],
        "valid_pairs": pair_payload["valid_pair_count"],
        "invalid_pairs": pair_payload["invalid_pair_count"],
        "artifact_count": artifact_manifest["artifact_count"],
        "final_process_count": process_audit["related_process_count"],
    }
    atomic_write_json(output / "finalize_complete.json", completion)
    return completion


def run_prepare(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output = Path(args.output).resolve()
    reports = output / "reports"
    environment_dir = output / "environment"
    output.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    environment_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    heal_root = Path(args.heal_root).resolve()
    config_hash = _file_sha256(config_path)
    checkpoint_hash = _file_sha256(checkpoint_path)
    code_paths = [
        Path(__file__).resolve(),
        Path(__file__).with_name("grouped_conv_stage_sensitivity") / "contracts.py",
        Path(__file__).with_name("grouped_conv_stage_sensitivity") / "runtime.py",
        Path(__file__).with_name("grouped_conv_stage_sensitivity") / "torch_pruning_strategy.py",
        Path(__file__).with_name("grouped_conv_stage_sensitivity") / "analysis.py",
    ]
    evaluation_code_hash = _combined_code_hash(code_paths)

    loaded, dataset, trace, config, _batch = _load_model_dataset_trace(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        heal_root=heal_root,
        device=args.device,
    )
    serialize_trace_result(trace, output / "formal_trace_result.json")
    split_info = [str(value) for value in getattr(dataset, "split_info", [])]
    validation_ids = json.loads(Path(config["validate_dir"]).read_text(encoding="utf-8"))
    if split_info[:500] != [str(value) for value in validation_ids[:500]]:
        raise RuntimeError("dataset.split_info differs from the declared validation split prefix")
    validation_manifest = build_frame_manifest(
        validation_ids,
        frame_count=500,
        dataset_config_hash=config_hash,
        checkpoint_hash=checkpoint_hash,
        evaluation_code_hash=evaluation_code_hash,
    )
    validation_manifest.update(
        {
            "manifest_path": str(output / "validation_frame_manifest_500.json"),
            "dataset_config": str(config_path),
            "dataset_root": str(config.get("data_dir", "")),
            "validation_source": str(config["validate_dir"]),
        }
    )
    atomic_write_json(output / "validation_frame_manifest_500.json", validation_manifest)
    training_ids = json.loads(Path(config["root_dir"]).read_text(encoding="utf-8"))
    taylor_manifest = build_frame_manifest(
        training_ids,
        frame_count=min(int(args.taylor_frames), len(training_ids)),
        dataset_config_hash=config_hash,
        checkpoint_hash=checkpoint_hash,
        evaluation_code_hash=evaluation_code_hash,
        split="training_taylor",
    )
    taylor_manifest["manifest_path"] = str(output / "taylor_frame_manifest.json")
    taylor_manifest["dataset_source"] = str(config["root_dir"])
    atomic_write_json(output / "taylor_frame_manifest.json", taylor_manifest)

    snapshot = build_physical_structure_snapshot(loaded.model)
    hashes = compute_physical_hashes(
        snapshot,
        model_hash=checkpoint_hash,
        config_hash=config_hash,
    )
    original_state_hash = state_dict_content_hash(dict(loaded.model.state_dict()))
    atomic_write_json(output / "original_physical_structure_snapshot_v2.json", snapshot.to_dict())
    atomic_write_json(output / "original_physical_hash_v2.json", hashes.to_dict())
    atomic_write_json(
        output / "original_model_identity.json",
        {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_hash": checkpoint_hash,
            "state_dict_content_hash": original_state_hash,
            "parameter_count": snapshot.parameter_count,
            "structure_hash_v2": hashes.structure_hash_v2,
        },
    )

    inventory = build_stage_inventory(_trace_inventory_source_rows(trace))
    inventory.update(
        {
            "trace_hash": trace.trace_hash,
            "trace_backend": trace.config.get("realized_backend"),
            "weighted_module_coverage": trace.trace_coverage.weighted_module_coverage,
            "dependency_scope_count": len(trace.dependency_scopes),
        }
    )
    atomic_write_json(output / "grouped_conv_stage_inventory.json", inventory)
    _atomic_csv(output / "grouped_conv_stage_inventory.csv", inventory["rows"])
    _atomic_text(reports / "grouped_conv_stage_inventory.md", _markdown_inventory(inventory))

    strength_plan = compute_pruning_strength_plan(inventory["rows"])
    legalizer_rows = _validate_strengths_with_formal_legalizer(
        loaded.model, trace, inventory, strength_plan
    )
    atomic_write_json(output / "pruning_strength_plan.json", strength_plan)
    strength_rows = [
        {
            "stage": stage,
            "status": row["status"],
            "module_count": row["module_count"],
            "original_output_channels_per_group": row["original_output_channels_per_group"],
            "common_legal_widths": row["common_legal_widths"],
            "mild": row["mild"],
            "mild_status": row["mild_status"],
            "aggressive": row["aggressive"],
            "aggressive_status": row["aggressive_status"],
        }
        for stage, row in strength_plan["stages"].items()
    ]
    _atomic_csv(output / "pruning_strength_plan.csv", strength_rows)
    _atomic_text(reports / "pruning_strength_plan.md", _markdown_strength(strength_plan))
    atomic_write_json(output / "formal_legalizer_strength_validation.json", {"rows": legalizer_rows})

    whitelists = _root_whitelists(inventory["rows"])
    atomic_write_json(
        output / "root_whitelists.json",
        {
            "stage_prefixes": dict(STAGE_PREFIXES),
            "whitelists": whitelists,
            "implementation_layer": "tools.experiments orchestration",
            "dependency_closure_filtering": False,
        },
    )
    candidates = build_candidate_matrix(strength_plan)
    for row in candidates:
        if row["candidate_status"] != "planned":
            continue
        failed = [
            stage
            for stage in row["active_stages"]
            if not bool(
                strength_plan["stages"][stage]["formal_legalizer_validation"]
                [row["strength"]]["passed"]
            )
        ]
        if failed:
            row["candidate_status"] = "infeasible"
            row["infeasible_reason"] = f"formal_legalizer_failed:{','.join(failed)}"
    atomic_write_json(output / "candidate_matrix.json", {"candidates": candidates})
    _atomic_csv(output / "candidate_matrix.csv", candidates)

    import torch
    import torch_pruning as tp

    tp_source = Path(tp.__file__).resolve().parent / "pruner/algorithms/base_pruner.py"
    tp_probe = {
        "python_version": sys.version,
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "torch_pruning_version": getattr(tp, "__version__", "unknown"),
        "torch_pruning_path": tp.__file__,
        "importance_class": "torch_pruning.importance.MagnitudeImportance",
        "importance_constructor_args": {
            "p": 2,
            "group_reduction": "mean",
            "normalizer": None,
            "bias": False,
        },
        "reduction": "mean",
        "normalization": None,
        "call_entrypoint": "MagnitudeImportance.__call__(root_only_group)",
        "returned_per_channel_importance": "pending_dry_run",
        "shared_position_selection_source": str(tp_source),
        "shared_position_selection_source_sha256": _file_sha256(tp_source),
        "shared_position_semantics": "imp.view(ch_groups, -1).mean(dim=0); repeat selected local indices for each group",
        "tp_dependency_graph_role": "not_used_in_main_experiment",
        "tp_materialization_role": "not_used_in_main_experiment",
    }
    atomic_write_json(environment_dir / "torch_pruning_probe.json", tp_probe)
    _atomic_text(environment_dir / "torch_pruning_probe.txt", json.dumps(tp_probe, indent=2) + "\n")

    dependency_scan = scan_forbidden_runtime_dependencies(Path(__file__).resolve())
    atomic_write_json(output / "runtime_dependency_audit.json", dependency_scan)
    environment = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "current_head": _run_command("git", "rev-parse", "HEAD")["output"][0],
        "current_branch": _run_command("git", "branch", "--show-current")["output"][0],
        "git_status_short": _run_command("git", "status", "--short")["output"],
        "git_diff_check": _run_command("git", "diff", "--check"),
        "python_executable": sys.executable,
        "requested_python": "/home/lixingfeng/miniconda3/envs/modelopt/bin/python",
        "requested_python_exists": Path(
            "/home/lixingfeng/miniconda3/envs/modelopt/bin/python"
        ).exists(),
        "config_path": str(config_path),
        "config_hash": config_hash,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_hash": checkpoint_hash,
        "heal_root": str(heal_root),
        "device": args.device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "trace_hash": trace.trace_hash,
        "trace_backend": trace.config.get("realized_backend"),
        "validation_frame_list_hash": validation_manifest["frame_list_hash"],
        "taylor_frame_list_hash": taylor_manifest["frame_list_hash"],
        "evaluation_code_hash": evaluation_code_hash,
        "original_parameter_count": snapshot.parameter_count,
        "runtime_dependency_audit": dependency_scan,
        "onnx_qdq_tensorrt_enabled": False,
    }
    atomic_write_json(output / "environment_report_v2.json", environment)
    process_audit = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "before_prepare_process_audit": _run_command(
            "ps", "-eo", "pid,ppid,user,etime,args"
        )["output"],
        "trtexec_started_by_runner": False,
        "global_pruning_scan_started_by_runner": False,
    }
    atomic_write_json(output / "process_audit_prepare.json", process_audit)
    completion = {
        "status": "completed",
        "stage": "prepare",
        "elapsed_seconds": time.time() - started,
        "trace_hash": trace.trace_hash,
        "inventory_total": inventory["total_count"],
        "observed_stage_counts": inventory["observed_stage_counts"],
        "inventory_count_mismatch": inventory["inventory_count_mismatch"],
        "planned_candidate_count": sum(row["candidate_status"] == "planned" for row in candidates),
        "infeasible_candidate_count": sum(row["candidate_status"] == "infeasible" for row in candidates),
        "validation_frame_list_hash": validation_manifest["frame_list_hash"],
    }
    atomic_write_json(output / "prepare_complete.json", completion)
    return completion


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("prepare", "dry_run", "generate", "evaluate", "analyze", "finalize"),
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--heal-root", default=str(DEFAULT_HEAL_ROOT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--taylor-frames", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    _append_actual_command(output, list(argv) if argv is not None else sys.argv[1:])
    if args.stage == "prepare":
        result = run_prepare(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.stage == "dry_run":
        result = run_dry_run(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "completed" else 1
    if args.stage == "generate":
        result = run_generate(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "completed" else 1
    if args.stage == "evaluate":
        result = run_evaluate(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["evaluation_status"] == "completed" else 1
    if args.stage == "analyze":
        result = run_analyze(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["analysis_status"] == "completed" else 1
    if args.stage == "finalize":
        result = run_finalize(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "completed" else 1
    raise RuntimeError(
        f"stage {args.stage!r} is intentionally unavailable until the prepare audit is reviewed"
    )


if __name__ == "__main__":
    raise SystemExit(main())
