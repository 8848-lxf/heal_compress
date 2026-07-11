#!/usr/bin/env python3
"""LiDAR-pyramid validation orchestration using only formal package APIs.

This module owns experiment contracts, manifests, resume decisions and report
assembly. Tracing, scoring, selection, physical materialization, ONNX/Q/DQ and
TensorRT operations remain delegated to ``tracer.api``, ``pruning.api`` and
``quantization.api``.
"""

from __future__ import annotations

import ast
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
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
    select_pruning_request,
    validate_physical_model,
)
from pruning.artifacts.io import atomic_write_json
from pruning.config import PruningConfig
from pruning.types import SamplingPruningRequest
from pruning.types import PhysicalPruningPlan
from quantization.api import (
    apply_canonical_node_names,
    build_canonical_precision_mapping,
    build_onnx_origin_map,
    build_trt_command,
    build_trt_engine,
    export_pruned_signal_maxk_onnx,
    generate_precision_profile,
    insert_explicit_qdq,
    run_engine_smoke,
    summarize_latency,
    validate_engine_provenance,
    validate_engine_structure,
    validate_precision_realization,
)
from tracer.api import serialize_trace_result, trace_model
from tracer.config import TraceConfig


class ExperimentContractError(RuntimeError):
    """Raised when a result would violate the controlled experiment design."""


def _plain(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _stable_hash(value: Any) -> str:
    raw = json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def select_target_grouped_conv_layers(
    inventory: Sequence[Mapping[str, Any]],
    *,
    expected_count: int = 3,
    pyramid_prefix: str = "pyramid_backbone",
) -> list[dict[str, Any]]:
    """Select dependency-safe pyramid grouped convolutions and require three."""

    selected = []
    for source in inventory:
        row = dict(source)
        path = str(row.get("module_path", ""))
        module_type = str(row.get("module_type", ""))
        groups = int(row.get("groups") or 1)
        if not path.startswith(pyramid_prefix):
            continue
        if module_type not in {"Conv2d", "ConvTranspose2d"} or groups <= 1:
            continue
        if not bool(row.get("root_pruning_allowed", True)):
            continue
        if bool(row.get("fixed_output_contract", False)):
            continue
        if not bool(row.get("dependency_input_pruning_allowed", True)):
            continue
        selected.append(row)
    selected.sort(key=lambda row: str(row["module_path"]))
    if len(selected) != int(expected_count):
        raise ExperimentContractError(
            f"expected exactly {expected_count} dependency-safe pyramid grouped convolutions, found {len(selected)}"
        )
    return selected


def validate_active_roots(request: SamplingPruningRequest, allowed_roots: set[str]) -> None:
    """Reject output/channel requests whose active root is outside the target set."""

    invalid = sorted(
        {
            entry.module_path
            for entry in request.entries
            if entry.axis in {"out", "channel"}
            and not bool(entry.metadata.get("dependency_driven", False))
            and entry.module_path not in allowed_roots
        }
    )
    if invalid:
        raise ExperimentContractError(f"non-target active root(s): {invalid}")


def _module_shapes(snapshot: Any) -> dict[str, dict[str, Any]]:
    payload = _plain(snapshot)
    modules = payload.get("modules", {}) if isinstance(payload, Mapping) else {}
    if isinstance(modules, Mapping):
        return {str(name): dict(row) for name, row in modules.items()}
    return {
        str(row.get("canonical_module_name")): {
            key: row.get(key)
            for key in (
                "module_type",
                "in_channels",
                "out_channels",
                "in_features",
                "out_features",
                "num_features",
                "groups",
                "weight_shape",
                "bias_shape",
                "parameter_count",
            )
        }
        for row in modules
    }


def compare_shape_snapshots(
    left: Any,
    right: Any,
    *,
    require_equivalent: bool = False,
) -> dict[str, Any]:
    """Compare physical module shapes and optionally block invalid AP comparison."""

    left_shapes = _module_shapes(left)
    right_shapes = _module_shapes(right)
    names = sorted(set(left_shapes) | set(right_shapes))
    mismatches = [
        {"module_path": name, "left": left_shapes.get(name), "right": right_shapes.get(name)}
        for name in names
        if left_shapes.get(name) != right_shapes.get(name)
    ]
    result = {"shape_equivalent": not mismatches, "mismatches": mismatches}
    if mismatches and require_equivalent:
        raise ExperimentContractError(f"shape mismatch blocks AP comparison: {mismatches[:3]}")
    return result


def width_feasibility(original_channels_per_group: int, requested_channels_per_group: int) -> dict[str, Any]:
    """Return an explicit feasible/skip decision without ever widening a layer."""

    original = int(original_channels_per_group)
    requested = int(requested_channels_per_group)
    feasible = original > requested and requested in PruningConfig().grouped_conv.allowed_channels_per_group
    return {
        "original_channels_per_group": original,
        "requested_channels_per_group": requested,
        "status": "feasible" if feasible else "skipped_infeasible",
        "reason": "" if feasible else "original_width_not_greater_than_requested_or_width_not_allowed",
    }


def build_frame_manifest(
    frame_ids: Sequence[str],
    *,
    frame_count: int,
    dataset_config_hash: str,
    split: str,
    seed: int,
) -> dict[str, Any]:
    """Select the deterministic prefix of a declared split and hash exact IDs."""

    available = [str(value) for value in frame_ids]
    count = int(frame_count)
    if count <= 0 or len(available) < count:
        raise ExperimentContractError(f"requested {count} frames but split has {len(available)}")
    selected = available[:count]
    frame_hash = _stable_hash(selected)
    return {
        "frame_ids": selected,
        "frame_count": len(selected),
        "frame_list_hash": frame_hash,
        "dataset_config_hash": str(dataset_config_hash),
        "validation_split": str(split),
        "frame_selection_policy": "ordered_split_prefix_v1",
        "random_seed": int(seed),
        "skipped_frame_policy": "fail_closed_no_replacement",
        "schema_version": "validation-frame-manifest-v1",
    }


def validate_dataset_frame_binding(
    manifest_frame_ids: Sequence[str],
    dataset_split_frame_ids: Sequence[str],
    dataset_local_sample_indices: Sequence[int],
) -> dict[str, Any]:
    """Bind authoritative split frame IDs to HEAL's dataset-local sample indices.

    DAIR-V2X datasets expose the real vehicle frame identity through
    ``dataset.split_info``.  Their collated ``sample_idx`` field is only the
    zero-based position in that split and must never be treated as a frame ID.
    """

    expected = [str(value) for value in manifest_frame_ids]
    split_ids = [str(value) for value in dataset_split_frame_ids[: len(expected)]]
    if len(split_ids) != len(expected):
        raise ExperimentContractError(
            f"dataset split has {len(split_ids)} entries for {len(expected)} manifest frames"
        )
    for index, (manifest_id, split_id) in enumerate(zip(expected, split_ids)):
        if manifest_id != split_id:
            raise ExperimentContractError(
                f"split frame mismatch at {index}: manifest={manifest_id}, dataset={split_id}"
            )
    local_indices = [int(value) for value in dataset_local_sample_indices]
    if len(local_indices) != len(expected):
        raise ExperimentContractError(
            f"observed {len(local_indices)} local sample indices for {len(expected)} frames"
        )
    for position, local_index in enumerate(local_indices):
        if local_index != position:
            raise ExperimentContractError(
                f"dataset-local sample index mismatch at {position}: observed={local_index}"
            )
    return {
        "frame_ids": expected,
        "frame_list_hash": _stable_hash(expected),
        "dataset_local_sample_indices": local_indices,
        "identity_source": "dataset.split_info",
        "local_index_source": "batch.ego.sample_idx",
    }


def _dataset_local_sample_index(value: Any) -> int:
    """Extract one collated HEAL dataset-local index without interpreting it as an ID."""

    if hasattr(value, "detach"):
        value = value.detach().cpu().reshape(-1).tolist()
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ExperimentContractError(f"expected one local sample index, got {value!r}")
        value = value[0]
    return int(value)


def assert_independent_model_origins(rows: Sequence[Mapping[str, Any]], checkpoint_hash: str) -> None:
    """Require every ratio model to originate from the same original checkpoint."""

    invalid = [
        str(row.get("model_id", ""))
        for row in rows
        if str(row.get("source_checkpoint_hash", "")) != str(checkpoint_hash)
    ]
    if invalid:
        raise ExperimentContractError(f"models do not originate from original checkpoint: {invalid}")


def physical_param_prune_ratio(before: Any, after: Any) -> float:
    """Compute parameter pruning only from physical snapshot counts."""

    before_payload, after_payload = _plain(before), _plain(after)
    before_count = int(before_payload.get("parameter_count", 0))
    after_count = int(after_payload.get("parameter_count", 0))
    if before_count <= 0 or after_count < 0 or after_count > before_count:
        raise ExperimentContractError(
            f"invalid physical parameter counts: before={before_count}, after={after_count}"
        )
    return 1.0 - float(after_count) / float(before_count)


def validate_one_shot_request(request: SamplingPruningRequest) -> None:
    """Require the formal one-shot selector contract."""

    if not request.one_shot or request.selector != "global_one_shot":
        raise ExperimentContractError("formal experiment requires a global one-shot request")


def validate_formal_defaults(config: PruningConfig) -> dict[str, Any]:
    """Return and validate the immutable pruning defaults used by the experiment."""

    report = {
        "importance_mode": config.importance.mode.value,
        "importance_normalization": config.importance.normalization.strategy.value,
        "selector": config.selection.strategy.value,
        "dense_conv_channel_alignment": config.alignment.dense_conv_channel_alignment,
        "allowed_channels_per_group": list(config.grouped_conv.allowed_channels_per_group),
        "grouped_conv_selection": config.grouped_conv.selection_policy.value,
        "dependency_input_pruning_allowed": config.protection.allow_dependency_input_pruning,
    }
    expected = {
        "importance_mode": "first_order_taylor",
        "importance_normalization": "coupled_dependency_mean_then_scope_mean_v1",
        "selector": "global_one_shot",
        "dense_conv_channel_alignment": 4,
        "allowed_channels_per_group": [4, 8, 16, 32, 64, 128, 256, 512],
        "grouped_conv_selection": "independent_group_topk",
        "dependency_input_pruning_allowed": True,
    }
    if report != expected:
        raise ExperimentContractError(f"formal defaults changed: {report}")
    return report


def validate_directional_contract(policy: Mapping[str, Any]) -> None:
    """Validate fixed-output/dynamic-input protection semantics."""

    if not bool(policy.get("fixed_output_contract")) or bool(policy.get("root_pruning_allowed")):
        raise ExperimentContractError("protected output contract is not enforced")
    if not bool(policy.get("input_dependency_pruning_allowed")):
        raise ExperimentContractError("protected module dependency input pruning is disabled")


def validate_latency_scope(specification: Mapping[str, Any]) -> None:
    """Require a forward-only latency boundary."""

    included = {str(value).lower() for value in specification.get("includes", [])}
    required = {"prepared_input", "model_forward", "cuda_synchronize"}
    forbidden = {"dataloader", "dataset_io", "voxelization", "postprocess", "nms", "ap"}
    if not required <= included or included & forbidden:
        raise ExperimentContractError(f"invalid forward-only latency boundary: {sorted(included)}")


def compute_speedup(*, original_p50_ms: float, candidate_p50_ms: float) -> float:
    """Compute the declared primary speedup from forward p50 latency."""

    if not math.isfinite(original_p50_ms) or not math.isfinite(candidate_p50_ms) or candidate_p50_ms <= 0:
        raise ExperimentContractError("latency values must be finite and positive")
    return float(original_p50_ms) / float(candidate_p50_ms)


def execute_build_if_preflight(
    preflight_passed: bool,
    builder: Callable[[], Any],
) -> dict[str, Any]:
    """Gate all engine building behind successful graph/structure preflight."""

    if not preflight_passed:
        return {"preflight_pass": False, "build_called": False, "result": None}
    return {"preflight_pass": True, "build_called": True, "result": builder()}


def validate_strict_precision_rows(mode: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Require exact requested/realized precision for every non-exempt layer."""

    expected = str(mode).removeprefix("strict_")
    if expected not in {"fp32", "fp16", "int8"}:
        raise ExperimentContractError(f"unknown strict precision mode: {mode}")
    mismatches = [
        dict(row)
        for row in rows
        if not bool(row.get("exempt", False))
        and (
            str(row.get("requested", "")) != expected
            or str(row.get("realized", "")) != expected
        )
    ]
    return {"passed": not mismatches, "expected_precision": expected, "mismatches": mismatches}


def validate_engine_stage_order(stages: Sequence[str]) -> None:
    """Require checker execution only after preflight and successful build."""

    expected = ["preflight", "build", "structure", "precision", "provenance", "smoke"]
    positions = {str(stage): index for index, stage in enumerate(stages)}
    if any(name not in positions for name in expected) or any(
        positions[left] >= positions[right] for left, right in zip(expected, expected[1:])
    ):
        raise ExperimentContractError(f"invalid engine stage order: {list(stages)}")


def scan_runtime_dependencies(path: str | Path) -> dict[str, Any]:
    """Audit imports without embedding historical algorithm module names."""

    source = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(str(node.module or ""))
    forbidden_roots = {"test" + "s", "quant_deploy"}
    forbidden = sorted(
        name
        for name in imported
        if name.split(".", 1)[0] in forbidden_roots or name.startswith("tools." + "latency_lut")
    )
    return {"imports": sorted(imported), "forbidden_imports": forbidden}


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command(*values: str) -> str:
    completed = subprocess.run(values, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return (completed.stdout or "").strip()


def _heal_config_with_absolute_data_paths(config_path: Path, heal_root: Path) -> dict[str, Any]:
    from opencood.hypes_yaml import yaml_utils

    config = yaml_utils.load_yaml(str(config_path))
    for key in ("data_dir", "root_dir", "validate_dir", "test_dir"):
        value = config.get(key)
        if isinstance(value, str) and value and not Path(value).is_absolute():
            config[key] = str(heal_root / value)
    return config


def _load_heal_model_and_real_trace(
    *,
    config_path: Path,
    checkpoint_path: Path,
    heal_root: Path,
    device: str,
) -> tuple[Any, Any, Any]:
    """Load through pruning.api and trace one real validation frame."""

    from opencood.data_utils.datasets import build_dataset
    from opencood.tools import train_utils

    loaded = load_model(
        lambda config: train_utils.create_model(config),
        checkpoint_path=checkpoint_path,
        model_config=config_path,
        device=device,
        strict_state_dict=True,
    )
    dataset_config = _heal_config_with_absolute_data_paths(config_path, heal_root)
    dataset = build_dataset(dataset_config, visualize=False, train=False)
    batch = dataset.collate_batch_test([dataset[0]])
    batch = train_utils.to_device(batch, device)
    trace = trace_model(
        loaded.model,
        batch["ego"],
        config=TraceConfig(fail_on_fx_trace_error=False),
    )
    return loaded, dataset, trace


def _target_inventory(trace: Any) -> list[dict[str, Any]]:
    policies = {row.module_path: row for row in trace.protection_policies}
    scopes_by_member: dict[str, set[str]] = defaultdict(set)
    units_by_member: dict[str, set[str]] = defaultdict(set)
    for scope in trace.dependency_scopes:
        for member in scope.members:
            scopes_by_member[member.module_path].add(scope.stable_id)
    for unit in trace.coupled_channel_units:
        for member in unit.members:
            units_by_member[member.module_path].add(unit.stable_id)
    rows = []
    for module in trace.module_inventory:
        if not module.module_path.startswith("pyramid_backbone"):
            continue
        if module.module_type not in {"Conv2d", "ConvTranspose2d"} or int(module.groups or 1) <= 1:
            continue
        policy = policies[module.module_path]
        in_per = int(module.in_channels or 0) // int(module.groups or 1)
        out_per = int(module.out_channels or 0) // int(module.groups or 1)
        stage = next(
            (int(part.removeprefix("layer")) for part in module.module_path.split(".") if part.startswith("layer") and part.removeprefix("layer").isdigit()),
            -1,
        )
        rows.append(
            {
                "module_path": module.module_path,
                "module_type": module.module_type,
                "in_channels": module.in_channels,
                "out_channels": module.out_channels,
                "groups": module.groups,
                "input_channels_per_group": in_per,
                "output_channels_per_group": out_per,
                "weight_shape": list(module.parameter_shapes.get("weight", ())),
                "dependency_scope_ids": sorted(scopes_by_member[module.module_path]),
                "coupled_channel_unit_ids": sorted(units_by_member[module.module_path]),
                "stage_index": stage,
                "belongs_to_stage1": stage == 1,
                "protection_policy": policy.to_dict(),
                "root_pruning_allowed": policy.root_pruning_allowed,
                "dependency_input_pruning_allowed": policy.input_dependency_pruning_allowed,
                "fixed_output_contract": policy.fixed_output_contract,
                "reachable_legal_channels_per_group": [
                    value
                    for value in PruningConfig().grouped_conv.allowed_channels_per_group
                    if value <= min(in_per, out_per)
                ],
            }
        )
    return sorted(rows, key=lambda row: row["module_path"])


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def prepare_experiment(args: argparse.Namespace) -> dict[str, Any]:
    """Generate environment/manifests, real trace and fail-closed A audit."""

    started = time.time()
    output = Path(args.output).resolve()
    reports = output / "reports"
    grouped_dir = output / "grouped_conv_sensitivity"
    output.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    heal_root = Path(args.heal_root).resolve()
    plugin_path = Path(args.plugin).resolve()
    trt_root = Path(args.trt_root).resolve()
    dataset_config_hash = _file_sha256(config_path)
    dataset_config = _heal_config_with_absolute_data_paths(config_path, heal_root)
    validation_ids = json.loads(Path(dataset_config["validate_dir"]).read_text(encoding="utf-8"))
    training_ids = json.loads(Path(dataset_config["root_dir"]).read_text(encoding="utf-8"))
    validation_manifest = build_frame_manifest(
        validation_ids,
        frame_count=500,
        dataset_config_hash=dataset_config_hash,
        split="validation",
        seed=0,
    )
    calibration_manifest = build_frame_manifest(
        training_ids,
        frame_count=min(200, len(training_ids)),
        dataset_config_hash=dataset_config_hash,
        split="training_calibration",
        seed=0,
    )
    taylor_manifest = build_frame_manifest(
        training_ids,
        frame_count=min(50, len(training_ids)),
        dataset_config_hash=dataset_config_hash,
        split="training_taylor",
        seed=0,
    )
    atomic_write_json(output / "validation_frame_manifest_500.json", validation_manifest)
    atomic_write_json(output / "calibration_frame_manifest_200.json", calibration_manifest)
    atomic_write_json(output / "taylor_frame_manifest_50.json", taylor_manifest)

    loaded, dataset, trace = _load_heal_model_and_real_trace(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        heal_root=heal_root,
        device=args.device,
    )
    serialize_trace_result(trace, output / "original_trace_result.json")
    original_snapshot = build_physical_structure_snapshot(loaded.model)
    original_hashes = compute_physical_hashes(
        original_snapshot,
        model_hash=_file_sha256(checkpoint_path),
        config_hash=dataset_config_hash,
    )
    atomic_write_json(output / "original_physical_structure_snapshot_v2.json", original_snapshot.to_dict())
    atomic_write_json(output / "original_physical_hash_v2.json", original_hashes.to_dict())

    candidates = _target_inventory(trace)
    selection_error = ""
    try:
        selected = select_target_grouped_conv_layers(candidates)
        selection_status = "passed"
    except ExperimentContractError as exc:
        selected = []
        selection_status = "failed_closed"
        selection_error = str(exc)
    target_report = {
        "selection_status": selection_status,
        "selection_error": selection_error,
        "expected_count": 3,
        "actual_count": len(candidates),
        "selected_layers": selected,
        "all_dependency_safe_pyramid_grouped_convs": candidates,
        "trace_hash": trace.trace_hash,
        "trace_backend": trace.config.get("realized_backend"),
        "weighted_module_coverage": trace.trace_coverage.weighted_module_coverage,
    }
    atomic_write_json(grouped_dir / "target_grouped_conv_layers.json", target_report)

    prior = {
        "status": "prior_artifacts_not_available",
        "searched_root": str(Path.cwd() / "outputs"),
        "ratio_060_artifacts": [],
        "ratio_070_artifacts": [],
        "conclusion": "No physical snapshot v2/ledger pair for both 0.6 and 0.7 was available; no historical causal claim is made.",
    }
    atomic_write_json(reports / "prior_060_070_grouped_conv_delta_audit.json", prior)
    _write_text(
        reports / "prior_060_070_grouped_conv_delta_audit.md",
        "# Prior 0.6/0.7 grouped-convolution audit\n\n"
        "Status: `prior_artifacts_not_available`. No historical 0.6/0.7 physical snapshot-v2 and ledger pair was found, so the stage1 causality hypothesis cannot be evaluated from prior artifacts.\n",
    )
    _write_text(
        reports / "grouped_conv_sensitivity_final.md",
        "# Grouped-convolution sensitivity experiment\n\n"
        f"Status: `{selection_status}`. The real formal trace found **{len(candidates)}** dependency-safe pyramid grouped Conv2d modules, not exactly three. "
        "Per the experiment contract, no three-layer subset was guessed and experiment A was not executed. This is not a passed or skipped model result; it is a fail-closed target-identification result.\n\n"
        f"Trace hash: `{trace.trace_hash}`; weighted-module coverage: `{trace.trace_coverage.weighted_module_coverage}`.\n",
    )

    gpu_report = _command(
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ).splitlines()
    environment = {
        "git_commit": _command("git", "rev-parse", "HEAD"),
        "git_branch": _command("git", "branch", "--show-current"),
        "git_dirty": bool(_command("git", "status", "--short")),
        "git_status_short": _command("git", "status", "--short").splitlines(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": __import__("torch").__version__,
        "torch_cuda_version": __import__("torch").version.cuda,
        "tensorrt_version": __import__("tensorrt").__version__,
        "gpu_inventory": gpu_report,
        "selected_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "selected_logical_device": args.device,
        "config_sha256": dataset_config_hash,
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "plugin_sha256": _file_sha256(plugin_path) if plugin_path.is_file() else "",
        "plugin_status": "available" if plugin_path.is_file() else "not_built",
        "tensorrt_root": str(trt_root),
        "trtexec_path": str(trt_root / "bin" / "trtexec"),
        "requested_paths": {
            "python": args.requested_python,
            "tensorrt_root": args.requested_trt_root,
            "plugin": args.plugin,
        },
        "resolved_paths": {
            "python": sys.executable,
            "tensorrt_root": str(trt_root),
            "plugin": str(plugin_path),
        },
        "formal_pruning_defaults": validate_formal_defaults(PruningConfig()),
        "validation_frame_list_hash": validation_manifest["frame_list_hash"],
        "calibration_frame_list_hash": calibration_manifest["frame_list_hash"],
        "taylor_frame_list_hash": taylor_manifest["frame_list_hash"],
        "original_parameter_count": original_snapshot.parameter_count,
        "original_structure_hash": original_hashes.structure_hash_v2,
        "dataset_length": len(dataset),
    }
    atomic_write_json(output / "environment_report.json", environment)
    process = {
        "stage": "prepare",
        "pid": os.getpid(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "started_unix": started,
        "ended_unix": time.time(),
        "status": "completed",
        "background": False,
    }
    atomic_write_json(output / "process_manifest.json", {"processes": [process]})
    return {
        "environment": environment,
        "target_report": target_report,
        "prior_audit": prior,
    }


def _atomic_torch_save(path: Path, state: Mapping[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(dict(state), temporary)
    os.replace(temporary, path)


def _snapshot_channel_count(snapshot: Any) -> int:
    payload = _plain(snapshot)
    total = 0
    for row in payload.get("modules", []):
        for key in ("out_channels", "out_features", "num_features"):
            if row.get(key) is not None:
                total += int(row[key])
                break
    return total


def _fixed_output_contracts(snapshot: Any) -> dict[str, int]:
    payload = _plain(snapshot)
    result: dict[str, int] = {}
    for row in payload.get("modules", []):
        policy = row.get("protection_policy", {})
        if not bool(policy.get("fixed_output_contract")):
            continue
        value = row.get("out_channels")
        if value is not None:
            result[str(row["canonical_module_name"])] = int(value)
    return result


def _append_process(output: Path, row: Mapping[str, Any]) -> None:
    path = output / "process_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"processes": []}
    payload.setdefault("processes", []).append(dict(row))
    atomic_write_json(path, payload)


def _load_dataset(config_path: Path, heal_root: Path, *, train: bool) -> tuple[Any, dict[str, Any]]:
    from opencood.data_utils.datasets import build_dataset

    config = _heal_config_with_absolute_data_paths(config_path, heal_root)
    return build_dataset(config, visualize=False, train=train), config


def _batch(dataset: Any, index: int, device: str, *, train: bool) -> Any:
    from opencood.tools import train_utils

    sample = dataset[int(index)]
    if sample is None:
        raise ExperimentContractError(f"dataset returned None for required frame index {index}")
    collate = dataset.collate_batch_train if train else dataset.collate_batch_test
    batch = collate([sample])
    if batch is None:
        raise ExperimentContractError(f"collate returned None for required frame index {index}")
    return train_utils.to_device(batch, device)


def _accumulate_taylor_gradients(
    model: Any,
    dataset: Any,
    config: Mapping[str, Any],
    *,
    frame_count: int,
    device: str,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from opencood.tools import train_utils

    criterion = train_utils.create_loss(config)
    model.eval()
    model.zero_grad(set_to_none=True)
    losses: list[float] = []
    torch.manual_seed(0)
    np.random.seed(0)
    for index in range(int(frame_count)):
        batch = _batch(dataset, index, device, train=True)
        output = model(batch["ego"])
        loss = criterion(output, batch["ego"]["label_dict"])
        if not torch.isfinite(loss):
            raise ExperimentContractError(f"non-finite Taylor loss at frame index {index}: {loss}")
        (loss / float(frame_count)).backward()
        losses.append(float(loss.detach().cpu()))
    return {
        "frame_count": len(losses),
        "loss_mean": sum(losses) / len(losses),
        "loss_min": min(losses),
        "loss_max": max(losses),
        "gradient_accumulation": "mean",
    }


def _search_parameter_budget(
    model: Any,
    atomic_units: Sequence[Any],
    importance: Any,
    *,
    target_ratio: float,
    original_parameter_count: int,
) -> tuple[int, Any, Any, list[dict[str, Any]]]:
    """Search immutable plans by exact predicted physical parameter count."""

    cache: dict[int, tuple[Any, Any, float, int]] = {}

    def evaluate(budget: int) -> tuple[Any, Any, float, int]:
        budget = max(int(budget), 0)
        if budget in cache:
            return cache[budget]
        config = PruningConfig()
        request = select_pruning_request(
            atomic_units,
            importance_result=importance,
            parameter_budget=budget,
            grouped_config=config.grouped_conv,
            selection_config=config.selection,
            alignment_config=config.alignment,
        )
        validate_one_shot_request(request)
        plan = legalize_pruning_plan(
            model,
            build_physical_pruning_plan(model, request),
            alignment_config=config.alignment,
            grouped_config=config.grouped_conv,
        )
        predicted_count = estimate_physical_parameter_count(model, plan)
        predicted_ratio = 1.0 - float(predicted_count) / float(original_parameter_count)
        cache[budget] = (request, plan, predicted_ratio, predicted_count)
        return cache[budget]

    upper = max(
        int(original_parameter_count * 2),
        sum(max(int(value), 0) for value in importance.unit_parameter_costs.values()),
    )
    lower = 0
    best_budget = int(round(original_parameter_count * float(target_ratio)))
    evaluate(best_budget)
    for _ in range(18):
        middle = (lower + upper) // 2
        _request, _plan, ratio, _count = evaluate(middle)
        if ratio < float(target_ratio):
            lower = middle + 1
        else:
            upper = middle - 1
        best_budget = min(
            cache,
            key=lambda value: (abs(cache[value][2] - float(target_ratio)), value),
        )
        if abs(cache[best_budget][2] - float(target_ratio)) <= 0.002:
            break
    # Probe around the best discrete budget because greedy atomic costs create
    # plateaus and binary-search boundaries need not land on a candidate cost.
    best_requested_cost = cache[best_budget][0].requested_parameter_cost
    for delta in (-65536, -32768, -16384, -8192, 8192, 16384, 32768, 65536):
        evaluate(max(best_requested_cost + delta, 0))
    best_budget = min(
        cache,
        key=lambda value: (abs(cache[value][2] - float(target_ratio)), value),
    )
    request, plan, _ratio, _count = cache[best_budget]
    attempts = [
        {
            "parameter_budget": budget,
            "requested_parameter_cost": row[0].requested_parameter_cost,
            "predicted_parameter_count": row[3],
            "predicted_parameter_prune_ratio": row[2],
            "target_error": row[2] - float(target_ratio),
        }
        for budget, row in sorted(cache.items())
    ]
    return best_budget, request, plan, attempts


def generate_pruning_sweep(args: argparse.Namespace) -> dict[str, Any]:
    """Generate independent physical models and full pruning artifacts."""

    import torch
    from opencood.tools import train_utils

    started = time.time()
    output = Path(args.output).resolve()
    sweep_root = output / "pruning_sweep"
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    heal_root = Path(args.heal_root).resolve()
    checkpoint_hash = _file_sha256(checkpoint_path)
    config_hash = _file_sha256(config_path)
    train_dataset, heal_config = _load_dataset(config_path, heal_root, train=True)
    validation_dataset, _ = _load_dataset(config_path, heal_root, train=False)
    original_snapshot_payload = json.loads(
        (output / "original_physical_structure_snapshot_v2.json").read_text(encoding="utf-8")
    )
    original_parameter_count = int(original_snapshot_payload["parameter_count"])
    original_channel_count = _snapshot_channel_count(original_snapshot_payload)
    fixed_outputs = _fixed_output_contracts(original_snapshot_payload)
    ratios = [float(value) for value in args.ratios.split(",") if value.strip()]
    summary_rows: list[dict[str, Any]] = []
    for ratio in ratios:
        model_id = f"prune_{ratio:.1f}"
        ratio_dir = sweep_root / f"ratio_{int(round(ratio * 10)):02d}"
        completion = ratio_dir / "generation_complete.json"
        if args.resume and completion.is_file() and not args.force:
            summary_rows.append(json.loads(completion.read_text(encoding="utf-8")))
            continue
        ratio_started = time.time()
        ratio_dir.mkdir(parents=True, exist_ok=True)
        if completion.is_file():
            previous = json.loads(completion.read_text(encoding="utf-8"))
            if not bool(previous.get("within_tolerance_0_02", False)):
                atomic_write_json(ratio_dir / "pilot_generation_complete.json", previous)
        manifest = {
            "model_id": model_id,
            "target_parameter_prune_ratio": ratio,
            "source_checkpoint": str(checkpoint_path),
            "source_checkpoint_hash": checkpoint_hash,
            "config_hash": config_hash,
            "trace_count": 1,
            "importance_count": 1,
            "materialization_transaction_count": 1,
            "formal_defaults": validate_formal_defaults(PruningConfig()),
            "status": "running",
        }
        atomic_write_json(ratio_dir / "experiment_manifest.json", manifest)
        try:
            loaded = load_model(
                lambda config: train_utils.create_model(config),
                checkpoint_path=checkpoint_path,
                model_config=config_path,
                device=args.device,
                strict_state_dict=True,
            )
            model = loaded.model
            validation_batch = _batch(validation_dataset, 0, args.device, train=False)
            trace = trace_model(
                model,
                validation_batch["ego"],
                config=TraceConfig(fail_on_fx_trace_error=False),
            )
            serialize_trace_result(trace, ratio_dir / "trace_result.json")
            taylor = _accumulate_taylor_gradients(
                model,
                train_dataset,
                heal_config,
                frame_count=int(args.taylor_frames),
                device=args.device,
            )
            importance = score_pruning_units(
                model,
                trace.coupled_channel_units,
                config=PruningConfig(),
                calibration_batches=int(args.taylor_frames),
                task_loss="HEAL point_pillar_pyramid_loss",
            )
            importance_payload = importance.to_dict()
            importance_payload["calibration_summary"] = taylor
            importance_payload["frame_manifest"] = str(output / "taylor_frame_manifest_50.json")
            atomic_write_json(ratio_dir / "importance_summary.json", importance_payload)
            parameter_budget, request, plan, budget_search_attempts = _search_parameter_budget(
                model,
                trace.atomic_prune_units,
                importance,
                target_ratio=ratio,
                original_parameter_count=original_parameter_count,
            )
            validate_one_shot_request(request)
            atomic_write_json(ratio_dir / "sampling_structure_request.json", request.to_dict())
            if not plan.indices_frozen_before_materialization:
                raise ExperimentContractError("physical plan indices are not frozen")
            atomic_write_json(ratio_dir / "pruning_plan.json", plan.to_dict())
            atomic_write_json(ratio_dir / "budget_search_report.json", {"attempts": budget_search_attempts})
            predicted_parameter_count = estimate_physical_parameter_count(model, plan)
            materialized = materialize_pruning(model, plan, in_place=True, build_snapshot=True)
            snapshot = materialized.snapshot or build_physical_structure_snapshot(materialized.model)
            hashes = compute_physical_hashes(
                snapshot,
                model_hash=checkpoint_hash,
                config_hash=config_hash,
            )
            validation = validate_physical_model(
                materialized.model,
                expected_snapshot=snapshot,
                grouped_config=PruningConfig().grouped_conv,
                fixed_output_contracts=fixed_outputs,
                example_inputs=(validation_batch["ego"],),
            )
            if not validation.passed:
                raise ExperimentContractError(f"physical validation failed: {validation.issues[:5]}")
            actual_ratio = physical_param_prune_ratio(original_snapshot_payload, snapshot)
            if predicted_parameter_count != snapshot.parameter_count:
                raise ExperimentContractError(
                    f"plan parameter prediction mismatch: predicted={predicted_parameter_count}, actual={snapshot.parameter_count}"
                )
            actual_channel_count = _snapshot_channel_count(snapshot)
            actual_channel_ratio = 1.0 - float(actual_channel_count) / float(original_channel_count)
            portable_state = {
                key: value.detach().cpu() if torch.is_tensor(value) else value
                for key, value in materialized.model.state_dict().items()
            }
            _atomic_torch_save(ratio_dir / "model_state_dict.pth", portable_state)
            atomic_write_json(
                ratio_dir / "model_reconstruction_config.json",
                {
                    "model_config": str(config_path),
                    "source_checkpoint": str(checkpoint_path),
                    "source_checkpoint_hash": checkpoint_hash,
                    "pruning_plan": "pruning_plan.json",
                    "replay_api": "pruning.api.replay_pruning",
                    "portable_state_dict": "model_state_dict.pth",
                    "full_pickled_model_saved": False,
                },
            )
            atomic_write_json(
                ratio_dir / "physical_pruning_application_ledger.json",
                materialized.ledger.to_dict(),
            )
            atomic_write_json(ratio_dir / "physical_structure_snapshot_v2.json", snapshot.to_dict())
            atomic_write_json(ratio_dir / "physical_hash_v2.json", hashes.to_dict())
            atomic_write_json(ratio_dir / "validation_report.json", _plain(validation))
            grouped_maps = [
                {
                    "module_path": entry.module_path,
                    "axis": entry.axis,
                    "group_keep_map": entry.group_keep_map,
                    "group_prune_map": entry.group_prune_map,
                }
                for entry in request.entries
                if entry.group_keep_map or entry.group_prune_map
            ]
            atomic_write_json(ratio_dir / "grouped_keep_maps.json", grouped_maps)
            terminal_statuses = materialized.ledger.to_dict()["status_counts"]
            row = {
                "model_id": model_id,
                "target_parameter_prune_ratio": ratio,
                "actual_parameter_count": snapshot.parameter_count,
                "actual_parameter_prune_ratio": actual_ratio,
                "target_error": actual_ratio - ratio,
                "within_tolerance_0_02": abs(actual_ratio - ratio) <= 0.02,
                "actual_channel_count": actual_channel_count,
                "actual_channel_prune_ratio": actual_channel_ratio,
                "parameter_budget": parameter_budget,
                "budget_search_plan_preflight_count": len(budget_search_attempts),
                "predicted_parameter_count": predicted_parameter_count,
                "requested_parameter_cost": request.requested_parameter_cost,
                "structure_hash_v2": hashes.structure_hash_v2,
                "shape_hash_v2": hashes.shape_hash_v2,
                "trace_hash": trace.trace_hash,
                "validation_passed": validation.passed,
                "ledger_status_counts": terminal_statuses,
                "elapsed_seconds": time.time() - ratio_started,
                "source_checkpoint_hash": checkpoint_hash,
                "status": "generated",
            }
            atomic_write_json(completion, row)
            summary_rows.append(row)
        except Exception as exc:
            failure = {
                "model_id": model_id,
                "target_parameter_prune_ratio": ratio,
                "source_checkpoint_hash": checkpoint_hash,
                "status": "failed",
                "failure_stage": "pruning_generation",
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": time.time() - ratio_started,
            }
            atomic_write_json(ratio_dir / "generation_failure.json", failure)
            summary_rows.append(failure)
            if args.fail_fast:
                raise
        finally:
            torch.cuda.empty_cache()
    assert_independent_model_origins(summary_rows, checkpoint_hash)
    atomic_write_json(sweep_root / "generation_summary.json", {"models": summary_rows})
    _append_process(
        output,
        {
            "stage": "sweep_generate",
            "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "started_unix": started,
            "ended_unix": time.time(),
            "status": "completed" if all(row["status"] == "generated" for row in summary_rows) else "completed_with_failures",
            "background": False,
            "ratios": ratios,
        },
    )
    return {"models": summary_rows}


def _restore_model(
    model_id: str,
    *,
    config_path: Path,
    checkpoint_path: Path,
    output: Path,
    device: str,
) -> tuple[Any, Any]:
    import torch
    from opencood.tools import train_utils

    loaded = load_model(
        lambda config: train_utils.create_model(config),
        checkpoint_path=checkpoint_path,
        model_config=config_path,
        device=device,
        strict_state_dict=True,
    )
    if model_id == "original":
        snapshot = build_physical_structure_snapshot(loaded.model)
        return loaded.model, snapshot
    ratio = float(model_id.removeprefix("prune_"))
    ratio_dir = output / "pruning_sweep" / f"ratio_{int(round(ratio * 10)):02d}"
    plan = PhysicalPruningPlan.from_dict(json.loads((ratio_dir / "pruning_plan.json").read_text(encoding="utf-8")))
    replayed = replay_pruning(loaded.model, plan, in_place=True)
    state = torch.load(ratio_dir / "model_state_dict.pth", map_location=device, weights_only=True)
    replayed.model.load_state_dict(state, strict=True)
    expected = json.loads((ratio_dir / "physical_structure_snapshot_v2.json").read_text(encoding="utf-8"))
    validation = validate_physical_model(replayed.model, expected_snapshot=expected)
    if not validation.passed:
        raise ExperimentContractError(f"restored model validation failed for {model_id}: {validation.issues}")
    return replayed.model, validation.snapshot


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def evaluate_pruning_sweep(args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate original plus seven physical models on one exact 500-frame list."""

    import torch
    from opencood.tools import inference_utils
    from opencood.utils import eval_utils

    started = time.time()
    output = Path(args.output).resolve()
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    heal_root = Path(args.heal_root).resolve()
    dataset, _config = _load_dataset(config_path, heal_root, train=False)
    frame_manifest = json.loads((output / "validation_frame_manifest_500.json").read_text(encoding="utf-8"))
    frame_ids = [str(value) for value in frame_manifest["frame_ids"]]
    if len(frame_ids) != 500:
        raise ExperimentContractError("validation manifest is not exactly 500 frames")
    split_info = getattr(dataset, "split_info", None)
    if not isinstance(split_info, (list, tuple)):
        raise ExperimentContractError("dataset does not expose authoritative split_info frame IDs")
    split_binding = validate_dataset_frame_binding(frame_ids, split_info, range(len(frame_ids)))
    if split_binding["frame_list_hash"] != frame_manifest["frame_list_hash"]:
        raise ExperimentContractError("dataset split frame hash differs from validation manifest")
    model_ids = ["original"] + [f"prune_{ratio:.1f}" for ratio in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)]
    thresholds = (0.03, 0.30, 0.50, 0.70)
    rows: list[dict[str, Any]] = []
    for model_id in model_ids:
        model_started = time.time()
        model, snapshot = _restore_model(
            model_id,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            output=output,
            device=args.device,
        )
        model.eval()
        result_stat = {
            threshold: {"tp": [], "fp": [], "gt": 0, "score": []}
            for threshold in thresholds
        }
        observed_ids: list[str] = []
        observed_local_indices: list[int] = []
        with torch.inference_mode():
            for index, expected_frame_id in enumerate(frame_ids):
                batch = _batch(dataset, index, args.device, train=False)
                if "sample_idx" not in batch["ego"]:
                    raise ExperimentContractError("HEAL batch is missing dataset-local sample_idx")
                observed_local_indices.append(_dataset_local_sample_index(batch["ego"]["sample_idx"]))
                observed_ids.append(expected_frame_id)
                inference = inference_utils.inference_intermediate_fusion(batch, model, dataset)
                for threshold in thresholds:
                    eval_utils.caluclate_tp_fp(
                        inference["pred_box_tensor"],
                        inference["pred_score"],
                        inference["gt_box_tensor"],
                        result_stat,
                        threshold,
                    )
        runtime_binding = validate_dataset_frame_binding(frame_ids, split_info, observed_local_indices)
        observed_hash = _stable_hash(observed_ids)
        if observed_hash != frame_manifest["frame_list_hash"]:
            raise ExperimentContractError(f"frame hash changed during evaluation for {model_id}")
        metrics = {
            f"ap_{threshold:.2f}": float(eval_utils.calculate_ap(result_stat, threshold)[0])
            for threshold in thresholds
        }
        metrics["mAP"] = sum(metrics.values()) / len(metrics)
        row = {
            "model_id": model_id,
            "target_parameter_prune_ratio": 0.0 if model_id == "original" else float(model_id.removeprefix("prune_")),
            "actual_parameter_count": int(snapshot.parameter_count),
            "actual_parameter_prune_ratio": physical_param_prune_ratio(
                json.loads((output / "original_physical_structure_snapshot_v2.json").read_text(encoding="utf-8")),
                snapshot,
            ),
            "frame_count": len(observed_ids),
            "frame_list_hash": observed_hash,
            "frame_identity_source": runtime_binding["identity_source"],
            "dataset_local_sample_index_min": min(observed_local_indices),
            "dataset_local_sample_index_max": max(observed_local_indices),
            "evaluation_wall_seconds": time.time() - model_started,
            **metrics,
        }
        atomic_write_json(output / "pruning_sweep" / model_id / "evaluation_500frames.json", row)
        rows.append(row)
        del model
        torch.cuda.empty_cache()
    baseline = rows[0]
    for row in rows:
        for key in ("ap_0.03", "ap_0.30", "ap_0.50", "ap_0.70", "mAP"):
            row[f"{key}_absolute_drop"] = baseline[key] - row[key]
        row["mAP_retention"] = row["mAP"] / baseline["mAP"] if baseline["mAP"] else None
    results = {
        "frame_list_hash": frame_manifest["frame_list_hash"],
        "frame_count": 500,
        "models": rows,
        "schema_version": "formal-pruning-sweep-evaluation-v1",
    }
    atomic_write_json(output / "pruning_sweep_500frames_results.json", results)
    _atomic_csv(output / "pruning_sweep_500frames_results.csv", rows)
    _append_process(
        output,
        {
            "stage": "sweep_evaluate",
            "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "started_unix": started,
            "ended_unix": time.time(),
            "status": "completed",
            "background": False,
            "frame_list_hash": frame_manifest["frame_list_hash"],
        },
    )
    return results


def benchmark_pruning_sweep(args: argparse.Namespace) -> dict[str, Any]:
    """Run forward-only CUDA-event microbenchmarks serially on one GPU."""

    import statistics
    import torch

    started = time.time()
    output = Path(args.output).resolve()
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    heal_root = Path(args.heal_root).resolve()
    dataset, _config = _load_dataset(config_path, heal_root, train=False)
    prepared = [_batch(dataset, index, args.device, train=False)["ego"] for index in range(10)]
    specification = {"includes": ["prepared_input", "model_forward", "cuda_synchronize"]}
    validate_latency_scope(specification)
    model_ids = ["original"] + [f"prune_{ratio:.1f}" for ratio in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)]
    rows: list[dict[str, Any]] = []
    for model_id in model_ids:
        model, _snapshot = _restore_model(
            model_id,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            output=output,
            device=args.device,
        )
        model.eval()
        repeat_summaries: list[dict[str, Any]] = []
        all_values: list[float] = []
        with torch.inference_mode():
            for repeat in range(3):
                for index in range(50):
                    model(prepared[index % len(prepared)])
                torch.cuda.synchronize()
                starts = [torch.cuda.Event(enable_timing=True) for _ in range(500)]
                ends = [torch.cuda.Event(enable_timing=True) for _ in range(500)]
                for index, (start_event, end_event) in enumerate(zip(starts, ends)):
                    start_event.record()
                    model(prepared[index % len(prepared)])
                    end_event.record()
                torch.cuda.synchronize()
                values = [float(start_event.elapsed_time(end_event)) for start_event, end_event in zip(starts, ends)]
                all_values.extend(values)
                repeat_summaries.append({"repeat": repeat, **summarize_latency(values).to_dict()})
        summary = summarize_latency(all_values).to_dict()
        repeat_means = [float(row["mean_ms"]) for row in repeat_summaries]
        repeat_cv = (
            statistics.pstdev(repeat_means) / statistics.mean(repeat_means)
            if statistics.mean(repeat_means) > 0
            else None
        )
        rows.append(
            {
                "model_id": model_id,
                **summary,
                "standard_deviation_ms": statistics.pstdev(all_values),
                "repeat_cv": repeat_cv,
                "warmup_per_repeat": 50,
                "measured_per_repeat": 500,
                "repeat_count": 3,
                "prepared_real_input_count": len(prepared),
                "latency_boundary": specification,
                "repeat_summaries": repeat_summaries,
            }
        )
        del model
        torch.cuda.empty_cache()
    baseline = rows[0]
    for row in rows:
        row["p50_speedup"] = compute_speedup(
            original_p50_ms=float(baseline["p50_ms"]),
            candidate_p50_ms=float(row["p50_ms"]),
        )
        row["mean_speedup"] = compute_speedup(
            original_p50_ms=float(baseline["mean_ms"]),
            candidate_p50_ms=float(row["mean_ms"]),
        )
    result = {
        "models": rows,
        "same_gpu_serial": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "schema_version": "formal-forward-latency-v1",
    }
    atomic_write_json(output / "pruning_sweep_latency_microbenchmark.json", result)
    _append_process(
        output,
        {
            "stage": "sweep_latency",
            "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "started_unix": started,
            "ended_unix": time.time(),
            "status": "completed",
            "background": False,
        },
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "sweep_generate", "sweep_evaluate", "sweep_latency"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--heal-root", required=True)
    parser.add_argument("--trt-root", required=True)
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--requested-python", default="")
    parser.add_argument("--requested-trt-root", default="")
    parser.add_argument("--ratios", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7")
    parser.add_argument("--taylor-frames", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.stage == "prepare":
        result = prepare_experiment(args)
        print(json.dumps({"stage": "prepare", "target_report": result["target_report"]}, indent=2))
        return 0
    if args.stage == "sweep_generate":
        result = generate_pruning_sweep(args)
        print(json.dumps({"stage": "sweep_generate", **result}, indent=2))
        return 0
    if args.stage == "sweep_evaluate":
        result = evaluate_pruning_sweep(args)
        print(json.dumps({"stage": "sweep_evaluate", "models": result["models"]}, indent=2))
        return 0
    if args.stage == "sweep_latency":
        result = benchmark_pruning_sweep(args)
        print(json.dumps({"stage": "sweep_latency", "models": result["models"]}, indent=2))
        return 0
    raise ExperimentContractError(f"unsupported stage: {args.stage}")


__all__ = [
    "ExperimentContractError",
    "assert_independent_model_origins",
    "build_frame_manifest",
    "compare_shape_snapshots",
    "compute_speedup",
    "execute_build_if_preflight",
    "physical_param_prune_ratio",
    "scan_runtime_dependencies",
    "select_target_grouped_conv_layers",
    "validate_active_roots",
    "validate_directional_contract",
    "validate_engine_stage_order",
    "validate_formal_defaults",
    "validate_latency_scope",
    "validate_one_shot_request",
    "validate_strict_precision_rows",
    "width_feasibility",
]


if __name__ == "__main__":
    raise SystemExit(main())
