"""Shared post-scatter deployment contract for HEAL TensorRT engines."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from quantization.config import CanonicalNamingConfig
from quantization.export.origin_mapping import (
    apply_canonical_node_names,
    build_onnx_origin_map,
)
from quantization.export.signal_maxk import capture_weighted_module_calls
from quantization.types import OnnxExportResult


POST_SCATTER_CONTRACT = "heal_post_scatter_dynamic_frontend_v1"
POST_SCATTER_INPUTS = ("spatial_features", "pairwise_t_matrix")
POST_SCATTER_INPUTS_WITH_MASK = (*POST_SCATTER_INPUTS, "agent_mask")
FRONTEND_MODULE_PREFIXES = ("encoder_m1",)
FORBIDDEN_ENGINE_INPUTS = {
    "voxel_features",
    "voxel_coords",
    "voxel_num_points",
    "valid_voxel_mask",
}


def is_external_frontend_module(module_path: str) -> bool:
    value = str(module_path)
    return any(value == prefix or value.startswith(f"{prefix}.") for prefix in FRONTEND_MODULE_PREFIXES)


def filter_post_scatter_quantization_groups(groups: Sequence[Any]) -> list[Any]:
    """Remove PFN genes and reject precision groups crossing the engine boundary."""

    retained = []
    for group in groups:
        members = tuple(str(value) for value in group.module_paths)
        external = tuple(value for value in members if is_external_frontend_module(value))
        if external and len(external) != len(members):
            raise RuntimeError(
                f"precision_group_crosses_post_scatter_boundary:{group.group_id}:{members}"
            )
        if not external:
            retained.append(group)
    if not retained:
        raise RuntimeError("post_scatter_quantization_space_empty")
    return retained


def filter_post_scatter_module_paths(paths: Sequence[str]) -> list[str]:
    return sorted(
        str(path) for path in paths if not is_external_frontend_module(str(path))
    )


def filter_post_scatter_pruning_units(
    units: Sequence[Any], *, require_nonempty: bool = True
) -> list[Any]:
    """Remove PFN pruning loci and reject graph dependencies across the boundary."""

    retained = []
    for unit in units:
        paths = {
            str(getattr(unit, "root_module_path", "")),
            *(
                str(getattr(member, "module_path", ""))
                for member in list(getattr(unit, "members", ()) or ())
            ),
        }
        paths.discard("")
        external = {path for path in paths if is_external_frontend_module(path)}
        if external and external != paths:
            raise RuntimeError(
                "pruning_unit_crosses_post_scatter_boundary:"
                f"{getattr(unit, 'stable_id', '')}:{sorted(paths)}"
            )
        if not external:
            retained.append(unit)
    if require_nonempty and not retained:
        raise RuntimeError("post_scatter_pruning_space_empty")
    return retained


def prepare_post_scatter_inputs(
    model: nn.Module,
    ego_batch: Mapping[str, Any],
    *,
    modality: str = "m1",
    max_agents: int = 2,
    include_agent_mask: bool = False,
) -> dict[str, torch.Tensor]:
    """Run the FP32 PFN/scatter frontend and construct engine inputs."""

    encoder = getattr(model, f"encoder_{modality}")
    source = ego_batch[f"inputs_{modality}"]
    with torch.no_grad():
        encoded = encoder.pillar_vfe(source)
        spatial = encoder.scatter(encoded)["spatial_features"].float().contiguous()
    record_len = int(ego_batch["record_len"][0].item())
    if record_len != int(spatial.shape[0]):
        raise RuntimeError(
            f"post_scatter_record_len_mismatch:{record_len}!={int(spatial.shape[0])}"
        )
    raw_pairwise = ego_batch["pairwise_t_matrix"].float()
    if not include_agent_mask:
        return {
            "spatial_features": spatial,
            "pairwise_t_matrix": raw_pairwise[
                :, :record_len, :record_len
            ].contiguous(),
        }
    if record_len > int(max_agents):
        raise RuntimeError(
            f"post_scatter_record_len_exceeds_max_agents:{record_len}:{max_agents}"
        )
    if record_len < int(max_agents):
        padding = spatial.new_zeros(
            (int(max_agents) - record_len, *tuple(spatial.shape[1:]))
        )
        spatial = torch.cat((spatial, padding), dim=0)
    pairwise = torch.eye(4, dtype=raw_pairwise.dtype, device=raw_pairwise.device)
    pairwise = pairwise.reshape(1, 1, 1, 4, 4).repeat(
        1, int(max_agents), int(max_agents), 1, 1
    )
    pairwise[:, :record_len, :record_len] = raw_pairwise[
        :, :record_len, :record_len
    ]
    agent_mask = spatial.new_zeros((1, int(max_agents)))
    agent_mask[:, :record_len] = 1.0
    return {
        "spatial_features": spatial.contiguous(),
        "pairwise_t_matrix": pairwise.contiguous(),
        "agent_mask": agent_mask.contiguous(),
    }


def post_scatter_shape_profiles(
    input_names: Sequence[str] = POST_SCATTER_INPUTS,
    *,
    channels: int = 64,
    height: int = 256,
    width: int = 512,
    max_agents: int = 2,
) -> dict[str, dict[str, tuple[int, ...]]]:
    names = set(str(value) for value in input_names)
    if names not in (set(POST_SCATTER_INPUTS), set(POST_SCATTER_INPUTS_WITH_MASK)):
        raise ValueError(f"unsupported_post_scatter_inputs:{sorted(names)}")
    fixed_agents = "agent_mask" in names
    if fixed_agents:
        # Masked family/V2X-ViT exports are static two-agent ONNX graphs.
        # TensorRT rejects explicit shape profiles for fully static networks.
        return {}
    minimum_agents = int(max_agents) if fixed_agents else 1
    profiles = {
        "spatial_features": {
            "min": (minimum_agents, int(channels), int(height), int(width)),
            "opt": (int(max_agents), int(channels), int(height), int(width)),
            "max": (int(max_agents), int(channels), int(height), int(width)),
        },
        "pairwise_t_matrix": {
            "min": (1, minimum_agents, minimum_agents, 4, 4),
            "opt": (1, int(max_agents), int(max_agents), 4, 4),
            "max": (1, int(max_agents), int(max_agents), 4, 4),
        },
    }
    return profiles


def audit_post_scatter_onnx(path: str | Path) -> dict[str, Any]:
    import onnx

    source = Path(path)
    graph = onnx.load(str(source), load_external_data=True)
    inputs = [str(value.name) for value in graph.graph.input]
    nodes = list(graph.graph.node)
    issues = []
    if set(inputs) not in (set(POST_SCATTER_INPUTS), set(POST_SCATTER_INPUTS_WITH_MASK)):
        issues.append(f"invalid_input_contract:{sorted(inputs)}")
    leaked = sorted(set(inputs) & FORBIDDEN_ENGINE_INPUTS)
    if leaked:
        issues.append(f"fixed_k_inputs_present:{leaked}")
    scatter = sum(node.op_type == "PointPillarScatterTRT" for node in nodes)
    if scatter:
        issues.append(f"scatter_nodes_present:{scatter}")
    custom = sorted({f"{node.domain}::{node.op_type}" for node in nodes if node.domain})
    if any("PointPillarScatter" in value for value in custom):
        issues.append(f"scatter_custom_op_present:{custom}")
    return {
        "schema_version": "heal-post-scatter-onnx-audit-v1",
        "passed": not issues,
        "issues": issues,
        "input_names": inputs,
        "fixed_k_inputs": leaked,
        "scatter_node_count": scatter,
        "custom_ops": custom,
        "runtime_max_k_dependency": False,
        "point_frontend": "dynamic_voxelization_pfn_scatter_outside_tensorrt",
        "engine_contract": POST_SCATTER_CONTRACT,
    }


def export_post_scatter_onnx(
    model: nn.Module,
    example_inputs: Mapping[str, torch.Tensor],
    output_path: str | Path,
    *,
    output_names: Sequence[str] = ("cls_preds", "reg_preds", "dir_preds"),
    naming_config: CanonicalNamingConfig | None = None,
    opset_version: int = 17,
    report_path: str | Path | None = None,
) -> OnnxExportResult:
    """Export a canonical graph whose runtime starts at dense BEV features."""

    input_names = tuple(str(value) for value in example_inputs)
    if set(input_names) not in (
        set(POST_SCATTER_INPUTS),
        set(POST_SCATTER_INPUTS_WITH_MASK),
    ):
        raise RuntimeError(f"post_scatter_export_input_contract:{input_names}")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes: dict[str, dict[int, str]] = {
        "spatial_features": {0: "num_agents"},
        "pairwise_t_matrix": {1: "num_agents", 2: "num_agents"},
        **{str(name): {0: "batch"} for name in output_names},
    }
    if "agent_mask" in input_names:
        # Masked family wrappers use a frozen two-agent export contract.
        dynamic_axes.pop("spatial_features")
        dynamic_axes.pop("pairwise_t_matrix")
    tensors = tuple(example_inputs[name] for name in input_names)
    with capture_weighted_module_calls(model) as calls:
        torch.onnx.export(
            model,
            tensors,
            str(destination),
            export_params=True,
            opset_version=int(opset_version),
            do_constant_folding=True,
            input_names=list(input_names),
            output_names=list(output_names),
            dynamic_axes=dynamic_axes,
        )
    origin = build_onnx_origin_map(
        destination,
        calls,
        naming_config=naming_config or CanonicalNamingConfig(),
    )
    rename = apply_canonical_node_names(
        destination,
        origin,
        output_path=destination,
        allow_custom_ops=False,
    )
    import onnx

    graph = onnx.load(str(destination), load_external_data=True)
    metadata = graph.metadata_props.add()
    metadata.key = "heal.engine_contract"
    metadata.value = POST_SCATTER_CONTRACT
    onnx.checker.check_model(graph)
    onnx.save(graph, str(destination))
    audit = audit_post_scatter_onnx(destination)
    if not audit["passed"]:
        raise RuntimeError(f"post_scatter_onnx_audit_failed:{audit['issues']}")
    result = OnnxExportResult(
        onnx_path=str(destination),
        input_names=list(input_names),
        output_names=list(output_names),
        fixed_k=0,
        dynamic_agent_dimension="agent_mask" not in input_names,
        checker_passed=True,
        origin_map=origin,
        canonical_rename=rename,
        schema_version="heal-post-scatter-onnx-export-v1",
    )
    if report_path is not None:
        payload = {**result.to_dict(), "post_scatter_audit": audit}
        Path(report_path).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return result


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "FORBIDDEN_ENGINE_INPUTS",
    "FRONTEND_MODULE_PREFIXES",
    "POST_SCATTER_CONTRACT",
    "POST_SCATTER_INPUTS",
    "POST_SCATTER_INPUTS_WITH_MASK",
    "audit_post_scatter_onnx",
    "export_post_scatter_onnx",
    "filter_post_scatter_module_paths",
    "filter_post_scatter_pruning_units",
    "filter_post_scatter_quantization_groups",
    "is_external_frontend_module",
    "post_scatter_shape_profiles",
    "prepare_post_scatter_inputs",
    "sha256_file",
]
