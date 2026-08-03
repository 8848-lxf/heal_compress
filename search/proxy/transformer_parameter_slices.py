"""Resolve Transformer width-domain atoms to exact parameter slices."""

from __future__ import annotations

from typing import Sequence

import torch.nn as nn

from ..pruning_space.local_domains import LocalPruningDomain
from .parameter_slice_resolver import ParameterSlice


def _append_linear_output(
    rows: list[ParameterSlice],
    model: nn.Module,
    path: str,
    indices: tuple[int, ...],
) -> None:
    module = model.get_submodule(path)
    if not isinstance(module, nn.Linear):
        raise RuntimeError(f"transformer_slice_dependency_not_linear:{path}")
    rows.append(ParameterSlice(f"{path}.weight", path, 0, indices, "transformer_prune_output_rows"))
    if module.bias is not None:
        rows.append(ParameterSlice(f"{path}.bias", path, 0, indices, "transformer_prune_output_bias"))


def _append_linear_input(
    rows: list[ParameterSlice],
    model: nn.Module,
    path: str,
    indices: tuple[int, ...],
) -> None:
    module = model.get_submodule(path)
    if not isinstance(module, nn.Linear):
        raise RuntimeError(f"transformer_slice_dependency_not_linear:{path}")
    rows.append(ParameterSlice(f"{path}.weight", path, 1, indices, "transformer_prune_input_columns"))


def _dedupe(rows: list[ParameterSlice]) -> list[ParameterSlice]:
    result: list[ParameterSlice] = []
    seen: set[tuple[str, int, tuple[int, ...]]] = set()
    for row in rows:
        key = (row.parameter_name, int(row.axis), tuple(row.indices))
        if key not in seen:
            seen.add(key)
            result.append(row)
    return result


def _attention_slices(
    model: nn.Module,
    domain: LocalPruningDomain,
) -> dict[str, list[ParameterSlice]]:
    heads = int(domain.constraints["heads"])
    width = int(domain.original_width)
    members = tuple(domain.dependency_members)
    role_paths = {
        role: tuple(dict.fromkeys(
            str(member["module_path"])
            for member in members
            if member["role"] == role
        ))
        for role in ("q", "k", "v", "out")
    }
    fused = str(domain.constraints.get("qkv_layout")) == "fused_qkv"
    result: dict[str, list[ParameterSlice]] = {}
    for head in range(heads):
        for local in range(width):
            flat = head * width + local
            qk_id = f"{domain.domain_id}::head{head}::qk::{local}"
            vo_id = f"{domain.domain_id}::head{head}::vo::{local}"
            qk_rows: list[ParameterSlice] = []
            vo_rows: list[ParameterSlice] = []
            if fused:
                if len(role_paths["q"]) != 1 or not (
                    role_paths["q"] == role_paths["k"] == role_paths["v"]
                ):
                    raise RuntimeError(f"transformer_fused_slice_paths_invalid:{domain.domain_id}")
                path = role_paths["q"][0]
                projection = heads * width
                _append_linear_output(qk_rows, model, path, (flat, projection + flat))
                _append_linear_output(vo_rows, model, path, (2 * projection + flat,))
            else:
                for path in role_paths["q"]:
                    _append_linear_output(qk_rows, model, path, (flat,))
                for path in role_paths["k"]:
                    _append_linear_output(qk_rows, model, path, (flat,))
                for path in role_paths["v"]:
                    _append_linear_output(vo_rows, model, path, (flat,))
            for path in role_paths["out"]:
                _append_linear_input(vo_rows, model, path, (flat,))

            if str(domain.constraints.get("adapter")) == "v2xvit_hgt":
                relation_att = str(domain.metadata.get("relation_att_path", ""))
                relation_msg = str(domain.metadata.get("relation_msg_path", ""))
                qk_rows.extend((
                    ParameterSlice(relation_att, domain.module_path, 2, (local,), "transformer_relation_qk_row"),
                    ParameterSlice(relation_att, domain.module_path, 3, (local,), "transformer_relation_qk_column"),
                ))
                vo_rows.extend((
                    ParameterSlice(relation_msg, domain.module_path, 2, (local,), "transformer_relation_vo_row"),
                    ParameterSlice(relation_msg, domain.module_path, 3, (local,), "transformer_relation_vo_column"),
                ))
            result[qk_id] = _dedupe(qk_rows)
            result[vo_id] = _dedupe(vo_rows)
    return result


def _ffn_slices(model: nn.Module, domain: LocalPruningDomain) -> dict[str, list[ParameterSlice]]:
    result: dict[str, list[ParameterSlice]] = {}
    for index in range(int(domain.original_width)):
        rows: list[ParameterSlice] = []
        for member in domain.dependency_members:
            path = str(member["module_path"])
            if str(member["axis"]) == "out":
                _append_linear_output(rows, model, path, (index,))
            elif str(member["axis"]) == "in":
                _append_linear_input(rows, model, path, (index,))
            else:
                raise RuntimeError(f"transformer_ffn_slice_axis_invalid:{domain.domain_id}:{member['axis']}")
        result[f"{domain.domain_id}::neuron::{index}"] = _dedupe(rows)
    return result


def build_transformer_unit_parameter_slices(
    model: nn.Module,
    domains: Sequence[LocalPruningDomain],
) -> dict[str, list[ParameterSlice]]:
    """Build the mask-proxy mapping used by the shared Stage-1 evaluator."""

    result: dict[str, list[ParameterSlice]] = {}
    for domain in domains:
        if domain.domain_type == "attention_dh":
            rows = _attention_slices(model, domain)
        elif domain.domain_type == "ffn_hidden":
            rows = _ffn_slices(model, domain)
        else:
            continue
        overlap = set(result).intersection(rows)
        if overlap:
            raise RuntimeError(f"transformer_parameter_slice_unit_overlap:{sorted(overlap)}")
        result.update(rows)
    expected = {
        unit_id
        for domain in domains
        if domain.domain_type in {"attention_dh", "ffn_hidden"}
        for unit_id in domain.ordered_unit_ids
    }
    if set(result) != expected:
        raise RuntimeError(
            f"transformer_parameter_slice_inventory_mismatch:missing={sorted(expected - set(result))}:"
            f"extra={sorted(set(result) - expected)}"
        )
    return result


__all__ = ["build_transformer_unit_parameter_slices"]
