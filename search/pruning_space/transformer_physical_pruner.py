"""Physical materialization for Transformer width-domain candidates.

Every rewrite replaces Parameters with smaller tensors.  It never installs a
mask, slices the module output after compute, or pads back to the original
``H * d_h``/``d_ff`` shape.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from .local_domains import LocalPruningDomain, legalize_domain_width_genes


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def state_dict_shape_hash(model: nn.Module) -> str:
    return _stable_hash({
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in model.state_dict().items()
    })


def _submodule(model: nn.Module, path: str) -> nn.Module:
    if path in {"", "__root__"}:
        return model
    return model.get_submodule(path)


def _replace_submodule(model: nn.Module, path: str, replacement: nn.Module) -> None:
    if path in {"", "__root__"}:
        raise ValueError("cannot_replace_root_module_in_place")
    parent_path, leaf = path.rsplit(".", 1) if "." in path else ("", path)
    parent = _submodule(model, parent_path)
    if leaf.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(leaf)] = replacement
    else:
        setattr(parent, leaf, replacement)


def _linear_with_slices(
    module: nn.Linear,
    *,
    output_indices: Sequence[int] | None = None,
    input_indices: Sequence[int] | None = None,
) -> nn.Linear:
    out_index = tuple(range(module.out_features)) if output_indices is None else tuple(int(value) for value in output_indices)
    in_index = tuple(range(module.in_features)) if input_indices is None else tuple(int(value) for value in input_indices)
    if not out_index or not in_index:
        raise ValueError("physical_linear_slice_empty")
    if tuple(sorted(set(out_index))) != out_index or out_index[-1] >= module.out_features:
        raise ValueError("physical_linear_output_indices_invalid")
    if tuple(sorted(set(in_index))) != in_index or in_index[-1] >= module.in_features:
        raise ValueError("physical_linear_input_indices_invalid")
    replacement = nn.Linear(
        len(in_index),
        len(out_index),
        bias=module.bias is not None,
        device=module.weight.device,
        dtype=module.weight.dtype,
    )
    out_tensor = torch.as_tensor(out_index, dtype=torch.long, device=module.weight.device)
    in_tensor = torch.as_tensor(in_index, dtype=torch.long, device=module.weight.device)
    with torch.no_grad():
        replacement.weight.copy_(module.weight.index_select(0, out_tensor).index_select(1, in_tensor))
        if module.bias is not None:
            replacement.bias.copy_(module.bias.index_select(0, out_tensor))
    replacement.train(module.training)
    return replacement


def _flatten_keep(rows: Sequence[Sequence[int]], original_d_h: int) -> tuple[int, ...]:
    return tuple(
        head * int(original_d_h) + int(local)
        for head, values in enumerate(rows)
        for local in values
    )


def _update_attention_metadata(module: nn.Module, *, target_d_h: int, heads: int) -> None:
    for name in ("head_dim", "dim_head", "d_qk", "d_v"):
        if hasattr(module, name) or name in {"d_qk", "d_v"}:
            setattr(module, name, int(target_d_h))
    for name in ("inner_dim", "inner_dim_qk", "inner_dim_v", "all_head_size"):
        if hasattr(module, name):
            setattr(module, name, int(heads) * int(target_d_h))
    if hasattr(module, "embed_dim_per_head"):
        setattr(module, "embed_dim_per_head", int(target_d_h))
    setattr(module, "scale", float(target_d_h) ** -0.5)


def _prune_fused_attention(
    model: nn.Module,
    domain: LocalPruningDomain,
    decoded: Mapping[str, Any],
) -> dict[str, Any]:
    module = _submodule(model, domain.module_path)
    heads = int(decoded["heads"])
    target = int(decoded["target_d_h"])
    original = int(domain.original_width)
    qk_flat = _flatten_keep(decoded["qk_keep_by_head"], original)
    vo_flat = _flatten_keep(decoded["vo_keep_by_head"], original)
    fused_paths = tuple(dict.fromkeys(
        member["module_path"]
        for member in domain.dependency_members
        if member["role"] in {"q", "k", "v"}
    ))
    if len(fused_paths) != 1:
        raise RuntimeError(f"fused_qkv_path_count_invalid:{domain.domain_id}:{fused_paths}")
    fused_path = fused_paths[0]
    fused = _submodule(model, fused_path)
    if not isinstance(fused, nn.Linear) or fused.out_features != 3 * heads * original:
        raise RuntimeError(f"fused_qkv_shape_invalid:{domain.domain_id}")
    projection_width = heads * original
    fused_rows = (
        qk_flat
        + tuple(projection_width + value for value in qk_flat)
        + tuple(2 * projection_width + value for value in vo_flat)
    )
    _replace_submodule(model, fused_path, _linear_with_slices(fused, output_indices=fused_rows))
    out_paths = tuple(member["module_path"] for member in domain.dependency_members if member["role"] == "out")
    if len(out_paths) != 1:
        raise RuntimeError(f"attention_output_path_count_invalid:{domain.domain_id}:{out_paths}")
    out = _submodule(model, out_paths[0])
    if not isinstance(out, nn.Linear) or out.in_features != heads * original:
        raise RuntimeError(f"attention_output_shape_invalid:{domain.domain_id}")
    _replace_submodule(model, out_paths[0], _linear_with_slices(out, input_indices=vo_flat))
    _update_attention_metadata(module, target_d_h=target, heads=heads)
    return {
        "module_path": domain.module_path,
        "domain_id": domain.domain_id,
        "domain_type": "attention_dh",
        "qkv_layout": "fused_qkv",
        "before_d_h": original,
        "after_d_h": target,
        "heads": heads,
        "qk_flattened_keep": list(qk_flat),
        "vo_flattened_keep": list(vo_flat),
        "bias_synchronized": fused.bias is not None,
        "scale_after": float(getattr(module, "scale")),
        "reshape_contract": "head_count_fixed_dynamic_local_dimension",
        "mask_only": False,
        "hidden_padding": False,
    }


def _prune_separate_attention(
    model: nn.Module,
    domain: LocalPruningDomain,
    decoded: Mapping[str, Any],
) -> dict[str, Any]:
    module = _submodule(model, domain.module_path)
    heads = int(decoded["heads"])
    target = int(decoded["target_d_h"])
    original = int(domain.original_width)
    qk_flat = _flatten_keep(decoded["qk_keep_by_head"], original)
    vo_flat = _flatten_keep(decoded["vo_keep_by_head"], original)
    role_paths = {
        role: tuple(member["module_path"] for member in domain.dependency_members if member["role"] == role)
        for role in ("q", "k", "v", "out")
    }
    if not all(role_paths.values()):
        raise RuntimeError(f"separate_attention_dependency_missing:{domain.domain_id}")
    for role in ("q", "k", "v"):
        keep = qk_flat if role in {"q", "k"} else vo_flat
        for path in role_paths[role]:
            linear = _submodule(model, path)
            if not isinstance(linear, nn.Linear) or linear.out_features != heads * original:
                raise RuntimeError(f"separate_projection_shape_invalid:{domain.domain_id}:{path}")
            _replace_submodule(model, path, _linear_with_slices(linear, output_indices=keep))
    for path in role_paths["out"]:
        linear = _submodule(model, path)
        if not isinstance(linear, nn.Linear) or linear.in_features != heads * original:
            raise RuntimeError(f"separate_output_shape_invalid:{domain.domain_id}:{path}")
        _replace_submodule(model, path, _linear_with_slices(linear, input_indices=vo_flat))

    adapter = str(domain.constraints.get("adapter", ""))
    if adapter == "v2xvit_hgt":
        relation_specs = (
            (str(domain.metadata.get("relation_att_path", "")), decoded["qk_keep_by_head"]),
            (str(domain.metadata.get("relation_msg_path", "")), decoded["vo_keep_by_head"]),
        )
        named_parameters = dict(model.named_parameters())
        for path, keep_by_head in relation_specs:
            if path not in named_parameters:
                raise RuntimeError(f"hgt_relation_parameter_missing:{path}")
            parameter = named_parameters[path]
            if parameter.ndim != 4 or parameter.shape[1] != heads or tuple(parameter.shape[-2:]) != (original, original):
                raise RuntimeError(f"hgt_relation_parameter_shape_invalid:{path}:{tuple(parameter.shape)}")
            head_values = []
            for head, keep in enumerate(keep_by_head):
                index = torch.as_tensor(keep, dtype=torch.long, device=parameter.device)
                head_values.append(parameter[:, head].index_select(1, index).index_select(2, index))
            replacement = nn.Parameter(
                torch.stack(head_values, dim=1).clone(),
                requires_grad=parameter.requires_grad,
            )
            leaf = path.rsplit(".", 1)[-1]
            setattr(module, leaf, replacement)
    _update_attention_metadata(module, target_d_h=target, heads=heads)
    return {
        "module_path": domain.module_path,
        "domain_id": domain.domain_id,
        "domain_type": "attention_dh",
        "qkv_layout": "separate_qkv",
        "before_d_h": original,
        "after_d_h": target,
        "heads": heads,
        "qk_flattened_keep": list(qk_flat),
        "vo_flattened_keep": list(vo_flat),
        "bias_synchronized": True,
        "relation_tensors_synchronized": adapter == "v2xvit_hgt",
        "scale_after": float(getattr(module, "scale")),
        "reshape_contract": "head_count_fixed_dynamic_local_dimension",
        "mask_only": False,
        "hidden_padding": False,
    }


def _prune_ffn(
    model: nn.Module,
    domain: LocalPruningDomain,
    decoded: Mapping[str, Any],
) -> dict[str, Any]:
    target = int(decoded["target_d_ff"])
    keep = tuple(int(value) for value in decoded["keep_indices"])
    original = int(domain.original_width)
    if len(keep) != target:
        raise RuntimeError(f"ffn_keep_count_mismatch:{domain.domain_id}")
    members = tuple(domain.dependency_members)
    output_roles = {"first", "gate", "up"}
    input_roles = {"second", "down"}
    for member in members:
        path = str(member["module_path"])
        linear = _submodule(model, path)
        if not isinstance(linear, nn.Linear):
            raise RuntimeError(f"ffn_dependency_not_linear:{domain.domain_id}:{path}")
        if member["role"] in output_roles:
            if linear.out_features != original:
                raise RuntimeError(f"ffn_output_shape_invalid:{domain.domain_id}:{path}")
            replacement = _linear_with_slices(linear, output_indices=keep)
        elif member["role"] in input_roles:
            if linear.in_features != original:
                raise RuntimeError(f"ffn_input_shape_invalid:{domain.domain_id}:{path}")
            replacement = _linear_with_slices(linear, input_indices=keep)
        else:
            raise RuntimeError(f"ffn_dependency_role_invalid:{domain.domain_id}:{member['role']}")
        _replace_submodule(model, path, replacement)
    return {
        "module_path": domain.module_path,
        "domain_id": domain.domain_id,
        "domain_type": "ffn_hidden",
        "ffn_type": decoded["ffn_type"],
        "before_d_ff": original,
        "after_d_ff": target,
        "keep_indices": list(keep),
        "d_model_fixed": True,
        "bias_synchronized": True,
        "mask_only": False,
        "hidden_padding": False,
    }


def _realized_width(model: nn.Module, domain: LocalPruningDomain) -> int:
    if domain.domain_type == "attention_dh":
        heads = int(domain.constraints["heads"])
        q_path = next(member["module_path"] for member in domain.dependency_members if member["role"] == "q")
        q = _submodule(model, q_path)
        if domain.constraints.get("qkv_layout") == "fused_qkv":
            return int(q.out_features) // 3 // heads
        return int(q.out_features) // heads
    if domain.domain_type == "ffn_hidden":
        member = next(value for value in domain.dependency_members if value["axis"] == "out")
        return int(_submodule(model, member["module_path"]).out_features)
    raise ValueError(f"unsupported_transformer_domain_type:{domain.domain_type}")


@dataclass(frozen=True)
class TransformerPhysicalPruneReport:
    model: str
    requested_widths: dict[str, int]
    realized_widths: dict[str, int]
    original_parameter_count: int
    physical_parameter_count: int
    operations: tuple[dict[str, Any], ...]
    state_dict_shape_hash: str
    structure_hash: str
    mask_only: bool
    hidden_padding: bool
    passed: bool
    issues: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def materialize_transformer_widths(
    model: nn.Module,
    domains: Sequence[LocalPruningDomain],
    width_genes: Mapping[str, int],
    *,
    model_name: str = "",
) -> TransformerPhysicalPruneReport:
    """Apply all requested Transformer widths and fail on any mismatch."""

    transformer_domains = tuple(
        domain for domain in domains if domain.domain_type in {"attention_dh", "ffn_hidden"}
    )
    requested = legalize_domain_width_genes(width_genes, transformer_domains)
    original_count = sum(int(parameter.numel()) for parameter in model.parameters())
    operations: list[dict[str, Any]] = []
    for domain in transformer_domains:
        target = requested[domain.domain_id]
        if target == domain.original_width:
            continue
        decoded = domain.decode_width(target)
        if domain.domain_type == "attention_dh":
            if domain.constraints.get("qkv_layout") == "fused_qkv":
                operations.append(_prune_fused_attention(model, domain, decoded))
            else:
                operations.append(_prune_separate_attention(model, domain, decoded))
        elif domain.domain_type == "ffn_hidden":
            operations.append(_prune_ffn(model, domain, decoded))
    realized = {domain.domain_id: _realized_width(model, domain) for domain in transformer_domains}
    issues = [
        f"requested_realized_width_conflict:{domain_id}:{requested[domain_id]}:{realized[domain_id]}"
        for domain_id in requested
        if requested[domain_id] != realized[domain_id]
    ]
    for domain in transformer_domains:
        if domain.domain_type != "attention_dh":
            continue
        target = realized[domain.domain_id]
        module = _submodule(model, domain.module_path)
        if not math.isclose(float(getattr(module, "scale", 0.0)), target**-0.5, rel_tol=0.0, abs_tol=1.0e-12):
            issues.append(f"attention_scale_mismatch:{domain.domain_id}")
        if int(domain.constraints.get("d_model", 0)) <= 0:
            issues.append(f"attention_d_model_contract_missing:{domain.domain_id}")
    physical_count = sum(int(parameter.numel()) for parameter in model.parameters())
    if any(requested[key] < next(row.original_width for row in transformer_domains if row.domain_id == key) for key in requested) and physical_count >= original_count:
        issues.append("physical_parameter_count_did_not_decrease")
    shape_hash = state_dict_shape_hash(model)
    structure_payload = {
        "recipe": "transformer-instance-width-physical-v1",
        "model": model_name,
        "requested_widths": requested,
        "realized_widths": realized,
        "operations": operations,
        "state_dict_shape_hash": shape_hash,
    }
    return TransformerPhysicalPruneReport(
        model=str(model_name),
        requested_widths=requested,
        realized_widths=realized,
        original_parameter_count=original_count,
        physical_parameter_count=physical_count,
        operations=tuple(operations),
        state_dict_shape_hash=shape_hash,
        structure_hash=_stable_hash(structure_payload),
        mask_only=False,
        hidden_padding=False,
        passed=not issues,
        issues=tuple(issues),
    )


__all__ = [
    "TransformerPhysicalPruneReport",
    "materialize_transformer_widths",
    "state_dict_shape_hash",
]
