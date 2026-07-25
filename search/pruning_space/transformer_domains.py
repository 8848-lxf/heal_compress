"""Instance-local Attention ``d_h`` and FFN ``d_ff`` width domains.

The public search engines only see :class:`LocalPruningDomain`.  Model-specific
structure is captured here as adapter metadata and an immutable decoder; a
family never implies parameter or width sharing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn

from .local_domains import LocalPruningDomain


TRANSFORMER_WIDTH_ALIGNMENT = 4


def _stable_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _legal_aligned_widths(
    original_width: int,
    *,
    minimum_width: int,
    alignment: int = TRANSFORMER_WIDTH_ALIGNMENT,
    label: str,
) -> tuple[int, ...]:
    """Return every aligned width plus the identity width.

    The original width is always retained so an unaligned pretrained model
    still has a lossless identity state.  Every pruned state is aligned.
    """

    original = int(original_width)
    alignment = int(alignment)
    minimum = int(minimum_width)
    if original <= 0:
        raise ValueError(f"{label}_original_width_invalid:{original}")
    if alignment <= 0:
        raise ValueError(f"{label}_width_alignment_invalid:{alignment}")
    first = max(alignment, ((minimum + alignment - 1) // alignment) * alignment)
    values = set(range(first, original + 1, alignment))
    values.add(original)
    return tuple(sorted(values))


def legal_attention_widths(
    original_d_h: int,
    *,
    minimum_width: int = 4,
    alignment: int = TRANSFORMER_WIDTH_ALIGNMENT,
) -> tuple[int, ...]:
    """All 4-aligned per-head dimensions up to the identity width."""

    return _legal_aligned_widths(
        original_d_h,
        minimum_width=minimum_width,
        alignment=alignment,
        label="attention_d_h",
    )


def legal_ffn_widths(
    original_d_ff: int,
    *,
    minimum_width: int = 4,
    alignment: int = TRANSFORMER_WIDTH_ALIGNMENT,
) -> tuple[int, ...]:
    """All 4-aligned FFN hidden widths up to the identity width."""

    return _legal_aligned_widths(
        original_d_ff,
        minimum_width=minimum_width,
        alignment=alignment,
        label="ffn_d_ff",
    )


@dataclass(frozen=True)
class AttentionInstanceSpec:
    model: str
    module_path: str
    block_path: str
    family: str
    qkv_layout: str
    heads: int
    original_d_h: int
    d_model: int
    q_projection_paths: tuple[str, ...]
    k_projection_paths: tuple[str, ...]
    v_projection_paths: tuple[str, ...]
    output_projection_paths: tuple[str, ...]
    softmax_paths: tuple[str, ...] = ()
    adapter: str = "generic"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if min(int(self.heads), int(self.original_d_h), int(self.d_model)) <= 0:
            raise ValueError(f"attention_instance_dimensions_invalid:{self.module_path}")
        if self.qkv_layout not in {"fused_qkv", "separate_qkv", "native_mha"}:
            raise ValueError(f"attention_qkv_layout_invalid:{self.module_path}:{self.qkv_layout}")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def domain_id(self) -> str:
        return f"attention_dh::{self.module_path}"

    @property
    def dependency_paths(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(
            self.q_projection_paths
            + self.k_projection_paths
            + self.v_projection_paths
            + self.output_projection_paths
        ))


@dataclass(frozen=True)
class FFNInstanceSpec:
    model: str
    module_path: str
    block_path: str
    family: str
    ffn_type: str
    original_d_ff: int
    d_model: int
    first_projection_path: str = ""
    second_projection_path: str = ""
    gate_projection_path: str = ""
    up_projection_path: str = ""
    down_projection_path: str = ""
    activation_path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.ffn_type not in {"standard", "gated"}:
            raise ValueError(f"ffn_type_invalid:{self.module_path}:{self.ffn_type}")
        if min(int(self.original_d_ff), int(self.d_model)) <= 0:
            raise ValueError(f"ffn_instance_dimensions_invalid:{self.module_path}")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def domain_id(self) -> str:
        return f"ffn_hidden::{self.module_path}"

    @property
    def dependency_paths(self) -> tuple[str, ...]:
        if self.ffn_type == "gated":
            return (self.gate_projection_path, self.up_projection_path, self.down_projection_path)
        return (self.first_projection_path, self.second_projection_path)


class SharedTransformerParameterError(RuntimeError):
    """Two nominal instances share a module, Parameter, or storage."""


def _parent(path: str) -> str:
    return path.rsplit(".", 1)[0] if "." in path else ""


def _first_linear_path(module_path: str, module: nn.Module, candidates: Sequence[str]) -> str:
    for name in candidates:
        child = getattr(module, name, None)
        if isinstance(child, nn.Linear):
            return f"{module_path}.{name}" if module_path else name
    return ""


def _out_linear_path(module_path: str, module: nn.Module) -> str:
    direct = _first_linear_path(module_path, module, ("out_proj", "proj", "o_proj"))
    if direct:
        return direct
    value = getattr(module, "to_out", None)
    if isinstance(value, nn.Linear):
        return f"{module_path}.to_out" if module_path else "to_out"
    if isinstance(value, (nn.Sequential, nn.ModuleList)):
        for index, child in enumerate(value):
            if isinstance(child, nn.Linear):
                return f"{module_path}.to_out.{index}" if module_path else f"to_out.{index}"
    return ""


def _softmax_paths(module_path: str, module: nn.Module) -> tuple[str, ...]:
    rows: list[str] = []
    for name, child in module.named_modules():
        if name and isinstance(child, nn.Softmax):
            rows.append(f"{module_path}.{name}" if module_path else name)
    return tuple(rows)


def _family_for_attention(path: str, module: nn.Module, model_name: str) -> str:
    if ".grid_attention." in path:
        return "cobevt_grid"
    if ".window_attention." in path:
        return "cobevt_window"
    if ".pwmsa." in path:
        window = int(getattr(module, "window_size", 0) or 0)
        return f"v2xvit_spatial_window_w{window}" if window else "v2xvit_spatial_window"
    if module.__class__.__name__ == "HGTCavAttention":
        return "v2xvit_agent_relation"
    return f"{model_name}_attention"


def discover_attention_instances(model: nn.Module, *, model_name: str) -> list[AttentionInstanceSpec]:
    """Discover real trainable Q/K/V/O instances without name-only guesses."""

    rows: list[AttentionInstanceSpec] = []
    seen_paths: set[str] = set()
    for raw_path, module in model.named_modules(remove_duplicate=False):
        path = raw_path or "__root__"
        if path in seen_paths:
            continue
        cls = module.__class__.__name__
        if cls == "HGTCavAttention" and all(hasattr(module, name) for name in ("q_linears", "k_linears", "v_linears", "a_linears")):
            heads = int(module.heads)
            q_rows = tuple(f"{raw_path}.q_linears.{index}" for index, _ in enumerate(module.q_linears))
            k_rows = tuple(f"{raw_path}.k_linears.{index}" for index, _ in enumerate(module.k_linears))
            v_rows = tuple(f"{raw_path}.v_linears.{index}" for index, _ in enumerate(module.v_linears))
            o_rows = tuple(f"{raw_path}.a_linears.{index}" for index, _ in enumerate(module.a_linears))
            d_h = int(module.q_linears[0].out_features) // heads
            rows.append(AttentionInstanceSpec(
                model=model_name,
                module_path=raw_path,
                block_path=_parent(_parent(raw_path)),
                family="v2xvit_agent_relation",
                qkv_layout="separate_qkv",
                heads=heads,
                original_d_h=d_h,
                d_model=int(module.q_linears[0].in_features),
                q_projection_paths=q_rows,
                k_projection_paths=k_rows,
                v_projection_paths=v_rows,
                output_projection_paths=o_rows,
                adapter="v2xvit_hgt",
                metadata={
                    "relation_att_path": f"{raw_path}.relation_att",
                    "relation_msg_path": f"{raw_path}.relation_msg",
                    "relation_count": int(module.relation_att.shape[0]),
                    "relation_att_operand_precision": "FP32",
                    "relation_msg_operand_precision_source": "av_precision_unit",
                },
            ))
            seen_paths.add(path)
            continue

        fused_path = _first_linear_path(raw_path, module, ("to_qkv", "qkv", "qkv_proj", "in_proj"))
        out_path = _out_linear_path(raw_path, module)
        if fused_path and out_path:
            fused = model.get_submodule(fused_path)
            out = model.get_submodule(out_path)
            heads = int(getattr(module, "heads", getattr(module, "num_heads", getattr(module, "n_heads", 0))) or 0)
            if heads > 0 and fused.out_features % (3 * heads) == 0:
                d_h = int(fused.out_features) // (3 * heads)
                if int(out.in_features) == heads * d_h:
                    rows.append(AttentionInstanceSpec(
                        model=model_name,
                        module_path=raw_path,
                        block_path=_parent(_parent(raw_path)),
                        family=_family_for_attention(raw_path, module, model_name),
                        qkv_layout="fused_qkv",
                        heads=heads,
                        original_d_h=d_h,
                        d_model=int(fused.in_features),
                        q_projection_paths=(fused_path,),
                        k_projection_paths=(fused_path,),
                        v_projection_paths=(fused_path,),
                        output_projection_paths=(out_path,),
                        softmax_paths=_softmax_paths(raw_path, module),
                        adapter=("cobevt" if cls == "Attention" and ".layers." in raw_path else "v2xvit_window" if cls == "BaseWindowAttention" else "generic_fused"),
                    ))
                    seen_paths.add(path)
                    continue

        q_path = _first_linear_path(raw_path, module, ("q_proj", "query", "to_q"))
        k_path = _first_linear_path(raw_path, module, ("k_proj", "key", "to_k"))
        v_path = _first_linear_path(raw_path, module, ("v_proj", "value", "to_v"))
        if q_path and k_path and v_path and out_path:
            q = model.get_submodule(q_path)
            k = model.get_submodule(k_path)
            v = model.get_submodule(v_path)
            out = model.get_submodule(out_path)
            heads = int(getattr(module, "heads", getattr(module, "num_heads", getattr(module, "n_heads", 0))) or 0)
            if heads > 0 and q.out_features == k.out_features and q.out_features % heads == 0:
                d_h = int(q.out_features) // heads
                if v.out_features == out.in_features == heads * d_h:
                    rows.append(AttentionInstanceSpec(
                        model=model_name,
                        module_path=raw_path,
                        block_path=_parent(raw_path),
                        family=_family_for_attention(raw_path, module, model_name),
                        qkv_layout="separate_qkv",
                        heads=heads,
                        original_d_h=d_h,
                        d_model=int(q.in_features),
                        q_projection_paths=(q_path,),
                        k_projection_paths=(k_path,),
                        v_projection_paths=(v_path,),
                        output_projection_paths=(out_path,),
                        adapter="generic_separate",
                    ))
                    seen_paths.add(path)
    return sorted(rows, key=lambda value: value.module_path)


def discover_ffn_instances(model: nn.Module, *, model_name: str) -> list[FFNInstanceSpec]:
    rows: list[FFNInstanceSpec] = []
    seen_paths: set[str] = set()
    for raw_path, module in model.named_modules(remove_duplicate=False):
        path = raw_path or "__root__"
        if path in seen_paths:
            continue
        gate = _first_linear_path(raw_path, module, ("gate_proj", "gate"))
        up = _first_linear_path(raw_path, module, ("up_proj", "up"))
        down = _first_linear_path(raw_path, module, ("down_proj", "down"))
        if gate and up and down:
            gate_module, up_module, down_module = (
                model.get_submodule(gate), model.get_submodule(up), model.get_submodule(down)
            )
            if gate_module.out_features == up_module.out_features == down_module.in_features:
                rows.append(FFNInstanceSpec(
                    model=model_name,
                    module_path=raw_path,
                    block_path=_parent(raw_path),
                    family="gated_transformer_ffn",
                    ffn_type="gated",
                    original_d_ff=int(gate_module.out_features),
                    d_model=int(down_module.out_features),
                    gate_projection_path=gate,
                    up_projection_path=up,
                    down_projection_path=down,
                ))
                seen_paths.add(path)
                continue

        first = _first_linear_path(raw_path, module, ("fc1", "linear1", "w1"))
        second = _first_linear_path(raw_path, module, ("fc2", "linear2", "w2"))
        activation = ""
        net = getattr(module, "net", None)
        if not (first and second) and isinstance(net, (nn.Sequential, nn.ModuleList)):
            linears = [(index, child) for index, child in enumerate(net) if isinstance(child, nn.Linear)]
            if len(linears) >= 2:
                first = f"{raw_path}.net.{linears[0][0]}"
                second = f"{raw_path}.net.{linears[-1][0]}"
                for index, child in enumerate(net):
                    if index > linears[0][0] and index < linears[-1][0] and not isinstance(child, nn.Dropout):
                        activation = f"{raw_path}.net.{index}"
                        break
        if first and second:
            first_module = model.get_submodule(first)
            second_module = model.get_submodule(second)
            if first_module.out_features == second_module.in_features and first_module.in_features == second_module.out_features:
                family = "standard_transformer_ffn"
                if ".window_ffd." in raw_path:
                    family = "cobevt_window_ffn"
                elif ".grid_ffd." in raw_path:
                    family = "cobevt_grid_ffn"
                rows.append(FFNInstanceSpec(
                    model=model_name,
                    module_path=raw_path,
                    block_path=_parent(raw_path),
                    family=family,
                    ffn_type="standard",
                    original_d_ff=int(first_module.out_features),
                    d_model=int(first_module.in_features),
                    first_projection_path=first,
                    second_projection_path=second,
                    activation_path=activation,
                ))
                seen_paths.add(path)
    return sorted(rows, key=lambda value: value.module_path)


def _ranking_rows(
    supplied: Sequence[Sequence[int]] | None,
    *,
    heads: int,
    width: int,
    label: str,
    allow_identity: bool,
) -> tuple[tuple[int, ...], ...]:
    if supplied is None:
        if not allow_identity:
            raise ValueError(f"transformer_fixed_ranking_missing:{label}")
        supplied = tuple(tuple(range(width)) for _ in range(heads))
    rows = tuple(tuple(int(value) for value in row) for row in supplied)
    expected = tuple(range(width))
    if len(rows) != heads or any(tuple(sorted(row)) != expected for row in rows):
        raise ValueError(f"transformer_head_ranking_not_permutation:{label}")
    return rows


def _ffn_ranking(
    supplied: Sequence[int] | None,
    *,
    width: int,
    label: str,
    allow_identity: bool,
) -> tuple[int, ...]:
    if supplied is None:
        if not allow_identity:
            raise ValueError(f"transformer_fixed_ranking_missing:{label}")
        supplied = tuple(range(width))
    order = tuple(int(value) for value in supplied)
    if tuple(sorted(order)) != tuple(range(width)):
        raise ValueError(f"transformer_ffn_ranking_not_permutation:{label}")
    return order


def _parameter_evidence(model: nn.Module, spec: AttentionInstanceSpec | FFNInstanceSpec) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for module_path in spec.dependency_paths:
        module = model.get_submodule(module_path)
        for name, parameter in module.named_parameters(recurse=False):
            storage = parameter.untyped_storage()
            evidence.append({
                "owner": spec.module_path,
                "parameter": f"{module_path}.{name}",
                "object_id": id(parameter),
                "storage_ptr": int(storage.data_ptr()),
                "storage_nbytes": int(storage.nbytes()),
            })
    return evidence


def assert_independent_transformer_instances(
    model: nn.Module,
    specs: Sequence[AttentionInstanceSpec | FFNInstanceSpec],
) -> None:
    """Fail closed when independent genes would mutate shared parameters."""

    by_object: dict[int, dict[str, Any]] = {}
    by_storage: dict[tuple[int, int], dict[str, Any]] = {}
    collisions: list[dict[str, Any]] = []
    for spec in specs:
        for row in _parameter_evidence(model, spec):
            for key, table in (
                (row["object_id"], by_object),
                ((row["storage_ptr"], row["storage_nbytes"]), by_storage),
            ):
                previous = table.setdefault(key, row)
                if previous["owner"] != row["owner"]:
                    collisions.append({"left": previous, "right": row})
    if collisions:
        raise SharedTransformerParameterError(
            "shared_transformer_parameter_requires_explicit_tied_domain:"
            + json.dumps(collisions, sort_keys=True)
        )


def build_attention_dh_domain(
    spec: AttentionInstanceSpec,
    *,
    qk_ranking_by_head: Sequence[Sequence[int]] | None,
    vo_ranking_by_head: Sequence[Sequence[int]] | None,
    minimum_width: int = 4,
    allow_identity_ranking: bool = False,
) -> LocalPruningDomain:
    diagnostic_identity = qk_ranking_by_head is None or vo_ranking_by_head is None
    qk = _ranking_rows(
        qk_ranking_by_head,
        heads=spec.heads,
        width=spec.original_d_h,
        label=f"{spec.module_path}:qk",
        allow_identity=allow_identity_ranking,
    )
    vo = _ranking_rows(
        vo_ranking_by_head,
        heads=spec.heads,
        width=spec.original_d_h,
        label=f"{spec.module_path}:vo",
        allow_identity=allow_identity_ranking,
    )
    legal = legal_attention_widths(spec.original_d_h, minimum_width=minimum_width)
    unit_ids: list[str] = []
    unit_indices: dict[str, tuple[int, ...]] = {}
    unit_scores: dict[str, float] = {}
    for role, ranking in (("qk", qk), ("vo", vo)):
        for head, order in enumerate(ranking):
            for rank, local in enumerate(order):
                unit_id = f"{spec.domain_id}::head{head}::{role}::{local}"
                unit_ids.append(unit_id)
                unit_indices[unit_id] = (head * spec.original_d_h + local,)
                unit_scores[unit_id] = float(rank)
    width_to_pruned: dict[int, tuple[str, ...]] = {}
    for width in legal:
        removed: list[str] = []
        count = spec.original_d_h - int(width)
        for role, ranking in (("qk", qk), ("vo", vo)):
            for head, order in enumerate(ranking):
                removed.extend(
                    f"{spec.domain_id}::head{head}::{role}::{local}"
                    for local in order[:count]
                )
        width_to_pruned[int(width)] = tuple(removed)
    dependency = []
    for role, paths, axis in (
        ("q", spec.q_projection_paths, "out"),
        ("k", spec.k_projection_paths, "out"),
        ("v", spec.v_projection_paths, "out"),
        ("out", spec.output_projection_paths, "in"),
    ):
        dependency.extend({"module_path": path, "role": role, "axis": axis} for path in paths)
    ranking_payload = {
        "domain_id": spec.domain_id,
        "qk_low_to_high_by_head": qk,
        "vo_low_to_high_by_head": vo,
        "method": (
            "diagnostic_identity_fixed_nested_per_head"
            if diagnostic_identity
            else "second_order_task_loss_fixed_nested_per_head"
        ),
    }
    return LocalPruningDomain(
        domain_id=spec.domain_id,
        root_module_path=spec.module_path,
        root_axis="head_local",
        scope_id=spec.module_path,
        kind="attention_dh",
        original_width=spec.original_d_h,
        total_original_width=spec.heads * spec.original_d_h,
        ordered_unit_ids=tuple(unit_ids),
        legal_widths=legal,
        width_to_pruned_unit_ids=width_to_pruned,
        unit_root_indices=unit_indices,
        groups=spec.heads,
        ranking_method=ranking_payload["method"],
        ranking_hash=_stable_hash(ranking_payload),
        unit_scores=unit_scores,
        constraints={
            "heads": spec.heads,
            "d_model": spec.d_model,
            "d_model_fixed": True,
            "qkv_layout": spec.qkv_layout,
            "qk_index_coupled": True,
            "vo_index_coupled": True,
            "equal_retained_count_per_head": True,
            "shared_qkvo_index": qk == vo,
            "residual_does_not_tie_d_h": True,
            "width_alignment": TRANSFORMER_WIDTH_ALIGNMENT,
            "adapter": spec.adapter,
        },
        domain_type="attention_dh",
        model=spec.model,
        module_path=spec.module_path,
        family=spec.family,
        block_path=spec.block_path,
        dependency_members=tuple(dependency),
        ranking_groups={
            "qk_low_to_high_by_head": [list(row) for row in qk],
            "vo_low_to_high_by_head": [list(row) for row in vo],
        },
        latency_mapping={
            "op_type": "attention",
            "H": spec.heads,
            "d_h": spec.original_d_h,
            "H_times_d_h": spec.heads * spec.original_d_h,
            "d_model": spec.d_model,
            "family": spec.family,
        },
        precision_units=("qk_projection", "v_projection", "output_projection", "softmax", "av"),
        metadata={
            **spec.metadata,
            "softmax_paths": list(spec.softmax_paths),
            "width_alignment": TRANSFORMER_WIDTH_ALIGNMENT,
            "width_policy": "all_4_multiples_plus_original_identity",
            "original_width_alignment_exception": bool(spec.original_d_h % TRANSFORMER_WIDTH_ALIGNMENT),
            "diagnostic_identity_ranking": diagnostic_identity,
        },
    )


def build_ffn_hidden_domain(
    spec: FFNInstanceSpec,
    *,
    ranking: Sequence[int] | None,
    minimum_width: int = 4,
    allow_identity_ranking: bool = False,
) -> LocalPruningDomain:
    diagnostic_identity = ranking is None
    order = _ffn_ranking(
        ranking,
        width=spec.original_d_ff,
        label=spec.module_path,
        allow_identity=allow_identity_ranking,
    )
    legal = legal_ffn_widths(spec.original_d_ff, minimum_width=minimum_width)
    unit_ids = tuple(f"{spec.domain_id}::neuron::{index}" for index in order)
    width_to_pruned = {
        int(width): tuple(
            f"{spec.domain_id}::neuron::{index}"
            for index in order[: spec.original_d_ff - int(width)]
        )
        for width in legal
    }
    axes = (
        ((spec.gate_projection_path, "gate", "out"), (spec.up_projection_path, "up", "out"), (spec.down_projection_path, "down", "in"))
        if spec.ffn_type == "gated"
        else ((spec.first_projection_path, "first", "out"), (spec.second_projection_path, "second", "in"))
    )
    ranking_payload = {
        "domain_id": spec.domain_id,
        "ffn_low_to_high": order,
        "method": (
            "diagnostic_identity_fixed_nested_ffn"
            if diagnostic_identity
            else "second_order_task_loss_fixed_nested_ffn"
        ),
    }
    return LocalPruningDomain(
        domain_id=spec.domain_id,
        root_module_path=spec.module_path,
        root_axis="ffn_hidden",
        scope_id=spec.module_path,
        kind="ffn_hidden",
        original_width=spec.original_d_ff,
        total_original_width=spec.original_d_ff,
        ordered_unit_ids=unit_ids,
        legal_widths=legal,
        width_to_pruned_unit_ids=width_to_pruned,
        unit_root_indices={unit_id: (int(unit_id.rsplit("::", 1)[1]),) for unit_id in unit_ids},
        groups=1,
        ranking_method=ranking_payload["method"],
        ranking_hash=_stable_hash(ranking_payload),
        unit_scores={f"{spec.domain_id}::neuron::{index}": float(rank) for rank, index in enumerate(order)},
        constraints={
            "ffn_type": spec.ffn_type,
            "d_model": spec.d_model,
            "d_model_fixed": True,
            "gated_coupling": spec.ffn_type == "gated",
            "first_action_removes_at_most_75_percent": True,
            "width_alignment": TRANSFORMER_WIDTH_ALIGNMENT,
        },
        domain_type="ffn_hidden",
        model=spec.model,
        module_path=spec.module_path,
        family=spec.family,
        block_path=spec.block_path,
        dependency_members=tuple(
            {"module_path": path, "role": role, "axis": axis}
            for path, role, axis in axes
        ),
        ranking_groups={"ffn_low_to_high": list(order)},
        latency_mapping={
            "op_type": "gated_ffn" if spec.ffn_type == "gated" else "ffn",
            "d_model": spec.d_model,
            "d_ff": spec.original_d_ff,
            "family": spec.family,
        },
        precision_units=("ffn1", "ffn_activation", "ffn2"),
        metadata={
            **spec.metadata,
            "activation_path": spec.activation_path,
            "width_alignment": TRANSFORMER_WIDTH_ALIGNMENT,
            "width_policy": "all_4_multiples_plus_original_identity",
            "original_width_alignment_exception": bool(spec.original_d_ff % TRANSFORMER_WIDTH_ALIGNMENT),
            "diagnostic_identity_ranking": diagnostic_identity,
        },
    )


def build_transformer_pruning_domains(
    model: nn.Module,
    *,
    model_name: str,
    attention_rankings: Mapping[str, Mapping[str, Sequence[Sequence[int]]]] | None = None,
    ffn_rankings: Mapping[str, Sequence[int]] | None = None,
    allow_identity_ranking: bool = False,
) -> tuple[list[LocalPruningDomain], list[AttentionInstanceSpec], list[FFNInstanceSpec]]:
    """Build independent instance domains on one shared search abstraction."""

    attention = discover_attention_instances(model, model_name=model_name)
    ffn = discover_ffn_instances(model, model_name=model_name)
    assert_independent_transformer_instances(model, [*attention, *ffn])
    attention_rankings = dict(attention_rankings or {})
    ffn_rankings = dict(ffn_rankings or {})
    domains: list[LocalPruningDomain] = []
    for spec in attention:
        ranking = dict(attention_rankings.get(spec.module_path) or {})
        domains.append(build_attention_dh_domain(
            spec,
            qk_ranking_by_head=ranking.get("qk"),
            vo_ranking_by_head=ranking.get("vo"),
            allow_identity_ranking=allow_identity_ranking,
        ))
    for spec in ffn:
        domains.append(build_ffn_hidden_domain(
            spec,
            ranking=ffn_rankings.get(spec.module_path),
            allow_identity_ranking=allow_identity_ranking,
        ))
    ids = [domain.domain_id for domain in domains]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate_transformer_domain_id")
    return sorted(domains, key=lambda value: value.domain_id), attention, ffn


def fixed_transformer_rankings_from_unit_scores(
    attention: Sequence[AttentionInstanceSpec],
    ffn: Sequence[FFNInstanceSpec],
    unit_scores: Mapping[str, float],
) -> tuple[dict[str, dict[str, tuple[tuple[int, ...], ...]]], dict[str, tuple[int, ...]], dict[str, Any]]:
    """Decode raw common-task-loss atom scores into immutable nested rankings.

    Scores are compared directly. No per-layer min/max, mean, or total-score
    normalization is performed. Missing or non-finite atoms fail closed.
    """

    scores = {str(key): float(value) for key, value in unit_scores.items()}
    missing: list[str] = []

    def ordered(ids: Sequence[tuple[str, int]]) -> tuple[int, ...]:
        for unit_id, _local in ids:
            value = scores.get(unit_id)
            if value is None or not torch.isfinite(torch.tensor(value)):
                missing.append(unit_id)
        return tuple(
            local
            for unit_id, local in sorted(
                ids,
                key=lambda row: (scores.get(row[0], float("inf")), row[1], row[0]),
            )
        )

    attention_rankings: dict[str, dict[str, tuple[tuple[int, ...], ...]]] = {}
    for spec in attention:
        by_role: dict[str, tuple[tuple[int, ...], ...]] = {}
        for role in ("qk", "vo"):
            by_role[role] = tuple(
                ordered(
                    tuple(
                        (
                            f"{spec.domain_id}::head{head}::{role}::{local}",
                            local,
                        )
                        for local in range(spec.original_d_h)
                    )
                )
                for head in range(spec.heads)
            )
        attention_rankings[spec.module_path] = by_role

    ffn_rankings: dict[str, tuple[int, ...]] = {}
    for spec in ffn:
        ffn_rankings[spec.module_path] = ordered(
            tuple(
                (f"{spec.domain_id}::neuron::{index}", index)
                for index in range(spec.original_d_ff)
            )
        )
    if missing:
        raise RuntimeError(f"transformer_fixed_ranking_missing_or_nonfinite_scores:{sorted(set(missing))}")
    payload = {
        "schema_version": "transformer-fixed-domain-ranking-v1",
        "method": "raw_common_task_loss_first_plus_second_order_taylor",
        "normalization_applied": False,
        "type_calibration": "identity",
        "attention": {
            path: {
                role: [list(row) for row in rankings]
                for role, rankings in sorted(roles.items())
            }
            for path, roles in sorted(attention_rankings.items())
        },
        "ffn": {path: list(order) for path, order in sorted(ffn_rankings.items())},
        "unit_scores": {key: scores[key] for key in sorted(scores)},
    }
    payload["ranking_manifest_hash"] = _stable_hash(payload)
    return attention_rankings, ffn_rankings, payload


__all__ = [
    "AttentionInstanceSpec",
    "TRANSFORMER_WIDTH_ALIGNMENT",
    "FFNInstanceSpec",
    "SharedTransformerParameterError",
    "assert_independent_transformer_instances",
    "build_attention_dh_domain",
    "build_ffn_hidden_domain",
    "build_transformer_pruning_domains",
    "discover_attention_instances",
    "discover_ffn_instances",
    "fixed_transformer_rankings_from_unit_scores",
    "legal_attention_widths",
    "legal_ffn_widths",
]
