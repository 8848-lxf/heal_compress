"""Transformer-aware BOPS, parameter and activation-memory proxy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn as nn

from ..pruning_space.transformer_domains import AttentionInstanceSpec, FFNInstanceSpec

from ..candidate import CandidatePhenotype
from ..pruning_space.local_domains import LocalPruningDomain
from .size_proxy import BIT_WIDTHS


@dataclass(frozen=True)
class AttentionWorkload:
    module_path: str
    projection_tokens: int
    query_tokens: int
    key_tokens: int
    attention_groups: int = 1

    def __post_init__(self) -> None:
        if min(
            int(self.projection_tokens),
            int(self.query_tokens),
            int(self.key_tokens),
            int(self.attention_groups),
        ) <= 0:
            raise ValueError(f"attention_workload_invalid:{self.module_path}")


@dataclass(frozen=True)
class FFNWorkload:
    module_path: str
    tokens: int

    def __post_init__(self) -> None:
        if int(self.tokens) <= 0:
            raise ValueError(f"ffn_workload_invalid:{self.module_path}")


@dataclass(frozen=True)
class ProjectionFreeAttentionWorkload:
    module_path: str
    query_tokens: int
    key_tokens: int
    feature_dimension: int
    attention_groups: int = 1

    def __post_init__(self) -> None:
        if min(
            int(self.query_tokens),
            int(self.key_tokens),
            int(self.feature_dimension),
            int(self.attention_groups),
        ) <= 0:
            raise ValueError(f"projection_free_attention_workload_invalid:{self.module_path}")


def _first_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return _first_tensor(item)
            except TypeError:
                pass
    if isinstance(value, Mapping):
        for item in value.values():
            try:
                return _first_tensor(item)
            except TypeError:
                pass
    raise TypeError("transformer_workload_input_tensor_missing")


def profile_transformer_workloads(
    model: nn.Module,
    sample: Any,
    *,
    forward_fn: Callable[[nn.Module, Any], Any],
    attention_instances: Sequence[AttentionInstanceSpec],
    ffn_instances: Sequence[FFNInstanceSpec],
) -> tuple[tuple[AttentionWorkload, ...], tuple[FFNWorkload, ...], dict[str, Any]]:
    """Measure real token/group shapes at every independent instance input."""

    attention_shapes: dict[str, list[tuple[int, ...]]] = {
        row.module_path: [] for row in attention_instances
    }
    ffn_shapes: dict[str, list[tuple[int, ...]]] = {
        row.module_path: [] for row in ffn_instances
    }
    handles: list[Any] = []

    def hook(target: list[tuple[int, ...]]):
        def capture(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
            target.append(tuple(int(value) for value in _first_tensor(inputs).shape))

        return capture

    for spec in attention_instances:
        handles.append(
            model.get_submodule(spec.module_path).register_forward_pre_hook(
                hook(attention_shapes[spec.module_path])
            )
        )
    for spec in ffn_instances:
        handles.append(
            model.get_submodule(spec.module_path).register_forward_pre_hook(
                hook(ffn_shapes[spec.module_path])
            )
        )
    try:
        with torch.no_grad():
            forward_fn(model, sample)
    finally:
        for handle in handles:
            handle.remove()

    attention_rows: list[AttentionWorkload] = []
    for spec in attention_instances:
        shapes = attention_shapes[spec.module_path]
        if not shapes:
            raise RuntimeError(f"attention_workload_instance_not_called:{spec.module_path}")
        call_rows: list[tuple[int, int, int, int]] = []
        for shape in shapes:
            if len(shape) < 2 or shape[-1] != spec.d_model:
                raise RuntimeError(
                    f"attention_workload_shape_contract:{spec.module_path}:{shape}:d_model={spec.d_model}"
                )
            projection_tokens = int(math.prod(shape[:-1]))
            if spec.adapter == "v2xvit_hgt":
                if len(shape) != 5:
                    raise RuntimeError(f"v2xvit_hgt_workload_shape:{spec.module_path}:{shape}")
                batch, agents, height, width, _channels = shape
                query_tokens = key_tokens = int(agents)
                groups = int(batch * height * width)
            elif spec.adapter == "v2xvit_window":
                if len(shape) != 5:
                    raise RuntimeError(f"v2xvit_window_workload_shape:{spec.module_path}:{shape}")
                window = int(getattr(model.get_submodule(spec.module_path), "window_size"))
                query_tokens = key_tokens = window * window
                if projection_tokens % query_tokens:
                    raise RuntimeError(f"v2xvit_window_workload_divisibility:{spec.module_path}")
                groups = projection_tokens // query_tokens
            else:
                query_tokens = key_tokens = int(shape[-2])
                groups = int(math.prod(shape[:-2])) if len(shape) > 2 else 1
            call_rows.append((projection_tokens, query_tokens, key_tokens, groups))
        q_values = {(row[1], row[2]) for row in call_rows}
        if len(q_values) != 1:
            raise RuntimeError(f"attention_workload_dynamic_token_shape:{spec.module_path}:{call_rows}")
        query_tokens, key_tokens = next(iter(q_values))
        attention_rows.append(
            AttentionWorkload(
                module_path=spec.module_path,
                projection_tokens=sum(row[0] for row in call_rows),
                query_tokens=query_tokens,
                key_tokens=key_tokens,
                attention_groups=sum(row[3] for row in call_rows),
            )
        )

    ffn_rows: list[FFNWorkload] = []
    for spec in ffn_instances:
        shapes = ffn_shapes[spec.module_path]
        if not shapes:
            raise RuntimeError(f"ffn_workload_instance_not_called:{spec.module_path}")
        for shape in shapes:
            if len(shape) < 2 or shape[-1] != spec.d_model:
                raise RuntimeError(
                    f"ffn_workload_shape_contract:{spec.module_path}:{shape}:d_model={spec.d_model}"
                )
        ffn_rows.append(
            FFNWorkload(
                module_path=spec.module_path,
                tokens=sum(int(math.prod(shape[:-1])) for shape in shapes),
            )
        )
    audit = {
        "schema_version": "real-transformer-workload-profile-v1",
        "attention_input_shapes": {
            key: [list(value) for value in values]
            for key, values in sorted(attention_shapes.items())
        },
        "ffn_input_shapes": {
            key: [list(value) for value in values]
            for key, values in sorted(ffn_shapes.items())
        },
        "attention_workloads": [asdict(row) for row in attention_rows],
        "ffn_workloads": [asdict(row) for row in ffn_rows],
        "real_forward_executed": True,
    }
    return tuple(attention_rows), tuple(ffn_rows), audit


def profile_projection_free_attention_workloads(
    model: nn.Module,
    sample: Any,
    *,
    forward_fn: Callable[[nn.Module, Any], Any],
    module_paths: Sequence[str],
) -> tuple[tuple[ProjectionFreeAttentionWorkload, ...], dict[str, Any]]:
    shapes: dict[str, list[tuple[int, ...]]] = {
        str(path): [] for path in module_paths
    }
    handles: list[Any] = []

    def make_hook(path: str):
        def capture(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
            if len(inputs) < 3:
                raise RuntimeError(f"projection_free_attention_inputs_missing:{path}")
            query, key, value = (_first_tensor(inputs[index]) for index in range(3))
            if query.shape[:-2] != key.shape[:-2] or key.shape[:-2] != value.shape[:-2]:
                raise RuntimeError(f"projection_free_attention_batch_shape_conflict:{path}")
            if query.shape[-1] != key.shape[-1] or key.shape[-1] != value.shape[-1]:
                raise RuntimeError(f"projection_free_attention_feature_shape_conflict:{path}")
            shapes[path].append(tuple(int(value) for value in query.shape))

        return capture

    for path in shapes:
        handles.append(model.get_submodule(path).register_forward_pre_hook(make_hook(path)))
    try:
        with torch.no_grad():
            forward_fn(model, sample)
    finally:
        for handle in handles:
            handle.remove()
    rows: list[ProjectionFreeAttentionWorkload] = []
    for path, values in sorted(shapes.items()):
        if not values:
            raise RuntimeError(f"projection_free_attention_not_called:{path}")
        signatures = {(shape[-2], shape[-1]) for shape in values}
        if len(signatures) != 1:
            raise RuntimeError(f"projection_free_attention_dynamic_shape:{path}:{values}")
        tokens, feature = next(iter(signatures))
        groups = sum(int(math.prod(shape[:-2])) if len(shape) > 2 else 1 for shape in values)
        rows.append(
            ProjectionFreeAttentionWorkload(
                module_path=path,
                query_tokens=tokens,
                key_tokens=tokens,
                feature_dimension=feature,
                attention_groups=groups,
            )
        )
    return tuple(rows), {
        "schema_version": "real-projection-free-attention-workload-v1",
        "input_shapes": {
            path: [list(shape) for shape in values]
            for path, values in sorted(shapes.items())
        },
        "workloads": [asdict(row) for row in rows],
        "real_forward_executed": True,
    }


def _precision(profile: Mapping[str, str], path: str, default: str = "FP32") -> str:
    return str(profile.get(path, default)).upper()


def _bits(profile: Mapping[str, str], path: str, default: str = "FP32") -> int:
    value = _precision(profile, path, default)
    if value not in BIT_WIDTHS:
        raise ValueError(f"transformer_bops_precision_invalid:{path}:{value}")
    return int(BIT_WIDTHS[value])


class TransformerBOPSProxy:
    """Exact formula template over explicit Attention/FFN workload shapes."""

    def __init__(
        self,
        domains: Sequence[LocalPruningDomain],
        *,
        attention_workloads: Sequence[AttentionWorkload],
        ffn_workloads: Sequence[FFNWorkload],
        projection_free_workloads: Sequence[ProjectionFreeAttentionWorkload] = (),
        softmax_ops_per_element: int = 5,
    ) -> None:
        self.domains = tuple(
            value for value in domains if value.domain_type in {"attention_dh", "ffn_hidden"}
        )
        self.by_module = {value.module_path: value for value in self.domains}
        self.attention_workloads = tuple(attention_workloads)
        self.ffn_workloads = tuple(ffn_workloads)
        self.projection_free_workloads = tuple(projection_free_workloads)
        self.softmax_ops_per_element = int(softmax_ops_per_element)
        expected_attention = {value.module_path for value in self.domains if value.domain_type == "attention_dh"}
        expected_ffn = {value.module_path for value in self.domains if value.domain_type == "ffn_hidden"}
        if {value.module_path for value in self.attention_workloads} != expected_attention:
            raise ValueError("attention_workload_domain_inventory_mismatch")
        if {value.module_path for value in self.ffn_workloads} != expected_ffn:
            raise ValueError("ffn_workload_domain_inventory_mismatch")

    @staticmethod
    def _role_paths(domain: LocalPruningDomain) -> dict[str, tuple[str, ...]]:
        return {
            role: tuple(dict.fromkeys(
                str(member["module_path"])
                for member in domain.dependency_members
                if member["role"] == role
            ))
            for role in ("q", "k", "v", "out", "first", "second", "gate", "up", "down")
        }

    def _rows(
        self,
        widths: Mapping[str, int],
        profile: Mapping[str, str],
    ) -> tuple[list[dict[str, Any]], float, float, float]:
        rows: list[dict[str, Any]] = []
        mixed_weight_bits = 0.0
        parameter_count = 0.0
        activation_memory_bits = 0.0
        for workload in self.attention_workloads:
            domain = self.by_module[workload.module_path]
            d_h = int(widths.get(domain.domain_id, domain.original_width))
            if d_h not in domain.legal_widths:
                raise ValueError(f"transformer_bops_illegal_width:{domain.domain_id}:{d_h}")
            heads = int(domain.constraints["heads"])
            d_model = int(domain.constraints["d_model"])
            inner = heads * d_h
            paths = self._role_paths(domain)
            fused = str(domain.constraints.get("qkv_layout")) == "fused_qkv"
            projection_components = (
                ("q_projection", paths["q"][0], workload.projection_tokens * d_model * inner, len(paths["q"])),
                ("k_projection", paths["k"][0], workload.projection_tokens * d_model * inner, len(paths["k"])),
                ("v_projection", paths["v"][0], workload.projection_tokens * d_model * inner, len(paths["v"])),
            )
            for component, path, macs, storage_copies in projection_components:
                weight_bits = _bits(profile, path)
                activation_bits = weight_bits
                bops = float(macs * weight_bits * activation_bits)
                # Type-specific HGT projections share one runtime workload but
                # own distinct parameter tensors. MAC uses the observed token
                # total once; storage/parameter accounting includes every path.
                parameters = d_model * inner * storage_copies
                rows.append({
                    "domain_id": domain.domain_id,
                    "module_path": path,
                    "family": domain.family,
                    "component": component,
                    "category": "attention_projection",
                    "MACs": float(macs),
                    "weight_bits": weight_bits,
                    "activation_bits": activation_bits,
                    "BOPS": bops,
                    "parameters": parameters,
                    "fused_qkv_storage": fused,
                    "parameter_storage_copies": storage_copies,
                    "T": workload.projection_tokens,
                    "H": heads,
                    "d_h": d_h,
                    "d_model": d_model,
                })
                mixed_weight_bits += parameters * weight_bits
                parameter_count += parameters
                activation_memory_bits += workload.projection_tokens * inner * activation_bits

            qk_macs = workload.attention_groups * heads * workload.query_tokens * workload.key_tokens * d_h
            qk_bops = float(qk_macs * 32 * 32)
            rows.append({
                "domain_id": domain.domain_id,
                "module_path": f"{domain.module_path}::__qk_matmul__",
                "family": domain.family,
                "component": "qk_matmul",
                "category": "qk",
                "MACs": float(qk_macs),
                "weight_bits": None,
                "activation_bits": None,
                "operand_a_bits": 32,
                "operand_b_bits": 32,
                "compute_precision": "FP32",
                "accumulator_precision": "FP32",
                "output_precision": "FP32",
                "BOPS": qk_bops,
                "T_q": workload.query_tokens,
                "T_k": workload.key_tokens,
                "attention_groups": workload.attention_groups,
                "H": heads,
                "d_h": d_h,
            })
            score_elements = workload.attention_groups * heads * workload.query_tokens * workload.key_tokens
            activation_memory_bits += score_elements * 32

            softmax_paths = tuple(domain.metadata.get("softmax_paths") or ())
            softmax_path = str(softmax_paths[0]) if softmax_paths else f"{domain.module_path}::__softmax_output__"
            softmax_bits = _bits(profile, softmax_path)
            softmax_ops = score_elements * self.softmax_ops_per_element
            softmax_cost = float(softmax_ops * softmax_bits)
            rows.append({
                "domain_id": domain.domain_id,
                "module_path": softmax_path,
                "family": domain.family,
                "component": "softmax",
                "category": "activation_op",
                "activation_operations": float(softmax_ops),
                "activation_bits": softmax_bits,
                "weight_bits": None,
                "BOPS": softmax_cost,
                "cost_semantics": "activation_operation_bit_cost_no_fake_weight_bits",
            })
            activation_memory_bits += score_elements * softmax_bits

            av_path = f"{domain.module_path}::__av_matmul__"
            av_bits = _bits(profile, av_path)
            av_macs = qk_macs
            av_bops = float(av_macs * av_bits * av_bits)
            rows.append({
                "domain_id": domain.domain_id,
                "module_path": av_path,
                "family": domain.family,
                "component": "av_matmul",
                "category": "av",
                "MACs": float(av_macs),
                "weight_bits": None,
                "activation_bits": None,
                "operand_a_bits": av_bits,
                "operand_b_bits": av_bits,
                "BOPS": av_bops,
                "T_q": workload.query_tokens,
                "T_k": workload.key_tokens,
                "H": heads,
                "d_h": d_h,
            })
            activation_memory_bits += workload.projection_tokens * inner * av_bits

            if str(domain.constraints.get("adapter")) == "v2xvit_hgt":
                relation_count = int(domain.metadata.get("relation_count", 0))
                if relation_count <= 0:
                    raise RuntimeError(f"v2xvit_hgt_relation_count_missing:{domain.domain_id}")
                pair_count = (
                    workload.attention_groups
                    * heads
                    * workload.query_tokens
                    * workload.key_tokens
                )
                relation_macs = pair_count * d_h * d_h
                relation_parameters = relation_count * heads * d_h * d_h
                rows.append({
                    "domain_id": domain.domain_id,
                    "module_path": str(domain.metadata["relation_att_path"]),
                    "family": domain.family,
                    "component": "qk_relation_transform",
                    "category": "qk_relation",
                    "MACs": float(relation_macs),
                    "weight_bits": 32,
                    "activation_bits": 32,
                    "operand_a_semantic": "learned_relation_att",
                    "operand_b_semantic": "q_or_k_activation",
                    "compute_precision": "FP32",
                    "accumulator_precision": "FP32",
                    "output_precision": "FP32",
                    "BOPS": float(relation_macs * 32 * 32),
                    "parameters": relation_parameters,
                    "T_q": workload.query_tokens,
                    "T_k": workload.key_tokens,
                    "attention_groups": workload.attention_groups,
                    "H": heads,
                    "d_h": d_h,
                })
                rows.append({
                    "domain_id": domain.domain_id,
                    "module_path": str(domain.metadata["relation_msg_path"]),
                    "family": domain.family,
                    "component": "message_relation_transform",
                    "category": "av_relation",
                    "MACs": float(relation_macs),
                    "weight_bits": av_bits,
                    "activation_bits": av_bits,
                    "operand_a_semantic": "learned_relation_msg",
                    "operand_b_semantic": "v_activation",
                    "compute_precision": _precision(profile, av_path),
                    "accumulator_precision": _precision(profile, av_path),
                    "output_precision": _precision(profile, av_path),
                    "BOPS": float(relation_macs * av_bits * av_bits),
                    "parameters": relation_parameters,
                    "T_q": workload.query_tokens,
                    "T_k": workload.key_tokens,
                    "attention_groups": workload.attention_groups,
                    "H": heads,
                    "d_h": d_h,
                })
                parameter_count += 2 * relation_parameters
                mixed_weight_bits += relation_parameters * (32 + av_bits)

            out_path = paths["out"][0]
            out_bits = _bits(profile, out_path)
            out_macs = workload.projection_tokens * inner * d_model
            out_parameters = inner * d_model * len(paths["out"])
            rows.append({
                "domain_id": domain.domain_id,
                "module_path": out_path,
                "family": domain.family,
                "component": "output_projection",
                "category": "output_projection",
                "MACs": float(out_macs),
                "weight_bits": out_bits,
                "activation_bits": out_bits,
                "BOPS": float(out_macs * out_bits * out_bits),
                "parameters": out_parameters,
                "parameter_storage_copies": len(paths["out"]),
                "T": workload.projection_tokens,
                "H": heads,
                "d_h": d_h,
                "d_model": d_model,
            })
            mixed_weight_bits += out_parameters * out_bits
            parameter_count += out_parameters
            activation_memory_bits += workload.projection_tokens * d_model * out_bits

        for workload in self.ffn_workloads:
            domain = self.by_module[workload.module_path]
            d_ff = int(widths.get(domain.domain_id, domain.original_width))
            if d_ff not in domain.legal_widths:
                raise ValueError(f"transformer_bops_illegal_width:{domain.domain_id}:{d_ff}")
            d_model = int(domain.constraints["d_model"])
            ffn_type = str(domain.constraints["ffn_type"])
            paths = self._role_paths(domain)
            first_paths = paths["gate"] + paths["up"] if ffn_type == "gated" else paths["first"]
            second_paths = paths["down"] if ffn_type == "gated" else paths["second"]
            for index, path in enumerate((*first_paths, *second_paths)):
                precision_bits = _bits(profile, path)
                macs = workload.tokens * d_model * d_ff
                component = (
                    "ffn_gate" if ffn_type == "gated" and index == 0
                    else "ffn_up" if ffn_type == "gated" and index == 1
                    else "ffn_down" if ffn_type == "gated"
                    else "ffn1" if index == 0
                    else "ffn2"
                )
                parameters = d_model * d_ff
                rows.append({
                    "domain_id": domain.domain_id,
                    "module_path": path,
                    "family": domain.family,
                    "component": component,
                    "category": "ffn",
                    "MACs": float(macs),
                    "weight_bits": precision_bits,
                    "activation_bits": precision_bits,
                    "BOPS": float(macs * precision_bits * precision_bits),
                    "parameters": parameters,
                    "T": workload.tokens,
                    "d_model": d_model,
                    "d_ff": d_ff,
                    "ffn_type": ffn_type,
                })
                mixed_weight_bits += parameters * precision_bits
                parameter_count += parameters
                activation_memory_bits += workload.tokens * (d_ff if component != "ffn2" and component != "ffn_down" else d_model) * precision_bits
        for workload in self.projection_free_workloads:
            prefix = workload.module_path
            qk_macs = (
                workload.attention_groups
                * workload.query_tokens
                * workload.key_tokens
                * workload.feature_dimension
            )
            rows.append({
                "domain_id": "",
                "module_path": f"{prefix}::__qk_bmm__",
                "family": "projection_free_attention",
                "component": "qk_matmul",
                "category": "qk",
                "MACs": float(qk_macs),
                "weight_bits": None,
                "operand_a_bits": 32,
                "operand_b_bits": 32,
                "compute_precision": "FP32",
                "accumulator_precision": "FP32",
                "output_precision": "FP32",
                "BOPS": float(qk_macs * 32 * 32),
            })
            score_elements = (
                workload.attention_groups
                * workload.query_tokens
                * workload.key_tokens
            )
            softmax_path = f"{prefix}::__softmax_output__"
            softmax_bits = _bits(profile, softmax_path)
            softmax_ops = score_elements * self.softmax_ops_per_element
            rows.append({
                "domain_id": "",
                "module_path": softmax_path,
                "family": "projection_free_attention",
                "component": "softmax",
                "category": "activation_op",
                "activation_operations": float(softmax_ops),
                "activation_bits": softmax_bits,
                "weight_bits": None,
                "BOPS": float(softmax_ops * softmax_bits),
                "cost_semantics": "activation_operation_bit_cost_no_fake_weight_bits",
            })
            av_path = f"{prefix}::__av_bmm__"
            av_bits = _bits(profile, av_path)
            rows.append({
                "domain_id": "",
                "module_path": av_path,
                "family": "projection_free_attention",
                "component": "av_matmul",
                "category": "av",
                "MACs": float(qk_macs),
                "weight_bits": None,
                "operand_a_bits": av_bits,
                "operand_b_bits": av_bits,
                "BOPS": float(qk_macs * av_bits * av_bits),
            })
            activation_memory_bits += score_elements * (32 + softmax_bits)
            activation_memory_bits += (
                workload.attention_groups
                * workload.query_tokens
                * workload.feature_dimension
                * av_bits
            )
        return rows, mixed_weight_bits, parameter_count, activation_memory_bits

    def evaluate_breakdown(self, phenotype: CandidatePhenotype) -> dict[str, Any]:
        widths = dict(phenotype.metadata.get("domain_width_profile") or {})
        profile = phenotype.realized_precision_profile
        rows, mixed_bits, parameters, activation_bits = self._rows(widths, profile)
        original_widths = {domain.domain_id: domain.original_width for domain in self.domains}
        baseline_rows, baseline_weight_bits, baseline_parameters, baseline_activation_bits = self._rows(original_widths, {})
        fp16_profile = {
            str(row["module_path"]): "FP16"
            for row in baseline_rows
            if row["category"] not in {"qk"}
        }
        fp16_rows, _, _, _ = self._rows(original_widths, fp16_profile)
        total = sum(float(row["BOPS"]) for row in rows)
        fp32_base = sum(float(row["BOPS"]) for row in baseline_rows)
        fp16_base = sum(float(row["BOPS"]) for row in fp16_rows)
        categories = {
            name: sum(float(row["BOPS"]) for row in rows if row["category"] == name)
            for name in ("attention_projection", "qk", "qk_relation", "activation_op", "av", "av_relation", "output_projection", "ffn")
        }
        weighted_macs = sum(float(row.get("MACs", 0.0)) for row in rows if row.get("weight_bits") is not None)
        int8_macs = sum(
            float(row.get("MACs", 0.0))
            for row in rows
            if row.get("weight_bits") == 8
        )
        return {
            "bops_formula_version": "transformer-bops-v2-hgt-relation-closure",
            "cnn_bops": 0.0,
            "attention_projection_bops": categories["attention_projection"],
            "qk_bops": categories["qk"],
            "qk_relation_bops": categories["qk_relation"],
            "softmax_activation_op_cost": categories["activation_op"],
            "av_bops": categories["av"],
            "av_relation_bops": categories["av_relation"],
            "output_projection_bops": categories["output_projection"],
            "ffn_bops": categories["ffn"],
            "transformer_bops_total": total,
            "bops_total": total,
            "bops_fp32_baseline": fp32_base,
            "bops_fp16_baseline": fp16_base,
            "R_bops_vs_fp32": total / max(fp32_base, 1.0),
            "R_bops_vs_fp16_deploy": total / max(fp16_base, 1.0),
            "R_bops": total / max(fp32_base, 1.0),
            "parameter_count": parameters,
            "parameter_count_original": baseline_parameters,
            "parameter_retention": parameters / max(baseline_parameters, 1.0),
            "mixed_weight_size_bytes": mixed_bits / 8.0,
            "original_fp32_weight_size_bytes": baseline_weight_bits / 8.0,
            "activation_memory_estimate_bytes": activation_bits / 8.0,
            "original_fp32_activation_memory_estimate_bytes": baseline_activation_bits / 8.0,
            "int8_macs_ratio": int8_macs / max(weighted_macs, 1.0),
            "breakdown": rows,
            "bias_parameter_cost_included": False,
            "qk_precision_contract": "F32A32O32",
        }

    def evaluate(self, phenotype: CandidatePhenotype) -> float:
        return float(self.evaluate_breakdown(phenotype)["R_bops_vs_fp32"])


class UnifiedBOPSProxy:
    """Combine an existing CNN proxy with the Transformer component proxy."""

    def __init__(self, transformer: TransformerBOPSProxy, cnn: Any | None = None) -> None:
        self.transformer = transformer
        self.cnn = cnn

    def evaluate_breakdown(self, phenotype: CandidatePhenotype) -> dict[str, Any]:
        transformer = self.transformer.evaluate_breakdown(phenotype)
        cnn = self.cnn.evaluate_breakdown(phenotype) if self.cnn is not None else {
            "bops_total": 0.0,
            "bops_fp32_baseline": 0.0,
            "bops_fp16_baseline": 0.0,
            "int8_macs_ratio": 0.0,
            "breakdown": [],
        }
        total = float(cnn["bops_total"]) + float(transformer["bops_total"])
        fp32 = float(cnn["bops_fp32_baseline"]) + float(transformer["bops_fp32_baseline"])
        fp16 = float(cnn["bops_fp16_baseline"]) + float(transformer["bops_fp16_baseline"])
        return {
            **transformer,
            "bops_formula_version": "unified-bops-v2-hgt-relation-closure",
            "cnn_bops": float(cnn["bops_total"]),
            "bops_total": total,
            "total_bops": total,
            "bops_fp32_baseline": fp32,
            "bops_fp16_baseline": fp16,
            "R_bops_vs_fp32": total / max(fp32, 1.0),
            "R_bops_vs_fp16_deploy": total / max(fp16, 1.0),
            "R_bops": total / max(fp32, 1.0),
            "breakdown": [*cnn.get("breakdown", []), *transformer["breakdown"]],
        }

    def evaluate(self, phenotype: CandidatePhenotype) -> float:
        return float(self.evaluate_breakdown(phenotype)["R_bops_vs_fp32"])


__all__ = [
    "AttentionWorkload",
    "FFNWorkload",
    "ProjectionFreeAttentionWorkload",
    "TransformerBOPSProxy",
    "UnifiedBOPSProxy",
    "profile_transformer_workloads",
    "profile_projection_free_attention_workloads",
]
