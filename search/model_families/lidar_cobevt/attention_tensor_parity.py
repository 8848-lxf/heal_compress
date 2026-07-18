"""Numerical parity metrics for CoBEVT Attention precision diagnostics."""

from __future__ import annotations

import math
import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch


_DIAGNOSTIC_OUTPUT_ROLES = (
    "layernorm",
    "q_projection",
    "k_projection",
    "v_projection",
    "qk_matmul",
    "softmax",
    "av_matmul",
    "output_projection",
    "residual_add",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_attention_diagnostic_outputs(
    source_onnx: str | Path,
    output_onnx: str | Path,
    outputs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Copy an ONNX graph and expose selected tensors for parity diagnostics."""

    import onnx

    source = Path(source_onnx).expanduser().resolve()
    destination = Path(output_onnx).expanduser().resolve()
    if source == destination:
        raise ValueError("attention_diagnostic_must_not_overwrite_source")
    model = onnx.load(str(source))
    values = {
        str(value.name): value
        for value in (
            list(model.graph.input)
            + list(model.graph.output)
            + list(model.graph.value_info)
        )
    }
    available = set(values)
    for node in model.graph.node:
        available.update(str(name) for name in node.output if str(name))
    existing_outputs = {str(value.name) for value in model.graph.output}
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in outputs:
        tensor_name = str(raw.get("tensor_name", ""))
        if not tensor_name or tensor_name not in available:
            raise ValueError(f"attention_diagnostic_tensor_missing:{tensor_name}")
        if tensor_name in seen:
            raise ValueError(f"attention_diagnostic_tensor_duplicate:{tensor_name}")
        seen.add(tensor_name)
        value = values.get(tensor_name)
        if tensor_name not in existing_outputs:
            if value is None or not value.type.HasField("tensor_type"):
                raise ValueError(f"attention_diagnostic_tensor_type_missing:{tensor_name}")
            model.graph.output.append(value)
            existing_outputs.add(tensor_name)
        selected.append(
            {
                "block_id": str(raw.get("block_id", "")),
                "diagnostic_latency_invalid": True,
                "role": str(raw.get("role", "")),
                "tensor_name": tensor_name,
            }
        )
    known_schema_bypass = sorted(
        {
            str(node.op_type)
            for node in model.graph.node
            if str(node.domain) == "" and str(node.op_type) == "PointPillarScatterTRT"
        }
    )
    checker_model = onnx.load_from_string(model.SerializeToString())
    if known_schema_bypass:
        for node in checker_model.graph.node:
            if str(node.op_type) in known_schema_bypass and str(node.domain) == "":
                node.domain = "trt"
        if not any(str(item.domain) == "trt" for item in checker_model.opset_import):
            checker_model.opset_import.append(onnx.helper.make_operatorsetid("trt", 1))
    onnx.checker.check_model(checker_model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(destination))
    return {
        "diagnostic_latency_invalid": True,
        "known_custom_op_schema_bypass": known_schema_bypass,
        "output_onnx": str(destination),
        "output_onnx_sha256": _sha256(destination),
        "outputs": selected,
        "source_onnx": str(source),
        "source_onnx_sha256": _sha256(source),
    }


def attention_diagnostic_output_specs(
    boundary_report: Mapping[str, Any],
    *,
    fused_bev_tensor: str = "/layers.2/Reshape_3_output_0",
    head_input_tensor: str = "/mlp_head/mlp_head.4/Transpose_output_0",
) -> list[dict[str, str]]:
    """Resolve public Attention stage tensors from a boundary rewrite report."""

    grouped: dict[str, dict[str, Mapping[str, Any]]] = {}
    for raw in boundary_report.get("node_records", []):
        block_id = str(raw.get("block_id", ""))
        role = str(raw.get("role", ""))
        if role not in _DIAGNOSTIC_OUTPUT_ROLES:
            continue
        if not block_id or role in grouped.setdefault(block_id, {}):
            raise ValueError(f"attention_diagnostic_role_duplicate:{block_id}:{role}")
        grouped[block_id][role] = raw
    if not grouped:
        raise ValueError("attention_diagnostic_blocks_missing")
    specs: list[dict[str, str]] = []
    seen: set[str] = set()

    def add(block_id: str, role: str, tensor_name: str) -> None:
        name = str(tensor_name)
        if not name:
            raise ValueError(f"attention_diagnostic_role_incomplete:{block_id}:{role}")
        if name in seen:
            return
        seen.add(name)
        specs.append({"block_id": block_id, "role": role, "tensor_name": name})

    for block_id in sorted(grouped):
        roles = grouped[block_id]
        missing = sorted(set(_DIAGNOSTIC_OUTPUT_ROLES) - set(roles))
        if missing:
            raise ValueError(
                f"attention_diagnostic_role_incomplete:{block_id}:{missing}"
            )
        for role in _DIAGNOSTIC_OUTPUT_ROLES:
            record = roles[role]
            inputs = list(record.get("input_tensors_before", []))
            outputs = list(record.get("output_tensors_before", []))
            if not outputs:
                raise ValueError(
                    f"attention_diagnostic_role_incomplete:{block_id}:{role}:output"
                )
            add(block_id, role, str(outputs[0]))
            if role == "softmax":
                if not inputs:
                    raise ValueError(
                        f"attention_diagnostic_role_incomplete:{block_id}:softmax:input"
                    )
                add(block_id, "scaled_qk_logits", str(inputs[0]))
            elif role == "residual_add":
                if len(inputs) != 2:
                    raise ValueError(
                        f"attention_diagnostic_role_incomplete:{block_id}:residual_add:inputs"
                    )
                add(block_id, "residual_attention_update", str(inputs[0]))
                add(block_id, "residual_input", str(inputs[1]))
    add("global", "fused_bev", fused_bev_tensor)
    add("global", "head_input", head_input_tensor)
    return specs


def attention_diagnostic_output_shards(
    outputs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    role_shards = {
        "pre": {"layernorm", "q_projection", "k_projection", "v_projection"},
        "qk": {"qk_matmul", "scaled_qk_logits", "softmax"},
        "post": {
            "av_matmul",
            "output_projection",
            "residual_add",
            "fused_bev",
            "head_input",
        },
    }
    reference_only_roles = {"residual_attention_update", "residual_input"}
    known_roles = set().union(*role_shards.values(), reference_only_roles)
    unknown = sorted(
        {str(row.get("role", "")) for row in outputs} - known_roles
    )
    if unknown:
        raise ValueError(f"attention_diagnostic_shard_role_unknown:{unknown}")
    tensor_names = [str(row.get("tensor_name", "")) for row in outputs]
    if any(not name for name in tensor_names) or len(tensor_names) != len(
        set(tensor_names)
    ):
        raise ValueError("attention_diagnostic_shard_tensor_invalid")
    shards = []
    for shard_id in ("pre", "qk", "post"):
        output_specs = [
            dict(row)
            for row in outputs
            if str(row.get("role", "")) in role_shards[shard_id]
        ]
        reference_only_specs = (
            [
                dict(row)
                for row in outputs
                if str(row.get("role", "")) in reference_only_roles
            ]
            if shard_id == "post"
            else []
        )
        if not output_specs:
            raise ValueError(f"attention_diagnostic_shard_empty:{shard_id}")
        shards.append(
            {
                "output_specs": output_specs,
                "reference_only_specs": reference_only_specs,
                "shard_id": shard_id,
            }
        )
    return shards


def metric_accumulation_spec(device_type: str) -> dict[str, Any]:
    return {
        "device_policy": "preserve",
        "dtype": torch.float32 if str(device_type) == "cuda" else torch.float64,
    }


def _metric_tensor(value: torch.Tensor) -> torch.Tensor:
    spec = metric_accumulation_spec(value.device.type)
    return value.detach().to(device=value.device, dtype=spec["dtype"])


def _finite_stats(value: torch.Tensor) -> dict[str, float]:
    finite = value[torch.isfinite(value)]
    if not finite.numel():
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan"), "std": float("nan"), "max_abs": float("nan")}
    return {
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "std": float(finite.std(unbiased=False)),
        "max_abs": float(finite.abs().max()),
    }


def tensor_error_metrics(
    reference: torch.Tensor, candidate: torch.Tensor
) -> dict[str, Any]:
    if tuple(reference.shape) != tuple(candidate.shape):
        raise ValueError(
            f"attention_parity_shape_mismatch:{tuple(reference.shape)}:{tuple(candidate.shape)}"
        )
    ref = _metric_tensor(reference)
    cand = _metric_tensor(candidate)
    ref_finite = torch.isfinite(ref)
    cand_finite = torch.isfinite(cand)
    common = ref_finite & cand_finite
    finite = bool(ref_finite.all() and cand_finite.all())
    if common.any():
        ref_values = ref[common]
        cand_values = cand[common]
        difference = cand_values - ref_values
        error_norm = float(torch.linalg.vector_norm(difference))
        reference_norm = float(torch.linalg.vector_norm(ref_values))
        if error_norm == 0.0:
            cosine = 1.0
        else:
            denominator = float(
                torch.linalg.vector_norm(ref_values)
                * torch.linalg.vector_norm(cand_values)
            )
            cosine = (
                float(torch.dot(ref_values.flatten(), cand_values.flatten()))
                / denominator
                if denominator
                else 0.0
            )
        max_error = float(difference.abs().max())
        mean_error = float(difference.abs().mean())
        relative_l2 = error_norm / max(reference_norm, 1e-30)
    else:
        cosine = float("nan")
        max_error = float("nan")
        mean_error = float("nan")
        relative_l2 = float("nan")
    ref_stats = _finite_stats(ref)
    cand_stats = _finite_stats(cand)
    return {
        "candidate_dtype": str(candidate.dtype),
        "candidate_inf_count": int(torch.isinf(cand).sum()),
        "candidate_max": cand_stats["max"],
        "candidate_max_abs": cand_stats["max_abs"],
        "candidate_mean": cand_stats["mean"],
        "candidate_min": cand_stats["min"],
        "candidate_nan_count": int(torch.isnan(cand).sum()),
        "candidate_std": cand_stats["std"],
        "cosine_similarity": cosine,
        "finite": finite,
        "maximum_absolute_error": max_error,
        "mean_absolute_error": mean_error,
        "reference_dtype": str(reference.dtype),
        "reference_inf_count": int(torch.isinf(ref).sum()),
        "reference_max": ref_stats["max"],
        "reference_max_abs": ref_stats["max_abs"],
        "reference_mean": ref_stats["mean"],
        "reference_min": ref_stats["min"],
        "reference_nan_count": int(torch.isnan(ref).sum()),
        "reference_std": ref_stats["std"],
        "relative_l2_error": relative_l2,
        "shape": list(reference.shape),
    }


def _topk_overlap(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    topk_values: Iterable[int],
) -> dict[str, float]:
    width = int(reference.shape[-1])
    rows = reference.reshape(-1, width)
    candidates = candidate.reshape(-1, width)
    result = {}
    for requested in topk_values:
        k = min(max(int(requested), 1), width)
        ref_indices = torch.topk(rows, k=k, dim=-1).indices
        cand_indices = torch.topk(candidates, k=k, dim=-1).indices
        overlap = (
            ref_indices.unsqueeze(-1) == cand_indices.unsqueeze(-2)
        ).any(dim=-1).sum(dim=-1)
        result[f"topk_overlap_{requested}"] = float(
            (overlap.to(dtype=rows.dtype) / float(k)).mean()
        )
    return result


def qk_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    topk_values: Sequence[int] = (1, 4, 8),
) -> dict[str, Any]:
    ref = _metric_tensor(reference)
    cand = _metric_tensor(candidate)
    if ref.shape != cand.shape or ref.ndim < 1:
        raise ValueError("qk_parity_shape_mismatch")
    width = int(ref.shape[-1])
    ref_rows = ref.reshape(-1, width)
    cand_rows = cand.reshape(-1, width)
    ref_ranks = torch.argsort(torch.argsort(ref_rows, dim=-1), dim=-1).to(
        dtype=ref.dtype
    )
    cand_ranks = torch.argsort(torch.argsort(cand_rows, dim=-1), dim=-1).to(
        dtype=ref.dtype
    )
    ref_centered = ref_ranks - ref_ranks.mean(dim=-1, keepdim=True)
    cand_centered = cand_ranks - cand_ranks.mean(dim=-1, keepdim=True)
    numerator = (ref_centered * cand_centered).sum(dim=-1)
    denominator = torch.sqrt(
        ref_centered.square().sum(dim=-1) * cand_centered.square().sum(dim=-1)
    ).clamp_min(1e-30)
    result = {
        "candidate_logits_max": float(cand.max()),
        "candidate_logits_min": float(cand.min()),
        "reference_logits_max": float(ref.max()),
        "reference_logits_min": float(ref.min()),
        "row_max_absolute_error": float(
            (ref_rows.max(dim=-1).values - cand_rows.max(dim=-1).values)
            .abs()
            .mean()
        ),
        "row_rank_correlation": float((numerator / denominator).mean()),
        "sign_flip_ratio": float(
            ((ref < 0) != (cand < 0)).to(dtype=ref.dtype).mean()
        ),
        "top1_index_agreement": float(
            (ref_rows.argmax(dim=-1) == cand_rows.argmax(dim=-1))
            .to(dtype=ref.dtype)
            .mean()
        ),
    }
    result.update(_topk_overlap(ref, cand, topk_values))
    return result


def softmax_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    topk_values: Sequence[int] = (1, 4, 8),
    epsilon: float = 1e-12,
    tiny_threshold: float = 1e-7,
) -> dict[str, Any]:
    ref = _metric_tensor(reference)
    cand = _metric_tensor(candidate)
    if ref.shape != cand.shape or ref.ndim < 1:
        raise ValueError("softmax_parity_shape_mismatch")
    ref_safe = ref.clamp_min(float(epsilon))
    cand_safe = cand.clamp_min(float(epsilon))
    midpoint = ((ref_safe + cand_safe) * 0.5).clamp_min(float(epsilon))
    ref_entropy = -(ref_safe * ref_safe.log()).sum(dim=-1)
    cand_entropy = -(cand_safe * cand_safe.log()).sum(dim=-1)
    indices = torch.arange(ref.shape[-1], dtype=ref.dtype, device=ref.device)
    ref_centroid = (ref * indices).sum(dim=-1) / ref.sum(dim=-1).clamp_min(epsilon)
    cand_centroid = (cand * indices).sum(dim=-1) / cand.sum(dim=-1).clamp_min(epsilon)
    result = {
        "attention_argmax_agreement": float(
            (ref.argmax(dim=-1) == cand.argmax(dim=-1)).to(dtype=ref.dtype).mean()
        ),
        "candidate_entropy": float(cand_entropy.mean()),
        "candidate_exact_zero_ratio": float((cand == 0).to(dtype=ref.dtype).mean()),
        "candidate_row_sum_max_deviation": float(
            (cand.sum(dim=-1) - 1.0).abs().max()
        ),
        "candidate_tiny_value_ratio": float(
            (cand < float(tiny_threshold)).to(dtype=ref.dtype).mean()
        ),
        "entropy_delta": float((cand_entropy - ref_entropy).mean()),
        "js_divergence": float(
            0.5
            * (
                (ref_safe * (ref_safe / midpoint).log()).sum(dim=-1)
                + (cand_safe * (cand_safe / midpoint).log()).sum(dim=-1)
            ).mean()
        ),
        "kl_fp32_to_candidate": float(
            (ref_safe * (ref_safe / cand_safe).log()).sum(dim=-1).mean()
        ),
        "reference_entropy": float(ref_entropy.mean()),
        "reference_exact_zero_ratio": float((ref == 0).to(dtype=ref.dtype).mean()),
        "reference_row_sum_max_deviation": float(
            (ref.sum(dim=-1) - 1.0).abs().max()
        ),
        "reference_tiny_value_ratio": float(
            (ref < float(tiny_threshold)).to(dtype=ref.dtype).mean()
        ),
        "spatial_attention_centroid_shift": float(
            (cand_centroid - ref_centroid).abs().mean()
        ),
    }
    result.update(_topk_overlap(ref, cand, topk_values))
    return result


def residual_metrics(
    residual_input: torch.Tensor,
    fp32_attention_update: torch.Tensor,
    fp32_output: torch.Tensor,
    candidate_output: torch.Tensor,
) -> dict[str, float]:
    residual = _metric_tensor(residual_input)
    update = _metric_tensor(fp32_attention_update)
    reference = _metric_tensor(fp32_output)
    candidate = _metric_tensor(candidate_output)
    if not (residual.shape == update.shape == reference.shape == candidate.shape):
        raise ValueError("residual_parity_shape_mismatch")
    candidate_update = candidate - residual
    update_norm = float(torch.linalg.vector_norm(update))
    residual_norm = float(torch.linalg.vector_norm(residual))
    return {
        "attention_to_residual_l2_ratio": update_norm / max(residual_norm, 1e-30),
        "candidate_output_equals_residual_ratio": float(
            (candidate == residual).to(dtype=residual.dtype).mean()
        ),
        "candidate_update_retention_ratio": float(
            torch.linalg.vector_norm(candidate_update)
        )
        / max(update_norm, 1e-30),
        "residual_output_relative_l2_error": float(
            torch.linalg.vector_norm(candidate - reference)
        )
        / max(float(torch.linalg.vector_norm(reference)), 1e-30),
    }


def select_failure_frames(
    rows: Sequence[Mapping[str, Any]], *, count: int = 3
) -> list[dict[str, Any]]:
    limit = max(int(count), 0)
    return sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            -float(row.get("maximum_absolute_error", -math.inf)),
            str(row.get("frame_id", "")),
        ),
    )[:limit]


__all__ = [
    "append_attention_diagnostic_outputs",
    "attention_diagnostic_output_specs",
    "qk_metrics",
    "residual_metrics",
    "select_failure_frames",
    "softmax_metrics",
    "tensor_error_metrics",
]
