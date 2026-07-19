"""Canonical synthetic Attention graphs for TensorRT capability evidence."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional

from .head_dim_capability import HeadDimCandidate


_ROLE_PRECISIONS: dict[str, dict[str, str]] = {
    "P0_strict_fp32": {
        "q_projection": "FP32",
        "k_projection": "FP32",
        "v_projection": "FP32",
        "qk_scale": "FP32",
        "qk_matmul": "FP32",
        "softmax": "FP32",
        "av_matmul": "FP32",
        "out_projection": "FP32",
        "qk_input_cast": "NONE",
    },
    "P1_strict_fp16_native": {
        "q_projection": "FP16",
        "k_projection": "FP16",
        "v_projection": "FP16",
        "qk_scale": "FP16",
        "qk_matmul": "FP16",
        "softmax": "FP16",
        "av_matmul": "FP16",
        "out_projection": "FP16",
        "qk_input_cast": "NONE",
    },
    "P2_f3_mixed": {
        "q_projection": "FP16",
        "k_projection": "FP16",
        "v_projection": "FP16",
        "qk_scale": "FP32",
        "qk_matmul": "FP32",
        "softmax": "FP16",
        "av_matmul": "FP16",
        "out_projection": "FP16",
        "qk_input_cast": "FP16_TO_FP32",
    },
    "P3_int8_projections_qk_fp32": {
        "q_projection": "INT8",
        "k_projection": "INT8",
        "v_projection": "INT8",
        "qk_scale": "FP32",
        "qk_matmul": "FP32",
        "softmax": "FP32",
        "av_matmul": "FP32",
        "out_projection": "INT8",
        "qk_input_cast": "DQ_TO_FP32",
    },
    "P4_int8_native_attention": {
        "q_projection": "INT8",
        "k_projection": "INT8",
        "v_projection": "INT8",
        "qk_scale": "INT8",
        "qk_matmul": "INT8",
        "softmax": "FP32",
        "av_matmul": "INT8",
        "out_projection": "INT8",
        "qk_input_cast": "QDQ_INT8",
    },
}


def requested_precision_manifest(profile: str) -> dict[str, str]:
    try:
        return dict(_ROLE_PRECISIONS[str(profile)])
    except KeyError as exc:
        raise ValueError(f"unsupported_synthetic_precision_profile:{profile}") from exc


def _safe_scale(value: torch.Tensor) -> torch.Tensor:
    flat = value.detach().float().abs()
    return torch.clamp(flat.amax() / 127.0, min=torch.finfo(torch.float32).eps)


def _weight_scales(weight: torch.Tensor) -> torch.Tensor:
    flat = weight.detach().float().abs().reshape(weight.shape[0], -1)
    return torch.clamp(
        flat.amax(dim=1) / 127.0, min=torch.finfo(torch.float32).eps
    )


def _qdq_tensor(value: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.fake_quantize_per_tensor_affine(value, scale, 0, -128, 127)


class _CpuHalfLinear(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any, value: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        del ctx
        return functional.linear(value.float(), weight.float()).half()

    @staticmethod
    def symbolic(graph: Any, value: Any, weight: Any) -> Any:
        transposed = graph.op("Transpose", weight, perm_i=[1, 0])
        return graph.op("MatMul", value, transposed)


class _CpuHalfMatMul(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any, left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        del ctx
        return torch.matmul(left.float(), right.float()).half()

    @staticmethod
    def symbolic(graph: Any, left: Any, right: Any) -> Any:
        return graph.op("MatMul", left, right)


class _CpuHalfSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        del ctx
        return torch.softmax(value.float(), dim=-1).half()

    @staticmethod
    def symbolic(graph: Any, value: Any) -> Any:
        return graph.op("Softmax", value, axis_i=-1)


def _linear(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if value.device.type == "cpu" and value.dtype == torch.float16:
        return _CpuHalfLinear.apply(value, weight)
    return functional.linear(value, weight)


def _matmul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.device.type == "cpu" and left.dtype == torch.float16:
        return _CpuHalfMatMul.apply(left, right)
    return torch.matmul(left, right)


def _softmax(value: torch.Tensor) -> torch.Tensor:
    if value.device.type == "cpu" and value.dtype == torch.float16:
        return _CpuHalfSoftmax.apply(value)
    return torch.softmax(value, dim=-1)


class CapabilityLinear(nn.Module):
    """Linear with optional deterministic explicit activation/weight Q/DQ."""

    def __init__(self, in_features: int, out_features: int, *, quantized: bool) -> None:
        super().__init__()
        source = nn.Linear(int(in_features), int(out_features), bias=False)
        self.weight = nn.Parameter(source.weight.detach().clone())
        self.quantized = bool(quantized)
        self.register_buffer("input_scale", torch.tensor(0.03125, dtype=torch.float32))
        self.register_buffer("output_scale", torch.tensor(0.03125, dtype=torch.float32))
        self.register_buffer("weight_scale", _weight_scales(self.weight))
        self.register_buffer(
            "weight_zero_point",
            torch.zeros(int(out_features), dtype=torch.int32),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if not self.quantized:
            return _linear(value, self.weight)
        activation = _qdq_tensor(value.float(), self.input_scale)
        weight = torch.fake_quantize_per_channel_affine(
            self.weight.float(),
            self.weight_scale,
            self.weight_zero_point,
            0,
            -128,
            127,
        )
        output = _linear(activation, weight)
        return _qdq_tensor(output, self.output_scale)


class _CoreAttention(nn.Module):
    def __init__(self, candidate: HeadDimCandidate) -> None:
        super().__init__()
        self.profile = candidate.precision_profile
        self.scale = float(candidate.d_qk) ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.register_buffer("q_scale", torch.tensor(0.03125, dtype=torch.float32))
        self.register_buffer("k_scale", torch.tensor(0.03125, dtype=torch.float32))
        self.register_buffer("v_scale", torch.tensor(0.03125, dtype=torch.float32))
        self.register_buffer("score_scale", torch.tensor(0.03125, dtype=torch.float32))
        self.register_buffer("attention_scale", torch.tensor(1.0 / 255.0, dtype=torch.float32))
        self.register_buffer("output_scale", torch.tensor(0.03125, dtype=torch.float32))

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        profile = self.profile
        if profile == "P2_f3_mixed":
            q_for_score = q.float()
            k_for_score = k.float()
        elif profile == "P4_int8_native_attention":
            q_for_score = _qdq_tensor(q.float(), self.q_scale)
            k_for_score = _qdq_tensor(k.float(), self.k_scale)
            v = _qdq_tensor(v.float(), self.v_scale)
        else:
            q_for_score = q
            k_for_score = k
        score = _matmul(q_for_score * self.scale, k_for_score.transpose(-1, -2))
        if profile == "P4_int8_native_attention":
            score = _qdq_tensor(score, self.score_scale)
        if profile == "P2_f3_mixed":
            attention = _softmax(score.half())
        else:
            attention = _softmax(score)
        if profile == "P4_int8_native_attention":
            attention = _qdq_tensor(attention.float(), self.attention_scale)
        output = _matmul(attention, v)
        if profile == "P4_int8_native_attention":
            output = _qdq_tensor(output.float(), self.output_scale)
        return output


class _ProjectionAttention(nn.Module):
    def __init__(self, candidate: HeadDimCandidate) -> None:
        super().__init__()
        self.profile = candidate.precision_profile
        self.num_heads = int(candidate.num_heads)
        self.d_qk = int(candidate.d_qk)
        self.d_v = int(candidate.d_v)
        self.scale = self.d_qk**-0.5
        quantized_projection = self.profile in {
            "P3_int8_projections_qk_fp32",
            "P4_int8_native_attention",
        }
        self.q_proj = CapabilityLinear(
            candidate.embed_dim,
            candidate.q_projection_out,
            quantized=quantized_projection,
        )
        self.k_proj = CapabilityLinear(
            candidate.embed_dim,
            candidate.k_projection_out,
            quantized=quantized_projection,
        )
        self.v_proj = CapabilityLinear(
            candidate.embed_dim,
            candidate.v_projection_out,
            quantized=quantized_projection,
        )
        self.out_proj = CapabilityLinear(
            candidate.out_projection_in,
            candidate.out_projection_out,
            quantized=quantized_projection,
        )
        self.softmax = nn.Softmax(dim=-1)
        self.register_buffer("q_scale", torch.tensor(0.03125, dtype=torch.float32))
        self.register_buffer("k_scale", torch.tensor(0.03125, dtype=torch.float32))
        self.register_buffer("v_scale", torch.tensor(0.03125, dtype=torch.float32))
        self.register_buffer("score_scale", torch.tensor(0.03125, dtype=torch.float32))
        self.register_buffer("attention_scale", torch.tensor(1.0 / 255.0, dtype=torch.float32))
        self.register_buffer("av_scale", torch.tensor(0.03125, dtype=torch.float32))

    def _reshape(self, value: torch.Tensor, width: int) -> torch.Tensor:
        groups, tokens, _ = value.shape
        return value.reshape(groups, tokens, self.num_heads, width).permute(0, 2, 1, 3)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        relative_position_bias: torch.Tensor,
    ) -> torch.Tensor:
        q = self._reshape(self.q_proj(x), self.d_qk)
        k = self._reshape(self.k_proj(x), self.d_qk)
        v = self._reshape(self.v_proj(x), self.d_v)
        if self.profile == "P2_f3_mixed":
            q_for_score = q.float()
            k_for_score = k.float()
        elif self.profile == "P4_int8_native_attention":
            q_for_score = _qdq_tensor(q.float(), self.q_scale)
            k_for_score = _qdq_tensor(k.float(), self.k_scale)
            v = _qdq_tensor(v.float(), self.v_scale)
        else:
            q_for_score = q
            k_for_score = k
        score = _matmul(
            q_for_score * self.scale, k_for_score.transpose(-1, -2)
        )
        score = score + relative_position_bias.to(score.dtype)
        score = score.masked_fill(~attention_mask[:, None, :, :], float("-inf"))
        if self.profile == "P4_int8_native_attention":
            score = _qdq_tensor(score, self.score_scale)
        if self.profile == "P2_f3_mixed":
            attention = _softmax(score.half())
        else:
            attention = _softmax(score)
        if self.profile == "P4_int8_native_attention":
            attention = _qdq_tensor(attention.float(), self.attention_scale)
        output = _matmul(attention, v)
        if self.profile == "P4_int8_native_attention":
            output = _qdq_tensor(output.float(), self.av_scale)
        output = output.permute(0, 2, 1, 3).reshape(
            output.shape[0], output.shape[2], self.num_heads * self.d_v
        )
        return self.out_proj(output)


def _input_dtype(profile: str) -> torch.dtype:
    return (
        torch.float16
        if profile in {"P1_strict_fp16_native", "P2_f3_mixed"}
        else torch.float32
    )


def build_synthetic_graph(
    candidate: HeadDimCandidate,
) -> tuple[nn.Module, dict[str, torch.Tensor], dict[str, str]]:
    if (
        candidate.graph_variant == "core_attention"
        and candidate.precision_profile == "P3_int8_projections_qk_fp32"
    ):
        raise ValueError("projection_int8_profile_requires_projection_graph")
    torch.manual_seed(20260718)
    dtype = _input_dtype(candidate.precision_profile)
    generator = torch.Generator(device="cpu").manual_seed(20260718)
    if candidate.graph_variant == "core_attention":
        graph: nn.Module = _CoreAttention(candidate)
        inputs = {
            "q": torch.randn(
                candidate.window_groups,
                candidate.num_heads,
                candidate.token_length,
                candidate.d_qk,
                generator=generator,
                dtype=dtype,
            ),
            "k": torch.randn(
                candidate.window_groups,
                candidate.num_heads,
                candidate.token_length,
                candidate.d_qk,
                generator=generator,
                dtype=dtype,
            ),
            "v": torch.randn(
                candidate.window_groups,
                candidate.num_heads,
                candidate.token_length,
                candidate.d_v,
                generator=generator,
                dtype=dtype,
            ),
        }
    else:
        graph = _ProjectionAttention(candidate)
        x = torch.randn(
            candidate.window_groups,
            candidate.token_length,
            candidate.embed_dim,
            generator=generator,
            dtype=dtype,
        )
        mask = torch.rand(
            candidate.window_groups,
            candidate.token_length,
            candidate.token_length,
            generator=generator,
        ) > 0.2
        diagonal = torch.arange(candidate.token_length)
        mask[:, diagonal, diagonal] = True
        bias_dtype = (
            torch.float32
            if candidate.precision_profile == "P2_f3_mixed"
            else dtype
        )
        inputs = {
            "x": x,
            "attention_mask": mask,
            "relative_position_bias": torch.randn(
                1,
                candidate.num_heads,
                candidate.token_length,
                candidate.token_length,
                generator=generator,
                dtype=bias_dtype,
            )
            * 0.01,
        }
    graph = graph.eval()
    if dtype == torch.float16:
        graph = graph.half()
    return graph, inputs, requested_precision_manifest(candidate.precision_profile)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_synthetic_onnx(
    candidate: HeadDimCandidate, destination: str | Path
) -> dict[str, Any]:
    import onnx

    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    graph, inputs, manifest = build_synthetic_graph(candidate)
    names = tuple(inputs)
    with torch.no_grad():
        output = graph(**inputs)
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("synthetic_reference_nonfinite")
    torch.onnx.export(
        graph,
        tuple(inputs[name] for name in names),
        path,
        input_names=names,
        output_names=("output",),
        opset_version=17,
        do_constant_folding=False,
    )
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    node_types = [str(node.op_type) for node in model.graph.node]
    return {
        "candidate_hash": candidate.candidate_hash,
        "candidate_id": candidate.candidate_id,
        "input_shapes": {name: list(value.shape) for name, value in inputs.items()},
        "node_type_counts": {
            name: node_types.count(name)
            for name in sorted(set(node_types))
        },
        "onnx_export_success": True,
        "onnx_path": str(path.resolve()),
        "onnx_sha256": _sha256(path),
        "output_dtype": str(output.dtype),
        "output_shape": list(output.shape),
        "requested_precision": manifest,
    }


__all__ = [
    "build_synthetic_graph",
    "export_synthetic_onnx",
    "requested_precision_manifest",
]
