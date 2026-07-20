"""Real-weight CoBEVT Attention deployment unit with the accepted F3 contract."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional


def _tensor_precision(tensor: dict[str, object]) -> str:
    value = str(tensor.get("Format/Datatype", "")).lower()
    if "half" in value or "fp16" in value:
        return "FP16"
    if "float" in value or "fp32" in value:
        return "FP32"
    if "int8" in value:
        return "INT8"
    return "unknown"


def _tensor_shape(tensor: dict[str, object]) -> tuple[int, ...]:
    return tuple(int(value) for value in tensor.get("Dimensions", ()))


def inspect_f3_deployment_unit_layers(
    layers: list[dict[str, object]],
    *,
    heads: int,
    tokens: int,
    d_qk: int,
    d_v: int,
    embed_dim: int,
) -> dict[str, object]:
    """Classify the realized F3 unit from tensor shapes and TRT evidence.

    TensorRT commonly combines the three projection GEMMs into one execution
    layer and removes their ONNX role names.  The generic synthetic-graph
    parser therefore cannot classify a real deployment unit.  This parser is
    deliberately strict about the deployment-unit shape sequence and fails
    closed when any role is missing or ambiguous.
    """

    gemms = [row for row in layers if str(row.get("LayerType", "")).lower() == "gemm"]
    projected_width = int(heads) * (2 * int(d_qk) + int(d_v))
    qk_batch = None
    projection = []
    qk = []
    av = []
    output = []
    for row in gemms:
        inputs = list(row.get("Inputs", ()))
        outputs = list(row.get("Outputs", ()))
        input_shapes = [_tensor_shape(value) for value in inputs]
        output_shapes = [_tensor_shape(value) for value in outputs]
        input_precisions = {_tensor_precision(value) for value in inputs}
        if len(inputs) == 1 and any(
            shape
            and (
                shape[-1] == projected_width
                or (
                    d_qk == d_v
                    and len(shape) == 3
                    and shape[0] == 3
                    and shape[-1] == heads * d_qk
                )
            )
            for shape in output_shapes
        ):
            projection.append(row)
        if (
            len(inputs) == 2
            and len(input_shapes[0]) >= 3
            and len(input_shapes[1]) >= 3
            and input_shapes[0][-2:] == (tokens, d_qk)
            and input_shapes[1][-2:] == (d_qk, tokens)
            and any(shape[-2:] == (tokens, tokens) for shape in output_shapes)
            and input_precisions == {"FP32"}
        ):
            qk.append(row)
            qk_batch = input_shapes[0][0]
        if (
            len(inputs) == 2
            and len(input_shapes[0]) >= 3
            and len(input_shapes[1]) >= 3
            and input_shapes[0][-2:] == (tokens, tokens)
            and input_shapes[1][-2:] == (tokens, d_v)
            and any(shape[-2:] == (tokens, d_v) for shape in output_shapes)
            and input_precisions == {"FP16"}
        ):
            av.append(row)
        if (
            len(inputs) == 1
            and any(shape and shape[-1] == heads * d_v for shape in input_shapes)
            and any(len(shape) == 2 and shape[-1] == embed_dim for shape in output_shapes)
            and input_precisions == {"FP16"}
        ):
            output.append(row)
    roles = {
        "projection": projection,
        "qk": qk,
        "av": av,
        "output": output,
    }
    ambiguous = {name: len(rows) for name, rows in roles.items() if len(rows) != 1}
    if ambiguous:
        raise ValueError(f"f3_deployment_unit_role_ambiguity:{ambiguous}")

    def role_precision(row: dict[str, object]) -> str:
        values = {
            _tensor_precision(tensor)
            for tensor in [*row.get("Inputs", ()), *row.get("Outputs", ())]
        }
        values.discard("unknown")
        if len(values) == 1:
            return next(iter(values))
        return "MIXED[" + ",".join(sorted(values)) + "]" if values else "unknown"

    softmax = []
    for row in layers:
        inputs = list(row.get("Inputs", ()))
        outputs = list(row.get("Outputs", ()))
        searchable = " ".join(
            str(row.get(key, "")) for key in ("Name", "LayerType", "TacticName")
        ).lower()
        if (
            "softmax" in searchable or all(token in searchable for token in ("exp", "sum", "div"))
        ) and any(_tensor_shape(tensor)[-2:] == (tokens, tokens) for tensor in outputs):
            softmax.append(row)
    if len(softmax) != 1:
        raise ValueError(f"f3_deployment_unit_softmax_ambiguity:{len(softmax)}")
    softmax_output_values = {
        _tensor_precision(tensor) for tensor in softmax[0].get("Outputs", ())
    }
    softmax_output_values.discard("unknown")
    softmax_precision = (
        next(iter(softmax_output_values))
        if len(softmax_output_values) == 1
        else "unknown"
    )
    qk_tactic = str(qk[0].get("TacticName", "")).lower()
    qk_accumulator = "FP32" if "f32f32" in qk_tactic and "_f32" in qk_tactic else "unknown"
    realized = {
        "q_projection": role_precision(projection[0]),
        "k_projection": role_precision(projection[0]),
        "v_projection": role_precision(projection[0]),
        "qk": role_precision(qk[0]),
        "softmax": softmax_precision,
        "av": role_precision(av[0]),
        "out_projection": role_precision(output[0]),
    }
    match = realized == {
        "q_projection": "FP16",
        "k_projection": "FP16",
        "v_projection": "FP16",
        "qk": "FP32",
        "softmax": "FP16",
        "av": "FP16",
        "out_projection": "FP16",
    }
    return {
        "attention_execution_layer_count": 4,
        "av_accumulator_precision": "unknown",
        "cast_count": sum("cast" in str(row.get("Name", "")).lower() for row in layers),
        "fused_mha_detected": False,
        "fusion_kind": "projection_fusion_with_primitives",
        "layer_count": len(layers),
        "plugin_used": any("plugin" in str(row.get("LayerType", "")).lower() for row in layers),
        "qk_accumulator_precision": qk_accumulator,
        "qk_execution_batch": qk_batch,
        "realized_precision": realized,
        "realized_q_projection_precision": realized["q_projection"],
        "realized_k_projection_precision": realized["k_projection"],
        "realized_v_projection_precision": realized["v_projection"],
        "realized_qk_precision": realized["qk"],
        "realized_softmax_precision": realized["softmax"],
        "realized_av_precision": realized["av"],
        "realized_out_projection_precision": realized["out_projection"],
        "reformat_count": sum("reformat" in str(row.get("Name", "")).lower() for row in layers),
        "requested_realized_match": match,
    }


class F3PrimitiveAttentionUnit(nn.Module):
    """Flattened CoBEVT Attention including LayerNorm and residual Add.

    Inputs retain the real CoBEVT token semantics: ``x`` is ``[groups, 32,
    embed]``, ``mask`` is ``[groups, 1, 1, 32]``, and ``relative_bias`` is
    ``[1, heads, 32, 32]``. Explicit casts encode the accepted F3 phenotype.
    """

    def __init__(self, attention: nn.Module, layernorm: nn.LayerNorm) -> None:
        super().__init__()
        self.heads = int(attention.heads)
        self.d_qk = int(attention.d_qk)
        self.d_v = int(attention.d_v)
        self.embed_dim = int(attention.embed_dim)
        self.scale = float(self.d_qk) ** -0.5
        self.layernorm = nn.LayerNorm(
            self.embed_dim, eps=float(layernorm.eps), elementwise_affine=True
        )
        self.q_proj = nn.Linear(self.embed_dim, self.heads * self.d_qk, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.heads * self.d_qk, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.heads * self.d_v, bias=False)
        self.out_proj = nn.Linear(self.heads * self.d_v, self.embed_dim, bias=False)
        with torch.no_grad():
            self.layernorm.weight.copy_(layernorm.weight)
            self.layernorm.bias.copy_(layernorm.bias)
            self.q_proj.weight.copy_(attention.q_proj.weight)
            self.k_proj.weight.copy_(attention.k_proj.weight)
            self.v_proj.weight.copy_(attention.v_proj.weight)
            self.out_proj.weight.copy_(attention.out_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        relative_bias: torch.Tensor,
    ) -> torch.Tensor:
        residual = x.to(dtype=torch.float16)
        normalized = functional.layer_norm(
            x.float(),
            (self.embed_dim,),
            self.layernorm.weight.float(),
            self.layernorm.bias.float(),
            self.layernorm.eps,
        )
        projection_input = normalized.to(dtype=torch.float16)
        q = functional.linear(projection_input, self.q_proj.weight.half())
        k = functional.linear(projection_input, self.k_proj.weight.half())
        v = functional.linear(projection_input, self.v_proj.weight.half())
        groups, tokens, _ = q.shape
        q = q.reshape(groups, tokens, self.heads, self.d_qk).permute(0, 2, 1, 3)
        k = k.reshape(groups, tokens, self.heads, self.d_qk).permute(0, 2, 1, 3)
        v = v.reshape(groups, tokens, self.heads, self.d_v).permute(0, 2, 1, 3)
        score = torch.matmul(q.float() * self.scale, k.float().transpose(-1, -2))
        score = score + relative_bias.float()
        score = torch.where(mask, score, torch.full_like(score, -1.0e4))
        attention = torch.softmax(score.half(), dim=-1)
        av = torch.matmul(attention, v)
        av = av.permute(0, 2, 1, 3).reshape(
            groups, tokens, self.heads * self.d_v
        )
        projected = functional.linear(av, self.out_proj.weight.half())
        return projected + residual


def f3_unit_requested_contract() -> dict[str, str]:
    return {
        "layernorm": "FP32",
        "q_projection": "FP16",
        "k_projection": "FP16",
        "v_projection": "FP16",
        "qk_recovery_cast": "FP32",
        "qk_scale": "FP32",
        "qk_matmul": "FP32",
        "softmax": "FP16",
        "av_matmul": "FP16",
        "output_projection": "FP16",
        "residual_add": "FP16",
        "complete_fused_mha": "false",
    }


__all__ = [
    "F3PrimitiveAttentionUnit",
    "f3_unit_requested_contract",
    "inspect_f3_deployment_unit_layers",
]
