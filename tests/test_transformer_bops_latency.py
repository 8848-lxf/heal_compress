"""Transformer BOPS formulas and fail-closed latency mappings."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from heal_compress.search.candidate import CandidatePhenotype, PrecisionDecision
from heal_compress.search.pruning_space.transformer_domains import build_transformer_pruning_domains
from heal_compress.search.proxy.transformer_bops import AttentionWorkload, FFNWorkload, ProjectionFreeAttentionWorkload, TransformerBOPSProxy, profile_transformer_workloads
from heal_compress.search.proxy.transformer_latency import TransformerLatencyLUT, TransformerLatencyProxy


class Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.heads = 4
        self.to_qkv = nn.Linear(32, 3 * 4 * 8, bias=False)
        self.to_out = nn.Sequential(nn.Linear(4 * 8, 32, bias=False), nn.Dropout(0.0))
        self.attend = nn.Softmax(dim=-1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, tokens, _channels = value.shape
        qkv = self.to_qkv(value).reshape(batch, tokens, 3, self.heads, 8)
        q, k, v = qkv.unbind(2)
        scores = torch.einsum("bthd,bshd->bhts", q, k)
        output = torch.einsum("bhts,bshd->bthd", self.attend(scores), v)
        return self.to_out(output.reshape(batch, tokens, -1))


class FFN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, 32)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(value)))


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = Attention()
        self.ffn = FFN()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.ffn(self.attn(value))


def _proxy():
    domains, _, _ = build_transformer_pruning_domains(
        Model(), model_name="toy", allow_identity_ranking=True
    )
    return TransformerBOPSProxy(
        domains,
        attention_workloads=(AttentionWorkload("attn", projection_tokens=16, query_tokens=8, key_tokens=8, attention_groups=2),),
        ffn_workloads=(FFNWorkload("ffn", tokens=16),),
    ), domains


def _phenotype(domains, *, compressed: bool) -> CandidatePhenotype:
    widths = {domain.domain_id: domain.original_width for domain in domains}
    profile = {}
    if compressed:
        widths["attention_dh::attn"] = 4
        widths["ffn_hidden::ffn"] = 64
        for path in ("attn.to_qkv", "attn.to_out.0", "ffn.fc1", "ffn.fc2"):
            profile[path] = PrecisionDecision("INT8", "INT8")
        profile["attn.attend"] = PrecisionDecision("INT8", "INT8")
        profile["attn::__av_matmul__"] = PrecisionDecision("INT8", "INT8")
    return CandidatePhenotype(
        precision_profile=profile,
        metadata={"domain_width_profile": widths},
    )


def test_transformer_bops_uses_attention_and_ffn_formulas_with_qk_fp32() -> None:
    proxy, domains = _proxy()
    baseline = proxy.evaluate_breakdown(_phenotype(domains, compressed=False))
    compressed = proxy.evaluate_breakdown(_phenotype(domains, compressed=True))
    assert compressed["R_bops_vs_fp32"] < baseline["R_bops_vs_fp32"]
    qk = next(row for row in compressed["breakdown"] if row["component"] == "qk_matmul")
    assert qk["MACs"] == 2 * 4 * 8 * 8 * 4
    assert qk["operand_a_bits"] == qk["operand_b_bits"] == 32
    assert qk["compute_precision"] == qk["accumulator_precision"] == "FP32"
    ffn = [row for row in compressed["breakdown"] if row["category"] == "ffn"]
    assert sum(row["MACs"] for row in ffn) == 2 * 16 * 32 * 64
    assert compressed["parameter_count"] < compressed["parameter_count_original"]
    assert compressed["mixed_weight_size_bytes"] < compressed["original_fp32_weight_size_bytes"]
    assert compressed["activation_memory_estimate_bytes"] > 0.0
    softmax = next(row for row in compressed["breakdown"] if row["component"] == "softmax")
    assert softmax["weight_bits"] is None
    assert softmax["cost_semantics"] == "activation_operation_bit_cost_no_fake_weight_bits"


def test_latency_proxy_returns_missing_unit_mapping_and_never_zero_default() -> None:
    bops, domains = _proxy()
    proxy = TransformerLatencyProxy(bops, TransformerLatencyLUT())
    candidate = _phenotype(domains, compressed=True)
    result = proxy.evaluate_breakdown(candidate, fail_on_missing=False)
    assert result["status"] == "missing_unit_mapping"
    assert result["latency_proxy_ms"] is None
    assert result["missing_unit_mapping"]
    assert result["full_engine_predictor"] is False
    assert result["unit_latency_additive"] is False
    with pytest.raises(RuntimeError, match="missing_unit_mapping"):
        proxy.evaluate(candidate)


def test_runtime_workload_profiler_uses_real_instance_input_shapes() -> None:
    model = Model().eval()
    _domains, attention, ffn = build_transformer_pruning_domains(
        model, model_name="toy", allow_identity_ranking=True
    )
    attention_rows, ffn_rows, audit = profile_transformer_workloads(
        model,
        torch.randn(2, 8, 32),
        forward_fn=lambda current, sample: current(sample),
        attention_instances=attention,
        ffn_instances=ffn,
    )
    assert attention_rows == (
        AttentionWorkload(
            "attn",
            projection_tokens=16,
            query_tokens=8,
            key_tokens=8,
            attention_groups=2,
        ),
    )
    assert ffn_rows == (FFNWorkload("ffn", tokens=16),)
    assert audit["real_forward_executed"] is True


def test_projection_free_attention_has_activation_cost_without_fake_weight_bits() -> None:
    proxy = TransformerBOPSProxy(
        (),
        attention_workloads=(),
        ffn_workloads=(),
        projection_free_workloads=(
            ProjectionFreeAttentionWorkload(
                "fusion.att",
                query_tokens=2,
                key_tokens=2,
                feature_dimension=32,
                attention_groups=64,
            ),
        ),
    )
    candidate = CandidatePhenotype(
        precision_profile={
            "fusion.att::__softmax_output__": PrecisionDecision("INT8", "INT8"),
            "fusion.att::__av_bmm__": PrecisionDecision("INT8", "INT8"),
        }
    )
    metrics = proxy.evaluate_breakdown(candidate)
    qk = next(row for row in metrics["breakdown"] if row["component"] == "qk_matmul")
    softmax = next(row for row in metrics["breakdown"] if row["component"] == "softmax")
    assert qk["compute_precision"] == "FP32"
    assert softmax["weight_bits"] is None
    assert softmax["activation_bits"] == 8
    assert metrics["parameter_count"] == 0
