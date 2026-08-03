"""Fail-closed Transformer unit latency mapping and auxiliary LUT proxy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping

from ..candidate import CandidatePhenotype
from .transformer_bops import TransformerBOPSProxy


@dataclass(frozen=True)
class TransformerLatencyKey:
    op_type: str
    family: str
    T: int
    T_q: int
    T_k: int
    H: int
    d_h: int
    H_times_d_h: int
    d_model: int
    d_ff: int
    weight_precision: str
    activation_precision: str
    compute_precision: str
    qdq_boundary: str
    fusion_profile: str

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


class TransformerLatencyLUT:
    def __init__(self, rows: Mapping[TransformerLatencyKey, float] | None = None) -> None:
        self._values: dict[str, float] = {}
        self._keys: dict[str, TransformerLatencyKey] = {}
        for key, value in dict(rows or {}).items():
            self.add(key, value)

    def add(self, key: TransformerLatencyKey, latency_ms: float) -> None:
        if float(latency_ms) <= 0.0:
            raise ValueError("transformer_latency_must_be_positive")
        digest = key.digest
        previous = self._keys.get(digest)
        if previous is not None and previous != key:
            raise RuntimeError(f"transformer_latency_key_hash_collision:{digest}")
        self._keys[digest] = key
        self._values[digest] = float(latency_ms)

    def lookup(self, key: TransformerLatencyKey) -> float | None:
        return self._values.get(key.digest)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "transformer-latency-lut-v1",
            "full_engine_predictor": False,
            "unit_latency_additive": False,
            "entries": [
                {"key": asdict(self._keys[digest]), "key_hash": digest, "latency_ms": self._values[digest]}
                for digest in sorted(self._values)
            ],
        }


def _latency_key(row: Mapping[str, Any]) -> TransformerLatencyKey:
    weight_bits = row.get("weight_bits")
    activation_bits = row.get("activation_bits")
    if weight_bits is None:
        weight_precision = "none"
        operand = int(row.get("operand_a_bits", activation_bits or 32))
        activation_precision = f"A{operand}"
    else:
        weight_precision = f"W{int(weight_bits)}"
        activation_precision = f"A{int(activation_bits)}"
    compute = str(row.get("compute_precision", "")) or (
        "FP32" if int(row.get("operand_a_bits", 0) or 0) == 32 else activation_precision
    )
    return TransformerLatencyKey(
        op_type=str(row.get("component", "")),
        family=str(row.get("family", "")),
        T=int(row.get("T", 0) or 0),
        T_q=int(row.get("T_q", 0) or 0),
        T_k=int(row.get("T_k", 0) or 0),
        H=int(row.get("H", 0) or 0),
        d_h=int(row.get("d_h", 0) or 0),
        H_times_d_h=int(row.get("H", 0) or 0) * int(row.get("d_h", 0) or 0),
        d_model=int(row.get("d_model", 0) or 0),
        d_ff=int(row.get("d_ff", 0) or 0),
        weight_precision=weight_precision,
        activation_precision=activation_precision,
        compute_precision=compute,
        qdq_boundary=("DQ_FP32_QK" if row.get("component") == "qk_matmul" else "explicit_qdq" if int(activation_bits or 32) == 8 else "none"),
        fusion_profile=str(row.get("fused_qkv_storage", False)).lower(),
    )


class TransformerLatencyProxy:
    """LUT is tie-break evidence only; Stage-2 must use a full engine."""

    def __init__(self, bops_proxy: TransformerBOPSProxy, lut: TransformerLatencyLUT) -> None:
        self.bops_proxy = bops_proxy
        self.lut = lut

    def evaluate_breakdown(
        self,
        phenotype: CandidatePhenotype,
        *,
        fail_on_missing: bool = True,
    ) -> dict[str, Any]:
        cost = self.bops_proxy.evaluate_breakdown(phenotype)
        rows = []
        missing = []
        total = 0.0
        for component in cost["breakdown"]:
            key = _latency_key(component)
            latency = self.lut.lookup(key)
            if latency is None:
                missing.append({"reason": "missing_unit_mapping", "key": asdict(key), "key_hash": key.digest})
            else:
                total += latency
            rows.append({"key": asdict(key), "key_hash": key.digest, "latency_ms": latency})
        if missing and fail_on_missing:
            raise RuntimeError("missing_unit_mapping:" + ",".join(value["key_hash"] for value in missing))
        return {
            "status": "missing_unit_mapping" if missing else "ok",
            "latency_proxy_ms": None if missing else total,
            "missing_unit_mapping": missing,
            "units": rows,
            "full_engine_predictor": False,
            "unit_latency_additive": False,
            "search_role": "tie_break_or_auxiliary_only",
            "stage2_full_engine_latency_required": True,
        }

    def evaluate(self, phenotype: CandidatePhenotype) -> float:
        value = self.evaluate_breakdown(phenotype, fail_on_missing=True)["latency_proxy_ms"]
        return float(value)


__all__ = [
    "TransformerLatencyKey",
    "TransformerLatencyLUT",
    "TransformerLatencyProxy",
]
