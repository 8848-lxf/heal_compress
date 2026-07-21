"""Requested full-model CoBEVT Attention accumulation profiles."""

from __future__ import annotations

from dataclasses import dataclass

from .attention_compute_contract import ComputeContract


@dataclass(frozen=True)
class AccumulationProfile:
    profile_id: str
    qk_contract: ComputeContract
    av_contract: ComputeContract
    implementation: str
    search_eligible: bool = False


def _contract(storage: str, operand: str, accumulator: str, output: str) -> ComputeContract:
    multiplication = "INT8" if operand == "INT8" else operand
    return ComputeContract(storage, operand, operand, multiplication, accumulator, output)


_F32 = _contract("FP32", "FP32", "FP32", "FP32")
_F16_DEFAULT = _contract("FP16", "FP16", "unknown", "FP16")
_F16_A32 = _contract("FP16", "FP16", "FP32", "FP32")
_F16_A16 = _contract("FP16", "FP16", "FP16", "FP16")
_I8 = _contract("INT8", "INT8", "INT32", "INT32")
_BF16 = _contract("BF16", "BF16", "FP32", "FP32")


_PROFILES = {
    row.profile_id: row
    for row in (
        AccumulationProfile("A0_F3_REFERENCE", _contract("FP16", "FP32", "FP32", "FP32"), _F16_DEFAULT, "native_trt"),
        AccumulationProfile("A1_R1_LOW_ATTENTION", _F16_DEFAULT, _F16_DEFAULT, "native_trt_fused"),
        AccumulationProfile("A2_QK_F16A32", _F16_A32, _F16_DEFAULT, "native_or_plugin_oracle"),
        AccumulationProfile("A3_QK_F16A16", _F16_A16, _F16_DEFAULT, "exact_only"),
        AccumulationProfile("A4_QK_I8A32I", _I8, _F16_DEFAULT, "native_or_plugin_oracle"),
        AccumulationProfile("A5_QK_BF16A32", _BF16, _F16_DEFAULT, "optional"),
        AccumulationProfile("B1_AV_F16A32", _F32, _F16_A32, "native_or_plugin_oracle"),
        AccumulationProfile("B2_AV_F16A16", _F32, _F16_A16, "exact_only"),
        AccumulationProfile("B3_AV_I8A32I", _F32, _I8, "experimental"),
        AccumulationProfile("C1_QK_F16A32_AV_F16A32", _F16_A32, _F16_A32, "gated_joint"),
        AccumulationProfile("C2_QK_F16A32_AV_F16A16", _F16_A32, _F16_A16, "gated_joint"),
        AccumulationProfile("C3_QK_I8A32I_AV_F16A32", _I8, _F16_A32, "gated_joint"),
    )
}


def accumulation_profile(profile_id: str) -> AccumulationProfile:
    try:
        return _PROFILES[str(profile_id)]
    except KeyError as exc:
        raise ValueError(f"unknown_accumulation_profile:{profile_id}") from exc


def accumulation_profiles() -> tuple[AccumulationProfile, ...]:
    return tuple(_PROFILES.values())


__all__ = ["AccumulationProfile", "accumulation_profile", "accumulation_profiles"]
