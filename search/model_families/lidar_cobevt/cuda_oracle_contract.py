"""Exact cuBLASLt/CUTLASS compute contracts for Attention GEMM oracles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .conda_cuda_toolchain import validate_cuda_toolchain_paths


@dataclass(frozen=True)
class OracleComputeContract:
    profile_id: str
    family: str
    operand_cuda_type: str
    compute_type: str
    scale_type: str
    output_cuda_type: str
    operand_precision: str
    accumulator_precision: str
    output_precision: str
    phenotype: str
    evidence_level: str = "A"
    implementation: str = "cuBLASLt_or_CUTLASS_oracle"

    def to_manifest(self) -> dict[str, str]:
        return asdict(self)


def _rows(family: str) -> tuple[OracleComputeContract, ...]:
    prefix = "O" if family == "QK" else "V"
    return (
        OracleComputeContract(
            f"{prefix}0_F32A32", family, "CUDA_R_32F", "CUBLAS_COMPUTE_32F",
            "CUDA_R_32F", "CUDA_R_32F", "FP32", "FP32", "FP32", "F32A32"
        ),
        OracleComputeContract(
            f"{prefix}1_F16A32", family, "CUDA_R_16F", "CUBLAS_COMPUTE_32F",
            "CUDA_R_32F", "CUDA_R_32F", "FP16", "FP32", "FP32", "F16A32"
        ),
        OracleComputeContract(
            f"{prefix}2_F16A16", family, "CUDA_R_16F", "CUBLAS_COMPUTE_16F",
            "CUDA_R_16F", "CUDA_R_16F", "FP16", "FP16", "FP16", "F16A16"
        ),
        OracleComputeContract(
            f"{prefix}3_BF16A32", family, "CUDA_R_16BF", "CUBLAS_COMPUTE_32F",
            "CUDA_R_32F", "CUDA_R_32F", "BF16", "FP32", "FP32", "BF16A32"
        ),
        OracleComputeContract(
            f"{prefix}4_I8A32I", family, "CUDA_R_8I", "CUBLAS_COMPUTE_32I",
            "CUDA_R_32I", "CUDA_R_32I", "INT8", "INT32", "INT32", "I8A32I"
        ),
    )


def oracle_contracts(family: str) -> tuple[OracleComputeContract, ...]:
    normalized = str(family).upper()
    if normalized not in {"QK", "AV"}:
        raise ValueError(f"unsupported_oracle_family:{family}")
    return _rows(normalized)


def oracle_contract(profile_id: str) -> OracleComputeContract:
    for family in ("QK", "AV"):
        for row in _rows(family):
            if row.profile_id == str(profile_id):
                return row
    raise ValueError(f"unknown_oracle_profile:{profile_id}")


def validate_oracle_toolchain(
    *, conda_prefix: str | Path, nvcc: str | Path, cxx: str | Path
) -> None:
    prefix = Path(conda_prefix)
    validate_cuda_toolchain_paths(
        conda_prefix=prefix,
        python_path=prefix / "bin/python",
        nvcc_path=Path(nvcc),
        cuda_home=prefix,
        cudacxx=Path(nvcc),
        cxx_path=Path(cxx),
    )


def accumulator_discriminative_probes(
    *, reduction_length: int
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    length = int(reduction_length)
    if length < 16:
        raise ValueError("probe_reduction_length_too_small")

    def shape(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(1, 1, 1, length).float()

    indices = torch.arange(length)
    alternating = torch.where(indices.remainder(2).eq(0), 1.0, -1.0)
    large_small = torch.where(indices.remainder(8).eq(0), 2048.0, 2.0 ** -10)
    tiny = torch.full((length,), torch.finfo(torch.float16).tiny / 2)
    underflow = torch.full((length,), 2.0 ** -13)
    small_margin = torch.ones(length)
    small_margin[-1] = 1.0 + 2.0 ** -10
    zero = torch.zeros(length)
    return {
        "alternating_cancellation": (shape(alternating), shape(torch.ones(length))),
        "large_small_mixture": (shape(large_small), shape(torch.ones(length))),
        "long_reduction": (shape(torch.ones(length)), shape(torch.full((length,), 1.0 / length))),
        "near_fp16_subnormal": (shape(tiny), shape(torch.ones(length))),
        "product_underflow": (shape(underflow), shape(underflow)),
        "softmax_small_margin": (shape(small_margin), shape(torch.ones(length))),
        "zero_uniform_logits": (shape(zero), shape(torch.ones(length))),
    }


__all__ = [
    "OracleComputeContract",
    "accumulator_discriminative_probes",
    "oracle_contract",
    "oracle_contracts",
    "validate_oracle_toolchain",
]
