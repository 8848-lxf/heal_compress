"""Run exact cuBLASLt GEMM contracts on captured CoBEVT Attention tensors."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from search.model_families.lidar_cobevt.cuda_oracle_contract import (
    accumulator_discriminative_probes,
    oracle_contracts,
    validate_oracle_toolchain,
)


def oracle_command(
    *,
    executable: str | Path,
    a_path: str | Path,
    b_path: str | Path,
    output_path: str | Path,
    m: int,
    n: int,
    k: int,
    batch: int,
    phenotype: str,
    warmup: int,
    iterations: int,
) -> list[str]:
    return [
        str(executable),
        "--a", str(a_path),
        "--b", str(b_path),
        "--out", str(output_path),
        "--m", str(int(m)),
        "--n", str(int(n)),
        "--k", str(int(k)),
        "--batch", str(int(batch)),
        "--profile", str(phenotype),
        "--warmup", str(int(warmup)),
        "--iterations", str(int(iterations)),
    ]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row.get(key), sort_keys=True)
                    if isinstance(row.get(key), (dict, list, tuple))
                    else row.get(key)
                    for key in fields
                }
            )


def _operand_bytes(value: torch.Tensor, precision: str) -> tuple[bytes, float]:
    cpu = value.detach().contiguous().cpu().float()
    if precision == "FP32":
        return cpu.numpy().astype(np.float32, copy=False).tobytes(), 1.0
    if precision == "FP16":
        return cpu.numpy().astype(np.float16).tobytes(), 1.0
    if precision == "BF16":
        raw = cpu.to(torch.bfloat16).view(torch.uint16).numpy().tobytes()
        return raw, 1.0
    if precision == "INT8":
        maximum = float(cpu.abs().max()) if cpu.numel() else 0.0
        scale = maximum / 127.0 if maximum > 0 else 1.0
        quantized = torch.clamp(torch.round(cpu / scale), -127, 127).to(torch.int8)
        return quantized.numpy().tobytes(), scale
    raise ValueError(f"unsupported_operand_precision:{precision}")


def _read_output(path: Path, precision: str, shape: tuple[int, ...]) -> torch.Tensor:
    if precision == "FP32":
        array = np.fromfile(path, dtype=np.float32)
        return torch.from_numpy(array.copy()).reshape(shape)
    if precision == "FP16":
        array = np.fromfile(path, dtype=np.float16)
        return torch.from_numpy(array.copy()).reshape(shape)
    if precision == "INT32":
        array = np.fromfile(path, dtype=np.int32)
        return torch.from_numpy(array.copy()).reshape(shape)
    raise ValueError(f"unsupported_output_precision:{precision}")


def _relative_l2(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    ref = reference.double().reshape(-1)
    value = candidate.double().reshape(-1)
    return float(torch.linalg.vector_norm(value - ref)) / max(
        float(torch.linalg.vector_norm(ref)), 1.0e-30
    )


def _cosine(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    ref = reference.double().reshape(-1)
    value = candidate.double().reshape(-1)
    denominator = float(torch.linalg.vector_norm(ref) * torch.linalg.vector_norm(value))
    return float(torch.dot(ref, value)) / max(denominator, 1.0e-30)


def _topk_overlap(reference: torch.Tensor, candidate: torch.Tensor, k: int) -> float:
    width = min(int(k), int(reference.shape[-1]))
    ref_indices = reference.topk(width, dim=-1).indices
    value_indices = candidate.topk(width, dim=-1).indices
    overlap = (ref_indices.unsqueeze(-1) == value_indices.unsqueeze(-2)).any(dim=-1)
    return float(overlap.float().mean())


def _rank_correlation(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    ref_order = reference.argsort(dim=-1).argsort(dim=-1).double()
    value_order = candidate.argsort(dim=-1).argsort(dim=-1).double()
    ref_centered = ref_order - ref_order.mean(dim=-1, keepdim=True)
    value_centered = value_order - value_order.mean(dim=-1, keepdim=True)
    numerator = (ref_centered * value_centered).sum(dim=-1)
    denominator = torch.sqrt(
        ref_centered.square().sum(dim=-1) * value_centered.square().sum(dim=-1)
    ).clamp_min(1.0e-30)
    return float((numerator / denominator).mean())


def _softmax_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    epsilon = 1.0e-12
    ref = reference.double().clamp_min(epsilon)
    value = candidate.double().clamp_min(epsilon)
    midpoint = 0.5 * (ref + value)
    kl = (ref * (ref.log() - value.log())).sum(dim=-1)
    js = 0.5 * (
        (ref * (ref.log() - midpoint.log())).sum(dim=-1)
        + (value * (value.log() - midpoint.log())).sum(dim=-1)
    )
    entropy_ref = -(ref * ref.log()).sum(dim=-1)
    entropy_value = -(value * value.log()).sum(dim=-1)
    return {
        "softmax_kl": float(kl.mean()),
        "softmax_js": float(js.mean()),
        "softmax_entropy_delta": float((entropy_value - entropy_ref).mean()),
        "softmax_argmax_agreement": float(
            reference.argmax(dim=-1).eq(candidate.argmax(dim=-1)).float().mean()
        ),
        "softmax_top4_overlap": _topk_overlap(reference, candidate, 4),
        "softmax_top8_overlap": _topk_overlap(reference, candidate, 8),
        "softmax_row_sum_error": float((candidate.sum(dim=-1) - 1).abs().max()),
    }


def _qk_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    difference = candidate.float() - reference.float()
    return {
        "absolute_l2": float(torch.linalg.vector_norm(difference.double())),
        "relative_l2": _relative_l2(reference, candidate),
        "max_abs": float(difference.abs().max()),
        "cosine": _cosine(reference, candidate),
        "sign_agreement": float(reference.sign().eq(candidate.sign()).float().mean()),
        "top1_agreement": float(
            reference.argmax(dim=-1).eq(candidate.argmax(dim=-1)).float().mean()
        ),
        "top4_overlap": _topk_overlap(reference, candidate, 4),
        "top8_overlap": _topk_overlap(reference, candidate, 8),
        "spearman": _rank_correlation(reference, candidate),
        "zero_ratio": float(candidate.eq(0).float().mean()),
        "finite_ratio": float(torch.isfinite(candidate).float().mean()),
    }


def _execute(
    *,
    executable: Path,
    scratch: Path,
    a: torch.Tensor,
    b_stored: torch.Tensor,
    reference: torch.Tensor,
    contract: Any,
    env: dict[str, str],
    warmup: int,
    iterations: int,
) -> tuple[dict[str, Any], torch.Tensor | None]:
    batch, m, k = map(int, a.shape)
    if tuple(b_stored.shape[:1]) != (batch,) or int(b_stored.shape[-1]) != k:
        raise ValueError("oracle_b_shape_mismatch")
    n = int(b_stored.shape[-2])
    run_id = hashlib.sha256(
        f"{contract.profile_id}:{batch}:{m}:{n}:{k}:{torch.sum(a).item()}".encode()
    ).hexdigest()[:16]
    a_path = scratch / f"{run_id}.a.bin"
    b_path = scratch / f"{run_id}.b.bin"
    out_path = scratch / f"{run_id}.out.bin"
    a_bytes, scale_a = _operand_bytes(a, contract.operand_precision)
    b_bytes, scale_b = _operand_bytes(b_stored, contract.operand_precision)
    a_path.write_bytes(a_bytes)
    b_path.write_bytes(b_bytes)
    command = oracle_command(
        executable=executable,
        a_path=a_path,
        b_path=b_path,
        output_path=out_path,
        m=m,
        n=n,
        k=k,
        batch=batch,
        phenotype=contract.phenotype,
        warmup=warmup,
        iterations=iterations,
    )
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        check=False,
    )
    metadata: dict[str, Any] = {
        "command": command,
        "returncode": int(completed.returncode),
        "stdout": completed.stdout,
        "a_sha256": _sha256_bytes(a_bytes),
        "b_sha256": _sha256_bytes(b_bytes),
        "scale_a": scale_a,
        "scale_b": scale_b,
    }
    last = next(
        (line for line in reversed(completed.stdout.splitlines()) if line.startswith("{")),
        "",
    )
    if last:
        try:
            metadata.update(json.loads(last))
        except json.JSONDecodeError:
            metadata["failure_reason"] = "oracle_json_parse_failed"
    candidate = None
    if completed.returncode == 0 and out_path.is_file():
        candidate = _read_output(
            out_path, contract.output_precision, (batch, m, n)
        ).float()
        if contract.operand_precision == "INT8":
            candidate = candidate * (scale_a * scale_b)
        metadata["output_sha256"] = _sha256_file(out_path)
        metadata.update(_qk_metrics(reference, candidate))
    for path in (a_path, b_path, out_path):
        path.unlink(missing_ok=True)
    return metadata, candidate


def run_oracle_matrix(
    *,
    capture_dir: str | Path,
    executable: str | Path,
    output_dir: str | Path,
    physical_gpu: int,
    conda_prefix: str | Path,
    nvcc: str | Path,
    cxx: str | Path,
    warmup: int = 20,
    iterations: int = 100,
) -> dict[str, Any]:
    validate_oracle_toolchain(
        conda_prefix=conda_prefix, nvcc=nvcc, cxx=cxx
    )
    captures = Path(capture_dir).expanduser().resolve()
    executable_path = Path(executable).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    scratch = output / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(int(physical_gpu))
    env["LD_LIBRARY_PATH"] = ":".join(
        [str(Path(conda_prefix).resolve() / "lib"), env.get("LD_LIBRARY_PATH", "")]
    )
    contracts = [*oracle_contracts("QK"), *oracle_contracts("AV")]
    contract_rows = [row.to_manifest() for row in contracts]
    numerical_rows: list[dict[str, Any]] = []
    latency_rows: list[dict[str, Any]] = []
    capture_paths = sorted((captures / "tensors").glob("*.pt"))
    if not capture_paths:
        raise ValueError("oracle_capture_files_missing")
    for capture_path in capture_paths:
        payload = torch.load(capture_path, map_location="cpu")
        tensors = payload["tensors"]
        qk_a = tensors["scaled_q"].reshape(-1, tensors["scaled_q"].shape[-2], tensors["scaled_q"].shape[-1])
        qk_b = tensors["k"].reshape(-1, tensors["k"].shape[-2], tensors["k"].shape[-1])
        qk_reference = tensors["qk_score"].reshape(-1, tensors["qk_score"].shape[-2], tensors["qk_score"].shape[-1])
        probability = tensors["probability"].reshape(-1, tensors["probability"].shape[-2], tensors["probability"].shape[-1])
        v = tensors["v"].reshape(-1, tensors["v"].shape[-2], tensors["v"].shape[-1])
        av_b = v.transpose(-1, -2).contiguous()
        av_reference = tensors["av"].reshape(-1, tensors["av"].shape[-2], tensors["av"].shape[-1])
        for contract in contracts:
            if contract.family == "QK":
                a, b, reference = qk_a, qk_b, qk_reference
            else:
                a, b, reference = probability, av_b, av_reference
            metadata, candidate = _execute(
                executable=executable_path,
                scratch=scratch,
                a=a,
                b_stored=b,
                reference=reference,
                contract=contract,
                env=env,
                warmup=warmup,
                iterations=iterations,
            )
            common = {
                "capture_path": str(capture_path),
                "frame_id": str(payload["frame_id"]),
                "module_name": str(payload["module_name"]),
                "family": contract.family,
                "profile": contract.profile_id,
                "phenotype": contract.phenotype,
                "operand_precision": contract.operand_precision,
                "accumulator_precision": contract.accumulator_precision,
                "output_precision": contract.output_precision,
                "evidence_level": "A" if metadata.get("status") == "success" else "none",
                **metadata,
            }
            if candidate is not None and contract.family == "QK":
                reference_masked = tensors["masked_logits"].reshape_as(qk_reference)
                delta = reference_masked - qk_reference
                candidate_masked = candidate + delta
                reference_probability = probability
                candidate_probability = torch.softmax(candidate_masked, dim=-1)
                common.update(_softmax_metrics(reference_probability, candidate_probability))
            numerical_rows.append(common)
            latency_rows.append(
                {
                    key: common.get(key)
                    for key in (
                        "frame_id", "module_name", "family", "profile", "phenotype",
                        "status", "mean_ms", "algorithm_id", "tile_id", "stages_id",
                        "split_k", "workspace_bytes", "m", "n", "k", "batch"
                    )
                }
            )
    probe_rows: list[dict[str, Any]] = []
    for probe_name, (a_value, b_value) in accumulator_discriminative_probes(
        reduction_length=1024
    ).items():
        a = a_value.reshape(1, 1, 1024)
        b = b_value.reshape(1, 1, 1024)
        reference = torch.matmul(a.float(), b.float().transpose(-1, -2))
        for contract in oracle_contracts("QK"):
            metadata, _candidate = _execute(
                executable=executable_path,
                scratch=scratch,
                a=a,
                b_stored=b,
                reference=reference,
                contract=contract,
                env=env,
                warmup=warmup,
                iterations=iterations,
            )
            probe_rows.append(
                {
                    "probe": probe_name,
                    "profile": contract.profile_id,
                    "phenotype": contract.phenotype,
                    **metadata,
                }
            )
    _write_csv(output / "oracle_compute_contract.csv", contract_rows)
    _write_csv(output / "oracle_numerical_results.csv", numerical_rows)
    _write_csv(output / "oracle_latency_results.csv", latency_rows)
    _write_csv(output / "../numerical_probes/accumulator_fingerprints.csv", probe_rows)
    manifest = {
        "binary": str(executable_path),
        "binary_sha256": _sha256_file(executable_path),
        "capture_dir": str(captures),
        "capture_count": len(capture_paths),
        "physical_gpu": int(physical_gpu),
        "conda_prefix": str(Path(conda_prefix).resolve()),
        "nvcc": str(Path(nvcc).resolve()),
        "cxx": str(Path(cxx).resolve()),
        "warmup": int(warmup),
        "iterations": int(iterations),
        "numerical_rows": len(numerical_rows),
        "probe_rows": len(probe_rows),
        "success_rows": sum(row.get("status") == "success" for row in numerical_rows),
    }
    (output / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    unsupported = sorted(
        {row["phenotype"] for row in numerical_rows if row.get("status") != "success"}
    )
    (output / "oracle_conclusion.md").write_text(
        "# cuBLASLt oracle conclusion\n\n"
        f"- Level-A successful records: `{manifest['success_rows']}/{len(numerical_rows)}`.\n"
        f"- Unsupported phenotypes: `{unsupported}`.\n"
        "- Successful records bind operand, compute, scale, output type and algorithm in the cuBLASLt descriptor.\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--executable", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--conda-prefix", required=True)
    parser.add_argument("--nvcc", required=True)
    parser.add_argument("--cxx", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    result = run_oracle_matrix(
        capture_dir=args.capture_dir,
        executable=args.executable,
        output_dir=args.output_dir,
        physical_gpu=args.physical_gpu,
        conda_prefix=args.conda_prefix,
        nvcc=args.nvcc,
        cxx=args.cxx,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
