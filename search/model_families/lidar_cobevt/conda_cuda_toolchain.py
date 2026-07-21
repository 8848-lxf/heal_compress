"""Fail-closed contracts for the CoBEVT Conda CUDA/SmoothQuant audit.

The helpers in this module are deliberately side-effect free.  The orchestration
entry point performs subprocesses and writes evidence; these functions decide
whether that evidence is admissible.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


SMOOTHQUANT_ALPHA_GRID = (0.5, 0.6, 0.7, 0.75, 0.8)
_SYSTEM_NVCC = (Path("/usr/bin/nvcc"), Path("/usr/local/cuda/bin/nvcc"))


def _resolved(path: Path | str) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _inside(path: Path | str, root: Path | str) -> bool:
    candidate = _resolved(path)
    prefix = _resolved(root)
    try:
        candidate.relative_to(prefix)
    except ValueError:
        return False
    return True


def validate_cuda_toolchain_paths(
    *,
    conda_prefix: Path,
    python_path: Path,
    nvcc_path: Path,
    cuda_home: Path,
    cudacxx: Path,
    cxx_path: Path | None = None,
) -> None:
    prefix = _resolved(conda_prefix)
    if not _inside(python_path, prefix):
        raise ValueError("python_outside_conda_prefix")
    if _resolved(nvcc_path) in {_resolved(path) for path in _SYSTEM_NVCC} or not _inside(
        nvcc_path, prefix
    ):
        raise ValueError("invalid_non_conda_cuda_toolchain")
    if not _inside(cuda_home, prefix):
        raise ValueError("cuda_home_outside_conda_prefix")
    if not _inside(cudacxx, prefix):
        raise ValueError("cudacxx_outside_conda_prefix")
    if _resolved(cudacxx) != _resolved(nvcc_path):
        raise ValueError("cudacxx_nvcc_mismatch")
    if cxx_path is not None and not _inside(cxx_path, prefix):
        raise ValueError("cxx_outside_conda_prefix")


def reject_non_conda_compiler_log(text: str, *, conda_prefix: Path) -> None:
    payload = str(text)
    if any(str(path) in payload for path in _SYSTEM_NVCC):
        raise ValueError("invalid_non_conda_cuda_toolchain")
    prefix = _resolved(conda_prefix)
    for token in re.findall(r"(?:^|\s)(/[^\s'\"]*/nvcc)(?=\s|$)", payload):
        if not _inside(token, prefix):
            raise ValueError("invalid_non_conda_cuda_toolchain")


def sanitize_tensorrt_builder_environment(
    environment: Mapping[str, str],
) -> dict[str, str]:
    """Do not propagate export-only synchronous execution into tactic timing."""

    cleaned = {str(key): str(value) for key, value in environment.items()}
    cleaned.pop("CUDA_LAUNCH_BLOCKING", None)
    return cleaned


def validate_cuda_extension_build(
    *,
    compile_log: str,
    conda_prefix: Path,
    runtime_backend: str,
    runtime_success: bool,
) -> None:
    reject_non_conda_compiler_log(compile_log, conda_prefix=conda_prefix)
    expected = str(_resolved(conda_prefix) / "bin/nvcc")
    if expected not in compile_log:
        raise ValueError("conda_nvcc_compile_evidence_missing")
    if not re.search(r"(?:compute_89|sm_89)", compile_log):
        raise ValueError("sm89_compile_flag_missing")
    backend = str(runtime_backend).strip().lower()
    if backend != "cuda" or "fallback" in backend:
        raise ValueError("cuda_extension_cpu_fallback")
    if not runtime_success:
        raise ValueError("cuda_extension_runtime_failed")


def extension_cache_identity(
    conda_prefix: Path, nvcc_sha256: str, architecture: str
) -> str:
    payload = {
        "architecture": str(architecture),
        "conda_prefix": str(_resolved(conda_prefix)),
        "nvcc_sha256": str(nvcc_sha256),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_toolchain_manifest(
    *,
    conda_prefix: Path,
    python_path: Path,
    nvcc_path: Path,
    nvcc_sha256: str,
    nvcc_version: str,
    cuda_home: Path,
    cudacxx: Path,
    sm89_supported: bool,
    extension_binary_sha256: str,
    cxx_path: Path | None = None,
) -> dict[str, Any]:
    validate_cuda_toolchain_paths(
        conda_prefix=conda_prefix,
        python_path=python_path,
        nvcc_path=nvcc_path,
        cuda_home=cuda_home,
        cudacxx=cudacxx,
        cxx_path=cxx_path,
    )
    for label, digest in (
        ("nvcc", nvcc_sha256),
        ("extension_binary", extension_binary_sha256),
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", str(digest)):
            raise ValueError(f"{label}_sha256_invalid")
    if not sm89_supported:
        raise ValueError("sm89_not_supported")
    return {
        "conda_prefix": str(_resolved(conda_prefix)),
        "python": str(_resolved(python_path)),
        "nvcc": str(_resolved(nvcc_path)),
        "nvcc_sha256": str(nvcc_sha256),
        "nvcc_version": str(nvcc_version),
        "CUDA_HOME": str(_resolved(cuda_home)),
        "CUDACXX": str(_resolved(cudacxx)),
        "CXX": str(_resolved(cxx_path)) if cxx_path is not None else None,
        "sm89_supported": True,
        "extension_binary_sha256": str(extension_binary_sha256),
    }


def smoothquant_scale(
    activation_channel_max: Any,
    weight_input_channel_max: Any,
    *,
    alpha: float,
    epsilon: float = 1e-12,
) -> Any:
    value = float(alpha)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("smoothquant_alpha_out_of_range")
    activation = activation_channel_max.clamp_min(float(epsilon))
    weight = weight_input_channel_max.clamp_min(float(epsilon))
    if tuple(activation.shape) != tuple(weight.shape):
        raise ValueError("smoothquant_channel_shape_mismatch")
    return activation.pow(value) / weight.pow(1.0 - value)


def smoothquant_reparameterize(x: Any, weight: Any, scale: Any) -> tuple[Any, Any]:
    if x.shape[-1] != weight.shape[-1] or scale.numel() != x.shape[-1]:
        raise ValueError("smoothquant_reparameterization_shape_mismatch")
    if not bool((scale > 0).all()):
        raise ValueError("smoothquant_scale_nonpositive")
    return x / scale, weight * scale.reshape(1, -1)


def validate_pre_quant_scale(scale: Sequence[float] | Any, *, input_features: int) -> None:
    count = int(scale.numel()) if hasattr(scale, "numel") else len(scale)
    if count != int(input_features):
        raise ValueError("pre_quant_scale_incomplete")


def validate_smoothquant_realization(
    *,
    projection_rows: Sequence[Mapping[str, Any]],
    required_projection_roles: Sequence[str],
    q_dtype: str,
    k_dtype: str,
    qk_input_dtypes: Sequence[str],
    qk_output_dtype: str,
    qk_accumulator_precision: str,
) -> None:
    qk_tokens = {
        str(q_dtype).upper(),
        str(k_dtype).upper(),
        *(str(value).upper() for value in qk_input_dtypes),
        str(qk_output_dtype).upper(),
        str(qk_accumulator_precision).upper(),
    }
    if "INT8" in qk_tokens or "INT32" in qk_tokens:
        raise ValueError("native_int8_qk_forbidden")
    by_role = {str(row["role"]): row for row in projection_rows}
    for role in required_projection_roles:
        row = by_role.get(str(role))
        if row is None:
            raise ValueError(f"smoothquant_projection_evidence_missing:{role}")
        if str(row.get("requested", "")).upper() != "INT8" or str(
            row.get("realized", "")
        ).upper() != "INT8":
            raise ValueError("smoothquant_projection_precision_fallback")
    if (
        str(q_dtype).upper() != "FP32"
        or str(k_dtype).upper() != "FP32"
        or tuple(str(value).upper() for value in qk_input_dtypes) != ("FP32", "FP32")
        or str(qk_output_dtype).upper() != "FP32"
        or str(qk_accumulator_precision).upper() != "FP32"
    ):
        raise ValueError("smoothquant_qk_fp32_contract_failed")


def validate_projection_qdq_adjacency(
    model: Any, projection_node_names: Sequence[str]
) -> list[dict[str, Any]]:
    """Require explicit activation/weight DQ to directly feed each projection."""

    nodes = {str(node.name): node for node in model.graph.node if str(node.name)}
    producers = {
        str(output): node for node in model.graph.node for output in node.output
    }
    records: list[dict[str, Any]] = []

    def dequantize_source(tensor_name: str, *, weight: bool) -> Any | None:
        producer = producers.get(str(tensor_name))
        if producer is None:
            return None
        if str(producer.op_type) == "DequantizeLinear":
            return producer
        if weight and str(producer.op_type) == "Transpose" and producer.input:
            candidate = producers.get(str(producer.input[0]))
            if candidate is not None and str(candidate.op_type) == "DequantizeLinear":
                return candidate
        return None

    for name in projection_node_names:
        node = nodes.get(str(name))
        if node is None:
            raise ValueError(f"projection_node_missing:{name}")
        if str(node.op_type) not in {"MatMul", "Gemm"} or len(node.input) < 2:
            raise ValueError(f"projection_node_unsupported:{name}:{node.op_type}")
        input_producers = [producers.get(str(node.input[index])) for index in (0, 1)]
        dequantize_nodes = [
            dequantize_source(str(node.input[0]), weight=False),
            dequantize_source(str(node.input[1]), weight=True),
        ]
        if any(value is None for value in dequantize_nodes):
            producer_types = [
                str(producer.op_type) if producer is not None else ""
                for producer in input_producers
            ]
            raise ValueError(
                f"projection_qdq_not_adjacent:{name}:{producer_types}"
            )
        records.append(
            {
                "projection_node": str(name),
                "activation_dequantize_node": str(dequantize_nodes[0].name),
                "weight_dequantize_node": str(dequantize_nodes[1].name),
                "validated": True,
            }
        )
    return records


def restore_projection_qdq_adjacency(
    model: Any,
    projection_node_names: Sequence[str],
    *,
    output_cast_precisions: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Bypass only FP16 Casts inserted between ModelOpt DQ and projections."""

    from onnx import TensorProto, helper

    nodes = {str(node.name): node for node in model.graph.node if str(node.name)}
    producers = {
        str(output): node for node in model.graph.node for output in node.output
    }

    def cast_target(node: Any) -> int | None:
        if node is None or str(node.op_type) != "Cast":
            return None
        return next(
            (int(value.i) for value in node.attribute if str(value.name) == "to"),
            None,
        )

    def is_dq_path(tensor_name: str, *, weight: bool) -> bool:
        producer = producers.get(str(tensor_name))
        if producer is None:
            return False
        if str(producer.op_type) == "DequantizeLinear":
            return True
        if weight and str(producer.op_type) == "Transpose" and producer.input:
            source = producers.get(str(producer.input[0]))
            return source is not None and str(source.op_type) == "DequantizeLinear"
        return False

    records: list[dict[str, Any]] = []
    removed_cast_outputs: set[str] = set()
    output_casts: dict[str, Any] = {}
    requested_output_casts = {
        str(name): str(precision).upper()
        for name, precision in (output_cast_precisions or {}).items()
    }

    def set_tensor_type(name: str, element_type: int) -> None:
        for value in [*model.graph.input, *model.graph.value_info, *model.graph.output]:
            if str(value.name) == str(name):
                value.type.tensor_type.elem_type = int(element_type)
                return
        model.graph.value_info.append(
            helper.make_tensor_value_info(str(name), int(element_type), None)
        )

    for name in projection_node_names:
        node = nodes.get(str(name))
        if node is None:
            raise ValueError(f"projection_node_missing:{name}")
        removed: list[str] = []
        for input_index in (0, 1):
            cast = producers.get(str(node.input[input_index]))
            if cast_target(cast) != int(TensorProto.FLOAT16) or not cast.input:
                raise ValueError(
                    f"projection_qdq_cast_pattern_missing:{name}:{input_index}"
                )
            source = str(cast.input[0])
            if not is_dq_path(source, weight=input_index == 1):
                raise ValueError(
                    f"projection_qdq_cast_source_invalid:{name}:{input_index}:{source}"
                )
            removed.append(str(cast.name))
            removed_cast_outputs.update(str(value) for value in cast.output)
            node.input[input_index] = source
        output_precision = requested_output_casts.get(str(name), "")
        output_cast_name = ""
        if output_precision:
            if output_precision not in {"FP16", "FP32"} or len(node.output) != 1:
                raise ValueError(
                    f"projection_output_cast_invalid:{name}:{output_precision}"
                )
            public = str(node.output[0])
            raw = f"{public}__before_smoothquant_output_cast"
            output_cast_name = f"{name}__output_{output_precision.lower()}"
            element_type = (
                int(TensorProto.FLOAT16)
                if output_precision == "FP16"
                else int(TensorProto.FLOAT)
            )
            node.output[0] = raw
            output_casts[str(name)] = helper.make_node(
                "Cast", [raw], [public], name=output_cast_name, to=element_type
            )
            set_tensor_type(raw, int(TensorProto.FLOAT))
            set_tensor_type(public, element_type)
        records.append(
            {
                "projection_node": str(name),
                "removed_cast_nodes": removed,
                "output_cast_node": output_cast_name,
                "output_cast_precision": output_precision,
                "validated": True,
            }
        )
    used = {
        str(value)
        for node in model.graph.node
        for value in node.input
        if str(value)
    } | {str(value.name) for value in model.graph.output}
    kept = []
    for node in model.graph.node:
        if (
            str(node.op_type) == "Cast"
            and set(str(value) for value in node.output) <= removed_cast_outputs
            and not any(str(value) in used for value in node.output)
        ):
            continue
        kept.append(node)
        output_cast = output_casts.get(str(node.name))
        if output_cast is not None:
            kept.append(output_cast)
    del model.graph.node[:]
    model.graph.node.extend(kept)
    validate_projection_qdq_adjacency(model, projection_node_names)
    return records


def validate_s2_retained_indices(*, expected: Mapping[str, Any], actual: Mapping[str, Any]) -> None:
    canonical = lambda value: json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    if canonical(expected) != canonical(actual):
        raise ValueError("s2_retained_indices_changed")


def build_latency_model_contract(
    *, unit_lut_status: str, full_engine_anchor_hashes: Sequence[str]
) -> dict[str, Any]:
    if str(unit_lut_status) != "not_additive":
        raise ValueError("unit_lut_status_must_remain_not_additive")
    hashes = [str(value) for value in full_engine_anchor_hashes]
    if not hashes:
        raise ValueError("full_engine_anchor_required")
    if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes):
        raise ValueError("full_engine_anchor_hash_invalid")
    return {
        "unit_lut": "not_additive",
        "formal_source": "full_engine_action_anchors",
        "full_engine_anchor_hashes": hashes,
    }


def summarize_staged_exports(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate fresh-process E0--E5 lineage and summarize the first failure."""

    stage_order = tuple(f"E{index}" for index in range(6))
    seen_processes: set[int] = set()
    first_failing_stage: str | None = None
    previous_index = -1
    normalized: list[dict[str, Any]] = []
    for raw in records:
        stage = str(raw.get("stage", ""))
        if stage not in stage_order:
            raise ValueError(f"unknown_export_stage:{stage}")
        stage_index = stage_order.index(stage)
        if stage_index <= previous_index:
            raise ValueError("staged_export_order_invalid")
        if first_failing_stage is not None:
            raise ValueError("stage_executed_after_failure")
        process_id = int(raw.get("process_id", -1))
        if process_id <= 0:
            raise ValueError("staged_export_process_id_invalid")
        if process_id in seen_processes:
            raise ValueError("staged_export_process_reused")
        seen_processes.add(process_id)
        return_code = int(raw.get("return_code", -1))
        normalized.append(
            {"stage": stage, "process_id": process_id, "return_code": return_code}
        )
        if return_code != 0:
            first_failing_stage = stage
        previous_index = stage_index
    completed_stages = [row["stage"] for row in normalized if row["return_code"] == 0]
    return {
        "records": normalized,
        "completed_stages": completed_stages,
        "first_failing_stage": first_failing_stage,
        "all_completed": first_failing_stage is None
        and completed_stages == list(stage_order),
        "fresh_processes": True,
    }


def classify_smoothquant_candidate(
    *,
    toolchain_pass: bool,
    extension_cuda: bool,
    export_success: bool,
    build_success: bool,
    realized_precision_pass: bool,
    smoke10_pass: bool,
    fixed50_pass: bool,
    formal_latency_pass: bool,
) -> str:
    """Admit a profile only after the full deployment evidence chain passes."""

    gates = (
        toolchain_pass,
        extension_cuda,
        export_success,
        build_success,
        realized_precision_pass,
        smoke10_pass,
        fixed50_pass,
        formal_latency_pass,
    )
    return "allowed" if all(bool(value) for value in gates) else "blocked"


__all__ = [
    "SMOOTHQUANT_ALPHA_GRID",
    "build_latency_model_contract",
    "build_toolchain_manifest",
    "classify_smoothquant_candidate",
    "extension_cache_identity",
    "reject_non_conda_compiler_log",
    "restore_projection_qdq_adjacency",
    "sanitize_tensorrt_builder_environment",
    "smoothquant_reparameterize",
    "smoothquant_scale",
    "summarize_staged_exports",
    "validate_cuda_extension_build",
    "validate_cuda_toolchain_paths",
    "validate_pre_quant_scale",
    "validate_projection_qdq_adjacency",
    "validate_s2_retained_indices",
    "validate_smoothquant_realization",
]
