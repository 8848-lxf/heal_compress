"""Precision audit helpers for original model and deployment checks."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import torch


def _empty_dtype_stats() -> dict[str, int]:
    return {"tensor_count": 0, "element_count": 0, "bytes": 0}


def _add_tensor(stats: dict[str, dict[str, int]], tensor: torch.Tensor) -> None:
    key = str(tensor.dtype)
    row = stats.setdefault(key, _empty_dtype_stats())
    row["tensor_count"] += 1
    row["element_count"] += int(tensor.numel())
    row["bytes"] += int(tensor.numel() * tensor.element_size())


def _finalize_stats(stats: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    preferred = ["torch.float32", "torch.float16", "torch.bfloat16", "torch.int8"]
    ordered: dict[str, dict[str, int]] = {}
    for key in preferred:
        if key in stats:
            ordered[key] = stats[key]
    for key in sorted(stats):
        if key not in ordered:
            ordered[key] = stats[key]
    return ordered


def _walk_tensors(value: Any, prefix: str = "") -> list[tuple[str, torch.Tensor]]:
    if torch.is_tensor(value):
        return [(prefix, value)]
    rows: list[tuple[str, torch.Tensor]] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = str(key) if not prefix else f"{prefix}.{key}"
            rows.extend(_walk_tensors(item, child))
    elif isinstance(value, tuple) and hasattr(value, "_fields"):
        for key in value._fields:
            child = str(key) if not prefix else f"{prefix}.{key}"
            rows.extend(_walk_tensors(getattr(value, key), child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            child = str(index) if not prefix else f"{prefix}.{index}"
            rows.extend(_walk_tensors(item, child))
    return rows


def _stats_for_tensors(rows: Sequence[tuple[str, torch.Tensor]]) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = {}
    for _path, tensor in rows:
        _add_tensor(stats, tensor)
    return _finalize_stats(stats)


def _is_probable_state_dict(value: Any) -> bool:
    if not isinstance(value, Mapping) or not value:
        return False
    tensor_items = 0
    string_tensor_keys = 0
    non_tensor_items = 0
    for key, item in value.items():
        if torch.is_tensor(item):
            tensor_items += 1
            if isinstance(key, str):
                string_tensor_keys += 1
        else:
            non_tensor_items += 1
    return tensor_items > 0 and string_tensor_keys == tensor_items and tensor_items >= non_tensor_items


def _state_dict_candidates(value: Any, prefix: str = "") -> list[tuple[str, Mapping[str, Any], int, int]]:
    candidates: list[tuple[str, Mapping[str, Any], int, int]] = []
    if _is_probable_state_dict(value):
        rows = _walk_tensors(value, prefix)
        candidates.append((prefix, value, len(rows), sum(int(t.numel()) for _p, t in rows)))
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = str(key) if not prefix else f"{prefix}.{key}"
            candidates.extend(_state_dict_candidates(item, child))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            child = str(index) if not prefix else f"{prefix}.{index}"
            candidates.extend(_state_dict_candidates(item, child))
    return candidates


def _choose_model_state_dict(payload: Any) -> tuple[str, Mapping[str, Any] | None]:
    candidates = _state_dict_candidates(payload)
    if not candidates:
        return "", None
    preferred_paths = {"model": 0, "state_dict": 1, "model_state_dict": 2, "net": 3, "module": 4}

    def sort_key(row: tuple[str, Mapping[str, Any], int, int]) -> tuple[int, int, int, str]:
        path, _mapping, tensor_count, element_count = row
        return (
            preferred_paths.get(path, 99),
            -int(element_count),
            -int(tensor_count),
            path,
        )

    path, mapping, _tensor_count, _element_count = sorted(candidates, key=sort_key)[0]
    return path, mapping


def audit_checkpoint_dtypes(checkpoint_path: str | Path) -> dict[str, Any]:
    path = Path(checkpoint_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    all_rows = _walk_tensors(payload)
    state_path, state_dict = _choose_model_state_dict(payload)
    state_rows = _walk_tensors(state_dict, state_path) if state_dict is not None else []
    return {
        "checkpoint_path": str(path),
        "top_level_type": type(payload).__name__,
        "top_level_keys": sorted(str(key) for key in payload.keys()) if isinstance(payload, Mapping) else [],
        "all_tensors": _stats_for_tensors(all_rows),
        "model_state_dict_path": state_path,
        "model_state_dict": _stats_for_tensors(state_rows),
        "tensor_paths_sample": [name for name, _tensor in all_rows[:50]],
    }


def audit_loaded_model_dtypes(model: torch.nn.Module) -> dict[str, Any]:
    parameter_rows = [(name, param.detach()) for name, param in model.named_parameters()]
    buffer_rows = [(name, buffer.detach()) for name, buffer in model.named_buffers()]
    return {
        "parameters": _stats_for_tensors(parameter_rows),
        "buffers": _stats_for_tensors(buffer_rows),
        "parameter_dtype_tensor_counts": _dtype_tensor_counts(parameter_rows),
        "buffer_dtype_tensor_counts": _dtype_tensor_counts(buffer_rows),
        "total_parameter_tensors": len(parameter_rows),
        "total_parameter_elements": sum(int(tensor.numel()) for _name, tensor in parameter_rows),
        "total_buffer_tensors": len(buffer_rows),
        "total_buffer_elements": sum(int(tensor.numel()) for _name, tensor in buffer_rows),
    }


def _dtype_tensor_counts(rows: Sequence[tuple[str, torch.Tensor]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _name, tensor in rows:
        counts[str(tensor.dtype)] = counts.get(str(tensor.dtype), 0) + 1
    return dict(sorted(counts.items()))


_PRECISION_PATTERNS = {
    "amp": re.compile(r"\bamp\b|mixed_precision|autocast", re.IGNORECASE),
    "autocast": re.compile(r"autocast|torch\.autocast|torch\.cuda\.amp", re.IGNORECASE),
    "GradScaler": re.compile(r"\bGradScaler\b"),
    "half": re.compile(r"\.half\(|float16|fp16", re.IGNORECASE),
    "bfloat16": re.compile(r"bfloat16|bf16", re.IGNORECASE),
}


def audit_training_precision_sources(paths: Sequence[str | Path]) -> dict[str, Any]:
    evidence: list[dict[str, Any]] = []
    scanned: list[str] = []
    for path_like in paths:
        path = Path(path_like)
        if not path.is_file():
            continue
        scanned.append(str(path))
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line_no, line in enumerate(text.splitlines(), start=1):
            for label, pattern in _PRECISION_PATTERNS.items():
                if pattern.search(line):
                    evidence.append({"path": str(path), "line": line_no, "pattern": label, "text": line.strip()[:240]})

    has_amp = any(hit["pattern"] in {"amp", "autocast"} for hit in evidence)
    has_autocast = any(hit["pattern"] == "autocast" for hit in evidence)
    has_scaler = any(hit["pattern"] == "GradScaler" for hit in evidence)
    has_half = any(hit["pattern"] == "half" for hit in evidence)
    has_bfloat16 = any(hit["pattern"] == "bfloat16" for hit in evidence)

    if has_amp or has_autocast:
        compute = "amp"
        forward = "mixed_precision_autocast" if has_autocast else "mixed_precision_unverified"
        backward = "mixed_precision_grad_scaler" if has_scaler else "mixed_precision_unverified"
    else:
        compute = "unverified"
        forward = "unverified"
        backward = "unverified"

    storage = "unverified"
    if has_half and not has_amp:
        storage = "possible_fp16"
    elif has_bfloat16 and not has_amp:
        storage = "possible_bfloat16"

    return {
        "scanned_paths": scanned,
        "training_weight_storage_precision": storage,
        "training_compute_precision": compute,
        "training_forward_compute_precision": forward,
        "training_backward_compute_precision": backward,
        "gradient_scaler_enabled": bool(has_scaler),
        "evidence": evidence,
    }


def _tensor_dtypes(value: Any) -> list[str]:
    return sorted({str(tensor.dtype) for _name, tensor in _walk_tensors(value)})


def audit_pytorch_eval_forward_precision(
    model: torch.nn.Module,
    sample_input: Any,
    *,
    module_name_patterns: Mapping[str, str],
    forward_fn: Callable[[torch.nn.Module, Any], Any] | None = None,
) -> dict[str, Any]:
    model.eval()
    module_samples: dict[str, dict[str, Any]] = {}
    hooks: list[Any] = []
    named_modules = dict(model.named_modules())
    for label, pattern in module_name_patterns.items():
        selected = None
        for name, module in named_modules.items():
            if name == pattern or pattern in name:
                selected = (name, module)
                break
        if selected is None:
            module_samples[label] = {"status": "module_not_found", "pattern": pattern}
            continue
        module_name, module = selected

        def hook(_module: torch.nn.Module, inputs: Any, output: Any, *, label: str = label, module_name: str = module_name) -> None:
            module_samples[label] = {
                "module_name": module_name,
                "input_dtypes": _tensor_dtypes(inputs),
                "output_dtypes": _tensor_dtypes(output),
                "autocast_enabled": bool(torch.is_autocast_enabled()),
            }

        hooks.append(module.register_forward_hook(hook))
    try:
        with torch.no_grad():
            if forward_fn is None:
                output = model(sample_input)
            else:
                output = forward_fn(model, sample_input)
    finally:
        for handle in hooks:
            handle.remove()
    parameter_rows = [(name, param.detach()) for name, param in model.named_parameters()]
    buffer_rows = [(name, buffer.detach()) for name, buffer in model.named_buffers()]
    return {
        "pytorch_eval_parameter_precision": _dtype_tensor_counts(parameter_rows),
        "pytorch_eval_buffer_precision": _dtype_tensor_counts(buffer_rows),
        "parameter_stats": _stats_for_tensors(parameter_rows),
        "buffer_stats": _stats_for_tensors(buffer_rows),
        "pytorch_eval_activation_precision": sorted({dtype for row in module_samples.values() for dtype in row.get("input_dtypes", []) + row.get("output_dtypes", [])}),
        "pytorch_eval_autocast_enabled": any(bool(row.get("autocast_enabled")) for row in module_samples.values()),
        "pytorch_eval_tf32_enabled": {
            "cuda_matmul_allow_tf32": bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False)),
            "cudnn_allow_tf32": bool(getattr(torch.backends.cudnn, "allow_tf32", False)),
        },
        "module_samples": module_samples,
        "output_dtypes": _tensor_dtypes(output),
    }
