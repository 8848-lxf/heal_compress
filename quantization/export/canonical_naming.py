"""Deterministic canonical names for weighted ONNX nodes."""

from __future__ import annotations

import hashlib
import re

from ..config import CanonicalNamingConfig


def normalize_module_path(module_path: str) -> str:
    """Normalize a PyTorch module path for use in a TensorRT layer name."""

    value = re.sub(r"[^0-9A-Za-z_]+", "_", str(module_path).strip("."))
    return value.strip("_") or "module"


def canonical_node_name(
    module_path: str,
    onnx_op_type: str,
    call_index: int,
    *,
    config: CanonicalNamingConfig | None = None,
    collision_salt: str = "",
) -> str:
    """Create ``__canonical__<module>__<op>__callNNNNN`` deterministically."""

    policy = config or CanonicalNamingConfig()
    normalized = normalize_module_path(module_path)
    suffix = f"__{onnx_op_type}__call{int(call_index):05d}"
    full = f"{policy.prefix}{normalized}{suffix}"
    if collision_salt:
        digest = hashlib.sha256(f"{module_path}|{onnx_op_type}|{call_index}|{collision_salt}".encode("utf-8")).hexdigest()[: policy.hash_length]
        suffix = f"__h{digest}{suffix}"
        full = f"{policy.prefix}{normalized}{suffix}"
    if len(full) <= policy.max_name_length:
        return full
    digest = hashlib.sha256(str(module_path).encode("utf-8")).hexdigest()[: policy.hash_length]
    suffix = f"_{digest}{suffix}"
    available = max(1, policy.max_name_length - len(policy.prefix) - len(suffix))
    return f"{policy.prefix}{normalized[:available]}{suffix}"
