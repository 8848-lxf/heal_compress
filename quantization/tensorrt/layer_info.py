"""TensorRT layer-info parsing helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def load_layer_info(value: str | Path | Sequence[Mapping[str, Any]] | Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize trtexec inspector JSON or already parsed rows."""

    if isinstance(value, (str, Path)):
        payload: Any = json.loads(Path(value).read_text(encoding="utf-8"))
    else:
        payload = value
    if isinstance(payload, Mapping):
        rows = payload.get("Layers") or payload.get("layers") or []
    else:
        rows = payload
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def layer_name(row: Mapping[str, Any]) -> str:
    return str(row.get("Name") or row.get("name") or row.get("LayerName") or "")


def layer_metadata(row: Mapping[str, Any]) -> str:
    return " ".join(
        value
        for value in (
            layer_name(row),
            str(row.get("Metadata") or row.get("metadata") or ""),
            str(row.get("LayerType") or row.get("type") or ""),
        )
        if value
    )


def has_canonical_identity(row: Mapping[str, Any], canonical_name: str) -> bool:
    """Match one ONNX compute identity without accepting its Q/DQ descendants."""

    canonical = str(canonical_name)
    metadata = str(row.get("Metadata") or row.get("metadata") or "")
    if f"[ONNX Layer: {canonical}]" in metadata:
        return True
    name = layer_name(row)
    return any(
        token.strip() == canonical or token.strip().startswith(f"{canonical}__member")
        for token in name.split(" + ")
    )


def is_weighted_compute_layer(row: Mapping[str, Any]) -> bool:
    """Return whether TensorRT metadata describes a weighted compute layer."""

    kind = str(row.get("LayerType") or row.get("type") or "").lower()
    return any(token in kind for token in ("conv", "gemm", "matmul", "matrix", "fully"))


def precision_name(row: Mapping[str, Any]) -> str:
    direct = str(row.get("Precision") or row.get("precision") or "")
    text = direct or json.dumps(row, sort_keys=True, default=str)
    upper = text.upper()
    if "INT8" in upper or "KINT8" in upper:
        return "int8"
    if "FP16" in upper or "HALF" in upper or "FLOAT16" in upper:
        return "fp16"
    if "FP32" in upper or "FLOAT32" in upper or '"FLOAT"' in upper or upper.strip() == "FLOAT":
        return "fp32"
    return ""
