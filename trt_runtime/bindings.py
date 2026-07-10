from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .plugins import load_tensorrt_plugin


def trt_dtype_to_torch(dtype: Any) -> torch.dtype:
    import tensorrt as trt

    if dtype == trt.float16:
        return torch.float16
    if dtype == trt.float32:
        return torch.float32
    if dtype == trt.int32:
        return torch.int32
    if dtype == trt.int64:
        return torch.int64
    if dtype == trt.bool:
        return torch.bool
    return torch.float32


def inspect_engine_bindings(engine_path: str | Path, plugin_path: str | Path | None = None) -> dict[str, Any]:
    import tensorrt as trt

    plugin_report = load_tensorrt_plugin(plugin_path)
    logger = trt.Logger(trt.Logger.ERROR)
    with Path(engine_path).open("rb") as handle:
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(handle.read())
    if engine is None:
        raise RuntimeError(f"failed to deserialize TensorRT engine: {engine_path}")
    rows = []
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        mode = engine.get_tensor_mode(name)
        rows.append(
            {
                "name": name,
                "mode": "input" if mode == trt.TensorIOMode.INPUT else "output",
                "dtype": str(engine.get_tensor_dtype(name)),
                "shape": list(engine.get_tensor_shape(name)),
            }
        )
    return {"engine_path": str(engine_path), "bindings": rows, "num_bindings": len(rows), "plugin_report": plugin_report}
