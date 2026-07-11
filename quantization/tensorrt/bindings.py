"""TensorRT binding inspection."""

from __future__ import annotations

from pathlib import Path

from .runtime import load_trt_engine


def inspect_engine_bindings(engine_path: str | Path, *, plugin_path: str | Path | None = None) -> list[dict[str, object]]:
    """Return binding names, modes, dtypes, and declared shapes."""

    handle = load_trt_engine(engine_path, plugin_path=plugin_path)
    import tensorrt as trt

    rows = []
    for index in range(handle.engine.num_io_tensors):
        name = handle.engine.get_tensor_name(index)
        rows.append(
            {
                "name": name,
                "mode": "input" if handle.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else "output",
                "dtype": str(handle.engine.get_tensor_dtype(name)),
                "shape": tuple(int(dim) for dim in handle.engine.get_tensor_shape(name)),
            }
        )
    return rows
