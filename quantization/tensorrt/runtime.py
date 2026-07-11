"""Lazy TensorRT engine loading, bindings, and smoke execution."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Mapping

from ..exceptions import TensorRTBuildError, TensorRTConfigurationError
from ..types import EngineSmokeResult, TensorRTEngineHandle
from .plugins import load_plugin


def load_trt_engine(engine_path: str | Path, *, plugin_path: str | Path | None = None) -> TensorRTEngineHandle:
    """Deserialize an engine; TensorRT is imported only on explicit use."""

    try:
        import tensorrt as trt
    except ImportError as exc:
        raise TensorRTConfigurationError("TensorRT Python bindings are unavailable") from exc
    plugin = load_plugin(plugin_path)
    path = Path(engine_path)
    if not path.is_file():
        raise TensorRTConfigurationError(f"TensorRT engine does not exist: {path}")
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(path.read_bytes())
    if engine is None:
        raise TensorRTBuildError(f"failed to deserialize TensorRT engine: {path}")
    return TensorRTEngineHandle(str(path), engine, runtime, logger, plugin)


def _torch_dtype(trt_module: Any, dtype: Any) -> Any:
    import torch

    mapping = {
        trt_module.float16: torch.float16,
        trt_module.float32: torch.float32,
        trt_module.int8: torch.int8,
        trt_module.int32: torch.int32,
        trt_module.bool: torch.bool,
    }
    return mapping.get(dtype, torch.float32)


class TensorRTEngineRunner:
    """Reusable dynamic-shape TensorRT execution context."""

    def __init__(self, handle: TensorRTEngineHandle, *, device: str = "cuda:0") -> None:
        import torch
        import tensorrt as trt

        self.handle = handle
        self.engine = handle.engine
        self.context = self.engine.create_execution_context()
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise TensorRTConfigurationError("TensorRT execution requires an available CUDA device")
        self.trt = trt
        self.stream = torch.cuda.Stream(device=self.device)

    def input_names(self) -> list[str]:
        return [
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(index)) == self.trt.TensorIOMode.INPUT
        ]

    def output_names(self) -> list[str]:
        return [
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(index)) == self.trt.TensorIOMode.OUTPUT
        ]

    def run(self, inputs: Mapping[str, Any]) -> tuple[dict[str, Any], float]:
        import torch

        buffers: dict[str, Any] = {}
        with torch.cuda.stream(self.stream):
            for name in self.input_names():
                if name not in inputs:
                    raise KeyError(f"missing TensorRT input binding: {name}")
                tensor = inputs[name].to(self.device, dtype=_torch_dtype(self.trt, self.engine.get_tensor_dtype(name))).contiguous()
                self.context.set_input_shape(name, tuple(int(dim) for dim in tensor.shape))
                self.context.set_tensor_address(name, int(tensor.data_ptr()))
                buffers[name] = tensor
            outputs: dict[str, Any] = {}
            for name in self.output_names():
                shape = tuple(int(dim) for dim in self.context.get_tensor_shape(name))
                if any(dim < 0 for dim in shape):
                    raise TensorRTBuildError(f"unresolved dynamic output shape for {name}: {shape}")
                output = torch.empty(math.prod(shape), device=self.device, dtype=_torch_dtype(self.trt, self.engine.get_tensor_dtype(name))).view(shape)
                self.context.set_tensor_address(name, int(output.data_ptr()))
                outputs[name] = output
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(self.stream)
            success = self.context.execute_async_v3(self.stream.cuda_stream)
            end.record(self.stream)
        if not success:
            raise TensorRTBuildError("TensorRT execute_async_v3 returned false")
        self.stream.synchronize()
        return outputs, float(start.elapsed_time(end))


def run_engine_smoke(
    engine: TensorRTEngineHandle | TensorRTEngineRunner | str | Path,
    inputs: Mapping[str, Any],
    *,
    plugin_path: str | Path | None = None,
    device: str = "cuda:0",
) -> EngineSmokeResult:
    """Run one caller-prepared frame; no dataset or checkpoint is loaded."""

    try:
        if isinstance(engine, TensorRTEngineRunner):
            runner = engine
        else:
            handle = engine if isinstance(engine, TensorRTEngineHandle) else load_trt_engine(engine, plugin_path=plugin_path)
            runner = TensorRTEngineRunner(handle, device=device)
        started = time.perf_counter()
        outputs, kernel_ms = runner.run(inputs)
        wall_ms = (time.perf_counter() - started) * 1000.0
        return EngineSmokeResult(True, {name: tuple(int(dim) for dim in value.shape) for name, value in outputs.items()}, kernel_ms or wall_ms)
    except Exception as exc:
        return EngineSmokeResult(False, {}, None, f"{type(exc).__name__}: {exc}")
