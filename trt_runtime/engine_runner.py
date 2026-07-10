from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .bindings import trt_dtype_to_torch
from .plugins import load_tensorrt_plugin


class TensorRTEngineRunner:
    def __init__(self, engine_path: str | Path, device: torch.device, plugin_path: str | Path | None = None) -> None:
        import tensorrt as trt

        if device.type != "cuda":
            raise RuntimeError("TensorRT execution requires a CUDA device.")
        self.plugin_report = load_tensorrt_plugin(plugin_path)
        self.trt = trt
        self.device = device
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.engine_path = str(engine_path)
        with Path(engine_path).open("rb") as handle:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(handle.read())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream(device=device)
        self.input_buffers: dict[str, torch.Tensor] = {}
        self.output_buffers: dict[str, torch.Tensor] = {}
        self.bound_ptrs: dict[str, int] = {}

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

    def _ensure_buffer(self, store: dict[str, torch.Tensor], name: str, numel: int, dtype: torch.dtype) -> torch.Tensor:
        existing = store.get(name)
        if existing is None or existing.dtype != dtype or existing.numel() < int(numel):
            store[name] = torch.empty((int(numel),), dtype=dtype, device=self.device)
            self.bound_ptrs.pop(name, None)
        return store[name]

    def _bind(self, name: str, tensor: torch.Tensor) -> None:
        ptr = int(tensor.data_ptr())
        if self.bound_ptrs.get(name) != ptr:
            self.context.set_tensor_address(name, ptr)
            self.bound_ptrs[name] = ptr

    def run_profiled(self, tensors_by_name: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        trt = self.trt
        profile: dict[str, Any] = {"input_shapes": {}, "output_shapes": {}}
        outputs: dict[str, torch.Tensor] = {}
        with torch.cuda.stream(self.stream):
            for name in self.input_names():
                if name not in tensors_by_name:
                    raise KeyError(f"missing TensorRT input binding tensor: {name}")
                source = tensors_by_name[name]
                dtype = trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
                tensor = source.to(device=self.device, dtype=dtype).contiguous()
                shape = tuple(int(dim) for dim in tensor.shape)
                profile["input_shapes"][name] = list(shape)
                self.context.set_input_shape(name, shape)
                max_numel = int(np.prod(shape, dtype=np.int64))
                buffer = self._ensure_buffer(self.input_buffers, name, max_numel, dtype)
                buffer[: tensor.numel()].view(shape).copy_(tensor, non_blocking=True)
                self._bind(name, buffer)
            for name in self.output_names():
                shape = tuple(int(dim) for dim in self.context.get_tensor_shape(name))
                dtype = trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
                numel = int(np.prod(shape, dtype=np.int64))
                buffer = self._ensure_buffer(self.output_buffers, name, numel, dtype)
                self._bind(name, buffer)
                outputs[name] = buffer[:numel].view(shape)
                profile["output_shapes"][name] = list(shape)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            wall_start = time.perf_counter()
            start.record(self.stream)
            ok = self.context.execute_async_v3(self.stream.cuda_stream)
            end.record(self.stream)
        if not ok:
            raise RuntimeError(f"TensorRT execution failed for engine: {self.engine_path}")
        self.stream.synchronize()
        profile["forward_latency_ms"] = float(start.elapsed_time(end))
        profile["runner_wall_ms"] = (time.perf_counter() - wall_start) * 1000.0
        return outputs, profile

    def run(self, tensors_by_name: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        outputs, _profile = self.run_profiled(tensors_by_name)
        return outputs
