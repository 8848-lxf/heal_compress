"""Reusable TensorRT execution context for formal dynamic-shape evaluation."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Mapping

import torch

from .bindings import trt_dtype_to_torch
from .plugins import load_tensorrt_plugin


class TensorRTEngineRunner:
    """Own one TensorRT context, stream and reusable binding buffers."""

    def __init__(
        self,
        engine_path: str | Path,
        device: torch.device,
        plugin_path: str | Path | None = None,
    ) -> None:
        import tensorrt as trt

        if device.type != "cuda":
            raise RuntimeError("TensorRT execution requires a CUDA device")
        self.plugin_report = load_tensorrt_plugin(plugin_path)
        self.device = device
        self.trt = trt
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
        self.last_input_shapes: dict[str, tuple[int, ...]] = {}
        self.allocation_stats = {
            "context_created_per_run": False,
            "stream_created_per_run": False,
            "context_create_count": 1,
            "stream_create_count": 1,
            "run_count": 0,
            "input_buffer_reallocation_count": 0,
            "output_buffer_reallocation_count": 0,
            "set_input_shape_call_count": 0,
            "output_shape_query_call_count": 0,
            "number_of_h2d_copies": 0,
            "number_of_d2d_input_copies": 0,
            "total_bytes_h2d": 0,
            "total_bytes_d2d_input": 0,
        }

    def input_names(self) -> list[str]:
        return [
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(index))
            == self.trt.TensorIOMode.INPUT
        ]

    def output_names(self) -> list[str]:
        return [
            self.engine.get_tensor_name(index)
            for index in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(index))
            == self.trt.TensorIOMode.OUTPUT
        ]

    def allocation_report(self) -> dict[str, Any]:
        runs = int(self.allocation_stats["run_count"])
        return {
            **self.allocation_stats,
            "input_buffer_reallocated_per_frame": bool(
                runs
                and int(self.allocation_stats["input_buffer_reallocation_count"])
                >= runs
            ),
            "output_buffer_reallocated_per_frame": bool(
                runs
                and int(self.allocation_stats["output_buffer_reallocation_count"])
                >= runs
            ),
            "input_buffers": {
                name: {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "numel": int(tensor.numel()),
                }
                for name, tensor in self.input_buffers.items()
            },
            "output_buffers": {
                name: {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "numel": int(tensor.numel()),
                }
                for name, tensor in self.output_buffers.items()
            },
        }

    def _profile_max_shape(
        self, name: str, fallback: tuple[int, ...]
    ) -> tuple[int, ...]:
        try:
            shapes = self.engine.get_tensor_profile_shape(name, 0)
            if shapes and len(shapes) == 3:
                maximum = tuple(int(value) for value in shapes[2])
                if all(value > 0 for value in maximum):
                    return maximum
        except Exception:
            pass
        return fallback

    def _ensure_buffer(
        self,
        store: dict[str, torch.Tensor],
        name: str,
        numel: int,
        dtype: torch.dtype,
        stat_key: str,
    ) -> torch.Tensor:
        current = store.get(name)
        if current is None or current.dtype != dtype or current.numel() < int(numel):
            store[name] = torch.empty(
                (int(numel),), dtype=dtype, device=self.device
            )
            self.bound_ptrs.pop(name, None)
            self.allocation_stats[stat_key] += 1
        return store[name]

    def _bind(self, name: str, tensor: torch.Tensor) -> None:
        pointer = int(tensor.data_ptr())
        if self.bound_ptrs.get(name) != pointer:
            self.context.set_tensor_address(name, pointer)
            self.bound_ptrs[name] = pointer

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        return (time.perf_counter() - started) * 1000.0

    def run(
        self,
        tensors_by_name: Mapping[str, torch.Tensor],
        *,
        cast_inputs_to_engine_dtype: bool = True,
    ) -> dict[str, torch.Tensor]:
        outputs, _ = self._run_impl(
            tensors_by_name,
            cast_inputs_to_engine_dtype=cast_inputs_to_engine_dtype,
            collect_profile=False,
        )
        return outputs

    def run_profiled(
        self,
        tensors_by_name: Mapping[str, torch.Tensor],
        *,
        cast_inputs_to_engine_dtype: bool = True,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        return self._run_impl(
            tensors_by_name,
            cast_inputs_to_engine_dtype=cast_inputs_to_engine_dtype,
            collect_profile=True,
        )

    def _run_impl(
        self,
        tensors_by_name: Mapping[str, torch.Tensor],
        *,
        cast_inputs_to_engine_dtype: bool,
        collect_profile: bool,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        self.allocation_stats["run_count"] += 1
        profile: dict[str, Any] = {
            key: 0.0
            for key in (
                "dtype_cast_ms",
                "contiguous_ms",
                "h2d_copy_ms",
                "input_device_copy_ms",
                "set_input_shape_ms",
                "bind_address_ms",
                "output_shape_query_ms",
                "execute_async_ms",
                "synchronize_ms",
                "output_wrap_ms",
            )
        }
        profile.update(
            {
                "input_buffer_reallocated": False,
                "output_buffer_reallocated": False,
                "set_input_shape_calls": 0,
                "output_shape_query_calls": 0,
                "h2d_copies": 0,
                "d2d_input_copies": 0,
                "bytes_h2d": 0,
                "bytes_d2d_input": 0,
                "input_shapes": {},
                "output_shapes": {},
            }
        )
        total_started = time.perf_counter()
        outputs: dict[str, torch.Tensor] = {}

        def synchronize_profile() -> None:
            if collect_profile:
                self.stream.synchronize()

        with torch.cuda.stream(self.stream):
            for name in self.input_names():
                if name not in tensors_by_name:
                    raise KeyError(f"missing TensorRT input binding tensor: {name}")
                source = tensors_by_name[name]
                target_dtype = (
                    trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
                    if cast_inputs_to_engine_dtype
                    else source.dtype
                )
                tensor = source
                if tensor.dtype != target_dtype:
                    started = time.perf_counter()
                    tensor = tensor.to(dtype=target_dtype)
                    synchronize_profile()
                    profile["dtype_cast_ms"] += self._elapsed_ms(started)
                if not tensor.is_contiguous():
                    started = time.perf_counter()
                    tensor = tensor.contiguous()
                    synchronize_profile()
                    profile["contiguous_ms"] += self._elapsed_ms(started)

                shape = tuple(int(value) for value in tensor.shape)
                profile["input_shapes"][name] = list(shape)
                if self.last_input_shapes.get(name) != shape:
                    started = time.perf_counter()
                    self.context.set_input_shape(name, shape)
                    profile["set_input_shape_ms"] += self._elapsed_ms(started)
                    profile["set_input_shape_calls"] += 1
                    self.allocation_stats["set_input_shape_call_count"] += 1
                    self.last_input_shapes[name] = shape

                maximum = self._profile_max_shape(name, shape)
                capacity = max(math.prod(maximum), int(tensor.numel()))
                previous = int(
                    self.allocation_stats["input_buffer_reallocation_count"]
                )
                buffer = self._ensure_buffer(
                    self.input_buffers,
                    name,
                    capacity,
                    target_dtype,
                    "input_buffer_reallocation_count",
                )
                if int(self.allocation_stats["input_buffer_reallocation_count"]) != previous:
                    profile["input_buffer_reallocated"] = True
                view = buffer[: int(tensor.numel())].view(shape)
                started = time.perf_counter()
                view.copy_(tensor, non_blocking=True)
                synchronize_profile()
                byte_count = int(tensor.numel() * tensor.element_size())
                if tensor.device != self.device:
                    profile["h2d_copy_ms"] += self._elapsed_ms(started)
                    profile["h2d_copies"] += 1
                    profile["bytes_h2d"] += byte_count
                    self.allocation_stats["number_of_h2d_copies"] += 1
                    self.allocation_stats["total_bytes_h2d"] += byte_count
                else:
                    profile["input_device_copy_ms"] += self._elapsed_ms(started)
                    profile["d2d_input_copies"] += 1
                    profile["bytes_d2d_input"] += byte_count
                    self.allocation_stats["number_of_d2d_input_copies"] += 1
                    self.allocation_stats["total_bytes_d2d_input"] += byte_count
                started = time.perf_counter()
                self._bind(name, buffer)
                profile["bind_address_ms"] += self._elapsed_ms(started)

            for name in self.output_names():
                started = time.perf_counter()
                shape = tuple(int(value) for value in self.context.get_tensor_shape(name))
                profile["output_shape_query_ms"] += self._elapsed_ms(started)
                profile["output_shape_query_calls"] += 1
                self.allocation_stats["output_shape_query_call_count"] += 1
                if any(value < 0 for value in shape):
                    raise RuntimeError(f"unresolved TensorRT output shape:{name}:{shape}")
                profile["output_shapes"][name] = list(shape)
                dtype = trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
                count = math.prod(shape)
                previous = int(
                    self.allocation_stats["output_buffer_reallocation_count"]
                )
                buffer = self._ensure_buffer(
                    self.output_buffers,
                    name,
                    count,
                    dtype,
                    "output_buffer_reallocation_count",
                )
                if int(self.allocation_stats["output_buffer_reallocation_count"]) != previous:
                    profile["output_buffer_reallocated"] = True
                started = time.perf_counter()
                self._bind(name, buffer)
                profile["bind_address_ms"] += self._elapsed_ms(started)
                outputs[name] = buffer[:count].view(shape)

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record(self.stream)
            success = self.context.execute_async_v3(self.stream.cuda_stream)
            end_event.record(self.stream)
        if not success:
            raise RuntimeError(f"TensorRT execution failed for engine: {self.engine_path}")
        started = time.perf_counter()
        self.stream.synchronize()
        profile["synchronize_ms"] = self._elapsed_ms(started)
        profile["execute_async_ms"] = float(start_event.elapsed_time(end_event))
        profile["total_runner_ms"] = self._elapsed_ms(total_started)
        # Preserve the original public timing keys for callers outside the
        # unified search workers.  The formal workers use the decomposed keys
        # above, while the generic evaluator still consumes these aliases.
        profile["forward_latency_ms"] = profile["execute_async_ms"]
        profile["runner_wall_ms"] = profile["total_runner_ms"]
        profile["allocation_report"] = self.allocation_report()
        return outputs, profile


__all__ = ["TensorRTEngineRunner"]
