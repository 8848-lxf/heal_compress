"""Runtime H/W shape profiling for weighted modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class RuntimeLayerShape:
    module_path: str
    call_index: int
    module_type: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    c_in: int | None
    c_out: int | None
    h_out: int
    w_out: int
    kernel_size: tuple[int, ...]
    stride: tuple[int, ...]
    padding: tuple[int, ...]
    dilation: tuple[int, ...]
    groups: int
    weight_shape: tuple[int, ...]
    precision_group_id: str = ""
    macs: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_path": self.module_path,
            "call_index": self.call_index,
            "module_type": self.module_type,
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "C_in": self.c_in,
            "C_out": self.c_out,
            "H_out": self.h_out,
            "W_out": self.w_out,
            "kernel_size": list(self.kernel_size),
            "stride": list(self.stride),
            "padding": list(self.padding),
            "dilation": list(self.dilation),
            "groups": self.groups,
            "weight_shape": list(self.weight_shape),
            "precision_group_id": self.precision_group_id,
            "MACs": self.macs,
        }


@dataclass(frozen=True)
class RuntimeShapeProfile:
    shapes: tuple[RuntimeLayerShape, ...]

    @property
    def by_module(self) -> dict[str, list[RuntimeLayerShape]]:
        result: dict[str, list[RuntimeLayerShape]] = {}
        for shape in self.shapes:
            result.setdefault(shape.module_path, []).append(shape)
        return result

    def to_dict(self) -> dict[str, Any]:
        return {"layers": [shape.to_dict() for shape in self.shapes]}


def _first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, dict):
        for item in value.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _call_model(model: nn.Module, example_inputs: Any, forward_fn: Any | None = None) -> Any:
    if forward_fn is not None:
        return forward_fn(model, example_inputs)
    if isinstance(example_inputs, tuple):
        return model(*example_inputs)
    if isinstance(example_inputs, dict):
        return model(**example_inputs)
    return model(example_inputs)


def _macs(module: nn.Module, output_shape: tuple[int, ...]) -> float:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        h = int(output_shape[-2]) if len(output_shape) >= 3 else 1
        w = int(output_shape[-1]) if len(output_shape) >= 4 else 1
        kh, kw = tuple(int(v) for v in module.kernel_size)
        return float(h * w * kh * kw * int(module.in_channels) * int(module.out_channels) / max(int(module.groups), 1))
    if isinstance(module, nn.Linear):
        return float(int(module.in_features) * int(module.out_features))
    return 0.0


def profile_runtime_layer_shapes(model: nn.Module, example_inputs: Any, *, forward_fn: Any | None = None) -> RuntimeShapeProfile:
    calls: dict[str, int] = {}
    rows: list[RuntimeLayerShape] = []
    handles = []

    def hook(name: str, module: nn.Module):
        def capture(_module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            input_tensor = _first_tensor(inputs)
            output_tensor = _first_tensor(output)
            if input_tensor is None or output_tensor is None:
                return
            call_index = calls.get(name, 0)
            calls[name] = call_index + 1
            input_shape = tuple(int(v) for v in input_tensor.shape)
            output_shape = tuple(int(v) for v in output_tensor.shape)
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                c_in = int(module.in_channels)
                c_out = int(module.out_channels)
                kernel = tuple(int(v) for v in module.kernel_size)
                stride = tuple(int(v) for v in module.stride)
                padding = tuple(int(v) for v in module.padding)
                dilation = tuple(int(v) for v in module.dilation)
                groups = int(module.groups)
                h_out = int(output_shape[-2]) if len(output_shape) >= 3 else 1
                w_out = int(output_shape[-1]) if len(output_shape) >= 4 else 1
            elif isinstance(module, nn.Linear):
                c_in = int(module.in_features)
                c_out = int(module.out_features)
                kernel = stride = padding = dilation = ()
                groups = 1
                h_out = 1
                w_out = 1
            else:
                return
            rows.append(
                RuntimeLayerShape(
                    module_path=name,
                    call_index=call_index,
                    module_type=type(module).__name__,
                    input_shape=input_shape,
                    output_shape=output_shape,
                    c_in=c_in,
                    c_out=c_out,
                    h_out=h_out,
                    w_out=w_out,
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=groups,
                    weight_shape=tuple(int(v) for v in module.weight.shape),
                    macs=_macs(module, output_shape),
                )
            )

        return capture

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            handles.append(module.register_forward_hook(hook(name, module)))
    try:
        with torch.no_grad():
            _call_model(model, example_inputs, forward_fn=forward_fn)
    finally:
        for handle in handles:
            handle.remove()
    return RuntimeShapeProfile(tuple(sorted(rows, key=lambda row: (row.module_path, row.call_index))))
