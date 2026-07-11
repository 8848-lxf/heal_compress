"""Representative-forward module call tracing without global monkeypatching."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping

import torch
import torch.nn as nn

from .types import ExampleInputContract, ExampleInputTensor, ModuleCallRecord


WEIGHTED_MODULE_TYPES = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.Linear,
)


def is_weighted_module(module: nn.Module) -> bool:
    """Return whether ``module`` has a supported weighted-layer identity."""

    return isinstance(module, WEIGHTED_MODULE_TYPES)


def _iter_tensors(value: Any) -> Iterator[torch.Tensor]:
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _iter_tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _iter_tensors(child)


def tensor_shapes(value: Any) -> list[tuple[int, ...]]:
    """Return tensor shapes in deterministic container traversal order."""

    return [tuple(int(dim) for dim in tensor.shape) for tensor in _iter_tensors(value)]


def invoke_model(
    model: nn.Module,
    example_inputs: Any,
    *,
    call_style: str = "auto",
    forward_fn: Callable[[nn.Module, Any], Any] | None = None,
) -> Any:
    """Invoke a representative forward using an explicit, serializable style."""

    if forward_fn is not None:
        return forward_fn(model, example_inputs)
    style = call_style
    if style == "auto":
        style = "args" if isinstance(example_inputs, tuple) else "single"
    if style == "single":
        return model(example_inputs)
    if style == "args":
        if not isinstance(example_inputs, (list, tuple)):
            raise TypeError("input_call_style='args' requires a list or tuple")
        return model(*tuple(example_inputs))
    if style == "kwargs":
        if not isinstance(example_inputs, Mapping):
            raise TypeError("input_call_style='kwargs' requires a mapping")
        return model(**dict(example_inputs))
    raise ValueError(f"unsupported input call style: {call_style}")


def _walk_contract(value: Any, path: str, tensors: list[ExampleInputTensor], non_tensors: list[str]) -> None:
    if torch.is_tensor(value):
        tensors.append(
            ExampleInputTensor(
                path=path,
                shape=tuple(int(dim) for dim in value.shape),
                dtype=str(value.dtype).replace("torch.", ""),
                device_type=value.device.type,
            )
        )
        return
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            _walk_contract(value[key], f"{path}.{key}", tensors, non_tensors)
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _walk_contract(child, f"{path}[{index}]", tensors, non_tensors)
        return
    non_tensors.append(path)


def build_example_input_contract(example_inputs: Any, *, call_style: str) -> ExampleInputContract:
    """Describe only shapes/dtypes/devices, never example tensor contents."""

    tensors: list[ExampleInputTensor] = []
    non_tensors: list[str] = []
    _walk_contract(example_inputs, "input", tensors, non_tensors)
    resolved_style = call_style
    if resolved_style == "auto":
        resolved_style = "args" if isinstance(example_inputs, tuple) else "single"
    return ExampleInputContract(
        call_style=resolved_style,
        tensors=tensors,
        non_tensor_paths=sorted(non_tensors),
    )


@dataclass
class ModuleCallTraceResult:
    """Raw result from one representative forward."""

    records: list[ModuleCallRecord]
    weighted_modules_total: int
    weighted_modules_called: int


def trace_module_calls(
    model: nn.Module,
    example_inputs: Any,
    *,
    call_style: str = "auto",
    forward_fn: Callable[[nn.Module, Any], Any] | None = None,
    include_non_weighted_leaf_calls: bool = True,
) -> ModuleCallTraceResult:
    """Execute one forward and record repeated module calls in call order.

    Hooks are removed in ``finally`` and the model's train/eval state is not
    modified. Per-module and global call indices make shared-module calls
    distinguishable without embedding runtime objects in the result.
    """

    modules = dict(model.named_modules())
    names_by_id = {id(module): name for name, module in modules.items() if name}
    weighted_names = {
        name for name, module in modules.items() if name and is_weighted_module(module)
    }
    per_module_count: dict[str, int] = defaultdict(int)
    pending: dict[int, list[ModuleCallRecord]] = defaultdict(list)
    records: list[ModuleCallRecord] = []
    handles: list[Any] = []

    def pre_hook(module: nn.Module, inputs: tuple[Any, ...]) -> None:
        name = names_by_id[id(module)]
        record = ModuleCallRecord(
            module_path=name,
            module_type=module.__class__.__name__,
            call_index=len(records),
            module_call_index=per_module_count[name],
            weighted=is_weighted_module(module),
            input_shapes=tensor_shapes(inputs),
        )
        per_module_count[name] += 1
        records.append(record)
        pending[id(module)].append(record)

    def post_hook(module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        stack = pending.get(id(module), [])
        if stack:
            stack.pop().output_shapes = tensor_shapes(output)

    for name, module in modules.items():
        if not name or any(module.children()):
            continue
        if not include_non_weighted_leaf_calls and not is_weighted_module(module):
            continue
        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(post_hook))
    try:
        with torch.no_grad():
            invoke_model(
                model,
                example_inputs,
                call_style=call_style,
                forward_fn=forward_fn,
            )
    finally:
        for handle in handles:
            handle.remove()

    called_weighted = {record.module_path for record in records if record.weighted}
    return ModuleCallTraceResult(
        records=records,
        weighted_modules_total=len(weighted_names),
        weighted_modules_called=len(called_weighted),
    )

