"""Forward-hook inventory for canonical weighted ONNX node naming."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator


@contextmanager
def capture_weighted_module_calls(
    model: Any,
) -> Iterator[list[dict[str, Any]]]:
    """Capture weighted calls without patching process-global exporter state."""

    import torch

    records: list[dict[str, Any]] = []
    handles = []
    counter = 0

    def register(module_path: str, module: Any) -> None:
        def hook(_module: Any, _inputs: tuple[Any, ...], _output: Any) -> None:
            nonlocal counter
            if isinstance(module, torch.nn.ConvTranspose2d):
                mapped = "ConvTranspose"
            elif isinstance(module, torch.nn.Conv2d):
                mapped = "Conv"
            else:
                mapped = "MatMul"
            weight = getattr(module, "weight", None)
            records.append(
                {
                    "module_path": module_path.removeprefix("model."),
                    "module_type": type(module).__name__,
                    "call_index": counter,
                    "mapped_onnx_op_type": mapped,
                    "weight_shape": (
                        list(weight.shape) if weight is not None else []
                    ),
                    "groups": int(getattr(module, "groups", 1) or 1),
                }
            )
            counter += 1

        handles.append(module.register_forward_hook(hook))

    weighted_types = (
        torch.nn.Conv2d,
        torch.nn.ConvTranspose2d,
        torch.nn.Linear,
    )
    for name, module in model.named_modules():
        if name and isinstance(module, weighted_types):
            register(str(name), module)
    try:
        yield records
    finally:
        for handle in handles:
            handle.remove()


__all__ = ["capture_weighted_module_calls"]
