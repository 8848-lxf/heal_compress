"""Model-agnostic runtime tensor-flow tracer.

Produces the canonical trace schema consumed by
:func:`heal_compress.tracer.op_graph.build_op_graph`::

    {"nodes": {name: {...}}, "edges": [{"src", "dst", "input_index", "kind"}]}

It installs forward hooks on every module and monkeypatches a set of torch
functions / tensor methods so that structural tensor operations (cat, add,
split, view, permute, interpolate, ...) become first-class nodes. Unlike
:class:`heal_compress.tracer.forward_wrapper.HealForwardWrapper`, it does not
require the HEAL export module, so it works for arbitrary ``nn.Module`` forwards
(including the toy models used in tests and any general LiDAR/camera backbone).
"""

from __future__ import annotations

import logging
import weakref
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _iter_tensors(obj: Any) -> Iterator[torch.Tensor]:
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_tensors(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _iter_tensors(value)


def _shape_of(t: torch.Tensor) -> List[int]:
    return [int(v) for v in t.shape]


class GenericTracer:
    """Records modules and structural tensor ops executed for a sample input.

    Args:
        model: The model to trace (not mutated).
        forward_fn: ``(model, sample) -> output``. Defaults to ``model(sample)``.
    """

    TORCH_FUNCTIONS: Tuple[Tuple[str, Any, str], ...] = (
        ("torch.cat", torch, "cat"),
        ("torch.stack", torch, "stack"),
        ("torch.add", torch, "add"),
        ("torch.relu", torch, "relu"),
        ("torch.sigmoid", torch, "sigmoid"),
        ("torch.softmax", torch, "softmax"),
        ("torch.where", torch, "where"),
        ("torch.sum", torch, "sum"),
        ("torch.mean", torch, "mean"),
        ("torch.split", torch, "split"),
        ("torch.chunk", torch, "chunk"),
        ("torch.tensor_split", torch, "tensor_split"),
        ("torch.bmm", torch, "bmm"),
        ("torch.matmul", torch, "matmul"),
        ("F.relu", F, "relu"),
        ("F.interpolate", F, "interpolate"),
        ("F.grid_sample", F, "grid_sample"),
    )
    TENSOR_METHODS: Tuple[str, ...] = (
        "view", "reshape", "permute", "transpose", "flatten",
        "unsqueeze", "squeeze", "expand", "repeat", "contiguous",
        "sum", "mean", "__getitem__",
        "split", "chunk",
        "__add__", "__radd__", "__iadd__", "__mul__", "__rmul__",
    )

    def __init__(
        self,
        model: nn.Module,
        forward_fn: Optional[Callable[[nn.Module, Any], Any]] = None,
    ):
        self.model = model
        self.forward_fn = forward_fn
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Dict[str, Any]] = []
        self.tensor_producers: Dict[int, Tuple[weakref.ReferenceType[torch.Tensor], str]] = {}
        self.module_stack: List[str] = []
        self._op_index = 0
        self._handles: List[Any] = []

    def trace(self, sample: Any) -> Dict[str, Any]:
        """Run one forward pass and return the trace graph dict."""
        self.nodes = {}
        self.edges = []
        self.tensor_producers = {}
        self.module_stack = []
        self._op_index = 0
        self._install_hooks()
        try:
            with torch.no_grad(), self._patched_ops():
                if self.forward_fn is not None:
                    self.forward_fn(self.model, sample)
                else:
                    self.model(sample)
        finally:
            self._remove_hooks()
        return {"nodes": self.nodes, "edges": self.edges}

    # -- hooks -------------------------------------------------------------- #
    def _install_hooks(self) -> None:
        for name, module in self.model.named_modules():
            if not name:
                continue
            self._add_module_node(name, module)
            # Hook leaf modules only. Container modules such as Sequential or a
            # whole residual block otherwise overwrite the tensor producer of
            # their last leaf output, hiding the real channel-carrying op from
            # Add/Cat dependency propagation.
            if any(module.children()):
                continue
            # Shared activation modules (HEAL Bottleneck reuses one in-place
            # ReLU three times) do not define a channel dimension. Hooking them
            # under one module name merges unrelated call sites and corrupts
            # producer tracking around grouped bottlenecks.
            if isinstance(module, (nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.GELU, nn.SiLU, nn.ELU)):
                continue
            self._handles.append(module.register_forward_pre_hook(self._pre_hook(name)))
            self._handles.append(module.register_forward_hook(self._post_hook(name)))

    def _remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def _add_module_node(self, name: str, module: nn.Module) -> None:
        if name in self.nodes:
            return
        info: Dict[str, Any] = {
            "type": module.__class__.__name__,
            "executed": False,
            "input_shapes": [],
            "output_shapes": [],
        }
        for attr in ("in_channels", "out_channels", "num_features",
                     "in_features", "out_features", "groups"):
            if hasattr(module, attr):
                try:
                    info[attr] = int(getattr(module, attr))
                except (TypeError, ValueError):
                    pass
        if isinstance(module, nn.MultiheadAttention):
            info["embed_dim"] = int(module.embed_dim)
            info["num_heads"] = int(module.num_heads)
        self.nodes[name] = info

    def _pre_hook(self, name: str) -> Callable:
        def hook(_m: nn.Module, inputs: tuple) -> None:
            self.nodes[name]["input_shapes"] = [_shape_of(tensor) for tensor in _iter_tensors(inputs)]
            for idx, tensor in enumerate(_iter_tensors(inputs)):
                producer = self._producer_of(tensor)
                if producer:
                    self._add_edge(producer, name, idx, "module_input")
            self.module_stack.append(name)
        return hook

    def _post_hook(self, name: str) -> Callable:
        def hook(_m: nn.Module, _inp: tuple, output: Any) -> None:
            self.nodes[name]["executed"] = True
            self.nodes[name]["output_shapes"] = [_shape_of(tensor) for tensor in _iter_tensors(output)]
            for tensor in _iter_tensors(output):
                self._set_producer(tensor, name)
            if self.module_stack and self.module_stack[-1] == name:
                self.module_stack.pop()
            elif name in self.module_stack:
                self.module_stack.remove(name)
        return hook

    # -- op patching -------------------------------------------------------- #
    @contextmanager
    def _patched_ops(self) -> Iterator[None]:
        patches: List[Tuple[Any, str, Any]] = []

        def patch(obj: Any, attr: str, replacement: Any) -> None:
            patches.append((obj, attr, getattr(obj, attr)))
            setattr(obj, attr, replacement)

        for op_name, obj, attr in self.TORCH_FUNCTIONS:
            original = getattr(obj, attr, None)
            if original is None:
                continue

            def make(name: str, func: Callable) -> Callable:
                def wrapper(*args: Any, **kwargs: Any) -> Any:
                    result = func(*args, **kwargs)
                    self._record_op(name, args, kwargs, result)
                    return result
                return wrapper

            patch(obj, attr, make(op_name, original))

        for attr in self.TENSOR_METHODS:
            original = getattr(torch.Tensor, attr)

            def make_m(name: str, method: Callable) -> Callable:
                def wrapper(tself: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
                    result = method(tself, *args, **kwargs)
                    self._record_op(f"Tensor.{name}", (tself, *args), kwargs, result)
                    return result
                return wrapper

            patch(torch.Tensor, attr, make_m(attr, original))

        try:
            yield
        finally:
            for obj, attr, original in reversed(patches):
                setattr(obj, attr, original)

    def _record_op(self, op_name: str, args: tuple, kwargs: dict, result: Any) -> None:
        outputs = list(_iter_tensors(result))
        if not outputs:
            return
        input_tensors = list(_iter_tensors(args)) + list(_iter_tensors(kwargs))
        node_name = f"op::{self._op_index:05d}::{op_name}"
        self._op_index += 1
        meta: Dict[str, Any] = {
            "type": "TensorOp",
            "op": op_name,
            "module_scope": list(self.module_stack),
            "input_shapes": [_shape_of(t) for t in input_tensors],
            "output_shapes": [_shape_of(t) for t in outputs],
        }
        if op_name == "torch.cat":
            dim = kwargs.get("dim", args[1] if len(args) > 1 else 0)
            if isinstance(dim, int) and dim < 0 and input_tensors:
                dim = dim + input_tensors[0].dim()
            meta["cat_dim"] = int(dim)
            meta["num_inputs"] = len(input_tensors)
        elif op_name in ("torch.split", "Tensor.split", "torch.chunk", "Tensor.chunk", "torch.tensor_split"):
            dim = kwargs.get("dim", -1)
            if op_name in ("torch.split", "Tensor.split"):
                dim = kwargs.get("dim", args[2] if len(args) > 2 else 0)
            elif op_name in ("torch.chunk", "Tensor.chunk"):
                dim = kwargs.get("dim", args[2] if len(args) > 2 else 0)
            elif op_name == "torch.tensor_split":
                dim = kwargs.get("dim", args[2] if len(args) > 2 else 0)
            if isinstance(dim, int) and dim < 0 and input_tensors:
                dim = dim + input_tensors[0].dim()
            meta["cat_dim"] = int(dim)
            meta["num_inputs"] = len(input_tensors)
        self.nodes[node_name] = meta
        for idx, tensor in enumerate(input_tensors):
            producer = self._producer_of(tensor)
            if producer:
                self._add_edge(producer, node_name, idx, "tensor_op_input")
        for tensor in outputs:
            self._set_producer(tensor, node_name)

    def _add_edge(self, src: str, dst: str, input_index: int, kind: str) -> None:
        self.edges.append({"src": src, "dst": dst, "input_index": input_index, "kind": kind})

    def _producer_of(self, tensor: torch.Tensor) -> Optional[str]:
        rec = self.tensor_producers.get(id(tensor))
        if rec is None:
            return None
        ref, producer = rec
        if ref() is tensor:
            return producer
        self.tensor_producers.pop(id(tensor), None)
        return None

    def _set_producer(self, tensor: torch.Tensor, producer: str) -> None:
        self.tensor_producers[id(tensor)] = (weakref.ref(tensor), producer)


def trace_model(
    model: nn.Module,
    sample: Any,
    forward_fn: Optional[Callable[[nn.Module, Any], Any]] = None,
) -> Dict[str, Any]:
    """Convenience wrapper: trace ``model`` on ``sample`` and return the graph."""
    return GenericTracer(model, forward_fn=forward_fn).trace(sample)
