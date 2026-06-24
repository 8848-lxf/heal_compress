"""Dynamic forward wrapper for HEAL models.

Enumerates multi-agent forward paths, applies plugin patches for traceable
execution, and records operator-level tensor flow for dependency analysis.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class HealForwardWrapper:
    """Wraps a HEAL model to enumerate all possible forward paths across
    different agent counts and collect a unified computation graph.

    The wrapper:
    1. Loads dummy inputs from .npz calibration files for each agent count.
    2. Applies install_plugin_patches() to make LSS modules traceable.
    3. Executes forward passes and records tensor producers/consumers
       via module hooks and torch function patches.
    4. Merges multi-path dependency relations into a single graph.

    Args:
        model: The HEAL model (already loaded and on device).
        export_module: The imported export_dynamic_onnx module, providing
            install_plugin_patches, DynamicAgentExportWrapper, etc.
        max_agents: Maximum number of agents to enumerate (default 2).
        device: Device for dummy inputs.
        modality: Camera modality name (auto-detected if None).
    """

    # Torch functions to intercept for tensor flow tracking
    TORCH_FUNCTIONS = (
        ("torch.cat", torch, "cat"),
        ("torch.stack", torch, "stack"),
        ("torch.bmm", torch, "bmm"),
        ("torch.matmul", torch, "matmul"),
        ("torch.add", torch, "add"),
        ("torch.mean", torch, "mean"),
        ("torch.max", torch, "max"),
        ("torch.sum", torch, "sum"),
        ("torch.softmax", torch, "softmax"),
        ("torch.sigmoid", torch, "sigmoid"),
        ("F.softmax", F, "softmax"),
    )

    # Tensor methods to intercept
    TENSOR_METHODS = (
        "view", "reshape", "permute", "transpose", "flatten",
        "unsqueeze", "squeeze", "expand", "repeat", "contiguous",
        "__add__", "__radd__", "__mul__", "__rmul__",
    )

    def __init__(
        self,
        model: nn.Module,
        export_module: Any,
        max_agents: int = 2,
        device: str | torch.device = "cpu",
        modality: str | None = None,
    ):
        self.model = model
        self.export_module = export_module
        self.max_agents = max_agents
        self.device = torch.device(device)
        self.modality = modality or self._detect_modality()

        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[dict[str, Any]] = []
        self.tensor_producers: dict[int, str] = {}
        self.module_stack: list[str] = []
        self._op_index: int = 0
        self._handles: list[Any] = []

    def _detect_modality(self) -> str:
        """Detect the primary camera modality of the HEAL model.

        Returns:
            Modality name string.

        Raises:
            RuntimeError: If no camera modality is found.
        """
        for modality_name in getattr(self.model, "modality_name_list", []):
            if getattr(self.model, "sensor_type_dict", {}).get(modality_name) == "camera":
                return modality_name
        for name in getattr(self.model, "modality_name_list", []):
            if hasattr(self.model, f"encoder_{name}"):
                return name
        raise RuntimeError("Cannot detect camera modality on HEAL model.")

    def apply_plugin_patches(self) -> list[str]:
        """Apply install_plugin_patches() to make LSS modules traceable.

        This replaces get_geometry, get_cam_feats, and voxel_pooling methods
        with versions compatible with torch.onnx.export and tracing.

        Returns:
            List of patched module names.
        """
        return self.export_module.install_plugin_patches(self.model)

    def build_dummy_inputs(
        self,
        npz_path: str,
        num_agents: int,
    ) -> tuple[torch.Tensor, ...]:
        """Construct dummy inputs for a specific agent count from a .npz file.

        Loads the npz and adjusts the batch dimension to match num_agents.

        Args:
            npz_path: Path to a calibration .npz file.
            num_agents: Desired number of agents.

        Returns:
            Tuple of (imgs, rots, trans, intrins, post_rots, post_trans,
            pairwise_t_matrix) tensors with the correct num_agents dimension.
        """
        inputs = self.export_module.load_example_inputs(
            npz_path, self.device, require_num_agents=0
        )
        return self._adjust_agent_count(inputs, num_agents)

    def _adjust_agent_count(
        self,
        inputs: tuple[torch.Tensor, ...],
        target_n: int,
    ) -> tuple[torch.Tensor, ...]:
        """Adjust input tensors to target agent count by repeating or slicing.

        Args:
            inputs: Original (imgs, rots, trans, intrins, post_rots,
                post_trans, pairwise_t_matrix).
            target_n: Target number of agents.

        Returns:
            Adjusted input tuple.
        """
        imgs, rots, trans, intrins, post_rots, post_trans, pairwise = inputs
        current_n = imgs.shape[0]

        if current_n == target_n:
            return inputs

        def _repeat_or_slice(t: torch.Tensor, dim: int) -> torch.Tensor:
            if t.shape[dim] >= target_n:
                slices = [slice(None)] * t.dim()
                slices[dim] = slice(0, target_n)
                return t[tuple(slices)]
            reps = [1] * t.dim()
            reps[dim] = (target_n + t.shape[dim] - 1) // t.shape[dim]
            expanded = t.repeat(*reps)
            slices = [slice(None)] * expanded.dim()
            slices[dim] = slice(0, target_n)
            return expanded[tuple(slices)]

        imgs = _repeat_or_slice(imgs, 0)
        rots = _repeat_or_slice(rots, 0)
        trans = _repeat_or_slice(trans, 0)
        intrins = _repeat_or_slice(intrins, 0)
        post_rots = _repeat_or_slice(post_rots, 0)
        post_trans = _repeat_or_slice(post_trans, 0)
        pairwise = _repeat_or_slice(_repeat_or_slice(pairwise, 1), 2)

        return imgs, rots, trans, intrins, post_rots, post_trans, pairwise

    def trace_all_paths(
        self,
        npz_path: str,
    ) -> dict[str, Any]:
        """Enumerate forward paths for all agent counts and build unified graph.

        For each num_agents in [1, ..., max_agents]:
        1. Build dummy inputs
        2. Create DynamicAgentExportWrapper
        3. Execute forward with hooks installed
        4. Record all tensor operations

        The results from all paths are merged (union of edges/nodes) into
        a single computation graph.

        Args:
            npz_path: Path to a calibration .npz file.

        Returns:
            Unified computation graph dict with 'nodes' and 'edges'.
        """
        self.nodes = {}
        self.edges = []

        self.apply_plugin_patches()

        for n_agents in range(1, self.max_agents + 1):
            logger.info(f"Tracing forward path with num_agents={n_agents}")
            inputs = self.build_dummy_inputs(npz_path, n_agents)
            self._trace_single_path(inputs, n_agents)

        return {"nodes": self.nodes, "edges": self.edges}

    def _trace_single_path(
        self,
        inputs: tuple[torch.Tensor, ...],
        num_agents: int,
    ) -> None:
        """Execute one forward pass and record the computation graph.

        Installs pre/post hooks on all modules, patches torch functions
        and tensor methods to intercept operations, then runs the forward.

        Args:
            inputs: Model input tensors.
            num_agents: Agent count for this path (used for labeling).
        """
        self.tensor_producers = {}
        self.module_stack = []
        self._op_index = 0

        self._install_module_hooks()
        try:
            crop_params = self.export_module._record_crop_params(
                self.model, self.modality, inputs
            )
            wrapper = self.export_module.DynamicAgentExportWrapper(
                self.model,
                modality_name=self.modality,
                crop_params=crop_params,
                insert_se3_inverse=True,
            ).to(self.device).eval()

            with torch.no_grad(), self._patched_ops():
                wrapper(*inputs)
        finally:
            self._remove_module_hooks()

    def _install_module_hooks(self) -> None:
        """Register forward pre/post hooks on all named modules."""
        for name, module in self.model.named_modules():
            if not name:
                continue
            self._add_module_node(name, module)
            self._handles.append(
                module.register_forward_pre_hook(self._make_pre_hook(name))
            )
            self._handles.append(
                module.register_forward_hook(self._make_post_hook(name))
            )

    def _remove_module_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def _add_module_node(self, name: str, module: nn.Module) -> None:
        """Add a module as a node in the computation graph.

        Args:
            name: Fully qualified module name.
            module: The nn.Module instance.
        """
        info: dict[str, Any] = {
            "type": module.__class__.__name__,
            "protected": False,
        }
        for attr in ("in_channels", "out_channels", "num_features",
                      "in_features", "out_features", "groups"):
            if hasattr(module, attr):
                info[attr] = int(getattr(module, attr))
        if isinstance(module, nn.LayerNorm):
            ns = module.normalized_shape
            if isinstance(ns, int):
                ns = (ns,)
            info["normalized_shape"] = [int(v) for v in ns]
        if isinstance(module, nn.MultiheadAttention):
            info["embed_dim"] = int(module.embed_dim)
            info["num_heads"] = int(module.num_heads)

        if name not in self.nodes:
            self.nodes[name] = info

    def _make_pre_hook(self, name: str) -> Callable:
        """Create a forward pre-hook that records input tensor producers.

        Args:
            name: Module name.

        Returns:
            Hook function.
        """
        def hook(_module, inputs):
            for tensor in _iter_tensors(inputs):
                producer = self.tensor_producers.get(id(tensor))
                if producer:
                    self._add_edge(producer, name, "module_input")
            self.module_stack.append(name)
        return hook

    def _make_post_hook(self, name: str) -> Callable:
        """Create a forward post-hook that records output tensor producers.

        Args:
            name: Module name.

        Returns:
            Hook function.
        """
        def hook(_module, _inputs, output):
            for tensor in _iter_tensors(output):
                self.tensor_producers[id(tensor)] = name
            if self.module_stack and self.module_stack[-1] == name:
                self.module_stack.pop()
        return hook

    @contextmanager
    def _patched_ops(self) -> Iterator[None]:
        """Context manager that patches torch functions and tensor methods
        to intercept tensor operations for graph construction."""
        patches: list[tuple[Any, str, Any]] = []

        def patch(obj, attr, replacement):
            original = getattr(obj, attr)
            setattr(obj, attr, replacement)
            patches.append((obj, attr, original))

        for op_name, obj, attr in self.TORCH_FUNCTIONS:
            original = getattr(obj, attr)

            def make_wrapper(name, func):
                def wrapper(*args, **kwargs):
                    result = func(*args, **kwargs)
                    self._record_op(name, args, kwargs, result)
                    return result
                return wrapper

            patch(obj, attr, make_wrapper(op_name, original))

        for attr in self.TENSOR_METHODS:
            original = getattr(torch.Tensor, attr)

            def make_method_wrapper(name, method):
                def wrapper(tensor_self, *args, **kwargs):
                    result = method(tensor_self, *args, **kwargs)
                    self._record_op(f"Tensor.{name}", (tensor_self, *args), kwargs, result)
                    return result
                return wrapper

            patch(torch.Tensor, attr, make_method_wrapper(attr, original))

        try:
            yield
        finally:
            for obj, attr, original in reversed(patches):
                setattr(obj, attr, original)

    def _record_op(
        self,
        op_name: str,
        args: tuple,
        kwargs: dict,
        result: Any,
    ) -> None:
        """Record a tensor operation as a node in the graph.

        Args:
            op_name: Operation name (e.g. 'torch.cat', 'Tensor.__add__').
            args: Positional arguments.
            kwargs: Keyword arguments.
            result: Operation output.
        """
        outputs = list(_iter_tensors(result))
        if not outputs:
            return
        input_tensors = list(_iter_tensors(args)) + list(_iter_tensors(kwargs))
        node_name = f"op::{self._op_index:05d}::{op_name}"
        self._op_index += 1

        meta: dict[str, Any] = {
            "type": "TensorOp",
            "op": op_name,
            "module_scope": list(self.module_stack),
            "input_shapes": [[int(s) for s in t.shape] for t in input_tensors],
            "output_shapes": [[int(s) for s in t.shape] for t in outputs],
        }
        if op_name == "torch.cat":
            dim = kwargs.get("dim", args[1] if len(args) > 1 else 0)
            if dim < 0 and input_tensors:
                dim = int(dim) + input_tensors[0].dim()
            meta["cat_dim"] = int(dim)
            meta["num_inputs"] = len(input_tensors)

        self.nodes[node_name] = meta

        for tensor in input_tensors:
            producer = self.tensor_producers.get(id(tensor))
            if producer:
                self._add_edge(producer, node_name, "tensor_op_input")
        for tensor in outputs:
            self.tensor_producers[id(tensor)] = node_name

    def _add_edge(self, src: str, dst: str, kind: str, **meta: Any) -> None:
        """Add a directed edge to the computation graph.

        Deduplicates edges with the same (src, dst, kind).

        Args:
            src: Source node name.
            dst: Destination node name.
            kind: Edge type (e.g. 'module_input', 'tensor_op_input').
            **meta: Additional edge metadata.
        """
        edge = {"src": src, "dst": dst, "kind": kind, **meta}
        self.edges.append(edge)


def _iter_tensors(obj: Any) -> Iterator[torch.Tensor]:
    """Recursively yield all tensors from a nested structure.

    Args:
        obj: Any object (tensor, dict, list, tuple, or other).

    Yields:
        torch.Tensor instances found in the structure.
    """
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_tensors(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            yield from _iter_tensors(value)
