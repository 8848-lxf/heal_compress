"""Physical channel-slicing primitives and the pruning-fn dispatch table.

Each ``pruning_fn`` is a callable ``(module, keep_indices) -> dict`` that
physically removes channels from a single module. Every function also exposes a
``.supports(module) -> bool`` predicate so that :meth:`PruningGroup.prune` can
validate an entire group *before* mutating any module (atomic pruning,
requirement #6).

The primitives intentionally mirror the proven slicing logic from the
model-specific backend, but here they are exposed as reusable, model-agnostic
operations keyed by (module-type, direction).
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn

PruningFn = Callable[[nn.Module, list[int]], dict[str, Any]]


def _index(indices: list[int], device: torch.device) -> torch.Tensor:
    return torch.as_tensor(indices, dtype=torch.long, device=device)


def _supports(*types: type) -> Callable[[nn.Module], bool]:
    def predicate(module: nn.Module) -> bool:
        return isinstance(module, types)
    return predicate


def _tag(fn: PruningFn, name: str, supports: Callable[[nn.Module], bool]) -> PruningFn:
    fn.__name__ = name  # type: ignore[attr-defined]
    fn.supports = supports  # type: ignore[attr-defined]
    return fn


# --------------------------------------------------------------------------- #
# Conv2d
# --------------------------------------------------------------------------- #
def _prune_conv_out(module: nn.Conv2d, keep: list[int]) -> dict[str, Any]:
    """Slice Conv2d output channels (axis 0). Handles depthwise sync."""
    before = module.out_channels
    idx = _index(keep, module.weight.device)
    if module.groups > 1:
        # Depthwise (groups == in == out): output slice must drive groups.
        if module.groups == before and module.in_channels == before:
            module.weight = nn.Parameter(module.weight.data.index_select(0, idx).clone())
            if module.bias is not None:
                module.bias = nn.Parameter(module.bias.data.index_select(0, idx).clone())
            module.out_channels = len(keep)
            module.in_channels = len(keep)
            module.groups = len(keep)
            return {"axis": "out", "before": before, "after": len(keep), "mode": "depthwise"}
        raise ValueError(
            f"grouped Conv2d out-slice needs explicit grouped handler "
            f"(groups={module.groups}, in={module.in_channels}, out={before})"
        )
    module.weight = nn.Parameter(module.weight.data.index_select(0, idx).clone())
    if module.bias is not None:
        module.bias = nn.Parameter(module.bias.data.index_select(0, idx).clone())
    module.out_channels = len(keep)
    return {"axis": "out", "before": before, "after": len(keep)}


def _prune_conv_in(module: nn.Conv2d, keep: list[int]) -> dict[str, Any]:
    """Slice Conv2d input channels (axis 1). Only standard (groups==1) convs."""
    if module.groups != 1:
        raise ValueError(f"grouped Conv2d in-slice unsupported (groups={module.groups})")
    before = module.in_channels
    idx = _index(keep, module.weight.device)
    module.weight = nn.Parameter(module.weight.data.index_select(1, idx).clone())
    module.in_channels = len(keep)
    return {"axis": "in", "before": before, "after": len(keep)}


def _conv_out_supports(module: nn.Module) -> bool:
    if not isinstance(module, nn.Conv2d):
        return False
    if module.groups == 1:
        return True
    # Depthwise is supported; other grouped convs need the grouped handler.
    return module.groups == module.out_channels == module.in_channels


def _conv_in_supports(module: nn.Module) -> bool:
    return isinstance(module, nn.Conv2d) and module.groups == 1


prune_conv_out = _tag(_prune_conv_out, "prune_conv_out", _conv_out_supports)
prune_conv_in = _tag(_prune_conv_in, "prune_conv_in", _conv_in_supports)


# --------------------------------------------------------------------------- #
# ConvTranspose2d  (weight shape: [in, out/groups, kH, kW])
# --------------------------------------------------------------------------- #
def _prune_convtranspose_out(module: nn.ConvTranspose2d, keep: list[int]) -> dict[str, Any]:
    if module.groups != 1:
        raise ValueError(f"grouped ConvTranspose2d out-slice unsupported (groups={module.groups})")
    before = module.out_channels
    idx = _index(keep, module.weight.device)
    # out channels live on axis 1 for ConvTranspose2d.
    module.weight = nn.Parameter(module.weight.data.index_select(1, idx).clone())
    if module.bias is not None:
        module.bias = nn.Parameter(module.bias.data.index_select(0, idx).clone())
    module.out_channels = len(keep)
    return {"axis": "out", "before": before, "after": len(keep)}


def _prune_convtranspose_in(module: nn.ConvTranspose2d, keep: list[int]) -> dict[str, Any]:
    if module.groups != 1:
        raise ValueError(f"grouped ConvTranspose2d in-slice unsupported (groups={module.groups})")
    before = module.in_channels
    idx = _index(keep, module.weight.device)
    # in channels live on axis 0 for ConvTranspose2d.
    module.weight = nn.Parameter(module.weight.data.index_select(0, idx).clone())
    module.in_channels = len(keep)
    return {"axis": "in", "before": before, "after": len(keep)}


def _convt_supports(module: nn.Module) -> bool:
    return isinstance(module, nn.ConvTranspose2d) and module.groups == 1


prune_convtranspose_out = _tag(_prune_convtranspose_out, "prune_convtranspose_out", _convt_supports)
prune_convtranspose_in = _tag(_prune_convtranspose_in, "prune_convtranspose_in", _convt_supports)


# --------------------------------------------------------------------------- #
# BatchNorm
# --------------------------------------------------------------------------- #
def _prune_bn(module: nn.modules.batchnorm._BatchNorm, keep: list[int]) -> dict[str, Any]:
    before = module.num_features
    idx = _index(keep, module.weight.device)
    if module.weight is not None:
        module.weight = nn.Parameter(module.weight.data.index_select(0, idx).clone())
    if module.bias is not None:
        module.bias = nn.Parameter(module.bias.data.index_select(0, idx).clone())
    if module.running_mean is not None:
        module.running_mean = module.running_mean.index_select(0, idx).clone()
    if module.running_var is not None:
        module.running_var = module.running_var.index_select(0, idx).clone()
    module.num_features = len(keep)
    return {"axis": "bn", "before": before, "after": len(keep)}


prune_bn = _tag(_prune_bn, "prune_bn", _supports(nn.modules.batchnorm._BatchNorm))


# --------------------------------------------------------------------------- #
# Linear
# --------------------------------------------------------------------------- #
def _prune_linear_out(module: nn.Linear, keep: list[int]) -> dict[str, Any]:
    before = module.out_features
    idx = _index(keep, module.weight.device)
    module.weight = nn.Parameter(module.weight.data.index_select(0, idx).clone())
    if module.bias is not None:
        module.bias = nn.Parameter(module.bias.data.index_select(0, idx).clone())
    module.out_features = len(keep)
    return {"axis": "out", "before": before, "after": len(keep)}


def _prune_linear_in(module: nn.Linear, keep: list[int]) -> dict[str, Any]:
    before = module.in_features
    idx = _index(keep, module.weight.device)
    module.weight = nn.Parameter(module.weight.data.index_select(1, idx).clone())
    module.in_features = len(keep)
    return {"axis": "in", "before": before, "after": len(keep)}


prune_linear_out = _tag(_prune_linear_out, "prune_linear_out", _supports(nn.Linear))
prune_linear_in = _tag(_prune_linear_in, "prune_linear_in", _supports(nn.Linear))


# --------------------------------------------------------------------------- #
# LayerNorm  (1-D normalized_shape only)
# --------------------------------------------------------------------------- #
def _prune_layernorm(module: nn.LayerNorm, keep: list[int]) -> dict[str, Any]:
    ns = module.normalized_shape
    if isinstance(ns, int):
        ns = (ns,)
    if len(ns) != 1:
        raise ValueError(f"multi-dim LayerNorm slice unsupported: normalized_shape={ns}")
    before = int(ns[0])
    idx = _index(keep, module.weight.device)
    if module.elementwise_affine:
        module.weight = nn.Parameter(module.weight.data.index_select(0, idx).clone())
        module.bias = nn.Parameter(module.bias.data.index_select(0, idx).clone())
    module.normalized_shape = (len(keep),)
    return {"axis": "norm", "before": before, "after": len(keep)}


def _layernorm_supports(module: nn.Module) -> bool:
    if not isinstance(module, nn.LayerNorm):
        return False
    ns = module.normalized_shape
    return (isinstance(ns, int)) or (len(ns) == 1)


prune_layernorm = _tag(_prune_layernorm, "prune_layernorm", _layernorm_supports)


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
def get_pruning_fn(module: nn.Module, direction: str) -> PruningFn | None:
    """Return the pruning_fn for ``(module, direction)`` or ``None`` if none.

    ``None`` means the op cannot be sliced on that axis by the general backend;
    callers should treat the enclosing group as protected.
    """
    if isinstance(module, nn.Conv2d):
        return prune_conv_out if direction == "out" else prune_conv_in
    if isinstance(module, nn.ConvTranspose2d):
        return prune_convtranspose_out if direction == "out" else prune_convtranspose_in
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        return prune_bn  # BN only has one axis
    if isinstance(module, nn.Linear):
        return prune_linear_out if direction == "out" else prune_linear_in
    if isinstance(module, nn.LayerNorm):
        return prune_layernorm
    return None


def is_prunable_module(module: nn.Module, direction: str) -> bool:
    """Whether ``(module, direction)`` can be physically sliced by this backend."""
    fn = get_pruning_fn(module, direction)
    return fn is not None and getattr(fn, "supports")(module)
