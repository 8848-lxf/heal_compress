"""Shape simulator for global one-shot physical pruning plans.

The simulator predicts module shapes from a ``GlobalPhysicalPrunePlan`` without
mutating the model.  It is intentionally conservative: fixed-shape contracts
such as PFN/scatter and unsupported ConvTranspose/deblock rewrites fail before
physical surgery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch.nn as nn

from .grouped_conv import resolve_grouped_conv_d_compact_frontfill_reblock, resolve_grouped_conv_input_keep


FIXED_SHAPE_KEYWORDS = (
    "pillar_vfe",
    "pfn_layers",
    "scatter",
    "voxel",
)
HEAD_KEYWORDS = ("cls_head", "reg_head", "dir_head")


@dataclass
class SimShape:
    module_name: str
    module_type: str
    in_channels_before: int | None = None
    out_channels_before: int | None = None
    groups_before: int | None = None
    in_channels_after: int | None = None
    out_channels_after: int | None = None
    groups_after: int | None = None
    weight_shape_before: list[int] | None = None
    weight_shape_after: list[int] | None = None
    bias_shape_before: list[int] | None = None
    bias_shape_after: list[int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_name": self.module_name,
            "module_type": self.module_type,
            "in_channels_before": self.in_channels_before,
            "out_channels_before": self.out_channels_before,
            "groups_before": self.groups_before,
            "in_channels_after": self.in_channels_after,
            "out_channels_after": self.out_channels_after,
            "groups_after": self.groups_after,
            "weight_shape_before": self.weight_shape_before,
            "weight_shape_after": self.weight_shape_after,
            "bias_shape_before": self.bias_shape_before,
            "bias_shape_after": self.bias_shape_after,
        }


@dataclass
class ShapeIssue:
    issue: str
    module_name: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"issue": self.issue, "module_name": self.module_name, **self.details}


def _shape_of(param: Any) -> list[int] | None:
    if param is None:
        return None
    return [int(v) for v in tuple(param.shape)]


def _axis_channels(module: nn.Module, axis: str) -> int:
    if isinstance(module, nn.Conv2d):
        if axis == "grouped_true_group_block":
            return int(module.groups)
        if axis == "grouped_d_compact_frontfill_reblock":
            return int(module.out_channels)
        return int(module.out_channels if axis in {"out", "grouped_coarsen_out"} else module.in_channels)
    if isinstance(module, nn.ConvTranspose2d):
        return int(module.out_channels if axis == "out" else module.in_channels)
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        return int(module.num_features)
    if isinstance(module, nn.Linear):
        return int(module.out_features if axis == "out" else module.in_features)
    raise TypeError(f"unsupported_module_for_shape_sim:{module.__class__.__name__}:{axis}")


def _request_dict(req: Any) -> dict[str, Any]:
    if isinstance(req, dict):
        return req
    return {
        "module_name": getattr(req, "module_name", ""),
        "axis": getattr(req, "axis", ""),
        "prune_indices": list(getattr(req, "prune_indices", [])),
        "source_recipe_ids": list(getattr(req, "source_recipe_ids", [])),
        "metadata": dict(getattr(req, "metadata", {}) or {}),
    }


def _is_fixed_shape_name(name: str) -> bool:
    low = name.lower()
    return any(key in low for key in FIXED_SHAPE_KEYWORDS)


def _is_head_name(name: str) -> bool:
    low = name.lower()
    return any(key in low for key in HEAD_KEYWORDS)


class GlobalPlanShapeSimulator:
    """Predict and validate shapes for a global physical prune plan."""

    def __init__(
        self,
        model: nn.Module,
        plan: Any,
        *,
        op_graph: Any | None = None,
        group_conv_align: int = 4,
        allow_convtranspose: bool = False,
        allow_fixed_shape_pruning: bool = False,
    ) -> None:
        self.model = model
        self.plan = plan
        self.op_graph = op_graph
        self.group_conv_align = int(group_conv_align)
        self.allow_convtranspose = bool(allow_convtranspose)
        self.allow_fixed_shape_pruning = bool(allow_fixed_shape_pruning)
        self.modules = dict(model.named_modules())
        self.shapes = self._initial_shapes()
        self.issues: list[ShapeIssue] = []
        self.operations: list[dict[str, Any]] = []

    def _initial_shapes(self) -> dict[str, SimShape]:
        shapes: dict[str, SimShape] = {}
        for name, module in self.modules.items():
            if not name:
                continue
            if isinstance(module, nn.Conv2d):
                shapes[name] = SimShape(
                    name,
                    "Conv2d",
                    module.in_channels,
                    module.out_channels,
                    module.groups,
                    module.in_channels,
                    module.out_channels,
                    module.groups,
                    _shape_of(module.weight),
                    _shape_of(module.weight),
                    _shape_of(module.bias),
                    _shape_of(module.bias),
                )
            elif isinstance(module, nn.ConvTranspose2d):
                shapes[name] = SimShape(
                    name,
                    "ConvTranspose2d",
                    module.in_channels,
                    module.out_channels,
                    module.groups,
                    module.in_channels,
                    module.out_channels,
                    module.groups,
                    _shape_of(module.weight),
                    _shape_of(module.weight),
                    _shape_of(module.bias),
                    _shape_of(module.bias),
                )
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                shapes[name] = SimShape(
                    name,
                    module.__class__.__name__,
                    None,
                    module.num_features,
                    None,
                    None,
                    module.num_features,
                    None,
                    _shape_of(module.weight),
                    _shape_of(module.weight),
                    _shape_of(module.bias),
                    _shape_of(module.bias),
                )
            elif isinstance(module, nn.Linear):
                shapes[name] = SimShape(
                    name,
                    "Linear",
                    module.in_features,
                    module.out_features,
                    None,
                    module.in_features,
                    module.out_features,
                    None,
                    _shape_of(module.weight),
                    _shape_of(module.weight),
                    _shape_of(module.bias),
                    _shape_of(module.bias),
                )
        return shapes

    def _requests(self) -> list[dict[str, Any]]:
        if hasattr(self.plan, "requests"):
            return [_request_dict(req) for req in self.plan.requests()]
        if isinstance(self.plan, dict):
            rows = self.plan.get("requests", [])
            return [_request_dict(row) for row in rows]
        return []

    def simulate(self) -> dict[str, Any]:
        for req in self._requests():
            self._apply_request(req)
        self._validate_internal_shapes()
        self._validate_op_graph_contracts()
        return self.report()

    def _apply_request(self, req: dict[str, Any]) -> None:
        name = str(req.get("module_name", ""))
        axis = str(req.get("axis", ""))
        module = self.modules.get(name)
        if module is None:
            self.issues.append(ShapeIssue("missing_module", name, {"request": req}))
            return
        shape = self.shapes.get(name)
        if shape is None:
            self.issues.append(ShapeIssue("unsupported_module", name, {"module_type": module.__class__.__name__, "request": req}))
            return
        metadata = dict(req.get("metadata", {}) or {})
        replay_axis = str(metadata.get("replay_axis", axis))
        original = _axis_channels(module, axis)
        prune = sorted({int(v) for v in req.get("prune_indices", []) if 0 <= int(v) < original})
        keep = [idx for idx in range(original) if idx not in set(prune)]
        if not keep:
            self.issues.append(ShapeIssue("empty_keep", name, {"axis": axis, "request": req}))
            return

        if _is_fixed_shape_name(name) and not self.allow_fixed_shape_pruning:
            self.issues.append(
                ShapeIssue(
                    "fixed_shape_contract_pruned",
                    name,
                    {
                        "axis": axis,
                        "replay_axis": replay_axis,
                        "before": original,
                        "after_planned": len(keep),
                        "prune_indices": prune,
                        "keep_indices": keep,
                    },
                )
            )
        if _is_head_name(name) and axis == "out":
            self.issues.append(
                ShapeIssue(
                    "head_final_output_pruned",
                    name,
                    {"axis": axis, "before": original, "after_planned": len(keep), "prune_indices": prune},
                )
            )
        if isinstance(module, nn.ConvTranspose2d) and not self.allow_convtranspose:
            self.issues.append(
                ShapeIssue(
                    "convtranspose_deblock_contract_requires_explicit_support",
                    name,
                    {"axis": axis, "before": original, "after_planned": len(keep), "prune_indices": prune},
                )
            )
        if isinstance(module, nn.ConvTranspose2d) and int(module.groups) > 1:
            self.issues.append(
                ShapeIssue(
                    "unsupported_grouped_convtranspose",
                    name,
                    {
                        "axis": axis,
                        "groups": int(module.groups),
                        "before": original,
                        "after_planned": len(keep),
                        "prune_indices": prune,
                    },
                )
            )

        if isinstance(module, nn.Conv2d):
            self._apply_conv2d(shape, axis, keep, metadata, replay_axis)
        elif isinstance(module, nn.ConvTranspose2d):
            self._apply_convtranspose(shape, axis, keep)
        elif isinstance(module, nn.modules.batchnorm._BatchNorm):
            shape.out_channels_after = len(keep)
            shape.weight_shape_after = [len(keep)] if shape.weight_shape_before is not None else None
            shape.bias_shape_after = [len(keep)] if shape.bias_shape_before is not None else None
        elif isinstance(module, nn.Linear):
            if axis == "out":
                shape.out_channels_after = len(keep)
            elif axis == "in":
                shape.in_channels_after = len(keep)
            shape.weight_shape_after = [int(shape.out_channels_after or 0), int(shape.in_channels_after or 0)]
            shape.bias_shape_after = [int(shape.out_channels_after or 0)] if shape.bias_shape_before is not None else None

        operation = {
                "module_name": name,
                "module_type": module.__class__.__name__,
                "physical_axis": axis,
                "replay_axis": replay_axis,
                "before": original,
                "after_planned": len(keep),
                "prune_indices": prune,
                "keep_indices": keep,
                "source_recipe_ids": req.get("source_recipe_ids", []),
                "metadata": metadata,
                "shape_after": shape.to_dict(),
            }
        if axis == "grouped_d_compact_frontfill_reblock":
            operation.update(
                {
                    "semantic_preserved": False,
                    "compact_first": True,
                    "frontfill_weight_transplant": True,
                    "weight_values_retained": True,
                    "zero_initialized_new_connections": True,
                    "requires_recovery_finetune": True,
                }
            )
        self.operations.append(operation)

    def _apply_conv2d(self, shape: SimShape, axis: str, keep: list[int], metadata: dict[str, Any], replay_axis: str) -> None:
        if axis == "grouped_true_group_block":
            old_groups = int(shape.groups_before or 1)
            if old_groups <= 1:
                self.issues.append(ShapeIssue("true_group_block_requires_grouped_conv", shape.module_name, shape.to_dict()))
                old_groups = max(old_groups, 1)
            if not keep:
                self.issues.append(ShapeIssue("true_group_block_requires_kept_group", shape.module_name, {"metadata": metadata}))
                keep = list(range(old_groups))
            in_before = int(shape.in_channels_before or 0)
            out_before = int(shape.out_channels_before or 0)
            if in_before % old_groups != 0 or out_before % old_groups != 0:
                self.issues.append(ShapeIssue("true_group_block_divisibility_violation", shape.module_name, shape.to_dict()))
                in_per = in_before // max(old_groups, 1)
                out_per = out_before // max(old_groups, 1)
            else:
                in_per = in_before // old_groups
                out_per = out_before // old_groups
            shape.groups_after = len(keep)
            shape.in_channels_after = len(keep) * in_per
            shape.out_channels_after = len(keep) * out_per
        elif axis == "grouped_d_compact_frontfill_reblock":
            old_output_keep = metadata.get("old_output_keep_indices", keep)
            old_input_keep = metadata.get("old_input_keep_indices", list(range(int(shape.in_channels_before or 0))))
            groups_new = int(metadata.get("groups_new") or 0)
            resolved = resolve_grouped_conv_d_compact_frontfill_reblock(
                self.modules[shape.module_name],  # type: ignore[arg-type]
                old_output_keep_indices=[int(v) for v in old_output_keep],
                old_input_keep_indices=[int(v) for v in old_input_keep],
                groups_new=groups_new,
            )
            if not resolved.get("legal", False):
                self.issues.append(
                    ShapeIssue(
                        str(resolved.get("reason") or "d_compact_frontfill_reblock_illegal"),
                        shape.module_name,
                        {"metadata": metadata, "resolved": resolved},
                    )
                )
            shape.groups_after = int(resolved.get("groups_new") or groups_new or shape.groups_before or 1)
            shape.in_channels_after = int(resolved.get("C_in_new") or len(old_input_keep))
            shape.out_channels_after = int(resolved.get("C_out_new") or len(old_output_keep))
        elif axis == "grouped_coarsen_out":
            groups_after = int(metadata.get("groups_new") or 0)
            if groups_after <= 0:
                self.issues.append(ShapeIssue("grouped_coarsen_missing_groups_new", shape.module_name, {"metadata": metadata}))
                groups_after = int(shape.groups_after or shape.groups_before or 1)
            shape.out_channels_after = len(keep)
            shape.groups_after = groups_after
        elif axis == "grouped_input_balanced" or replay_axis == "grouped_input_balanced":
            resolved = resolve_grouped_conv_input_keep(module=self.modules[shape.module_name], keep=keep, allow_repair=False)  # type: ignore[arg-type]
            if not resolved.get("legal", False):
                self.issues.append(
                    ShapeIssue(
                        "grouped_input_balance_violation",
                        shape.module_name,
                        {
                            "reason": resolved.get("reason", ""),
                            "groups": resolved.get("groups"),
                            "per_group_kept_count": resolved.get("per_group_kept_count", {}),
                        },
                    )
                )
            shape.in_channels_after = len(keep)
        elif replay_axis in {"grouped_independent_keep", "grouped_keep"}:
            shape.in_channels_after = len(keep)
            shape.out_channels_after = len(keep)
        elif replay_axis == "grouped_remove":
            old_groups = int(shape.groups_before or 1)
            old_out = int(shape.out_channels_before or len(keep))
            old_per = old_out // max(old_groups, 1)
            kept_groups = sorted({idx // max(old_per, 1) for idx in keep})
            shape.out_channels_after = len(keep)
            shape.in_channels_after = len(kept_groups) * int((shape.in_channels_before or 0) // max(old_groups, 1))
            shape.groups_after = len(kept_groups)
        else:
            if axis == "out":
                shape.out_channels_after = len(keep)
            elif axis == "in":
                shape.in_channels_after = len(keep)
        groups = int(shape.groups_after or 1)
        in_after = int(shape.in_channels_after or 0)
        out_after = int(shape.out_channels_after or 0)
        in_per = in_after // groups if groups else 0
        kernel = (shape.weight_shape_before or [0, 0, 1, 1])[2:]
        shape.weight_shape_after = [out_after, in_per, *kernel]
        shape.bias_shape_after = [out_after] if shape.bias_shape_before is not None else None

    def _apply_convtranspose(self, shape: SimShape, axis: str, keep: list[int]) -> None:
        if axis == "out":
            shape.out_channels_after = len(keep)
        elif axis == "in":
            shape.in_channels_after = len(keep)
        groups = int(shape.groups_after or 1)
        out_per = int(shape.out_channels_after or 0) // groups if groups else 0
        kernel = (shape.weight_shape_before or [0, 0, 1, 1])[2:]
        shape.weight_shape_after = [int(shape.in_channels_after or 0), out_per, *kernel]
        shape.bias_shape_after = [int(shape.out_channels_after or 0)] if shape.bias_shape_before is not None else None

    def _validate_internal_shapes(self) -> None:
        for name, shape in self.shapes.items():
            if shape.module_type == "Conv2d":
                groups = int(shape.groups_after or 0)
                c_in = int(shape.in_channels_after or 0)
                c_out = int(shape.out_channels_after or 0)
                if groups <= 0 or c_in <= 0 or c_out <= 0:
                    self.issues.append(ShapeIssue("invalid_conv2d_channels", name, shape.to_dict()))
                    continue
                if c_in % groups != 0 or c_out % groups != 0:
                    self.issues.append(ShapeIssue("grouped_conv_divisibility", name, shape.to_dict()))
                if groups > 1 and self.group_conv_align > 1:
                    in_per = c_in // groups if groups else 0
                    out_per = c_out // groups if groups else 0
                    if groups % self.group_conv_align != 0:
                        self.issues.append(
                            ShapeIssue(
                                "grouped_conv_group_count_alignment",
                                name,
                                {"groups_after": groups, "align": self.group_conv_align, **shape.to_dict()},
                            )
                        )
                    if in_per % self.group_conv_align != 0 or out_per % self.group_conv_align != 0:
                        self.issues.append(
                            ShapeIssue(
                                "grouped_conv_inner_channel_alignment",
                                name,
                                {
                                    "in_channels_per_group_after": in_per,
                                    "out_channels_per_group_after": out_per,
                                    "align": self.group_conv_align,
                                    **shape.to_dict(),
                                },
                            )
                        )
                expected = [c_out, c_in // groups, *((shape.weight_shape_before or [0, 0, 1, 1])[2:])]
                if shape.weight_shape_after != expected:
                    self.issues.append(ShapeIssue("conv2d_weight_shape_mismatch", name, {"expected": expected, "actual": shape.weight_shape_after, **shape.to_dict()}))
            elif shape.module_type == "ConvTranspose2d":
                groups = int(shape.groups_after or 0)
                c_in = int(shape.in_channels_after or 0)
                c_out = int(shape.out_channels_after or 0)
                if groups <= 0 or c_in % groups != 0 or c_out % groups != 0:
                    self.issues.append(ShapeIssue("convtranspose_divisibility", name, shape.to_dict()))
                expected = [c_in, c_out // max(groups, 1), *((shape.weight_shape_before or [0, 0, 1, 1])[2:])]
                if shape.weight_shape_after != expected:
                    self.issues.append(ShapeIssue("convtranspose_weight_shape_mismatch", name, {"expected": expected, "actual": shape.weight_shape_after, **shape.to_dict()}))
            elif shape.module_type in {"BatchNorm1d", "BatchNorm2d", "BatchNorm3d"}:
                c = int(shape.out_channels_after or 0)
                if shape.weight_shape_after is not None and shape.weight_shape_after != [c]:
                    self.issues.append(ShapeIssue("bn_weight_shape_mismatch", name, shape.to_dict()))
            elif shape.module_type == "Linear":
                expected = [int(shape.out_channels_after or 0), int(shape.in_channels_after or 0)]
                if shape.weight_shape_after != expected:
                    self.issues.append(ShapeIssue("linear_weight_shape_mismatch", name, {"expected": expected, "actual": shape.weight_shape_after, **shape.to_dict()}))

    def _node_channel_after(self, node_name: str) -> int | None:
        shape = self.shapes.get(node_name)
        if shape is None:
            return None
        return shape.out_channels_after

    def _node_input_after(self, node_name: str) -> int | None:
        shape = self.shapes.get(node_name)
        if shape is None:
            return None
        return shape.in_channels_after

    def _validate_op_graph_contracts(self) -> None:
        if self.op_graph is None:
            return
        for node in getattr(self.op_graph, "nodes", {}).values():
            op_type = getattr(node, "op_type", "")
            name = getattr(node, "name", "")
            if op_type in {"Conv", "ConvTranspose2d", "Linear"}:
                self._validate_parametric_input_contract(name, op_type)
            elif op_type in {"BN", "Norm"}:
                self._validate_norm_contract(name)
            if op_type == "Add":
                widths = []
                for src, _idx in self.op_graph.incoming(name):
                    ch = self._walk_upstream_channel(src)
                    if ch is not None:
                        widths.append(ch)
                if len(widths) >= 2 and len(set(widths)) != 1:
                    self.issues.append(ShapeIssue("residual_add_branch_channel_mismatch", name, {"branch_channels_after": widths}))
            elif op_type == "Cat":
                cat_dim = getattr(node, "cat_dim", 1)
                if cat_dim != 1:
                    continue
                branch_widths = []
                for src, _idx in self.op_graph.incoming(name):
                    ch = self._walk_upstream_channel(src)
                    if ch is not None:
                        branch_widths.append(ch)
                expected = sum(branch_widths)
                for dst, _idx in self.op_graph.outgoing(name):
                    c_in = self._walk_downstream_input(dst)
                    if c_in is not None and expected and c_in != expected:
                        self.issues.append(
                            ShapeIssue(
                                "concat_downstream_input_mismatch",
                                name,
                                {"branch_channels_after": branch_widths, "sum_after": expected, "downstream": dst, "downstream_in_after": c_in},
                            )
                        )

    def _validate_parametric_input_contract(self, name: str, op_type: str) -> None:
        c_in = self._node_input_after(name)
        if c_in is None:
            return
        incoming = self.op_graph.incoming(name)
        if not incoming:
            return
        if op_type == "Linear":
            # Linear often follows flatten/view where in_features is not a pure
            # channel width. Only enforce direct Linear->Linear contracts here.
            incoming = [
                pair for pair in incoming
                if getattr(getattr(self.op_graph, "nodes", {}).get(pair[0]), "op_type", "") == "Linear"
            ]
        for src, _idx in incoming:
            src_op = getattr(getattr(self.op_graph, "nodes", {}).get(src), "op_type", "")
            if src_op == "Cat":
                # Handled by concat sum validation.
                continue
            expected = self._walk_upstream_channel(src)
            if expected is None or int(expected) == int(c_in):
                continue
            self.issues.append(
                ShapeIssue(
                    "downstream_input_channel_mismatch",
                    name,
                    {
                        "upstream": src,
                        "upstream_channels_after": int(expected),
                        "downstream_in_after": int(c_in),
                        "op_type": op_type,
                    },
                )
            )

    def _validate_norm_contract(self, name: str) -> None:
        c = self._node_channel_after(name)
        if c is None:
            return
        for src, _idx in self.op_graph.incoming(name):
            expected = self._walk_upstream_channel(src)
            if expected is None or int(expected) == int(c):
                continue
            self.issues.append(
                ShapeIssue(
                    "norm_channel_mismatch",
                    name,
                    {
                        "upstream": src,
                        "upstream_channels_after": int(expected),
                        "norm_channels_after": int(c),
                    },
                )
            )

    def _walk_upstream_channel(self, node_name: str, seen: set[str] | None = None) -> int | None:
        seen = seen or set()
        if node_name in seen:
            return None
        seen.add(node_name)
        ch = self._node_channel_after(node_name)
        if ch is not None:
            return int(ch)
        node = getattr(self.op_graph, "nodes", {}).get(node_name) if self.op_graph is not None else None
        if node is None:
            return None
        incoming = self.op_graph.incoming(node_name)
        for src, _idx in incoming:
            ch = self._walk_upstream_channel(src, seen)
            if ch is not None:
                return ch
        return None

    def _walk_downstream_input(self, node_name: str, seen: set[str] | None = None) -> int | None:
        seen = seen or set()
        if node_name in seen:
            return None
        seen.add(node_name)
        c_in = self._node_input_after(node_name)
        if c_in is not None:
            return int(c_in)
        node = getattr(self.op_graph, "nodes", {}).get(node_name) if self.op_graph is not None else None
        if node is None:
            return None
        for dst, _idx in self.op_graph.outgoing(node_name):
            c_in = self._walk_downstream_input(dst, seen)
            if c_in is not None:
                return c_in
        return None

    def report(self) -> dict[str, Any]:
        return {
            "legal": not self.issues,
            "num_issues": len(self.issues),
            "issues": [issue.to_dict() for issue in self.issues],
            "operations": self.operations,
            "shapes": {name: shape.to_dict() for name, shape in sorted(self.shapes.items())},
        }
