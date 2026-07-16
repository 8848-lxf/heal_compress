"""Batched virtual channel and MAC resolver."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .runtime_shape_profiler import RuntimeLayerShape


@dataclass(frozen=True)
class BatchChannelTables:
    base_cin: torch.Tensor
    base_cout: torch.Tensor
    base_groups: torch.Tensor
    base_params: torch.Tensor
    action_cin_reduction: torch.Tensor
    action_cout_reduction: torch.Tensor
    action_param_reduction: torch.Tensor
    runtime_shapes: tuple[RuntimeLayerShape, ...]
    layer_ids: tuple[str, ...]
    layer_kind: torch.Tensor | None = None
    kernel_elems: torch.Tensor | None = None
    bias_out_multiplier: torch.Tensor | None = None
    channel_param_multiplier: torch.Tensor | None = None
    action_out_mask: torch.Tensor | None = None
    action_in_mask: torch.Tensor | None = None
    max_channels: int = 0
    constant_base_params: float = 0.0


class BatchChannelResolver:
    """Resolve candidate virtual widths with tensor matrix products."""

    def __init__(self, tables: BatchChannelTables, *, device: torch.device) -> None:
        self.layer_ids = tuple(tables.layer_ids)
        self.layer_index = {name: idx for idx, name in enumerate(self.layer_ids)}
        self.runtime_shapes = tuple(tables.runtime_shapes)
        self.base_cin = tables.base_cin.to(device)
        self.base_cout = tables.base_cout.to(device)
        self.base_groups = tables.base_groups.to(device)
        self.base_params = tables.base_params.to(device).clamp_min(1.0)
        self.constant_base_params = torch.tensor(
            max(float(tables.constant_base_params), 0.0),
            dtype=self.base_params.dtype,
            device=device,
        )
        self.action_cin_reduction = tables.action_cin_reduction.to(device)
        self.action_cout_reduction = tables.action_cout_reduction.to(device)
        self.action_param_reduction = tables.action_param_reduction.to(device)
        self.layer_kind = tables.layer_kind.to(device) if tables.layer_kind is not None else torch.zeros_like(self.base_params)
        self.kernel_elems = tables.kernel_elems.to(device) if tables.kernel_elems is not None else torch.ones_like(self.base_params)
        self.bias_out_multiplier = tables.bias_out_multiplier.to(device) if tables.bias_out_multiplier is not None else torch.zeros_like(self.base_params)
        self.channel_param_multiplier = tables.channel_param_multiplier.to(device) if tables.channel_param_multiplier is not None else torch.zeros_like(self.base_params)
        self.action_out_mask = tables.action_out_mask.to(device) if tables.action_out_mask is not None else None
        self.action_in_mask = tables.action_in_mask.to(device) if tables.action_in_mask is not None else None
        if self.action_out_mask is not None:
            self.max_channels = int(self.action_out_mask.shape[-1])
        elif self.action_in_mask is not None:
            self.max_channels = int(self.action_in_mask.shape[-1])
        else:
            self.max_channels = int(tables.max_channels)
        if self.max_channels > 0:
            arange = torch.arange(self.max_channels, dtype=torch.float32, device=device)
            self.valid_out_mask = arange.unsqueeze(0) < self.base_cout.unsqueeze(1)
            self.valid_in_mask = arange.unsqueeze(0) < self.base_cin.unsqueeze(1)
        else:
            self.valid_out_mask = torch.empty((len(self.layer_ids), 0), dtype=torch.bool, device=device)
            self.valid_in_mask = torch.empty((len(self.layer_ids), 0), dtype=torch.bool, device=device)
        self.shape_layer_indices = torch.tensor(
            [self.layer_index.get(shape.module_path, 0) for shape in self.runtime_shapes],
            dtype=torch.long,
            device=device,
        )
        self.shape_h = torch.tensor([float(shape.h_out) for shape in self.runtime_shapes], dtype=torch.float32, device=device)
        self.shape_w = torch.tensor([float(shape.w_out) for shape in self.runtime_shapes], dtype=torch.float32, device=device)
        self.shape_k = torch.tensor(
            [float((shape.kernel_size or (1, 1))[0] * (shape.kernel_size or (1, 1))[1]) for shape in self.runtime_shapes],
            dtype=torch.float32,
            device=device,
        )
        self.shape_is_linear = torch.tensor(
            [1.0 if shape.module_type == "Linear" else 0.0 for shape in self.runtime_shapes],
            dtype=torch.float32,
            device=device,
        )
        base = self.resolve(torch.zeros((1, self.action_cin_reduction.shape[0]), dtype=torch.float32, device=device))
        macs = self.macs_after(base)
        self.fp16_bops_baseline = (macs * 16.0 * 16.0).sum()
        self.fp32_bops_baseline = (macs * 32.0 * 32.0).sum()

    def resolve(self, pruning_choice_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        out_pruned_mask = None
        in_pruned_mask = None
        if self.action_out_mask is not None and self.action_in_mask is not None:
            batch = int(pruning_choice_tensor.shape[0])
            out_hits = pruning_choice_tensor.matmul(self.action_out_mask.reshape(self.action_out_mask.shape[0], -1)).reshape(batch, len(self.layer_ids), self.max_channels)
            in_hits = pruning_choice_tensor.matmul(self.action_in_mask.reshape(self.action_in_mask.shape[0], -1)).reshape(batch, len(self.layer_ids), self.max_channels)
            out_pruned_mask = (out_hits > 0.0) & self.valid_out_mask.unsqueeze(0)
            in_pruned_mask = (in_hits > 0.0) & self.valid_in_mask.unsqueeze(0)
            cout_after = (self.base_cout.unsqueeze(0) - out_pruned_mask.float().sum(dim=2)).clamp_min(1.0)
            cin_after = (self.base_cin.unsqueeze(0) - in_pruned_mask.float().sum(dim=2)).clamp_min(1.0)
        else:
            cin_after = (self.base_cin.unsqueeze(0) - pruning_choice_tensor.matmul(self.action_cin_reduction)).clamp_min(1.0)
            cout_after = (self.base_cout.unsqueeze(0) - pruning_choice_tensor.matmul(self.action_cout_reduction)).clamp_min(1.0)
        groups_after = self.base_groups.unsqueeze(0).expand_as(cin_after).clamp_min(1.0)
        cin_per_group = torch.floor(cin_after / groups_after.clamp_min(1.0)).clamp_min(1.0)
        cout_per_group = torch.floor(cout_after / groups_after.clamp_min(1.0)).clamp_min(1.0)
        conv_params = cout_after * cin_per_group * self.kernel_elems.unsqueeze(0) + cout_after * self.bias_out_multiplier.unsqueeze(0)
        conv_t_params = cin_after * cout_per_group * self.kernel_elems.unsqueeze(0) + cout_after * self.bias_out_multiplier.unsqueeze(0)
        linear_params = cin_after * cout_after + cout_after * self.bias_out_multiplier.unsqueeze(0)
        channel_params = cout_after * self.channel_param_multiplier.unsqueeze(0)
        fallback_params = (self.base_params.unsqueeze(0) - pruning_choice_tensor.matmul(self.action_param_reduction)).clamp_min(1.0)
        params_after = torch.where(
            self.layer_kind.unsqueeze(0) == 1,
            conv_params,
            torch.where(
                self.layer_kind.unsqueeze(0) == 2,
                conv_t_params,
                torch.where(
                    self.layer_kind.unsqueeze(0) == 3,
                    linear_params,
                    torch.where(self.layer_kind.unsqueeze(0) == 4, channel_params, fallback_params),
                ),
            ),
        ).clamp_min(1.0)
        return {
            "C_in_after": cin_after,
            "C_out_after": cout_after,
            "groups_after": groups_after,
            "params_after": params_after,
            "out_pruned_mask": out_pruned_mask,
            "in_pruned_mask": in_pruned_mask,
        }

    def resolve_explicit_masks(
        self,
        out_pruned_mask: torch.Tensor,
        in_pruned_mask: torch.Tensor,
        *,
        pruning_choice_tensor: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Resolve exact candidate channel masks without a dense action table.

        Domain-width candidates expand to their frozen atomic masks before proxy
        scoring.  Materializing an ``atomic x layer x channel`` action tensor for
        thousands of traced units is both wasteful and error prone, so this path
        consumes the already-expanded candidate masks directly.
        """

        expected = (len(self.layer_ids), self.max_channels)
        if tuple(out_pruned_mask.shape[1:]) != expected:
            raise ValueError(
                f"invalid_explicit_out_mask_shape:{tuple(out_pruned_mask.shape)}:{expected}"
            )
        if tuple(in_pruned_mask.shape[1:]) != expected:
            raise ValueError(
                f"invalid_explicit_in_mask_shape:{tuple(in_pruned_mask.shape)}:{expected}"
            )
        out_pruned = out_pruned_mask.to(self.base_cout.device, dtype=torch.bool)
        in_pruned = in_pruned_mask.to(self.base_cin.device, dtype=torch.bool)
        out_pruned = out_pruned & self.valid_out_mask.unsqueeze(0)
        in_pruned = in_pruned & self.valid_in_mask.unsqueeze(0)
        cout_after = (
            self.base_cout.unsqueeze(0) - out_pruned.float().sum(dim=2)
        ).clamp_min(1.0)
        cin_after = (
            self.base_cin.unsqueeze(0) - in_pruned.float().sum(dim=2)
        ).clamp_min(1.0)
        groups_after = self.base_groups.unsqueeze(0).expand_as(cin_after).clamp_min(1.0)
        cin_per_group = torch.floor(cin_after / groups_after).clamp_min(1.0)
        cout_per_group = torch.floor(cout_after / groups_after).clamp_min(1.0)
        conv_params = (
            cout_after * cin_per_group * self.kernel_elems.unsqueeze(0)
            + cout_after * self.bias_out_multiplier.unsqueeze(0)
        )
        conv_t_params = (
            cin_after * cout_per_group * self.kernel_elems.unsqueeze(0)
            + cout_after * self.bias_out_multiplier.unsqueeze(0)
        )
        linear_params = (
            cin_after * cout_after
            + cout_after * self.bias_out_multiplier.unsqueeze(0)
        )
        channel_params = cout_after * self.channel_param_multiplier.unsqueeze(0)
        if pruning_choice_tensor is None:
            fallback_params = self.base_params.unsqueeze(0).expand_as(cin_after)
        else:
            fallback_params = (
                self.base_params.unsqueeze(0)
                - pruning_choice_tensor.to(self.base_params.device).matmul(
                    self.action_param_reduction
                )
            ).clamp_min(1.0)
        params_after = torch.where(
            self.layer_kind.unsqueeze(0) == 1,
            conv_params,
            torch.where(
                self.layer_kind.unsqueeze(0) == 2,
                conv_t_params,
                torch.where(
                    self.layer_kind.unsqueeze(0) == 3,
                    linear_params,
                    torch.where(
                        self.layer_kind.unsqueeze(0) == 4,
                        channel_params,
                        fallback_params,
                    ),
                ),
            ),
        ).clamp_min(1.0)
        return {
            "C_in_after": cin_after,
            "C_out_after": cout_after,
            "groups_after": groups_after,
            "params_after": params_after,
            "out_pruned_mask": out_pruned,
            "in_pruned_mask": in_pruned,
        }

    def macs_after(self, resolved: dict[str, torch.Tensor]) -> torch.Tensor:
        cin = resolved["C_in_after"][:, self.shape_layer_indices]
        cout = resolved["C_out_after"][:, self.shape_layer_indices]
        groups = resolved["groups_after"][:, self.shape_layer_indices].clamp_min(1.0)
        conv_macs = self.shape_h.unsqueeze(0) * self.shape_w.unsqueeze(0) * self.shape_k.unsqueeze(0) * cin * cout / groups
        linear_macs = cin * cout
        return torch.where(self.shape_is_linear.unsqueeze(0) > 0.0, linear_macs, conv_macs)
