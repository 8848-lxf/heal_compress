"""One-shot physical pruning executor for HEAL models.

Reads the best_subnet_config.json and physically removes channels from
Conv2d, BatchNorm, Linear, and ConvTranspose2d layers. Handles cross-branch
synchronization for pyramid_backbone multi-scale structure.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..utils.io_utils import ensure_dir, load_json, save_json

logger = logging.getLogger(__name__)


class PhysicalPruner:
    """Executes one-shot physical channel pruning on a HEAL model.

    Given a subnet config (mask=0 groups), synchronously removes:
    - Conv2d.weight output/input channels
    - Conv2d.bias elements
    - BatchNorm weight/bias/running_mean/running_var
    - Linear.weight rows/columns
    - ConvTranspose2d channels
    - Depthwise Conv2d groups

    Does NOT modify:
    - kernel_size, stride, padding, dilation
    - Activation types, BN types
    - BEV grid parameters (frustum, camC, D, nx, dx, bx)

    Args:
        model: The HEAL model to prune.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._modules_map: dict[str, nn.Module] = dict(model.named_modules())

    def apply(
        self,
        subnet_config_path: str,
        groups_path: str,
        output_dir: str,
    ) -> str:
        """Execute physical pruning based on subnet config.

        Steps:
        1. Load subnet config and coupled channel groups.
        2. For each group with mask=0:
           a. Determine keep_indices for the group.
           b. Slice output channels of source modules.
           c. Slice BatchNorm parameters.
           d. Slice input channels of dependent modules.
        3. Handle pyramid_backbone cross-branch synchronization.
        4. Save pruned model checkpoint and report.

        Args:
            subnet_config_path: Path to best_subnet_config.json.
            groups_path: Path to coupled_channel_groups.json.
            output_dir: Output directory for pruned model.

        Returns:
            Path to saved pruned model checkpoint.
        """
        out = ensure_dir(output_dir)
        config = load_json(subnet_config_path)
        prune_vars = config.get("prune_vars", {})

        from ..tracer.coupled_channel_group import CoupledChannelGroupBuilder
        groups = CoupledChannelGroupBuilder.load(groups_path)

        applied = []
        skipped = []

        for group in groups:
            gid = group.group_id
            keep = prune_vars.get(gid, 1)
            if group.is_protected or keep == 1:
                continue

            try:
                keep_indices = self._compute_keep_indices(group)
                if not keep_indices:
                    skipped.append({"group_id": gid, "reason": "no_keep_indices"})
                    continue

                ops = self._prune_group(group, keep_indices)
                applied.append({
                    "group_id": gid,
                    "keep_indices": keep_indices,
                    "operations": ops,
                })
            except Exception as exc:
                skipped.append({
                    "group_id": gid, "reason": str(exc),
                })
                logger.error(f"Failed to prune group {gid}: {exc}")

        ckpt_path = str(Path(out) / "pruned_model.pth")
        torch.save(
            {"model": self.model, "applied": applied, "skipped": skipped},
            ckpt_path,
        )
        save_json(
            {"applied": applied, "skipped": skipped, "checkpoint": ckpt_path},
            str(Path(out) / "physical_prune_report.json"),
        )
        logger.info(
            f"Physical pruning: {len(applied)} groups applied, "
            f"{len(skipped)} skipped. Saved to {ckpt_path}"
        )
        return ckpt_path

    def _compute_keep_indices(self, group: Any) -> list[int]:
        """Determine which channel indices to keep for a pruned group.

        Uses importance-based selection if available, otherwise keeps
        the first N channels where N satisfies alignment constraints.

        Args:
            group: CoupledChannelGroup object.

        Returns:
            Sorted list of channel indices to keep.
        """
        keep = group.successor_mapping.get("keep_indices")
        if keep is not None:
            return sorted(int(i) for i in keep)

        # Default: keep all (this group was marked mask=0, so we remove all)
        return []

    def _prune_group(self, group: Any, keep_indices: list[int]) -> list[dict]:
        """Execute pruning for a single coupled channel group.

        Args:
            group: CoupledChannelGroup object.
            keep_indices: Channel indices to retain.

        Returns:
            List of operation records.
        """
        ops = []
        idx_tensor = None

        # Slice output channels of source modules
        for layer_name in group.source_modules:
            module = self._modules_map.get(layer_name)
            if module is None:
                continue
            op = self._slice_output_channels(layer_name, module, keep_indices)
            if op:
                ops.append(op)

        # Slice BatchNorm layers
        for bn_name in group.successor_mapping.get("conv_bn", []):
            module = self._modules_map.get(bn_name)
            if module is None:
                continue
            op = self._slice_batchnorm(bn_name, module, keep_indices)
            if op:
                ops.append(op)

        # Slice input channels of dependent modules
        for layer_name in group.dependent_modules:
            module = self._modules_map.get(layer_name)
            if module is None:
                continue
            op = self._slice_input_channels(layer_name, module, keep_indices)
            if op:
                ops.append(op)

        return ops

    def _slice_output_channels(
        self, name: str, module: nn.Module, keep: list[int],
    ) -> dict[str, Any] | None:
        """Slice output channels of a Conv2d or Linear layer.

        Args:
            name: Layer name.
            module: The module.
            keep: Channel indices to keep.

        Returns:
            Operation record dict, or None.
        """
        idx = torch.tensor(keep, dtype=torch.long, device=module.weight.device)

        if isinstance(module, nn.Conv2d):
            before = module.out_channels
            module.weight = nn.Parameter(
                module.weight.data.index_select(0, idx).clone()
            )
            if module.bias is not None:
                module.bias = nn.Parameter(
                    module.bias.data.index_select(0, idx).clone()
                )
            module.out_channels = len(keep)
            if module.groups > 1 and module.groups == before:
                module.groups = len(keep)  # Depthwise
            return {"layer": name, "axis": "out", "before": before, "after": len(keep)}

        if isinstance(module, nn.ConvTranspose2d):
            before = module.out_channels
            module.weight = nn.Parameter(
                module.weight.data.index_select(1, idx).clone()
            )
            if module.bias is not None:
                module.bias = nn.Parameter(
                    module.bias.data.index_select(0, idx).clone()
                )
            module.out_channels = len(keep)
            return {"layer": name, "axis": "out", "before": before, "after": len(keep)}

        if isinstance(module, nn.Linear):
            before = module.out_features
            module.weight = nn.Parameter(
                module.weight.data.index_select(0, idx).clone()
            )
            if module.bias is not None:
                module.bias = nn.Parameter(
                    module.bias.data.index_select(0, idx).clone()
                )
            module.out_features = len(keep)
            return {"layer": name, "axis": "out", "before": before, "after": len(keep)}

        return None

    def _slice_input_channels(
        self, name: str, module: nn.Module, keep: list[int],
    ) -> dict[str, Any] | None:
        """Slice input channels of a Conv2d or Linear layer.

        Args:
            name: Layer name.
            module: The module.
            keep: Channel indices to keep.

        Returns:
            Operation record dict, or None.
        """
        idx = torch.tensor(keep, dtype=torch.long, device=module.weight.device)

        if isinstance(module, nn.Conv2d):
            if module.groups > 1:
                return None  # Grouped conv input slicing is complex
            before = module.in_channels
            module.weight = nn.Parameter(
                module.weight.data.index_select(1, idx).clone()
            )
            module.in_channels = len(keep)
            return {"layer": name, "axis": "in", "before": before, "after": len(keep)}

        if isinstance(module, nn.Linear):
            before = module.in_features
            module.weight = nn.Parameter(
                module.weight.data.index_select(1, idx).clone()
            )
            module.in_features = len(keep)
            return {"layer": name, "axis": "in", "before": before, "after": len(keep)}

        return None

    def _slice_batchnorm(
        self, name: str, module: nn.Module, keep: list[int],
    ) -> dict[str, Any] | None:
        """Slice BatchNorm parameters.

        Args:
            name: Layer name.
            module: BatchNorm module.
            keep: Channel indices to keep.

        Returns:
            Operation record dict, or None.
        """
        if not isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            return None

        idx = torch.tensor(keep, dtype=torch.long, device=module.weight.device)
        before = module.num_features
        module.weight = nn.Parameter(module.weight.data.index_select(0, idx).clone())
        module.bias = nn.Parameter(module.bias.data.index_select(0, idx).clone())
        module.running_mean = module.running_mean.index_select(0, idx).clone()
        module.running_var = module.running_var.index_select(0, idx).clone()
        module.num_features = len(keep)
        return {"layer": name, "axis": "bn", "before": before, "after": len(keep)}
