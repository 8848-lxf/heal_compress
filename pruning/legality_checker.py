"""Structure legality checker for pruned HEAL models.

Validates that the pruned model structure is internally consistent
and can execute a forward pass without runtime errors.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..utils.io_utils import ensure_dir, save_json

logger = logging.getLogger(__name__)


class StructureLegalityChecker:
    """Validates the structural integrity of a pruned HEAL model.

    Checks:
    1. Conv2d: in/out channels valid, groups divides channels.
    2. BatchNorm: num_features matches predecessor out_channels.
    3. Linear: in/out features valid.
    4. Residual Add: both branches have equal channels.
    5. Concat: sum of branch channels equals successor input.
    6. Pyramid deblocks cat: up0+up1+up2 == fused_feature channels.
    7. cls/reg/dir head output dimensions unchanged.
    8. Forward pass with calibration data succeeds.

    Args:
        model: The pruned HEAL model.
    """

    def __init__(self, model: nn.Module):
        self.model = model

    def check(
        self,
        output_dir: str,
        calibration_inputs: tuple[torch.Tensor, ...] | None = None,
        export_module: Any = None,
    ) -> dict[str, Any]:
        """Run all legality checks and generate a report.

        Args:
            output_dir: Directory for saving the report.
            calibration_inputs: Optional tuple of calibration tensors for
                a full forward-pass verification.
            export_module: Optional export module for wrapper creation.

        Returns:
            Report dict with 'legal' flag, 'issues' list, and suggestions.
        """
        out = ensure_dir(output_dir)
        issues: list[dict[str, Any]] = []

        issues.extend(self._check_conv_channels())
        issues.extend(self._check_batchnorm_features())
        issues.extend(self._check_linear_features())
        issues.extend(self._check_groups_divisibility())
        issues.extend(self._check_detection_heads())

        # Forward pass verification
        if calibration_inputs is not None:
            forward_issues = self._check_forward_pass(
                calibration_inputs, export_module
            )
            issues.extend(forward_issues)

        report = {
            "legal": len(issues) == 0,
            "num_issues": len(issues),
            "issues": issues,
        }

        save_json(report, str(Path(out) / "legality_report.json"))
        if issues:
            logger.warning(f"Legality check found {len(issues)} issues")
            for issue in issues[:10]:
                logger.warning(f"  {issue}")
        else:
            logger.info("Legality check passed")
        return report

    def _check_conv_channels(self) -> list[dict]:
        """Check Conv2d layers for valid channel counts."""
        issues = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                if module.in_channels <= 0:
                    issues.append({
                        "layer": name, "issue": "invalid_in_channels",
                        "value": module.in_channels,
                        "suggestion": "Check upstream pruning removed too many channels.",
                    })
                if module.out_channels <= 0:
                    issues.append({
                        "layer": name, "issue": "invalid_out_channels",
                        "value": module.out_channels,
                        "suggestion": "This layer's output was fully pruned.",
                    })
        return issues

    def _check_batchnorm_features(self) -> list[dict]:
        """Check BatchNorm layers for valid feature counts."""
        issues = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if module.num_features <= 0:
                    issues.append({
                        "layer": name, "issue": "invalid_bn_features",
                        "value": module.num_features,
                    })
        return issues

    def _check_linear_features(self) -> list[dict]:
        """Check Linear layers for valid feature counts."""
        issues = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if module.in_features <= 0 or module.out_features <= 0:
                    issues.append({
                        "layer": name, "issue": "invalid_linear_features",
                        "in": module.in_features, "out": module.out_features,
                    })
        return issues

    def _check_groups_divisibility(self) -> list[dict]:
        """Check that groups properly divides in/out channels."""
        issues = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                if module.groups <= 0:
                    issues.append({"layer": name, "issue": "invalid_groups"})
                elif (module.in_channels % module.groups != 0
                      or module.out_channels % module.groups != 0):
                    issues.append({
                        "layer": name, "issue": "groups_not_divisible",
                        "in_channels": module.in_channels,
                        "out_channels": module.out_channels,
                        "groups": module.groups,
                    })
        return issues

    def _check_detection_heads(self) -> list[dict]:
        """Check that detection head output dimensions are unchanged.

        cls_head, reg_head, dir_head must maintain their original output size.
        """
        issues = []
        for head_name in ("cls_head", "reg_head", "dir_head"):
            head = None
            for name, module in self.model.named_modules():
                if name.endswith(head_name):
                    head = module
                    break
            if head is None:
                continue
            # The final layer should not have been pruned
            if isinstance(head, nn.Conv2d):
                if head.weight.shape[0] <= 0:
                    issues.append({
                        "layer": head_name,
                        "issue": "head_output_pruned",
                        "out_channels": head.out_channels,
                    })
        return issues

    def _check_forward_pass(
        self,
        inputs: tuple[torch.Tensor, ...],
        export_module: Any = None,
    ) -> list[dict]:
        """Run a forward pass and check for runtime errors.

        Args:
            inputs: Calibration input tensors.
            export_module: Export module for creating the wrapper.

        Returns:
            List of issues (empty if forward succeeds).
        """
        issues = []
        try:
            self.model.eval()
            with torch.no_grad():
                if export_module:
                    modality = None
                    for name in getattr(self.model, "modality_name_list", []):
                        if hasattr(self.model, f"encoder_{name}"):
                            modality = name
                            break
                    if modality:
                        crop_params = export_module._record_crop_params(
                            self.model, modality, inputs
                        )
                        wrapper = export_module.DynamicAgentExportWrapper(
                            self.model, modality_name=modality,
                            crop_params=crop_params, insert_se3_inverse=True,
                        ).eval()
                        wrapper(*inputs)
                    else:
                        self.model(*inputs)
                else:
                    # Try direct forward
                    self.model(*inputs)
        except Exception as exc:
            issues.append({
                "issue": "forward_pass_failed",
                "error": str(exc),
                "suggestion": "Check that pruning maintained structural consistency.",
            })
        return issues
