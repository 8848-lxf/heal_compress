"""BEV fusion feature KL divergence distillation trainer.

Teacher (original model) parameters frozen.
Student (pruned proxy model) parameters optimized.
Distillation target: BEV fused feature from pyramid_backbone output.
"""

from __future__ import annotations

import logging
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..quantization.pseudo_quant import PseudoQuantManager
from ..utils.io_utils import ensure_dir, save_json

logger = logging.getLogger(__name__)


def bev_kl_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    temperature: float = 4.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute BEV KL divergence distillation loss.

    Channel-dimension Softmax with temperature coefficient:
        P_T(c|u,v) = exp(F_T(c,u,v)/tau) / sum_c' exp(F_T(c',u,v)/tau)
        P_S(c|u,v) = exp(F_S(c,u,v)/tau) / sum_c' exp(F_S(c',u,v)/tau)

    L_BEV_KL = tau^2 / (H*W) * sum_{u,v,c} P_T * log(P_T / (P_S + eps))

    Args:
        student: Student BEV feature [B, C_S, H, W].
        teacher: Teacher BEV feature [B, C_T, H, W].
        temperature: Softmax temperature (default 4.0).
        eps: Small constant for numerical stability.

    Returns:
        Scalar KL divergence loss.
    """
    tau = max(float(temperature), 1e-6)
    # Flatten spatial dims: [B, C, H*W]
    s = student.flatten(2) / tau
    t = teacher.flatten(2) / tau
    return F.kl_div(
        F.log_softmax(s, dim=1),
        F.softmax(t, dim=1),
        reduction="batchmean",
    ) * (tau * tau)


def align_bev_feature(
    student: torch.Tensor,
    teacher: torch.Tensor,
) -> torch.Tensor:
    """Align student BEV feature to teacher dimensions.

    Spatial alignment: bilinear interpolation.
    Channel alignment: 1x1 conv (only during distillation, not deployed).

    Args:
        student: Student feature [B, C_S, H_S, W_S].
        teacher: Teacher feature [B, C_T, H_T, W_T].

    Returns:
        Aligned student feature with teacher's spatial dimensions.
        Channel alignment is handled separately.
    """
    if student.shape[-2:] != teacher.shape[-2:]:
        student = F.interpolate(
            student, size=teacher.shape[-2:],
            mode="bilinear", align_corners=False,
        )
    return student


class FeatureAligner(nn.Module):
    """1x1 conv feature aligner for channel dimension matching.

    Used only during distillation training. NOT preserved in final model.

    Args:
        in_channels: Student feature channels.
        out_channels: Teacher feature channels.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class BEVKLDistillationTrainer:
    """BEV KL divergence distillation trainer for accuracy recovery.

    Teacher: original model (frozen).
    Student: proxy model with channel masks and pseudo-quantization.

    Total loss: L_recover = L_task + lambda_bev * L_BEV_KL

    Constraints during finetuning:
    - mask=0 groups remain zeroed.
    - Per-layer pseudo-quantization applied during forward.
    - Aligner parameters optimized but NOT saved to final model.

    Args:
        teacher: Original HEAL model (frozen).
        student: Proxy model (masks + pseudo-quant applied).
        export_module: The export_dynamic_onnx module for wrapper creation.
        modality: Camera modality name.
        epochs: Number of finetuning epochs.
        lr: Learning rate.
        tau: KL divergence temperature.
        lambda_bev: Weight for BEV KL loss.
        device: Target device.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        export_module: Any,
        modality: str,
        epochs: int = 10,
        lr: float = 1e-4,
        tau: float = 4.0,
        lambda_bev: float = 0.5,
        device: str | torch.device = "cpu",
    ):
        self.teacher = teacher.to(device).eval()
        self.student = student.to(device)
        self.export_module = export_module
        self.modality = modality
        self.epochs = epochs
        self.lr = lr
        self.tau = tau
        self.lambda_bev = lambda_bev
        self.device = torch.device(device)

        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.aligner: FeatureAligner | None = None
        self._bev_features: dict[str, torch.Tensor] = {}

    def run(
        self,
        train_loader: Any,
        subnet_config: dict[str, Any],
        output_dir: str,
    ) -> str:
        """Execute the distillation finetuning.

        Args:
            train_loader: DataLoader yielding calibration batches (with labels).
            subnet_config: Subnet config dict with prune_vars and bitwidth_vars.
            output_dir: Output directory for finetuned model.

        Returns:
            Path to saved finetuned model checkpoint.
        """
        out = ensure_dir(output_dir)
        bitwidth_vars = subnet_config.get("bitwidth_vars", {})

        pseudo_quant = PseudoQuantManager(self.student)
        optimizer = torch.optim.Adam(self.student.parameters(), lr=self.lr)
        if self.aligner is not None:
            optimizer.add_param_group({"params": self.aligner.parameters()})

        history = []
        for epoch in range(self.epochs):
            self.student.train()
            total_loss = 0.0
            steps = 0

            for batch in train_loader:
                batch = _to_device(batch, str(self.device))
                optimizer.zero_grad(set_to_none=True)

                # Teacher BEV feature (frozen)
                with torch.no_grad():
                    teacher_bev = self._extract_bev(self.teacher, batch)

                # Student forward with pseudo-quant
                with pseudo_quant.apply(bitwidth_vars):
                    student_bev = self._extract_bev(self.student, batch)
                    student_bev = align_bev_feature(student_bev, teacher_bev)

                    # Channel alignment if needed
                    if student_bev.shape[1] != teacher_bev.shape[1]:
                        if self.aligner is None:
                            self.aligner = FeatureAligner(
                                student_bev.shape[1], teacher_bev.shape[1]
                            ).to(self.device)
                            optimizer.add_param_group({"params": self.aligner.parameters()})
                        student_bev = self.aligner(student_bev)

                    loss_bev = bev_kl_loss(student_bev, teacher_bev, self.tau)

                    # Task loss (placeholder - uses model outputs)
                    loss_task = torch.tensor(0.0, device=self.device)

                    loss = loss_task + self.lambda_bev * loss_bev
                    loss.backward()
                    optimizer.step()

                total_loss += float(loss.detach().cpu())
                steps += 1

            avg_loss = total_loss / max(steps, 1)
            history.append({"epoch": epoch, "loss": avg_loss})
            logger.info(f"Epoch {epoch+1}/{self.epochs}: loss={avg_loss:.6f}")

        ckpt_path = str(Path(out) / "proxy_model_finetuned.pth")
        torch.save(
            {"model": self.student, "history": history},
            ckpt_path,
        )
        save_json(
            {"history": history, "checkpoint": ckpt_path},
            str(Path(out) / "distill_report.json"),
        )
        logger.info(f"Finetuned model saved to {ckpt_path}")
        return ckpt_path

    def _extract_bev(self, model: nn.Module, batch: Any) -> torch.Tensor:
        """Extract BEV fused feature from a HEAL model.

        Uses a forward hook on pyramid_backbone to capture the fused feature.

        Args:
            model: HEAL model.
            batch: Input batch (tuple of tensors).

        Returns:
            BEV feature tensor [B, C, H, W].
        """
        features = {}

        def hook_fn(name):
            def hook(module, input, output):
                if torch.is_tensor(output):
                    features[name] = output
                elif isinstance(output, dict):
                    for k, v in output.items():
                        if torch.is_tensor(v):
                            features[f"{name}.{k}"] = v
            return hook

        handles = []
        for name, module in model.named_modules():
            if "pyramid_backbone" in name or "backbone" in name:
                handles.append(module.register_forward_hook(hook_fn(name)))

        try:
            if isinstance(batch, (tuple, list)):
                crop_params = self.export_module._record_crop_params(
                    model, self.modality, batch
                )
                wrapper = self.export_module.DynamicAgentExportWrapper(
                    model, modality_name=self.modality,
                    crop_params=crop_params, insert_se3_inverse=True,
                ).eval()
                wrapper(*batch)
            else:
                model(batch)
        finally:
            for h in handles:
                h.remove()

        # Return the largest spatial feature found
        best = None
        for name, feat in features.items():
            if feat.dim() == 4:
                if best is None or feat.numel() > best.numel():
                    best = feat
        if best is None:
            raise RuntimeError("Failed to extract BEV feature from model.")
        return best


def _to_device(obj: Any, device: str) -> Any:
    """Recursively move tensors to device."""
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_device(v, device) for v in obj)
    return obj
