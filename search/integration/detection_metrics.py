"""Detection metric accumulation shared by formal PyTorch and TensorRT workers."""

from __future__ import annotations

from typing import Any

import torch


def calculate_tp_fp_for_threshold(
    det_boxes: Any,
    det_score: Any,
    gt_boxes: Any,
    result_stat: dict[float, dict[str, Any]],
    iou_thresh: float,
    backend: str,
    device: torch.device,
) -> None:
    """Accumulate score-ordered one-to-one BEV matches for one IoU threshold."""

    gt_count = 0 if gt_boxes is None else int(gt_boxes.shape[0])
    if det_boxes is None or det_score is None or int(det_boxes.shape[0]) == 0:
        result_stat[iou_thresh]["gt"] += gt_count
        return
    if str(backend).lower() == "cpu":
        from opencood.utils import eval_utils

        eval_utils.caluclate_tp_fp(
            det_boxes, det_score, gt_boxes, result_stat, iou_thresh
        )
        return
    if str(backend).lower() != "gpu":
        raise ValueError(f"unknown AP IoU backend:{backend}")
    _calculate_tp_fp_gpu_bev(
        det_boxes,
        det_score,
        gt_boxes,
        result_stat,
        float(iou_thresh),
        device,
    )


def _calculate_tp_fp_gpu_bev(
    det_boxes: torch.Tensor,
    det_score: torch.Tensor,
    gt_boxes: torch.Tensor | None,
    result_stat: dict[float, dict[str, Any]],
    iou_thresh: float,
    device: torch.device,
) -> None:
    from opencood.pcdet_utils.iou3d_nms.iou3d_nms_utils import boxes_iou_bev
    from opencood.utils import box_utils

    gt_count = 0 if gt_boxes is None else int(gt_boxes.shape[0])
    det_boxes = det_boxes.to(device=device, dtype=torch.float32)
    det_score = det_score.to(device=device, dtype=torch.float32).reshape(-1)
    order = torch.argsort(det_score, descending=True)
    det_score = det_score[order]
    det_boxes = det_boxes[order]
    result_stat[iou_thresh]["score"] += det_score.detach().cpu().tolist()
    if gt_count == 0:
        count = int(det_boxes.shape[0])
        result_stat[iou_thresh]["fp"] += [1] * count
        result_stat[iou_thresh]["tp"] += [0] * count
        return

    assert gt_boxes is not None
    gt_boxes = gt_boxes.to(device=device, dtype=torch.float32)
    detected = box_utils._corners_to_nms_boxes_torch(det_boxes).contiguous()
    targets = box_utils._corners_to_nms_boxes_torch(gt_boxes).contiguous()
    iou_matrix = boxes_iou_bev(detected, targets)
    available = torch.ones((gt_count,), dtype=torch.bool, device=device)
    false_positive: list[int] = []
    true_positive: list[int] = []
    for index in range(int(iou_matrix.shape[0])):
        if not bool(available.any()):
            false_positive.append(1)
            true_positive.append(0)
            continue
        row = iou_matrix[index].clone()
        row[~available] = -1.0
        maximum, target_index = torch.max(row, dim=0)
        if float(maximum.item()) < float(iou_thresh):
            false_positive.append(1)
            true_positive.append(0)
        else:
            false_positive.append(0)
            true_positive.append(1)
            available[target_index] = False
    result_stat[iou_thresh]["fp"] += false_positive
    result_stat[iou_thresh]["tp"] += true_positive
    result_stat[iou_thresh]["gt"] += gt_count


__all__ = ["calculate_tp_fp_for_threshold"]
