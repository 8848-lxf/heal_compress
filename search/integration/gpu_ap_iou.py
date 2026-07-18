"""GPU-only BEV IoU matching used by production evaluation runners."""

from __future__ import annotations

from typing import Any, MutableMapping

import torch


def calculate_gpu_tp_fp_for_threshold(
    det_boxes: torch.Tensor | None,
    det_score: torch.Tensor | None,
    gt_boxes: torch.Tensor | None,
    result_stat: MutableMapping[float, MutableMapping[str, Any]],
    iou_thresh: float,
    device: torch.device,
) -> None:
    threshold = float(iou_thresh)
    gt = 0 if gt_boxes is None else int(gt_boxes.shape[0])
    if det_boxes is None or det_score is None or int(det_boxes.shape[0]) == 0:
        result_stat[threshold]["gt"] += gt
        return
    if device.type != "cuda":
        raise RuntimeError("gpu_ap_iou_requires_cuda_device")
    from opencood.pcdet_utils.iou3d_nms.iou3d_nms_utils import boxes_iou_bev
    from opencood.utils import box_utils

    det_boxes = det_boxes.to(device=device, dtype=torch.float32)
    det_score = det_score.to(device=device, dtype=torch.float32).reshape(-1)
    order = torch.argsort(det_score, descending=True)
    det_score = det_score[order]
    det_boxes = det_boxes[order]
    result_stat[threshold]["score"] += det_score.detach().cpu().tolist()
    if gt == 0:
        count = int(det_boxes.shape[0])
        result_stat[threshold]["fp"] += [1] * count
        result_stat[threshold]["tp"] += [0] * count
        return
    gt_boxes = gt_boxes.to(device=device, dtype=torch.float32)
    det_nms = box_utils._corners_to_nms_boxes_torch(det_boxes).contiguous()
    gt_nms = box_utils._corners_to_nms_boxes_torch(gt_boxes).contiguous()
    iou_matrix = boxes_iou_bev(det_nms, gt_nms)
    available = torch.ones((gt,), dtype=torch.bool, device=device)
    false_positives: list[int] = []
    true_positives: list[int] = []
    for row in iou_matrix:
        if not bool(available.any()):
            false_positives.append(1)
            true_positives.append(0)
            continue
        available_iou = row.clone()
        available_iou[~available] = -1.0
        maximum, index = torch.max(available_iou, dim=0)
        if float(maximum.item()) < threshold:
            false_positives.append(1)
            true_positives.append(0)
        else:
            false_positives.append(0)
            true_positives.append(1)
            available[index] = False
    result_stat[threshold]["fp"] += false_positives
    result_stat[threshold]["tp"] += true_positives
    result_stat[threshold]["gt"] += gt


__all__ = ["calculate_gpu_tp_fp_for_threshold"]
