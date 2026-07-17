"""Typed ONNX export recipe owned by the LiDAR CoBEVT family."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import onnx
import torch
import torch.nn as nn

from quantization.export.heal_lidar_cobevt import HEALLiDARCoBEVTSignalMaxK

from .operator_probe import audit_onnx_operators

COBEVT_INPUT_NAMES = (
    "voxel_features",
    "voxel_coords",
    "voxel_num_points",
    "pairwise_t_matrix",
    "valid_voxel_mask",
    "record_len",
)
COBEVT_OUTPUT_NAMES = ("cls_preds", "reg_preds", "dir_preds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class CobevtTypedExportReport:
    onnx_path: str
    onnx_sha256: str
    checker_passed: bool
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    node_count: int
    op_counts: dict[str, int]
    registered_custom_ops: tuple[str, ...]
    unregistered_custom_ops: tuple[str, ...]


class CobevtExportRecipe:
    def __init__(
        self,
        *,
        fixed_k: int,
        max_cav: int = 2,
        opset_version: int = 17,
    ) -> None:
        if fixed_k <= 0 or max_cav <= 0:
            raise ValueError("fixed_k_and_max_cav_must_be_positive")
        self.fixed_k = int(fixed_k)
        self.max_cav = int(max_cav)
        self.opset_version = int(opset_version)

    def build_module(self, model: nn.Module) -> HEALLiDARCoBEVTSignalMaxK:
        return HEALLiDARCoBEVTSignalMaxK(
            model,
            fixed_k=self.fixed_k,
            max_cav=self.max_cav,
            output_names=COBEVT_OUTPUT_NAMES,
        )

    def export(
        self,
        model: nn.Module,
        inputs: Mapping[str, torch.Tensor],
        output_path: str | Path,
    ) -> CobevtTypedExportReport:
        missing = [name for name in COBEVT_INPUT_NAMES if name not in inputs]
        if missing:
            raise RuntimeError(f"cobevt_export_inputs_missing:{','.join(missing)}")
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        wrapper = self.build_module(model).eval()
        args = tuple(inputs[name] for name in COBEVT_INPUT_NAMES)
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                args,
                str(destination),
                export_params=True,
                opset_version=self.opset_version,
                do_constant_folding=True,
                input_names=list(COBEVT_INPUT_NAMES),
                output_names=list(COBEVT_OUTPUT_NAMES),
                custom_opsets={"trt": 1},
            )
        model_proto = onnx.load(str(destination))
        onnx.checker.check_model(model_proto)
        operator_report = audit_onnx_operators(destination)
        return CobevtTypedExportReport(
            onnx_path=str(destination),
            onnx_sha256=_sha256(destination),
            checker_passed=True,
            input_names=tuple(str(value.name) for value in model_proto.graph.input),
            output_names=tuple(str(value.name) for value in model_proto.graph.output),
            node_count=operator_report.node_count,
            op_counts=operator_report.op_counts,
            registered_custom_ops=operator_report.registered_custom_ops,
            unregistered_custom_ops=operator_report.unregistered_custom_ops,
        )

