"""Typed ONNX export recipe owned by the LiDAR CoBEVT family."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import onnx
import torch
import torch.nn as nn

from quantization.export.heal_lidar_cobevt import HEALLiDARCoBEVTSignalMaxK
from quantization.export.origin_mapping import (
    apply_canonical_node_names,
    build_onnx_origin_map,
)
from quantization.export.signal_maxk import capture_weighted_module_calls
from quantization.types import OnnxOriginMapResult

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
    weighted_entry_count: int
    functional_compute_count: int
    canonical_rename_count: int
    origin_map_hash: str
    origin_map: dict


def _cobevt_origin_map(origin: OnnxOriginMapResult) -> OnnxOriginMapResult:
    functional = []
    for ordinal, group in enumerate(origin.functional_compute_groups):
        if "affine_grid" in group.module_path:
            module_path = "lidar_cobevt.functional_affine_grid_matmul"
            source_call = "quantization.export.heal_lidar_cobevt._warp_agents"
            reason = "cobevt_grid_coordinates_require_fp16"
        else:
            module_path = f"lidar_cobevt.functional_matmul.{ordinal:03d}"
            source_call = "lidar_cobevt_parameter_free_functional_matmul"
            reason = "cobevt_functional_matmul_requires_fp16"
        functional.append(
            replace(
                group,
                module_path=module_path,
                canonical_node_name=f"__canonical__cobevt_functional_{ordinal:03d}",
                source_call=source_call,
                protection_reason=reason,
            )
        )
    return OnnxOriginMapResult(
        entries=list(origin.entries),
        source_onnx=origin.source_onnx,
        unresolved_weighted_nodes=list(origin.unresolved_weighted_nodes),
        functional_matmul_nodes=list(origin.functional_matmul_nodes),
        functional_compute_groups=functional,
        naming_policy_version=origin.naming_policy_version,
    )


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
            with capture_weighted_module_calls(wrapper) as calls:
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
        origin = _cobevt_origin_map(build_onnx_origin_map(destination, calls))
        rename = apply_canonical_node_names(
            destination,
            origin,
            output_path=destination,
            allow_custom_ops=True,
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
            weighted_entry_count=len(origin.entries),
            functional_compute_count=len(origin.functional_compute_groups),
            canonical_rename_count=rename.renamed_node_count,
            origin_map_hash=origin.origin_map_hash,
            origin_map=origin.to_dict(),
        )

