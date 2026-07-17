"""Strongly typed deployment ownership for the LiDAR CoBEVT family."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quantization.config import TensorRTBuildConfig
from quantization.tensorrt.command import build_trt_command
from quantization.types import (
    CanonicalPrecisionMappingResult,
    TensorRTCommandResult,
)
from search.hashing import canonical_json_hash


def model_family_deployment_identity(
    *,
    model_family: str,
    physical_hash: str,
    precision_hash: str,
    calibration_signature: str,
    build_signature: str,
    recipe_version: str = "",
    fixed_k: int = 0,
) -> str:
    fields = {
        "build_signature": str(build_signature),
        "calibration_signature": str(calibration_signature),
        "fixed_k": int(fixed_k),
        "model_family": str(model_family),
        "physical_hash": str(physical_hash),
        "precision_hash": str(precision_hash),
        "recipe_version": str(recipe_version),
    }
    missing = [
        key
        for key in (
            "build_signature",
            "calibration_signature",
            "model_family",
            "physical_hash",
            "precision_hash",
        )
        if not fields[key]
    ]
    if missing:
        raise ValueError(f"model_family_deployment_identity_missing:{','.join(missing)}")
    return canonical_json_hash(fields)


@dataclass(frozen=True)
class CobevtDeploymentRecipe:
    tensorrt_root: Path
    trtexec_path: Path
    plugin_path: Path
    fixed_k: int
    workspace_mib: int = 512

    MODEL_FAMILY = "lidar_cobevt"
    RECIPE_VERSION = "lidar-cobevt-strongly-typed-deployment-v1"

    def __post_init__(self) -> None:
        if int(self.fixed_k) <= 0:
            raise ValueError("cobevt_fixed_k_must_be_positive")

    @property
    def model_family(self) -> str:
        return self.MODEL_FAMILY

    @property
    def plugin_boundary_dtype(self) -> str:
        return "fp32"

    def build_config(self) -> TensorRTBuildConfig:
        return TensorRTBuildConfig(
            trtexec_path=Path(self.trtexec_path),
            plugin_path=Path(self.plugin_path),
            workspace_mib=int(self.workspace_mib),
            shape_profiles={},
            precision_constraints="none",
            enable_fp16=False,
            enable_int8=False,
            no_tf32=True,
            skip_inference=True,
            export_layer_info=True,
            strongly_typed=True,
            production_mode=True,
            plugin_boundary_dtype=self.plugin_boundary_dtype,
            policy_version=self.RECIPE_VERSION,
        )

    def builder_command(
        self,
        *,
        onnx_path: str | Path,
        engine_path: str | Path,
        mapping: CanonicalPrecisionMappingResult,
        layer_info_path: str | Path | None = None,
    ) -> TensorRTCommandResult:
        return build_trt_command(
            onnx_path,
            engine_path,
            mapping,
            config=self.build_config(),
            layer_info_path=layer_info_path,
        )

    def deployment_identity(
        self,
        *,
        physical_hash: str,
        precision_hash: str,
        calibration_signature: str,
        build_signature: str,
    ) -> str:
        return model_family_deployment_identity(
            model_family=self.model_family,
            physical_hash=physical_hash,
            precision_hash=precision_hash,
            calibration_signature=calibration_signature,
            build_signature=build_signature,
            recipe_version=self.RECIPE_VERSION,
            fixed_k=self.fixed_k,
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "fixed_k": int(self.fixed_k),
            "model_family": self.model_family,
            "plugin_boundary_dtype": self.plugin_boundary_dtype,
            "production_mode": True,
            "recipe_version": self.RECIPE_VERSION,
            "strongly_typed": True,
            "tensorrt_root": str(self.tensorrt_root),
        }


__all__ = ["CobevtDeploymentRecipe", "model_family_deployment_identity"]

