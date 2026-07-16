#!/usr/bin/env python3
"""Rebuild only Q/DQ graph and TensorRT engine for strong-type parser diagnosis.

This command intentionally reuses an existing base ONNX, canonical mapping,
calibration scales, physical snapshot and plugin.  It is not an acceptance
evaluation; use the production runner with fresh calibration after parsing is
closed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _mapping(payload: dict[str, object]) -> object:
    from quantization.types import CanonicalPrecisionEntry, CanonicalPrecisionMappingResult

    data = dict(payload)
    data["entries"] = [CanonicalPrecisionEntry(**dict(row)) for row in data.get("entries", [])]
    return CanonicalPrecisionMappingResult(**data)


def _shape_profiles(fixed_k: int) -> dict[str, dict[str, tuple[int, ...]]]:
    return {
        "pairwise_t_matrix": {"min": (1, 1, 1, 4, 4), "opt": (1, 2, 2, 4, 4), "max": (1, 2, 2, 4, 4)},
        "valid_voxel_mask": {"min": (fixed_k,), "opt": (fixed_k,), "max": (fixed_k,)},
        "voxel_coords": {"min": (fixed_k, 4), "opt": (fixed_k, 4), "max": (fixed_k, 4)},
        "voxel_features": {"min": (fixed_k, 32, 4), "opt": (fixed_k, 32, 4), "max": (fixed_k, 32, 4)},
        "voxel_num_points": {"min": (fixed_k,), "opt": (fixed_k,), "max": (fixed_k,)},
    }


def main(args: argparse.Namespace) -> int:
    from quantization.config import QDQConfig, TensorRTBuildConfig
    from quantization.precision.qdq_inserter import insert_explicit_qdq
    from search.integration.trt_compatible_export import make_pointpillar_domain_compatible
    from search.stage2.trt_modelopt import build_engine_modelopt

    source = args.source_artifacts.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    required = {
        "base_onnx": source / "pruned_fp32.onnx",
        "mapping": source / "canonical_layer_map.json",
        "scales": source / "calibration_scales.json",
        "snapshot": source / "physical_snapshot.json",
        "plugin": args.plugin.expanduser().resolve(),
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"strong_type_diagnostic_inputs_missing:{missing}")
    mapping = _mapping(json.loads(required["mapping"].read_text(encoding="utf-8")))
    scales = json.loads(required["scales"].read_text(encoding="utf-8"))
    qdq = insert_explicit_qdq(
        required["base_onnx"],
        output / "qdq.onnx",
        mapping,
        scales=scales,
        config=QDQConfig(
            allowed_precisions=("fp32", "fp16", "int8"),
            grouped_conv_int8_allowed_channels_per_group=(4, 8, 16, 32, 64, 128, 256, 512),
        ),
        calibration_metadata={
            "diagnostic_only": True,
            "scale_reuse_source": str(required["scales"]),
        },
    )
    compatibility = make_pointpillar_domain_compatible(
        output / "qdq.onnx",
        output / "qdq_trt_compatible.onnx",
    )
    root = args.tensorrt_root.expanduser().resolve()
    trtexec = root / "targets/x86_64-linux-gnu/bin/trtexec"
    if not trtexec.is_file():
        trtexec = root / "bin/trtexec"
    build_config = TensorRTBuildConfig(
        trtexec_path=trtexec,
        plugin_path=required["plugin"],
        shape_profiles=_shape_profiles(args.fixed_k),
        strongly_typed=True,
        policy_version="strongly-typed-parser-diagnostic-v1",
    )
    result = build_engine_modelopt(
        qdq_onnx=output / "qdq_trt_compatible.onnx",
        engine_path=output / "engine.plan",
        precision_mapping=mapping,
        build_config=build_config,
        physical_snapshot=json.loads(required["snapshot"].read_text(encoding="utf-8")),
        output_dir=output,
        tensorrt_root=root,
        conda_env=args.conda_env,
        gpu_id=args.gpu,
    )
    _write_json(output / "qdq_report.json", qdq.to_dict())
    _write_json(output / "onnx_domain_compatibility_report.json", compatibility)
    _write_json(
        output / "diagnostic_result.json",
        {
            "status": result.get("status"),
            "diagnostic_only": True,
            "source_artifacts": str(source),
            "engine": result,
        },
    )
    print(json.dumps({"status": result.get("status"), "output": str(output)}, sort_keys=True))
    return 0 if result.get("status") == "ok" else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=6)
    parser.add_argument("--fixed-k", type=int, default=29696)
    parser.add_argument("--conda-env", default="modelopt")
    parser.add_argument(
        "--tensorrt-root",
        type=Path,
        default=Path("/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118"),
    )
    parser.add_argument(
        "--plugin",
        type=Path,
        default=REPO / "quantization/plugins/pointpillar_scatter_trt/build/libpointpillar_scatter_trt.so",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
