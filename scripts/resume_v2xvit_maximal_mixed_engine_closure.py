#!/usr/bin/env python3
"""Resume the single maximal-mixed audit at the TensorRT build boundary.

The initial audit run deliberately failed before TensorRT because its
canonical functional mapping was incomplete.  This runner consumes the
already provenance-bound ONNX/QDQ/train200 artifacts and performs exactly one
engine build; it never exports, calibrates, searches, or mutates a candidate.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from quantization.config import TensorRTBuildConfig
from quantization.types import CanonicalPrecisionEntry, CanonicalPrecisionMappingResult
from search.stage2.transformer_precision_export import audit_trt_attention_fp32_contract
from search.stage2.trt_modelopt import build_engine_modelopt
from search.stage2.v2xvit_functional_precision import audit_trt_v2xvit_functional_precision


def _read(path: Path) -> dict:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _write(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _mapping(payload: dict) -> CanonicalPrecisionMappingResult:
    value = dict(payload)
    value["entries"] = [
        CanonicalPrecisionEntry(**dict(row)) for row in value.get("entries", ())
    ]
    return CanonicalPrecisionMappingResult(**value)


def run(args: argparse.Namespace) -> int:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise RuntimeError(
            "resume_maximal_mixed_gpu_visibility_mismatch:"
            f"{os.environ.get('CUDA_VISIBLE_DEVICES')}!={args.physical_gpu}"
        )
    root = args.engine_dir.resolve()
    required = (
        "physical_mixed_qdq.onnx",
        "canonical_precision_mapping.json",
        "maximal_mixed_manifest.json",
        "calibration_manifest.json",
        "functional_precision_onnx_mapping.json",
        "qdq_attention_fp32_audit.json",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise RuntimeError(f"resume_maximal_mixed_missing_artifacts:{missing}")
    calibration = _read(root / "calibration_manifest.json")
    if (
        int(calibration.get("processed_frames", -1)) != 200
        or int(calibration.get("skipped_frames", -1)) != 0
    ):
        raise RuntimeError("resume_maximal_mixed_train200_contract_failed")
    functional_onnx = _read(root / "functional_precision_onnx_mapping.json")
    if not bool(functional_onnx.get("passed")):
        raise RuntimeError("resume_maximal_mixed_functional_onnx_not_exact")
    build_dir = root / "engine_build"
    if build_dir.exists():
        raise RuntimeError(f"resume_maximal_mixed_build_already_attempted:{build_dir}")
    trtexec = args.tensorrt_root / "bin" / "trtexec"
    if not trtexec.is_file():
        trtexec = args.tensorrt_root / "targets/x86_64-linux-gnu/bin/trtexec"
    if not trtexec.is_file() or not args.plugin.is_file():
        raise RuntimeError(f"resume_maximal_mixed_tool_missing:{trtexec}:{args.plugin}")
    manifest = _read(root / "maximal_mixed_manifest.json")
    build = build_engine_modelopt(
        qdq_onnx=root / "physical_mixed_qdq.onnx",
        engine_path=root / "candidate.plan",
        precision_mapping=_mapping(_read(root / "canonical_precision_mapping.json")),
        build_config=TensorRTBuildConfig(
            trtexec_path=trtexec,
            plugin_path=args.plugin,
            workspace_mib=4096,
            timeout_seconds=3600,
            no_tf32=True,
            skip_inference=True,
            export_layer_info=True,
            strongly_typed=True,
            enable_fp16=False,
            enable_int8=False,
            policy_version="v2xvit-greedy005-w8a8-qk-fp32-v1",
        ),
        physical_snapshot=manifest["physical_snapshot"],
        output_dir=build_dir,
        tensorrt_root=args.tensorrt_root,
        conda_env="modelopt",
        gpu_id=args.physical_gpu,
    )
    _write(root / "engine_build_acceptance.json", build)
    if build.get("status") != "ok":
        raise RuntimeError(
            f"resume_maximal_mixed_trt_build_failed:{build.get('failure_reason', '')}"
        )
    layer_info = build_dir / "engine_layer_info.json"
    attention = audit_trt_attention_fp32_contract(
        layer_info, _read(root / "qdq_attention_fp32_audit.json")
    )
    _write(root / "trt_attention_fp32_audit.json", attention)
    functional = audit_trt_v2xvit_functional_precision(layer_info, functional_onnx)
    _write(root / "functional_precision_trt_audit.json", functional)
    weighted = dict(build.get("precision_realization_validation") or {})
    acceptance = {
        "schema_version": "v2xvit-maximal-mixed-closure-acceptance-v2",
        "passed": bool(
            weighted.get("passed")
            and attention.get("passed")
            and functional.get("passed")
            and int(functional.get("conflict_count", 1)) == 0
            and int(functional.get("fallback_count", 1)) == 0
            and int(functional.get("unmapped_count", 1)) == 0
        ),
        "resumed_at_engine_boundary": True,
        "export_repeated": False,
        "calibration_repeated": False,
        "engine_build_count": 1,
        "train200_processed": int(calibration.get("processed_frames", -1)),
        "train200_skipped": int(calibration.get("skipped_frames", -1)),
        "weighted_requested_realized_exact": bool(weighted.get("passed")),
        "attention_fp32_contract_exact": bool(attention.get("passed")),
        "functional_requested_realized_exact": bool(functional.get("passed")),
        "conflict_count": int(functional.get("conflict_count", 0))
        + len(weighted.get("mismatches", ())),
        "fallback_count": int(functional.get("fallback_count", 0)),
        "unmapped_count": int(functional.get("unmapped_count", 0))
        + int(weighted.get("unresolved_layer_count", 0)),
        "engine_path": str(root / "candidate.plan"),
        "candidate_hash": manifest.get("candidate_hash"),
        "physical_hash": manifest.get("physical_hash"),
        "worker_gpu": (build.get("worker_invocation") or {}).get(
            "cuda_visible_devices", ""
        ),
    }
    _write(root / "maximal_mixed_closure_acceptance.json", acceptance)
    print(json.dumps(acceptance, sort_keys=True), flush=True)
    if not acceptance["passed"]:
        raise RuntimeError(f"resume_maximal_mixed_closure_failed:{acceptance}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-dir", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--plugin", type=Path, required=True)
    parser.add_argument(
        "--tensorrt-root",
        type=Path,
        default=Path("/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
