#!/usr/bin/env python3
"""Build representative strongly-typed engines for derived window joins."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from quantization.config import TensorRTBuildConfig
from quantization.types import (
    CanonicalPrecisionEntry,
    CanonicalPrecisionMappingResult,
    stable_json_hash,
)
from search.quantization_space.v2xvit_av_merge import derive_merge_precision
from search.stage2.trt_modelopt import build_engine_modelopt


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_graph(path: Path, *, left: str, right: str, derived: str, scales: tuple[float, float]) -> None:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    inputs = [
        helper.make_tensor_value_info("left", TensorProto.FLOAT, [1, 2, 64, 128, 256]),
        helper.make_tensor_value_info("right", TensorProto.FLOAT, [1, 2, 64, 128, 256]),
    ]
    nodes = []
    initializers = []

    def boundary(source: str, precision: str, scale: float, side: str) -> str:
        if precision == "FP32":
            return source
        if precision == "FP16":
            output = f"{source}_{side}_fp16"
            nodes.append(helper.make_node("Cast", [source], [output], name=f"{side}_to_fp16", to=TensorProto.FLOAT16))
            return output
        scale_name = f"{side}_scale"
        zero_name = f"{side}_zero"
        quantized = f"{source}_{side}_q"
        dequantized = f"{source}_{side}_dq"
        initializers.extend([
            numpy_helper.from_array(np.asarray(scale, dtype=np.float32), scale_name),
            numpy_helper.from_array(np.asarray(0, dtype=np.int8), zero_name),
        ])
        nodes.extend([
            helper.make_node("QuantizeLinear", [source, scale_name, zero_name], [quantized], name=f"{side}_q"),
            helper.make_node("DequantizeLinear", [quantized, scale_name, zero_name], [dequantized], name=f"{side}_dq"),
        ])
        return dequantized

    left_tensor = boundary("left", left, scales[0], "left")
    right_tensor = boundary("right", right, scales[1], "right")
    if derived == "FP16":
        for side, value in (("left", left_tensor), ("right", right_tensor)):
            precision = left if side == "left" else right
            if precision != "FP16":
                output = f"{value}_{side}_join_fp16"
                nodes.append(helper.make_node("Cast", [value], [output], name=f"{side}_join_to_fp16", to=TensorProto.FLOAT16))
                if side == "left":
                    left_tensor = output
                else:
                    right_tensor = output
    elif derived == "FP32":
        for side, value in (("left", left_tensor), ("right", right_tensor)):
            precision = left if side == "left" else right
            if precision != "FP32":
                output = f"{value}_{side}_join_fp32"
                nodes.append(helper.make_node("Cast", [value], [output], name=f"{side}_join_to_fp32", to=TensorProto.FLOAT))
                if side == "left":
                    left_tensor = output
                else:
                    right_tensor = output
    nodes.append(helper.make_node(
        "Add", [left_tensor, right_tensor], ["merged"],
        name="__canonical__window_merge",
    ))
    output_type = {"FP32": TensorProto.FLOAT, "FP16": TensorProto.FLOAT16, "INT8": TensorProto.FLOAT}[derived]
    output = helper.make_tensor_value_info("merged", output_type, [1, 2, 64, 128, 256])
    graph = helper.make_graph(nodes, "v2xvit_window_merge", inputs, [output], initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)])
    onnx.checker.check_model(model)
    onnx.save(model, path)


def inspector_exact(layer_info: Path, requested: str) -> dict[str, Any]:
    from quantization.tensorrt.layer_info import load_layer_info, layer_metadata, layer_name

    rows = [row for row in load_layer_info(layer_info) if "window_merge" in layer_metadata(row) or "window_merge" in layer_name(row)]
    token = {"FP32": "float", "FP16": "half", "INT8": "int8"}[requested]
    exact = []
    for row in rows:
        inputs = [str(value.get("Format/Datatype", "")).lower() for value in row.get("Inputs", ())]
        outputs = [str(value.get("Format/Datatype", "")).lower() for value in row.get("Outputs", ())]
        if len(inputs) >= 2 and all(token in value for value in inputs[:2]) and outputs:
            exact.append(row)
    return {
        "requested": requested,
        "matched_layer_names": [layer_name(row) for row in rows],
        "exact_layer_names": [layer_name(row) for row in exact],
        "requested_realized_exact": len(exact) == 1,
        "unmapped": not rows,
        "conflict": len(exact) != 1,
        "fallback": bool(rows) and not exact,
    }


def engine_was_built(build: dict[str, Any], engine_path: Path) -> bool:
    """Accept a raw successful micro-engine before weighted-layer validation.

    The shared wrapper subsequently applies a weighted canonical structure
    checker.  A parameter-free Add intentionally has no weighted module, so
    that checker reports ``engine_structure_validation_failed`` even though
    trtexec produced the engine and layer inspector successfully.  This audit
    uses its own exact Add inspector below and never relaxes production model
    validation.
    """

    return bool(
        engine_path.is_file()
        and isinstance(build.get("build"), dict)
        and build["build"].get("success") is True
        and int(build["build"].get("returncode", -1)) == 0
    )


def run(args: argparse.Namespace) -> int:
    if args.physical_gpu not in {2, 3}:
        raise RuntimeError(f"v2xvit_merge_audit_gpu_not_allowed:{args.physical_gpu}")
    root = args.output_root.resolve() / "merge_audit"
    root.mkdir(parents=True, exist_ok=True)
    cases = [
        ("all_fp32", "FP32", "FP32", (None, None), {}),
        ("all_fp16", "FP16", "FP16", (None, None), {}),
        ("all_int8_compatible", "INT8", "INT8", (0.1, 0.1), {"add_int8": True}),
        ("mixed_int8_fp16", "INT8", "FP16", (0.1, None), {"add_int8": True}),
        ("mixed_fp16_fp32", "FP16", "FP32", (None, None), {}),
    ]
    results = []
    for name, left, right, scales, capability in cases:
        case = root / name
        case.mkdir(parents=True, exist_ok=True)
        result_path = case / "result.json"
        if result_path.is_file():
            results.append(json.loads(result_path.read_text()))
            continue
        derived = derive_merge_precision("Add", [left, right], scales, trt_capability=capability)
        requested = str(derived["derived_precision"])
        onnx_path = case / "window_merge.onnx"
        build_graph(
            onnx_path,
            left=left,
            right=right,
            derived=requested,
            scales=(float(scales[0] or 0.1), float(scales[1] or 0.1)),
        )
        mapping = CanonicalPrecisionMappingResult(
            entries=[CanonicalPrecisionEntry(
                module_path="v2xvit.window_merge.audit",
                canonical_node_name="__canonical__window_merge",
                precision_group="derived_window_merge",
                requested_precision=requested.lower(),
                realized_request_precision=requested.lower(),
                realized_output_precision=requested.lower(),
                onnx_op_type="Add",
            )],
            profile_id=f"window_merge_{name}",
            profile_hash=stable_json_hash({"case": name, "derived": derived}),
            origin_map_hash=stable_json_hash({"synthetic_real_shape": [1, 2, 64, 128, 256]}),
            policy_version="v2xvit-window-merge-derived-join-v1",
        )
        engine = case / "window_merge.plan"
        build = build_engine_modelopt(
            qdq_onnx=onnx_path,
            engine_path=engine,
            precision_mapping=mapping,
            build_config=TensorRTBuildConfig(
                trtexec_path=args.tensorrt_root / "bin/trtexec",
                workspace_mib=1024,
                timeout_seconds=1200,
                no_tf32=True,
                skip_inference=True,
                export_layer_info=True,
                strongly_typed=True,
                enable_fp16=False,
                enable_int8=False,
                policy_version="v2xvit-window-merge-derived-join-v1",
            ),
            physical_snapshot={
                "snapshot_schema_version": "physical-structure-snapshot-v2",
                "schema_version": "physical-structure-snapshot-v2",
                "model_family": "v2xvit_window_merge_micro_contract",
                "modules": [],
                "parameter_count": 0,
                "snapshot_hash": stable_json_hash({"case": name}),
            },
            output_dir=case / "engine_build",
            tensorrt_root=args.tensorrt_root,
            conda_env="modelopt",
            gpu_id=args.physical_gpu,
        )
        raw_engine_built = engine_was_built(build, engine)
        audit = (
            inspector_exact(case / "engine_build/engine_layer_info.json", requested)
            if raw_engine_built else
            {"requested": requested, "requested_realized_exact": False, "unmapped": True, "conflict": True, "fallback": False}
        )
        # If a genuine INT8 Add is unavailable, freeze the derived capability
        # to FP16 and prove that promoted state in a separate engine.
        promoted = None
        if name == "all_int8_compatible" and not audit["requested_realized_exact"]:
            promoted_dir = case / "promoted_fp16"
            promoted_dir.mkdir(parents=True, exist_ok=True)
            promoted_derived = derive_merge_precision(
                "Add", [left, right], scales, trt_capability={"add_int8": False}
            )
            promoted_onnx = promoted_dir / "window_merge.onnx"
            build_graph(promoted_onnx, left=left, right=right, derived="FP16", scales=(0.1, 0.1))
            promoted_mapping = CanonicalPrecisionMappingResult(
                entries=[CanonicalPrecisionEntry(
                    module_path="v2xvit.window_merge.audit",
                    canonical_node_name="__canonical__window_merge",
                    precision_group="derived_window_merge",
                    requested_precision="fp16",
                    realized_request_precision="fp16",
                    realized_output_precision="fp16",
                    onnx_op_type="Add",
                )],
                profile_id="window_merge_all_int8_promoted_fp16",
                profile_hash=stable_json_hash(promoted_derived),
                origin_map_hash=mapping.origin_map_hash,
                policy_version=mapping.policy_version,
            )
            promoted_engine = promoted_dir / "window_merge.plan"
            promoted_build = build_engine_modelopt(
                qdq_onnx=promoted_onnx, engine_path=promoted_engine,
                precision_mapping=promoted_mapping,
                build_config=TensorRTBuildConfig(
                    trtexec_path=args.tensorrt_root / "bin/trtexec", workspace_mib=1024,
                    timeout_seconds=1200, no_tf32=True, skip_inference=True,
                    export_layer_info=True, strongly_typed=True, enable_fp16=False,
                    enable_int8=False, policy_version=mapping.policy_version,
                ),
                physical_snapshot={"snapshot_schema_version": "physical-structure-snapshot-v2", "schema_version": "physical-structure-snapshot-v2", "model_family": "v2xvit_window_merge_micro_contract", "modules": [], "parameter_count": 0, "snapshot_hash": stable_json_hash({"case": name, "promoted": True})},
                output_dir=promoted_dir / "engine_build", tensorrt_root=args.tensorrt_root,
                conda_env="modelopt", gpu_id=args.physical_gpu,
            )
            promoted_raw_engine_built = engine_was_built(promoted_build, promoted_engine)
            promoted_audit = inspector_exact(
                promoted_dir / "engine_build/engine_layer_info.json", "FP16"
            ) if promoted_raw_engine_built else {"requested_realized_exact": False}
            promoted = {"derived": promoted_derived, "build": promoted_build, "audit": promoted_audit, "engine_sha256": sha256(promoted_engine) if promoted_engine.is_file() else None}
        final_exact = bool(promoted["audit"]["requested_realized_exact"] if promoted else audit["requested_realized_exact"])
        row = {
            "case": name,
            "input_precisions": [left, right],
            "derived": derived,
            "build_status": build.get("status"),
            "raw_engine_built": raw_engine_built,
            "audit": audit,
            "promoted": promoted,
            "final_requested_realized_exact": final_exact,
            "final_derived_precision": "FP16" if promoted else requested,
            "engine_sha256": sha256(engine) if engine.is_file() else None,
        }
        write_json(result_path, row)
        results.append(row)
    acceptance = {
        "cases": results,
        "case_count": len(results),
        "derived_join_valid": len(results) == 5 and all(row["final_requested_realized_exact"] for row in results),
        "requested_realized_exact": all(row["final_requested_realized_exact"] for row in results),
        "all_int8_add_supported": next(row for row in results if row["case"] == "all_int8_compatible")["promoted"] is None,
    }
    write_json(args.output_root / "reports/window_merge_acceptance.json", acceptance)
    write_json(args.output_root / "reports/window_merge_requested_realized.json", {"cases": results})
    write_json(args.output_root / "reports/window_merge_derived_join_cases.json", {"cases": results})
    csv_path = args.output_root / "reports/window_merge_derived_join_cases.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fields = (
            "case", "input_precisions", "requested_derived_precision",
            "final_derived_precision", "build_status",
            "requested_realized_exact", "unmapped", "conflict", "fallback",
            "promoted_from_unsupported_int8", "engine_sha256",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in results:
            audit = row["audit"]
            writer.writerow({
                "case": row["case"],
                "input_precisions": "+".join(row["input_precisions"]),
                "requested_derived_precision": row["derived"]["derived_precision"],
                "final_derived_precision": row["final_derived_precision"],
                "build_status": row["build_status"],
                "requested_realized_exact": row["final_requested_realized_exact"],
                "unmapped": audit.get("unmapped", False),
                "conflict": audit.get("conflict", False),
                "fallback": audit.get("fallback", False),
                "promoted_from_unsupported_int8": row["promoted"] is not None,
                "engine_sha256": row["engine_sha256"],
            })
    if not acceptance["derived_join_valid"]:
        raise RuntimeError("v2xvit_window_merge_derived_join_real_engine_failed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--tensorrt-root", type=Path, default=Path("/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118"))
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
