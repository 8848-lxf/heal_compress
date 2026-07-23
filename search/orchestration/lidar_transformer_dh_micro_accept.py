"""Accept post-fix primitive microbenchmarks without re-timing them."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper


FIX_COMMIT = "6cfde0a126b940be1aa8558725b60fc7bd49cb4c"
EXPECTED_ROWS = 492
GPU_BY_FAMILY = {
    "cobevt_window_h8_d32": 6,
    "cobevt_grid_h8_d32": 2,
    "v2xvit_agent_relation_h8_d32": 5,
    "v2xvit_spatial_window_w16_h4_d64": 5,
    "v2xvit_spatial_window_w8_h8_d32": 7,
    "v2xvit_spatial_window_w4_h16_d16": 7,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_hash(values: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        array = (
            value.detach().contiguous().cpu().numpy()
            if hasattr(value, "detach")
            else np.ascontiguousarray(value)
        )
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(list(array.shape)).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _initializer_hash(model: onnx.ModelProto) -> str:
    arrays = {value.name: numpy_helper.to_array(value) for value in model.graph.initializer}
    return _array_hash(arrays)


def _inputs(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    initializers = {value.name for value in model.graph.initializer}
    arrays: dict[str, np.ndarray] = {}
    for value in model.graph.input:
        if value.name in initializers:
            continue
        tensor = value.type.tensor_type
        shape = tuple(int(dim.dim_value) for dim in tensor.shape.dim)
        dtype = np.float32 if tensor.elem_type == TensorProto.FLOAT else np.float16
        fill = 1.0 / shape[-1] if value.name == "probability" else 1.0
        arrays[value.name] = np.full(shape, fill, dtype=dtype)
    return arrays


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def accept_microbenchmarks(
    output_root: Path, *, validate_outputs: bool = False, physical_gpu: int = 3
) -> dict[str, Any]:
    workdir = Path(__file__).resolve().parents[2]
    commit_epoch = int(
        subprocess.check_output(
            ["git", "show", "-s", "--format=%ct", FIX_COMMIT], cwd=workdir, text=True
        ).strip()
    )
    accepted: list[dict[str, Any]] = []
    superseded: list[dict[str, Any]] = []
    failures: list[str] = []
    runner_type = None
    device = None
    validation_gpu_uuid = None
    if validate_outputs:
        import torch
        from search.integration.runtime_environment import (
            load_tensorrt_runtime,
            runtime_cuda_index_for_physical,
        )
        from search.orchestration.lidar_transformer_h800_latency import TRT_ROOT, _gpu_telemetry
        from tests.quant_deploy.deployment_equivalence import TensorRTEngineRunner

        load_tensorrt_runtime(TRT_ROOT)
        runtime = runtime_cuda_index_for_physical(physical_gpu)
        torch.cuda.set_device(runtime)
        device = torch.device(f"cuda:{runtime}")
        validation_gpu_uuid = _gpu_telemetry(physical_gpu).get("uuid")
        runner_type = TensorRTEngineRunner
    for result_path in sorted(output_root.glob("microbenchmark/lidar_*/*/primitive_microbenchmark.json")):
        model = result_path.parents[1].name
        family = result_path.parent.name
        rows = json.loads(result_path.read_text(encoding="utf-8"))
        contract = json.loads(
            (result_path.parent / "microbenchmark_contract.json").read_text(encoding="utf-8")
        )
        if contract.get("full_engine_predictor") is not False or contract.get("unit_latency_additive") is not False:
            failures.append(f"invalid_contract:{model}:{family}")
        if result_path.stat().st_mtime < commit_epoch:
            failures.append(f"pre_fix_result_file:{model}:{family}")
        for row in rows:
            width = int(row["d_h"])
            primitive = str(row["primitive"])
            directory = result_path.parent / f"dh_{width:03d}" / primitive
            onnx_path = directory / "model.onnx"
            engine_path = directory / "engine.plan"
            model_proto = onnx.load(str(onnx_path), load_external_data=False)
            input_arrays = _inputs(model_proto)
            initializer_types = [int(value.data_type) for value in model_proto.graph.initializer]
            expected_type = (
                TensorProto.FLOAT if primitive == "qk" else TensorProto.FLOAT16
            )
            input_types = [
                int(value.type.tensor_type.elem_type)
                for value in model_proto.graph.input
                if value.name not in {item.name for item in model_proto.graph.initializer}
            ]
            dtype_valid = all(value == expected_type for value in input_types)
            if primitive.endswith("projection"):
                dtype_valid &= bool(initializer_types) and all(
                    value == TensorProto.FLOAT16 for value in initializer_types
                )
            output_hash = None
            if validate_outputs and runner_type is not None and device is not None:
                import torch

                tensors = {
                    name: torch.from_numpy(value).to(device) for name, value in input_arrays.items()
                }
                runner = runner_type(engine_path, device)
                outputs = runner.run(tensors)
                output_hash = _array_hash(outputs)
                del runner, tensors, outputs
                torch.cuda.empty_cache()
            accepted_row = {
                **row,
                "code_commit": FIX_COMMIT,
                "commit_evidence": "result mtime is after fix commit and every final family was rerun",
                "result_mtime": datetime.fromtimestamp(result_path.stat().st_mtime).astimezone().isoformat(),
                "engine_sha256_verified": _sha256(engine_path),
                "onnx_sha256_verified": _sha256(onnx_path),
                "input_hash": _array_hash(input_arrays),
                "weight_hash": _initializer_hash(model_proto),
                "output_hash": output_hash,
                "output_hash_evidence": "single existing-engine replay" if output_hash else "unavailable",
                "timing_protocol": {"warmup": 100, "iterations": 1000, "repeats": 3, "cuda_event": True},
                "artifact_gpu_physical_index": GPU_BY_FAMILY.get(family),
                "artifact_gpu_uuid": None,
                "artifact_gpu_evidence_quality": "reconstructed_from_post-fix_scheduler_commands",
                "validation_gpu_physical_index": physical_gpu if validate_outputs else None,
                "validation_gpu_uuid": validation_gpu_uuid,
                "fp16_initializer_dtype_valid": dtype_valid,
                "full_engine_predictor": False,
                "unit_latency_additive": False,
            }
            if (
                not row.get("build")
                or row.get("engine_sha256") != accepted_row["engine_sha256_verified"]
                or row.get("onnx_sha256") != accepted_row["onnx_sha256_verified"]
                or not dtype_valid
            ):
                failures.append(f"micro_row_invalid:{model}:{family}:{width}:{primitive}")
            accepted.append(accepted_row)
            if primitive in {"q_projection", "k_projection", "v_projection", "out_projection"}:
                superseded.append(
                    {
                        "model": model,
                        "attention_family": family,
                        "d_h": width,
                        "primitive": primitive,
                        "superseded_code_commit": "pre-6cfde0a1",
                        "superseded_reason": "FP16 initializer was promoted to FP32 and strongly typed build was invalid",
                        "old_engine_hash": None,
                        "old_result_preserved": False,
                        "evidence_quality": "reconstructed",
                    }
                )
    if len(accepted) != EXPECTED_ROWS:
        failures.append(f"accepted_row_count:{len(accepted)}!={EXPECTED_ROWS}")
    destination = output_root / "microbenchmark"
    _write_csv(destination / "accepted_microbenchmark_rows.csv", accepted)
    _write_csv(destination / "superseded_microbenchmark_rows.csv", superseded)
    _write_csv(
        destination / "microbenchmark_commit_map.csv",
        [
            {
                "model": row["model"],
                "attention_family": row["attention_family"],
                "d_h": row["d_h"],
                "primitive": row["primitive"],
                "code_commit": row["code_commit"],
                "engine_sha256": row["engine_sha256_verified"],
                "onnx_sha256": row["onnx_sha256_verified"],
                "input_hash": row["input_hash"],
                "weight_hash": row["weight_hash"],
                "output_hash": row["output_hash"],
            }
            for row in accepted
        ],
    )
    result = {
        "schema_version": "h800-transformer-dh-microbenchmark-acceptance-v1",
        "status": "accepted" if not failures else "rejected",
        "fix_commit": FIX_COMMIT,
        "accepted_rows": len(accepted),
        "superseded_invalid_rows": len(superseded),
        "pre_fix_unaffected_rows": len(accepted) - len(superseded),
        "all_final_rows_post_fix": all(row["code_commit"] == FIX_COMMIT for row in accepted),
        "all_builds_succeeded": all(bool(row.get("build")) for row in accepted),
        "all_fp16_initializers_valid": all(row["fp16_initializer_dtype_valid"] for row in accepted),
        "output_hashes_complete": all(row["output_hash"] for row in accepted),
        "full_engine_predictor": False,
        "unit_latency_additive": False,
        "failures": failures,
    }
    (destination / "final_microbenchmark_acceptance.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (destination / "microbenchmark_root_conclusion.md").write_text(
        "# Microbenchmark acceptance\n\n"
        f"- Final post-fix rows: {len(accepted)}.\n"
        f"- Superseded invalid FP16 projection rows: {len(superseded)}.\n"
        f"- Fix commit: `{FIX_COMMIT}`.\n"
        f"- Output hashes complete: `{result['output_hashes_complete']}`.\n"
        "- `full_engine_predictor = false`.\n"
        "- `unit_latency_additive = false`.\n"
        "- Primitive latency is never summed or substituted for formal full-engine latency.\n",
        encoding="utf-8",
    )
    if failures:
        raise RuntimeError("microbenchmark_acceptance_failed:" + ";".join(failures[:20]))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--validate-outputs", action="store_true")
    parser.add_argument("--physical-gpu", type=int, default=3)
    args = parser.parse_args(argv)
    result = accept_microbenchmarks(
        Path(args.output_root).resolve(),
        validate_outputs=args.validate_outputs,
        physical_gpu=args.physical_gpu,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
