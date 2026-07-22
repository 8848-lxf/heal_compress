"""Primitive TensorRT micro-engines for real Transformer attention shapes.

The microbenchmarks explain tactic/alignment transitions only.  They are not
summed to predict full-engine latency and never replace the full-engine timing
contract.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from search.model_families.transformer.dh_alignment_audit import (
    _walk_layer_info,
    audit_engine_alignment,
)
from search.model_families.transformer.dh_candidate_grid import (
    adjacent_microbenchmark_pairs,
    dense_head_dimension_grid,
)
from search.orchestration.lidar_transformer_h800_latency import _time_engine


TRT_ROOT = Path("/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118")
TRTEXEC = TRT_ROOT / "bin" / "trtexec"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})


def _attention_shape(model: str, family: Mapping[str, Any], runtime: Mapping[str, Any]) -> dict[str, int | str]:
    family_paths = set(str(value) for value in family["module_paths"])
    row = next(value for value in runtime["attention_primitives"] if str(value["module_path"]) in family_paths)
    shape = [int(value) for value in row["input_shape"]]
    if str(family["attention_kind"]) == "agent_relation":
        batch, agents, height, width, embed = shape
        return {"kind": "agent_relation", "tokens": batch * agents * height * width, "instances": batch * height * width, "sequence": agents, "embed": embed}
    if model == "lidar_cobevt":
        batch, agents, grid_h, grid_w, win_h, win_w, embed = shape
        return {"kind": "standard", "tokens": batch * agents * grid_h * grid_w * win_h * win_w, "instances": batch * grid_h * grid_w, "sequence": agents * win_h * win_w, "embed": embed}
    batch, agents, height, width, embed = shape
    window = int(family["window_size"])
    return {"kind": "standard", "tokens": batch * agents * height * width, "instances": batch * agents * (height // window) * (width // window), "sequence": window * window, "embed": embed}


def _linear_model(path: Path, *, name: str, tokens: int, input_width: int, output_width: int, dtype: int) -> dict[str, np.ndarray]:
    np_dtype = np.float32 if dtype == TensorProto.FLOAT else np.float16
    # ``float16_array / Python int`` may promote the initializer to float32.
    # Strongly typed TensorRT then rejects the MatMul because the runtime input
    # remains FP16.  Construct the initializer at the requested dtype directly.
    weight = np.full(
        (input_width, output_width),
        1.0 / max(input_width, 1),
        dtype=np_dtype,
    )
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["input", "weight"], ["output"], name=f"{name}_projection")],
        name,
        [helper.make_tensor_value_info("input", dtype, [tokens, input_width])],
        [helper.make_tensor_value_info("output", dtype, [tokens, output_width])],
        [numpy_helper.from_array(weight, "weight")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return {"input": np.ones((tokens, input_width), dtype=np_dtype)}


def _standard_attention_model(path: Path, *, name: str, instances: int, heads: int, sequence: int, d_h: int, qk: bool, dtype: int) -> dict[str, np.ndarray]:
    np_dtype = np.float32 if dtype == TensorProto.FLOAT else np.float16
    if qk:
        inputs = [
            helper.make_tensor_value_info("q", dtype, [instances, heads, sequence, d_h]),
            helper.make_tensor_value_info("k", dtype, [instances, heads, sequence, d_h]),
        ]
        nodes = [
            helper.make_node("Transpose", ["k"], ["kt"], perm=[0, 1, 3, 2], name="qk_transpose"),
            helper.make_node("MatMul", ["q", "kt"], ["output"], name="qk_matmul"),
        ]
        output = helper.make_tensor_value_info("output", dtype, [instances, heads, sequence, sequence])
        arrays = {
            "q": np.ones((instances, heads, sequence, d_h), dtype=np_dtype),
            "k": np.ones((instances, heads, sequence, d_h), dtype=np_dtype),
        }
    else:
        inputs = [
            helper.make_tensor_value_info("probability", dtype, [instances, heads, sequence, sequence]),
            helper.make_tensor_value_info("value", dtype, [instances, heads, sequence, d_h]),
        ]
        nodes = [helper.make_node("MatMul", ["probability", "value"], ["output"], name="av_matmul")]
        output = helper.make_tensor_value_info("output", dtype, [instances, heads, sequence, d_h])
        arrays = {
            "probability": np.full((instances, heads, sequence, sequence), 1.0 / sequence, dtype=np_dtype),
            "value": np.ones((instances, heads, sequence, d_h), dtype=np_dtype),
        }
    graph = helper.make_graph(nodes, name, inputs, [output])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return arrays


def _relation_model(path: Path, *, name: str, instances: int, heads: int, sequence: int, d_h: int, qk: bool, dtype: int) -> dict[str, np.ndarray]:
    np_dtype = np.float32 if dtype == TensorProto.FLOAT else np.float16
    if qk:
        shapes = {"q": (1, heads, instances, sequence, d_h), "relation": (1, heads, sequence, sequence, d_h, d_h), "k": (1, heads, instances, sequence, d_h)}
        equation = "bmlip,bmijpq,bmljq->bmlij"
    else:
        shapes = {"probability": (1, heads, instances, sequence, sequence), "relation": (1, heads, sequence, sequence, d_h, d_h), "value": (1, heads, instances, sequence, d_h)}
        equation = "bmlij,bmijpc,bmljp->bmlic"
    inputs = [helper.make_tensor_value_info(key, dtype, list(shape)) for key, shape in shapes.items()]
    output_shape = [1, heads, instances, sequence, sequence if qk else d_h]
    graph = helper.make_graph(
        [helper.make_node("Einsum", list(shapes), ["output"], equation=equation, name="relation_qk" if qk else "relation_av")],
        name, inputs, [helper.make_tensor_value_info("output", dtype, output_shape)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return {key: np.ones(shape, dtype=np_dtype) for key, shape in shapes.items()}


def _build(directory: Path, *, physical_gpu: int) -> tuple[bool, str]:
    command = [
        str(TRTEXEC), f"--onnx={directory / 'model.onnx'}", f"--saveEngine={directory / 'engine.plan'}",
        "--stronglyTyped", "--noTF32", "--skipInference", "--profilingVerbosity=detailed",
        "--memPoolSize=workspace:8192", f"--exportLayerInfo={directory / 'engine_layer_info.json'}",
    ]
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    environment["LD_LIBRARY_PATH"] = f"{TRT_ROOT / 'lib'}:{environment.get('LD_LIBRARY_PATH', '')}"
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment)
    (directory / "engine_build.log").write_text(completed.stdout, encoding="utf-8")
    return completed.returncode == 0 and (directory / "engine.plan").is_file(), " ".join(command)


def run(*, output_root: Path, model: str, family_id: str, physical_gpu: int, warmup: int = 100, iterations: int = 1000, repeats: int = 3) -> list[dict[str, Any]]:
    import torch
    from search.integration.runtime_environment import load_tensorrt_runtime, runtime_cuda_index_for_physical

    families = json.loads((output_root / "inventory" / f"{model.removeprefix('lidar_')}_attention_families.json").read_text(encoding="utf-8"))
    family = next(row for row in families if str(row["family_id"]) == family_id)
    d0 = int(family["original_d_h"])
    width_values = [
        row.d_h
        for row in dense_head_dimension_grid(
            d0, heads=int(family["heads"]), low_width_extension=d0 <= 16
        )
    ]
    pairs = adjacent_microbenchmark_pairs(width_values)
    widths = sorted({value for pair in pairs for value in pair}, reverse=True)
    runtime = json.loads((output_root / "structures" / model / family_id / f"dh_{d0:03d}" / "runtime_costs.json").read_text(encoding="utf-8"))
    shape = _attention_shape(model, family, runtime)
    destination = output_root / "microbenchmark" / model / family_id
    load_tensorrt_runtime(TRT_ROOT)
    runtime_gpu = runtime_cuda_index_for_physical(physical_gpu)
    torch.cuda.set_device(runtime_gpu)
    device = torch.device(f"cuda:{runtime_gpu}")
    rows: list[dict[str, Any]] = []
    for d_h in widths:
        projection = int(family["heads"]) * d_h
        for primitive in ("q_projection", "k_projection", "v_projection", "qk", "av", "out_projection"):
            directory = destination / f"dh_{d_h:03d}" / primitive
            directory.mkdir(parents=True, exist_ok=True)
            model_path = directory / "model.onnx"
            if primitive.endswith("projection") and primitive != "out_projection":
                arrays = _linear_model(model_path, name=primitive, tokens=int(shape["tokens"]), input_width=int(shape["embed"]), output_width=projection, dtype=TensorProto.FLOAT16)
            elif primitive == "out_projection":
                arrays = _linear_model(model_path, name=primitive, tokens=int(shape["tokens"]), input_width=projection, output_width=int(shape["embed"]), dtype=TensorProto.FLOAT16)
            elif str(shape["kind"]) == "agent_relation":
                arrays = _relation_model(model_path, name=primitive, instances=int(shape["instances"]), heads=int(family["heads"]), sequence=int(shape["sequence"]), d_h=d_h, qk=primitive == "qk", dtype=TensorProto.FLOAT if primitive == "qk" else TensorProto.FLOAT16)
            else:
                arrays = _standard_attention_model(model_path, name=primitive, instances=int(shape["instances"]), heads=int(family["heads"]), sequence=int(shape["sequence"]), d_h=d_h, qk=primitive == "qk", dtype=TensorProto.FLOAT if primitive == "qk" else TensorProto.FLOAT16)
            built, command = _build(directory, physical_gpu=physical_gpu)
            row: dict[str, Any] = {"model": model, "attention_family": family_id, "d_h": d_h, "primitive": primitive, "strongly_typed": True, "no_tf32": True, "build": built, "micro_latency_additive": False, "shape_source": shape, "command": command}
            if built:
                tensors = {name: torch.from_numpy(value).to(device) for name, value in arrays.items()}
                aggregate, repetition_rows = _time_engine(directory / "engine.plan", tensors, device, warmup=warmup, iterations=iterations, repeats=repeats)
                audit = audit_engine_alignment(directory / "engine_layer_info.json", logical_d_h=d_h, projection_width=projection)
                layer_rows = _walk_layer_info(json.loads((directory / "engine_layer_info.json").read_text(encoding="utf-8")))
                row.update(aggregate)
                row.update({"repetitions": repetition_rows, "engine_sha256": _sha256(directory / "engine.plan"), "onnx_sha256": _sha256(model_path), "padding_status": audit["padding_status"], "tensor_core_hint": audit["tensor_core_hint"], "fallback_hint": audit["fallback_hint"], "workspace_mib": 8192, "tactics": sorted({str(value.get("TacticName", value.get("Tactic", ""))) for value in layer_rows if value.get("TacticName", value.get("Tactic", ""))})})
            rows.append(row)
            _write_json(destination / "microbenchmark_checkpoint.json", rows)
            torch.cuda.empty_cache()
    by_key = {(int(row["d_h"]), str(row["primitive"])): row for row in rows if row.get("build")}
    for high, low in pairs:
        for primitive in ("q_projection", "k_projection", "v_projection", "qk", "av", "out_projection"):
            if (high, primitive) in by_key and (low, primitive) in by_key:
                by_key[(low, primitive)]["speedup_neighbor"] = float(by_key[(high, primitive)]["p50_ms"]) / float(by_key[(low, primitive)]["p50_ms"])
    _write_json(destination / "primitive_microbenchmark.json", rows)
    _write_csv(destination / "primitive_microbenchmark.csv", rows)
    _write_json(destination / "microbenchmark_contract.json", {"pairs": pairs, "full_engine_predictor": False, "unit_latency_additive": False, "precision_contract": "projection/AV/Out FP16; QK FP32", "shape": shape})
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model", choices=("lidar_cobevt", "lidar_v2xvit"), required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    rows = run(output_root=Path(args.output_root).resolve(), model=args.model, family_id=args.family, physical_gpu=args.physical_gpu, warmup=args.warmup, iterations=args.iterations, repeats=args.repeats)
    print(json.dumps({"rows": len(rows), "build_failures": sum(not row["build"] for row in rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
