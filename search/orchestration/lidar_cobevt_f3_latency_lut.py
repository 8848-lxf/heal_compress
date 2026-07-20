"""Build and time the 54 real-weight CoBEVT F3 Attention deployment units."""

from __future__ import annotations

import argparse
import ctypes
import csv
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from search.model_families.lidar_cobevt.attention_dim_pruning import (
    PrunableCobevtAttention,
)
from search.model_families.lidar_cobevt.attention_taylor import (
    keep_indices_from_scores,
)
from search.model_families.lidar_cobevt.f3_attention_unit import (
    F3PrimitiveAttentionUnit,
    f3_unit_requested_contract,
    inspect_f3_deployment_unit_layers,
)
from search.model_families.lidar_cobevt.head_dim_capability import HeadDimCandidate
from search.model_families.lidar_cobevt.minimal_structure_quant_latency import (
    aggregate_latency_repetitions,
    validate_f3_lut_admission,
    validate_lut_full_engine_deltas,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def _row_dir(root: Path, block_id: str, d_qk: int, d_v: int) -> Path:
    safe = block_id.replace(".", "_")
    return root / "latency_lut/engines" / safe / f"qk{d_qk}_v{d_v}"


def _build_environment(trt_root: Path, physical_gpu: int) -> dict[str, str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    libraries = (
        trt_root / "targets/x86_64-linux-gnu/lib",
        trt_root / "lib",
        Path("/home/lixingfeng/anaconda3/envs/modelopt/lib"),
    )
    env["LD_LIBRARY_PATH"] = ":".join(
        [*(str(path) for path in libraries), env.get("LD_LIBRARY_PATH", "")]
    )
    return env


def _hardware(physical_gpu: int) -> dict[str, str]:
    query = subprocess.run(
        [
            "nvidia-smi",
            f"--id={physical_gpu}",
            "--query-gpu=uuid,name,compute_cap,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip().split(", ")
    return {
        "gpu_uuid": query[0],
        "gpu_model": query[1],
        "compute_capability": query[2],
        "driver": query[3],
    }


def build_lut(
    root: Path,
    *,
    experiment_root: Path,
    checkpoint: Path,
    config: Path,
    heal_root: Path,
    trt_root: Path,
    physical_gpu: int,
) -> dict[str, int]:
    from search.model_families.lidar_cobevt.model_capability import CobevtModelCapability

    hardware = _hardware(physical_gpu)
    ranking = json.loads(
        (experiment_root / "structure_experiment/attention_pruning_ranking.json").read_text()
    )
    bundle = CobevtModelCapability(checkpoint, config, heal_root).load(
        device=torch.device("cpu")
    )
    modules = dict(bundle.model.named_modules())
    block_ids = sorted(ranking)
    if len(block_ids) != 6:
        raise RuntimeError(f"f3_lut_attention_block_count:{len(block_ids)}")
    trtexec = trt_root / "targets/x86_64-linux-gnu/bin/trtexec"
    rows: list[dict[str, Any]] = []
    for block_id in block_ids:
        stock = modules[block_id]
        parent = modules[block_id.rsplit(".", 1)[0]]
        norm = parent.norm
        scores = ranking[block_id]
        for d_qk in (16, 24, 32):
            for d_v in (16, 24, 32):
                destination = _row_dir(root, block_id, d_qk, d_v)
                destination.mkdir(parents=True, exist_ok=True)
                qk_keep = keep_indices_from_scores(
                    scores["qk_by_head"], keep_width=d_qk
                )
                vo_keep = keep_indices_from_scores(
                    scores["vo_by_head"], keep_width=d_v
                )
                attention = PrunableCobevtAttention.from_stock_attention(
                    stock,
                    qk_keep_by_head=qk_keep,
                    vo_keep_by_head=vo_keep,
                )
                unit = F3PrimitiveAttentionUnit(attention, norm).eval().cuda(physical_gpu)
                generator = torch.Generator(device=f"cuda:{physical_gpu}").manual_seed(
                    20260720
                )
                x = torch.randn(
                    (512, 32, 256),
                    generator=generator,
                    device=f"cuda:{physical_gpu}",
                    dtype=torch.float32,
                )
                mask = torch.ones(
                    (512, 1, 1, 32), device=f"cuda:{physical_gpu}", dtype=torch.bool
                )
                bias = attention.relative_position_bias_table(
                    attention.relative_position_index
                ).permute(2, 0, 1).unsqueeze(0).float().cuda(physical_gpu)
                onnx_path = destination / "attention_f3.onnx"
                engine_path = destination / "engine.plan"
                layer_path = destination / "engine_layer_info.json"
                profile_path = destination / "engine_profile.json"
                try:
                    with torch.no_grad():
                        torch.onnx.export(
                            unit,
                            (x, mask, bias),
                            str(onnx_path),
                            input_names=["x", "attention_mask", "relative_bias"],
                            output_names=["output"],
                            opset_version=17,
                            do_constant_folding=True,
                        )
                    command = [
                        str(trtexec),
                        f"--onnx={onnx_path}",
                        f"--saveEngine={engine_path}",
                        "--skipInference",
                        "--stronglyTyped",
                        "--noTF32",
                        "--noBuilderCache",
                        "--profilingVerbosity=detailed",
                        "--dumpLayerInfo",
                        "--dumpProfile",
                        f"--exportLayerInfo={layer_path}",
                        f"--exportProfile={profile_path}",
                        "--memPoolSize=workspace:512",
                        "--verbose",
                    ]
                    completed = subprocess.run(
                        command,
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        env=_build_environment(trt_root, physical_gpu),
                        check=False,
                        timeout=1200,
                    )
                    (destination / "build.log").write_text(completed.stdout or "")
                    success = (
                        completed.returncode == 0
                        and engine_path.is_file()
                        and layer_path.is_file()
                    )
                    inspector: dict[str, Any] = {}
                    if layer_path.is_file():
                        payload = json.loads(layer_path.read_text())
                        inspector = inspect_f3_deployment_unit_layers(
                            payload.get("Layers", payload),
                            heads=8,
                            tokens=32,
                            d_qk=d_qk,
                            d_v=d_v,
                            embed_dim=256,
                        )
                    realized_match = bool(success and inspector.get("requested_realized_match"))
                    row = {
                        "block_id": block_id,
                        "attention_kind": "window" if "window" in block_id else "grid",
                        "batch": 512,
                        "num_heads": 8,
                        "sq": 32,
                        "skv": 32,
                        "d_qk": d_qk,
                        "d_v": d_v,
                        "external_embed_dim": 256,
                        "input_layout": "flattened_(b*x*y),token,embed_from_real_cobevt",
                        "mask_kind": "cobevt_relation_mask",
                        "precision_profile_id": "F3_PRIMITIVE_V1",
                        "requested_precision": f3_unit_requested_contract(),
                        "requested_realized_match": realized_match,
                        "build_success": success,
                        "build_returncode": completed.returncode,
                        "builder_command": command,
                        "onnx_path": str(onnx_path),
                        "onnx_sha256": _sha256(onnx_path),
                        "engine_path": str(engine_path),
                        "engine_sha256": _sha256(engine_path) if engine_path.is_file() else "",
                        "evidence_directory": str(destination),
                        "gpu_uuid": hardware["gpu_uuid"],
                        "gpu_arch": "SM89",
                        "tensorrt_version": "10.9.0.34",
                        "cuda_version": "11.8",
                        "driver": hardware["driver"],
                        **inspector,
                    }
                except Exception as exc:  # noqa: BLE001
                    row = {
                        "block_id": block_id,
                        "d_qk": d_qk,
                        "d_v": d_v,
                        "build_success": False,
                        "requested_realized_match": False,
                        "failure_reason": f"{type(exc).__name__}:{exc}",
                        "evidence_directory": str(destination),
                    }
                _write_json(destination / "record.json", row)
                rows.append(row)
                del unit, attention, x, mask, bias
                torch.cuda.empty_cache()
    _write_json(root / "latency_lut/candidate_matrix.json", rows)
    return {
        "attempted": len(rows),
        "built": sum(bool(row.get("build_success")) for row in rows),
        "phenotype_matched": sum(bool(row.get("requested_realized_match")) for row in rows),
    }


def reparse_lut(root: Path) -> dict[str, int]:
    """Reclassify already-built engines without rebuilding or changing evidence."""

    rows = json.loads((root / "latency_lut/candidate_matrix.json").read_text())
    reparsed: list[dict[str, Any]] = []
    failures = 0
    for original in rows:
        row = dict(original)
        destination = Path(str(row["evidence_directory"]))
        try:
            payload = json.loads((destination / "engine_layer_info.json").read_text())
            inspector = inspect_f3_deployment_unit_layers(
                payload.get("Layers", payload),
                heads=int(row["num_heads"]),
                tokens=int(row["sq"]),
                d_qk=int(row["d_qk"]),
                d_v=int(row["d_v"]),
                embed_dim=int(row["external_embed_dim"]),
            )
            row.update(inspector)
            row["parser_schema"] = "f3-deployment-unit-shape-sequence-v1"
            row.pop("phenotype_parse_failure", None)
        except Exception as exc:  # noqa: BLE001
            row["requested_realized_match"] = False
            row["phenotype_parse_failure"] = f"{type(exc).__name__}:{exc}"
            failures += 1
        _write_json(destination / "record.json", row)
        reparsed.append(row)
    _write_json(root / "latency_lut/candidate_matrix.json", reparsed)
    return {
        "attempted": len(reparsed),
        "build_success": sum(bool(row.get("build_success")) for row in reparsed),
        "phenotype_matched": sum(bool(row.get("requested_realized_match")) for row in reparsed),
        "parse_failures": failures,
    }


def _gpu_processes(physical_gpu: int) -> list[str]:
    output = subprocess.run(
        [
            "nvidia-smi",
            f"--id={physical_gpu}",
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    return [line.strip() for line in output.splitlines() if line.strip()]


def audit_isolation(physical_gpu: int, seconds: int) -> dict[str, Any]:
    samples = []
    started = time.monotonic()
    while time.monotonic() - started < seconds:
        processes = _gpu_processes(physical_gpu)
        samples.append({"elapsed": time.monotonic() - started, "processes": processes})
        if processes:
            return {"isolated": False, "samples": samples}
        time.sleep(min(5, max(0.1, seconds - (time.monotonic() - started))))
    return {"isolated": True, "samples": samples}


def time_lut(
    root: Path,
    *,
    physical_gpu: int,
    isolation_seconds: int,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, Any]:
    from search.orchestration.lidar_cobevt_attention_microbenchmark import (
        load_tensorrt_runtime,
    )
    from tests.quant_deploy.deployment_equivalence import TensorRTEngineRunner

    run = json.loads((root / "run_manifest.json").read_text())
    trt_root = Path(run["tensorrt_root"])
    isolation = audit_isolation(physical_gpu, isolation_seconds)
    _write_json(root / "latency_lut/isolation_audit.json", isolation)
    formal = bool(isolation["isolated"])
    load_tensorrt_runtime(trt_root)
    torch.cuda.set_device(physical_gpu)
    rows = json.loads((root / "latency_lut/candidate_matrix.json").read_text())
    timed = []
    for row in rows:
        if not row.get("build_success"):
            continue
        fusion = str(row.get("fusion_kind", "primitive"))
        try:
            validate_f3_lut_admission(
                requested_realized_match=bool(row.get("requested_realized_match")),
                fusion_kind=fusion,
                isolated_gpu=formal,
                formal=formal,
            )
            runner = TensorRTEngineRunner(row["engine_path"], torch.device("cuda", physical_gpu))
            generator = torch.Generator(device=f"cuda:{physical_gpu}").manual_seed(20260720)
            inputs = {
                "x": torch.randn((512, 32, 256), generator=generator, device=f"cuda:{physical_gpu}"),
                "attention_mask": torch.ones((512, 1, 1, 32), dtype=torch.bool, device=f"cuda:{physical_gpu}"),
                "relative_bias": torch.zeros((1, 8, 32, 32), device=f"cuda:{physical_gpu}"),
            }
            runner.run(inputs)
            # Inputs and outputs are now allocated and bound.  Time only execute_async_v3.
            stream = runner.stream
            for _ in range(warmup):
                with torch.cuda.stream(stream):
                    if not runner.context.execute_async_v3(stream.cuda_stream):
                        raise RuntimeError("lut_warmup_execute_failed")
            stream.synchronize()
            repetitions = []
            for _ in range(repeats):
                samples = []
                for _ in range(iterations):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    with torch.cuda.stream(stream):
                        start.record(stream)
                        if not runner.context.execute_async_v3(stream.cuda_stream):
                            raise RuntimeError("lut_timed_execute_failed")
                        end.record(stream)
                    end.synchronize()
                    samples.append(float(start.elapsed_time(end)))
                repetitions.append(
                    {
                        "p50_ms": float(np.percentile(samples, 50)),
                        "p90_ms": float(np.percentile(samples, 90)),
                        "p95_ms": float(np.percentile(samples, 95)),
                        "p99_ms": float(np.percentile(samples, 99)),
                        "mean_ms": float(statistics.mean(samples)),
                        "std_ms": float(statistics.pstdev(samples)),
                    }
                )
            aggregate = aggregate_latency_repetitions(repetitions)
            timed.append({**row, **aggregate, "formal": formal, "repetitions": repetitions})
        except Exception as exc:  # noqa: BLE001
            timed.append({**row, "timing_failure": f"{type(exc).__name__}:{exc}", "formal": formal})
    filename = "f3_primitive_latency_lut.csv" if formal else "f3_primitive_latency_lut_screening.csv"
    json_name = filename.replace(".csv", ".json")
    _write_json(root / "latency_lut" / json_name, timed)
    fields = sorted({key for row in timed for key in row if not isinstance(row.get(key), (dict, list))})
    with (root / "latency_lut" / filename).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in timed:
            writer.writerow({key: row.get(key, "") for key in fields})
    return {"formal": formal, "timed": sum("p50_ms" in row for row in timed), "rows": len(timed)}


def time_full_models(
    root: Path,
    *,
    checkpoint: Path,
    config: Path,
    heal_root: Path,
    plugin: Path,
    physical_gpu: int,
    isolation_seconds: int,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, Any]:
    from search.orchestration.lidar_cobevt_attention_microbenchmark import load_tensorrt_runtime
    from search.orchestration.lidar_cobevt_attention_pruning import (
        _export_inputs,
        _load_bundle,
        candidate_engine_directory,
    )
    from tests.quant_deploy.deployment_equivalence import TensorRTEngineRunner

    run = json.loads((root / "run_manifest.json").read_text())
    trt_root = Path(run["tensorrt_root"])
    isolation = audit_isolation(physical_gpu, isolation_seconds)
    _write_json(root / "latency_lut/full_engine_isolation_audit.json", isolation)
    if not isolation["isolated"]:
        raise RuntimeError("full_engine_formal_latency_requires_isolated_gpu")
    load_tensorrt_runtime(trt_root)
    ctypes.CDLL(str(plugin), mode=ctypes.RTLD_GLOBAL)
    torch.cuda.set_device(physical_gpu)
    device = torch.device("cuda", physical_gpu)
    bundle = _load_bundle(
        checkpoint=checkpoint, config=config, heal_root=heal_root, device=torch.device("cpu")
    )
    inputs = _export_inputs(bundle, config, device, fixed_k=29696)
    experiment = json.loads((root / "experiment_config.json").read_text())
    structure = {row["candidate_id"]: row for row in experiment["candidates"]}
    timed = []
    for candidate_id in ("S0", "S1", "S2", "S3", "S4"):
        directory = candidate_engine_directory(
            root,
            candidate_id,
            29696,
            precision="FP16",
            profile_name="F3_rest_fp16_qk_fp32_minimal_island",
        )
        build = json.loads((directory / "build_report.json").read_text())
        runner = TensorRTEngineRunner(build["engine_path"], device)
        runner.run(inputs)
        stream = runner.stream
        for _ in range(warmup):
            with torch.cuda.stream(stream):
                if not runner.context.execute_async_v3(stream.cuda_stream):
                    raise RuntimeError("full_engine_warmup_execute_failed")
        stream.synchronize()
        repetitions = []
        for _ in range(repeats):
            samples = []
            for _ in range(iterations):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                with torch.cuda.stream(stream):
                    start.record(stream)
                    if not runner.context.execute_async_v3(stream.cuda_stream):
                        raise RuntimeError("full_engine_timed_execute_failed")
                    end.record(stream)
                end.synchronize()
                samples.append(float(start.elapsed_time(end)))
            repetitions.append(
                {
                    "p50_ms": float(np.percentile(samples, 50)),
                    "p90_ms": float(np.percentile(samples, 90)),
                    "p95_ms": float(np.percentile(samples, 95)),
                    "p99_ms": float(np.percentile(samples, 99)),
                    "mean_ms": float(statistics.mean(samples)),
                    "std_ms": float(statistics.pstdev(samples)),
                }
            )
        timed.append(
            {
                "candidate_id": candidate_id,
                "d_qk": int(structure[candidate_id]["d_qk"]),
                "d_v": int(structure[candidate_id]["d_v"]),
                "engine_sha256": build["engine_sha256"],
                "formal": True,
                "gpu_uuid": _hardware(physical_gpu)["gpu_uuid"],
                **aggregate_latency_repetitions(repetitions),
                "repetitions": repetitions,
            }
        )
    _write_json(root / "latency_lut/full_engine_formal_latency.json", timed)
    lut = json.loads((root / "latency_lut/f3_primitive_latency_lut.json").read_text())
    lut_sum: dict[tuple[int, int], float] = {}
    for d_qk in (16, 24, 32):
        for d_v in (16, 24, 32):
            lut_sum[(d_qk, d_v)] = sum(
                float(row["p50_ms"])
                for row in lut
                if int(row["d_qk"]) == d_qk and int(row["d_v"]) == d_v
            )
    baseline_full = next(row for row in timed if row["candidate_id"] == "S0")
    baseline_lut = lut_sum[(32, 32)]
    validation_rows = []
    for row in timed:
        predicted = lut_sum[(int(row["d_qk"]), int(row["d_v"]))] - baseline_lut
        actual = float(row["p50_ms"]) - float(baseline_full["p50_ms"])
        validation_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "predicted_delta_ms": predicted,
                "actual_delta_ms": actual,
                "absolute_error_ms": abs(predicted - actual),
            }
        )
    nonbaseline = [row for row in validation_rows if row["candidate_id"] != "S0"]
    summary = validate_lut_full_engine_deltas(nonbaseline)
    _write_json(
        root / "latency_lut/f3_lut_full_engine_validation.json",
        {"rows": validation_rows, "summary": summary},
    )
    fields = sorted({key for row in validation_rows for key in row})
    with (root / "latency_lut/f3_lut_full_engine_validation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(validation_rows)
    return {"formal": True, "timed": len(timed), **summary}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("build", "reparse", "time", "validate-full"), required=True
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--heal-root", required=True)
    parser.add_argument("--physical-gpu", type=int, default=7)
    parser.add_argument("--isolation-seconds", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--plugin", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.output_dir)
    run = json.loads((root / "run_manifest.json").read_text())
    if args.phase == "build":
        result = build_lut(
            root,
            experiment_root=Path(args.experiment_root),
            checkpoint=Path(args.checkpoint),
            config=Path(args.config),
            heal_root=Path(args.heal_root),
            trt_root=Path(run["tensorrt_root"]),
            physical_gpu=args.physical_gpu,
        )
    elif args.phase == "reparse":
        result = reparse_lut(root)
    elif args.phase == "time":
        result = time_lut(
            root,
            physical_gpu=args.physical_gpu,
            isolation_seconds=args.isolation_seconds,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
    else:
        if not args.plugin:
            raise ValueError("full_engine_latency_plugin_required")
        result = time_full_models(
            root,
            checkpoint=Path(args.checkpoint),
            config=Path(args.config),
            heal_root=Path(args.heal_root),
            plugin=Path(args.plugin),
            physical_gpu=args.physical_gpu,
            isolation_seconds=args.isolation_seconds,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
