from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from deployment_equivalence import TensorRTEngineRunner, _load_model_context, _record_len_value
from export_lidar_pyramid_onnx import _extract_inputs, _input_names_for_export_mode, _prepare_export_tensors, _to_device
from quant_deploy_utils import (
    DEFAULT_CHECKPOINT,
    DEFAULT_HEAL_REPO,
    DEFAULT_HYPES_YAML,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TRT_ROOT,
    ensure_quant_deploy_run_dirs,
    find_trtexec_report,
    read_json,
    run_command,
    save_json,
)


MODES = ("padded_agent_static", "dynamic_agent_dim")
PRECISIONS = ("fp32", "fp16")
LATENCY_FIELDS = [
    "input_prepare_ms",
    "dtype_cast_ms",
    "contiguous_ms",
    "h2d_copy_ms",
    "input_device_copy_ms",
    "set_input_shape_ms",
    "output_shape_query_ms",
    "bind_address_ms",
    "execute_async_ms",
    "synchronize_ms",
    "d2h_copy_ms",
    "output_wrap_ms",
    "alloc_ms",
    "total_runner_ms",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decompose lidar_pyramid TensorRT latency for agent-aware export modes.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--padded_root", default=None)
    parser.add_argument("--dynamic_root", default=None)
    parser.add_argument("--hypes_yaml", default=str(DEFAULT_HYPES_YAML))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--heal_repo", default=str(DEFAULT_HEAL_REPO))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_frames", type=int, default=50)
    parser.add_argument("--warmup_ms", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--duration", type=int, default=3)
    parser.add_argument("--max_cav", type=int, default=2)
    parser.add_argument("--trt_root", default=str(DEFAULT_TRT_ROOT))
    parser.add_argument("--trtexec_path", default=None)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--skip_trtexec", action="store_true")
    parser.add_argument("--skip_runner", action="store_true")
    return parser.parse_args(argv)


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(v) for v in values)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[index]


def _stats(values: list[float]) -> dict[str, float | None]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {"mean": None, "p50": None, "p90": None, "p95": None, "max": None}
    return {
        "mean": float(sum(vals) / len(vals)),
        "p50": _percentile(vals, 50),
        "p90": _percentile(vals, 90),
        "p95": _percentile(vals, 95),
        "max": max(vals),
    }


def summarize_latency_rows(rows: list[dict[str, Any]], fields: list[str] | None = None) -> dict[str, Any]:
    fields = fields or LATENCY_FIELDS
    summary = {
        "num_frames": len(rows),
        "overall": {field: _stats([float(row.get(field, 0.0) or 0.0) for row in rows]) for field in fields},
        "by_record_len": {},
    }
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(int(row.get("record_len", 0))), []).append(row)
    for group, group_rows in sorted(groups.items(), key=lambda item: int(item[0])):
        summary["by_record_len"][group] = {
            "record_len": int(group),
            "num_frames": len(group_rows),
            **{field: _stats([float(row.get(field, 0.0) or 0.0) for row in group_rows]) for field in fields},
        }
    return summary


def _extract_latency_block(log_text: str, label: str) -> dict[str, float | None]:
    escaped = re.escape(label)
    line_re = re.compile(
        escaped
        + r".*?min\s*=\s*([0-9.]+)\s*ms.*?mean\s*=\s*([0-9.]+)\s*ms.*?median\s*=\s*([0-9.]+)\s*ms"
        + r".*?percentile\(90%\)\s*=\s*([0-9.]+)\s*ms.*?percentile\(95%\)\s*=\s*([0-9.]+)\s*ms",
        flags=re.IGNORECASE,
    )
    match = line_re.search(log_text)
    if not match:
        return {"min": None, "mean": None, "p50": None, "p90": None, "p95": None}
    return {
        "min": float(match.group(1)),
        "mean": float(match.group(2)),
        "p50": float(match.group(3)),
        "p90": float(match.group(4)),
        "p95": float(match.group(5)),
    }


def parse_trtexec_latency_metrics(log_text: str) -> dict[str, Any]:
    blocks = {
        "total_host_latency": _extract_latency_block(log_text, "Host Latency"),
        "latency": _extract_latency_block(log_text, "Latency"),
        "enqueue": _extract_latency_block(log_text, "Enqueue Time"),
        "h2d": _extract_latency_block(log_text, "H2D Latency"),
        "gpu_compute": _extract_latency_block(log_text, "GPU Compute Time"),
        "d2h": _extract_latency_block(log_text, "D2H Latency"),
    }
    throughput = None
    match = re.search(r"Throughput:\s*([0-9.]+)\s*qps", log_text, flags=re.IGNORECASE)
    if match:
        throughput = float(match.group(1))
    result: dict[str, Any] = {"throughput_qps": throughput}
    for prefix, block in blocks.items():
        for key, value in block.items():
            result[f"{prefix}_{key}_ms"] = value
    if result.get("total_host_latency_p50_ms") is None and result.get("latency_p50_ms") is not None:
        for key in ("min", "mean", "p50", "p90", "p95"):
            result[f"total_host_latency_{key}_ms"] = result.get(f"latency_{key}_ms")
        result["total_host_latency_source"] = "trtexec Latency line"
    else:
        result["total_host_latency_source"] = "trtexec Host Latency line"
    result["forward_p50_ms"] = result.get("gpu_compute_p50_ms")
    return result


def _shape_spec(profile_shapes: dict[str, Any], which: str = "opt") -> str:
    items = []
    for name, profile in profile_shapes.items():
        shape = profile.get(which) or profile.get("opt")
        items.append(f"{name}:{'x'.join(str(int(v)) for v in shape)}")
    return ",".join(items)


def _mode_source_root(args: argparse.Namespace, mode: str) -> Path:
    if mode == "padded_agent_static" and args.padded_root:
        return Path(args.padded_root)
    if mode == "dynamic_agent_dim" and args.dynamic_root:
        return Path(args.dynamic_root)
    return Path(args.output_root)


def _engine_path(root: Path, mode: str, precision: str) -> Path:
    return root / "artifacts" / "engines" / precision / f"lidar_pyramid_{mode}_{precision}.engine"


def _onnx_path(root: Path, mode: str) -> Path:
    return root / "artifacts" / "onnx" / "fp32" / f"lidar_pyramid_{mode}_fp32_dynamic.onnx"


def _profile_shapes(root: Path) -> dict[str, Any]:
    return read_json(root / "configs" / "profile_shapes.json", default={}) or {}


def _engine_size_mb(path: Path) -> float | None:
    return path.stat().st_size / (1024 * 1024) if path.exists() else None


def _load_dataset_context(args: argparse.Namespace):
    hypes, device, model, modality = _load_model_context(args)
    from opencood.data_utils.datasets import build_dataset

    dataset = build_dataset(hypes, visualize=True, train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=dataset.collate_batch_test)
    return hypes, device, model, modality, dataset, loader


def _first_sample_tensors(args: argparse.Namespace, mode: str) -> tuple[list[str], tuple[torch.Tensor, ...], dict[str, Any]]:
    _hypes, device, _model, modality, _dataset, loader = _load_dataset_context(args)
    for batch in loader:
        if batch is None:
            continue
        ego = batch["ego"] if isinstance(batch, dict) and "ego" in batch else batch
        ego = _to_device(ego, device)
        original_tensors, _agent_modalities = _extract_inputs(ego, modality)
        input_names = _input_names_for_export_mode(mode)
        tensors = _prepare_export_tensors(original_tensors, export_mode=mode, max_cav=int(args.max_cav))
        return input_names, tensors, ego
    raise RuntimeError("No real sample available for TensorRT latency decomposition.")


def _write_trtexec_input_files(args: argparse.Namespace, dirs: dict[str, Path], mode: str) -> tuple[dict[str, Path], dict[str, list[int]]]:
    input_names, tensors, _ego = _first_sample_tensors(args, mode)
    input_dir = dirs["debug"] / f"trtexec_inputs_{mode}"
    input_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}
    shapes: dict[str, list[int]] = {}
    for name, tensor in zip(input_names, tensors):
        arr = tensor.detach().cpu().contiguous().numpy()
        path = input_dir / f"{name}.raw"
        arr.tofile(path)
        files[name] = path
        shapes[name] = list(arr.shape)
    save_json({"mode": mode, "inputs": {name: {"path": str(path), "shape": shapes[name]} for name, path in files.items()}}, input_dir / "manifest.json")
    return files, shapes


def run_trtexec_benchmark(args: argparse.Namespace, dirs: dict[str, Path], mode: str, precision: str) -> dict[str, Any]:
    source_root = _mode_source_root(args, mode)
    engine_path = _engine_path(source_root, mode, precision)
    profile_shapes = _profile_shapes(source_root)
    input_files, input_shapes = _write_trtexec_input_files(args, dirs, mode)
    trtexec_report = find_trtexec_report(trt_root=args.trt_root, explicit_trtexec=args.trtexec_path)
    result: dict[str, Any] = {
        "scheme": mode,
        "precision": precision,
        "engine_path": str(engine_path),
        "engine_size_MB": _engine_size_mb(engine_path),
        "profile_shapes": profile_shapes,
        "input_shapes": input_shapes,
        "trtexec": trtexec_report,
        "success": False,
        "error": None,
    }
    if not engine_path.exists():
        result["error"] = f"engine file does not exist: {engine_path}"
    elif not trtexec_report.get("trtexec_found"):
        result["error"] = f"trtexec not found. {trtexec_report.get('suggestion')}"
    else:
        load_inputs = ",".join(f"{name}:{path}" for name, path in input_files.items())
        cmd = [
            trtexec_report["trtexec_path"],
            f"--loadEngine={engine_path}",
            f"--shapes={_shape_spec({name: {'opt': shape} for name, shape in input_shapes.items()}, 'opt')}",
            f"--loadInputs={load_inputs}",
            f"--warmUp={int(args.warmup_ms)}",
            f"--iterations={int(args.iterations)}",
            f"--duration={int(args.duration)}",
            "--avgRuns=1",
            "--percentile=50,90,95",
            "--useSpinWait",
            "--verbose",
        ]
        log_path = dirs["logs_benchmark"] / f"trtexec_{mode}_{precision}.log"
        command = run_command(cmd, log_path, timeout=int(args.timeout))
        log_text = log_path.read_text(encoding="utf-8")
        result.update(parse_trtexec_latency_metrics(log_text))
        result.update({"success": bool(command.get("success")), "error": command.get("error"), "command": cmd, "log_path": str(log_path)})
    save_json(result, dirs["benchmark"] / f"trtexec_{mode}_{precision}.json")
    return result


def _copy_outputs_to_cpu(outputs: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], float, int, int]:
    start = time.perf_counter()
    copied = {name: tensor.detach().cpu() for name, tensor in outputs.items()}
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000.0
    num_copies = len(copied)
    num_bytes = sum(int(tensor.numel() * tensor.element_size()) for tensor in copied.values())
    return copied, elapsed, num_copies, num_bytes


def run_runner_breakdown(args: argparse.Namespace, dirs: dict[str, Path], mode: str) -> dict[str, Any]:
    source_root = _mode_source_root(args, mode)
    hypes, device, _model, modality, _dataset, loader = _load_dataset_context(args)
    if device.type != "cuda":
        raise RuntimeError("TensorRT latency decomposition requires CUDA.")
    reports_by_precision: dict[str, Any] = {}
    allocation_reports: dict[str, Any] = {}
    for precision in PRECISIONS:
        engine_path = _engine_path(source_root, mode, precision)
        runner = TensorRTEngineRunner(engine_path, device)
        rows: list[dict[str, Any]] = []
        actual = 0
        for frame_idx, batch in enumerate(loader):
            if actual >= int(args.num_frames):
                break
            if batch is None:
                continue
            input_start = time.perf_counter()
            ego = batch["ego"] if isinstance(batch, dict) and "ego" in batch else batch
            ego = _to_device(ego, device)
            original_tensors, _agent_modalities = _extract_inputs(ego, modality)
            input_names = _input_names_for_export_mode(mode)
            tensors = _prepare_export_tensors(original_tensors, export_mode=mode, max_cav=int(args.max_cav))
            tensors_by_name = {name: tensor for name, tensor in zip(input_names, tensors)}
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            input_prepare_ms = (time.perf_counter() - input_start) * 1000.0
            outputs, profile = runner.run_profiled(tensors_by_name)
            _cpu_outputs, d2h_ms, d2h_copies, d2h_bytes = _copy_outputs_to_cpu(outputs)
            alloc_ms = 0.0
            row = {
                "frame_id": frame_idx,
                "record_len": _record_len_value(ego),
                "precision": precision,
                "scheme": mode,
                "input_prepare_ms": input_prepare_ms,
                "dtype_cast_ms": profile.get("dtype_cast_ms", 0.0),
                "contiguous_ms": profile.get("contiguous_ms", 0.0),
                "h2d_copy_ms": profile.get("h2d_copy_ms", 0.0),
                "input_device_copy_ms": profile.get("input_device_copy_ms", 0.0),
                "set_input_shape_ms": profile.get("set_input_shape_ms", 0.0),
                "output_shape_query_ms": profile.get("output_shape_query_ms", 0.0),
                "bind_address_ms": profile.get("bind_address_ms", 0.0),
                "execute_async_ms": profile.get("execute_async_ms", 0.0),
                "synchronize_ms": profile.get("synchronize_ms", 0.0),
                "d2h_copy_ms": d2h_ms,
                "output_wrap_ms": profile.get("output_wrap_ms", 0.0),
                "alloc_ms": alloc_ms,
                "total_runner_ms": float(profile.get("total_runner_ms", 0.0)) + d2h_ms,
                "input_buffer_reallocated": bool(profile.get("input_buffer_reallocated")),
                "output_buffer_reallocated": bool(profile.get("output_buffer_reallocated")),
                "set_input_shape_calls": int(profile.get("set_input_shape_calls", 0)),
                "output_shape_query_calls": int(profile.get("output_shape_query_calls", 0)),
                "h2d_copies": int(profile.get("h2d_copies", 0)),
                "d2d_input_copies": int(profile.get("d2d_input_copies", 0)),
                "d2h_copies": d2h_copies,
                "bytes_h2d": int(profile.get("bytes_h2d", 0)),
                "bytes_d2d_input": int(profile.get("bytes_d2d_input", 0)),
                "bytes_d2h": d2h_bytes,
                "input_shapes": profile.get("input_shapes", {}),
                "output_shapes": profile.get("output_shapes", {}),
            }
            rows.append(row)
            actual += 1
        allocation_reports[precision] = runner.allocation_report()
        allocation_reports[precision]["number_of_d2h_copies"] = sum(int(row.get("d2h_copies", 0)) for row in rows)
        allocation_reports[precision]["total_bytes_d2h"] = sum(int(row.get("bytes_d2h", 0)) for row in rows)
        reports_by_precision[precision] = {
            "scheme": mode,
            "precision": precision,
            "engine_path": str(engine_path),
            "num_frames": int(args.num_frames),
            "actual_frames": actual,
            "latency_fields": LATENCY_FIELDS,
            "frames": rows,
            "summary": summarize_latency_rows(rows),
            "allocation_report": allocation_reports[precision],
        }
        loader = DataLoader(_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=_dataset.collate_batch_test)
    report = {"scheme": mode, "precisions": reports_by_precision}
    save_json(report, dirs["debug"] / f"trt_latency_breakdown_{mode}.json")
    return report


def _latency_scope_report(dirs: dict[str, Path]) -> dict[str, Any]:
    return {
        "pytorch_forward_p50_scope": {
            "source": "evaluate_lidar_pyramid_ort_ap.py::_timed(lambda: model(ego))",
            "includes_data_preparation": False,
            "includes_h2d": False,
            "includes_d2h": False,
            "includes_postprocess": False,
            "includes_synchronization": True,
        },
        "onnxruntime_forward_p50_scope": {
            "source": "evaluate_lidar_pyramid_ort_ap.py::_timed(lambda: session.run(...))",
            "includes_data_preparation": False,
            "includes_h2d": "ORT internal provider copies if needed",
            "includes_d2h": "ORT output materialization to CPU then torch.to(device) in evaluator path",
            "includes_postprocess": False,
            "includes_synchronization": True,
        },
        "trt_forward_p50_scope_before_breakdown": {
            "source": "TensorRTEngineRunner.run inside evaluator",
            "includes_data_preparation": False,
            "includes_dtype_cast_contiguous": True,
            "includes_h2d": True,
            "includes_d2h": False,
            "includes_postprocess": False,
            "includes_synchronization": True,
            "old_runner_allocated_outputs_per_frame": True,
        },
        "trt_breakdown_scope": {
            "source": "latency_decomposition.py",
            "includes_input_prepare_ms": True,
            "execute_async_ms": "CUDA event elapsed time around execute_async_v3",
            "total_runner_ms": "profiled runner call plus explicit output D2H copy for measurement",
        },
    }


def _p50(report: dict[str, Any], key: str) -> float | None:
    return (((report.get("summary") or {}).get("overall") or {}).get(key) or {}).get("p50")


def _read_five_way_pytorch_p50(root: Path, mode: str) -> float | None:
    report = read_json(root / "evaluation" / f"five_way_ap_report_{mode}.json", default={}) or {}
    return ((report.get("reports") or {}).get("pytorch_original") or {}).get("forward_p50_ms")


def classify_latency_bottleneck(row: dict[str, Any]) -> dict[str, Any]:
    execute = float(row.get("execute_cuda_event_p50") or 0.0)
    total = float(row.get("total_runner_p50") or 0.0)
    d2h = float(row.get("d2h_p50") or 0.0)
    set_shape = float(row.get("set_shape_p50") or 0.0)
    engine_only = float(row.get("engine_only_p50") or 0.0)
    if engine_only and execute > engine_only * 10.0 and set_shape < 1.0:
        bottleneck = "dynamic_shape_profile_execution"
        recommendation = "Engine is fast for fixed shape, but real varying voxel shapes are slow in TensorRT execution; bucket/pad voxel count near opt shape or rebuild tighter profiles."
    elif execute and total > execute * 3.0:
        bottleneck = "runner_overhead"
        recommendation = "Optimize runner/buffer/copy; reuse context, stream, buffers, and tensor addresses."
    elif d2h and total and d2h > total * 0.3:
        bottleneck = "d2h_copy"
        recommendation = "Copy only postprocess-required outputs, or move postprocess to GPU."
    elif set_shape and total and set_shape > total * 0.2:
        bottleneck = "dynamic_shape_profile"
        recommendation = "Prefer padded static shapes or reduce shape changes; dynamic_agent_dim likely pays high profile overhead."
    elif execute and engine_only and abs(execute - engine_only) / max(engine_only, 1.0e-6) < 0.5 and execute > 50.0:
        bottleneck = "engine_compute"
        recommendation = "Inspect TensorRT layer profile and slow ops."
    else:
        bottleneck = "mixed_or_measurement_scope"
        recommendation = "Compare timing scopes and inspect layer profile, copy, and synchronization together."
    return {**row, "bottleneck": bottleneck, "recommendation": recommendation}


def allocation_report_with_explicit_d2h(precision_report: dict[str, Any]) -> dict[str, Any]:
    allocation = dict(precision_report.get("allocation_report") or {})
    frames = precision_report.get("frames") or []
    allocation["number_of_d2h_copies"] = sum(int(row.get("d2h_copies", 0)) for row in frames)
    allocation["total_bytes_d2h"] = sum(int(row.get("bytes_d2h", 0)) for row in frames)
    return allocation


def run_fixed_shape_control(args: argparse.Namespace, dirs: dict[str, Path], mode: str) -> dict[str, Any]:
    input_names, tensors, ego = _first_sample_tensors(args, mode)
    tensors_by_name = {name: tensor for name, tensor in zip(input_names, tensors)}
    source_root = _mode_source_root(args, mode)
    record_len = _record_len_value(ego)
    report: dict[str, Any] = {
        "scheme": mode,
        "record_len": record_len,
        "input_shapes": {name: list(tensor.shape) for name, tensor in tensors_by_name.items()},
        "precisions": {},
    }
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    for precision in PRECISIONS:
        runner = TensorRTEngineRunner(_engine_path(source_root, mode, precision), device)
        rows = []
        for _idx in range(int(args.iterations)):
            _outputs, profile = runner.run_profiled(tensors_by_name)
            rows.append(
                {
                    "record_len": record_len,
                    "execute_async_ms": profile.get("execute_async_ms", 0.0),
                    "total_runner_ms": profile.get("total_runner_ms", 0.0),
                    "set_input_shape_ms": profile.get("set_input_shape_ms", 0.0),
                    "input_device_copy_ms": profile.get("input_device_copy_ms", 0.0),
                }
            )
        report["precisions"][precision] = {
            "iterations": int(args.iterations),
            "summary": summarize_latency_rows(rows, ["execute_async_ms", "total_runner_ms", "set_input_shape_ms", "input_device_copy_ms"]),
            "allocation_report": runner.allocation_report(),
        }
    save_json(report, dirs["debug"] / f"trt_fixed_shape_control_{mode}.json")
    return report


def _write_markdown(path: Path, rows: list[dict[str, Any]], conclusion: str) -> None:
    lines = [
        "# Latency Decomposition Report",
        "",
        f"- conclusion: {conclusion}",
        "- int8_qdq_modelopt_plugin: not run",
        "",
        "scheme | precision | engine_only_p50 | fixed_shape_execute_p50 | execute_cuda_event_p50 | h2d_p50 | d2h_p50 | set_shape_p50 | alloc_p50 | total_runner_p50 | PyTorch_forward_p50 | bottleneck | recommendation",
        "--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---",
    ]
    for row in rows:
        lines.append(
            " | ".join(
                [
                    str(row.get("scheme")),
                    str(row.get("precision")),
                    _fmt(row.get("engine_only_p50")),
                    _fmt(row.get("fixed_shape_execute_p50")),
                    _fmt(row.get("execute_cuda_event_p50")),
                    _fmt(row.get("h2d_p50")),
                    _fmt(row.get("d2h_p50")),
                    _fmt(row.get("set_shape_p50")),
                    _fmt(row.get("alloc_p50")),
                    _fmt(row.get("total_runner_p50")),
                    _fmt(row.get("PyTorch_forward_p50")),
                    str(row.get("bottleneck")),
                    str(row.get("recommendation")),
                ]
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_latency_summary(args: argparse.Namespace, dirs: dict[str, Path], trtexec_reports: dict[str, dict[str, Any]], runner_reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for mode in MODES:
        source_root = _mode_source_root(args, mode)
        pytorch_p50 = _read_five_way_pytorch_p50(source_root, mode)
        for precision in PRECISIONS:
            trt = trtexec_reports.get(mode, {}).get(precision, {})
            runner = ((runner_reports.get(mode, {}) or {}).get("precisions") or {}).get(precision, {})
            row = {
                "scheme": mode,
                "precision": precision,
                "engine_only_p50": trt.get("gpu_compute_p50_ms"),
                "engine_only_host_latency_p50": trt.get("total_host_latency_p50_ms"),
                "engine_only_enqueue_p50": trt.get("enqueue_p50_ms"),
                "engine_only_h2d_p50": trt.get("h2d_p50_ms"),
                "engine_only_d2h_p50": trt.get("d2h_p50_ms"),
                "engine_only_throughput_qps": trt.get("throughput_qps"),
                "execute_cuda_event_p50": _p50(runner, "execute_async_ms"),
                "h2d_p50": _p50(runner, "h2d_copy_ms"),
                "d2h_p50": _p50(runner, "d2h_copy_ms"),
                "set_shape_p50": _p50(runner, "set_input_shape_ms"),
                "alloc_p50": _p50(runner, "alloc_ms"),
                "total_runner_p50": _p50(runner, "total_runner_ms"),
                "PyTorch_forward_p50": pytorch_p50,
            }
            fixed_control = (((runner_reports.get(mode, {}) or {}).get("fixed_shape_control") or {}).get("precisions") or {}).get(precision, {})
            fixed_summary = (fixed_control.get("summary") or {}).get("overall") or {}
            row["fixed_shape_execute_p50"] = (fixed_summary.get("execute_async_ms") or {}).get("p50")
            row["fixed_shape_runner_p50"] = (fixed_summary.get("total_runner_ms") or {}).get("p50")
            rows.append(classify_latency_bottleneck(row))
    conclusion = _overall_conclusion(rows)
    report = {
        "output_root": str(dirs["output_root"]),
        "num_frames": int(args.num_frames),
        "rows": rows,
        "trtexec_reports": trtexec_reports,
        "runner_reports": runner_reports,
        "latency_scope_report": _latency_scope_report(dirs),
        "conclusion": conclusion,
    }
    save_json(report, dirs["summary"] / "latency_decomposition_report.json")
    _write_markdown(dirs["summary"] / "latency_decomposition_report.md", rows, conclusion)
    save_json(report["latency_scope_report"], dirs["debug"] / "latency_scope_report.json")
    allocation = {
        mode: {
            precision: allocation_report_with_explicit_d2h((((runner_reports.get(mode, {}) or {}).get("precisions") or {}).get(precision, {}) or {}))
            for precision in PRECISIONS
        }
        for mode in MODES
    }
    save_json(allocation, dirs["debug"] / "trt_runner_allocation_report.json")
    return report


def _overall_conclusion(rows: list[dict[str, Any]]) -> str:
    if any(row.get("bottleneck") == "dynamic_shape_profile_execution" for row in rows):
        return "TRT engine-only and fixed-shape runner are fast, but real 50-frame varying voxel shapes are slow inside TensorRT execution; the main issue is dynamic shape/profile execution, not Python copy/allocation overhead."
    if any(row.get("bottleneck") == "runner_overhead" for row in rows):
        return "TRT real-sample latency is dominated by runner/copy/synchronization overhead rather than pure engine compute for at least one scheme."
    if any(row.get("bottleneck") == "engine_compute" for row in rows):
        return "TensorRT engine compute time itself is high; inspect layer profile."
    return "Latency is mixed; compare engine-only, CUDA event, copy, and timing scopes per scheme."


def run_latency_decomposition(args: argparse.Namespace) -> dict[str, Any]:
    dirs = ensure_quant_deploy_run_dirs(args.output_root)
    trtexec_reports: dict[str, dict[str, Any]] = {mode: {} for mode in MODES}
    runner_reports: dict[str, dict[str, Any]] = {}
    if not args.skip_trtexec:
        for mode in MODES:
            for precision in PRECISIONS:
                trtexec_reports[mode][precision] = run_trtexec_benchmark(args, dirs, mode, precision)
    if not args.skip_runner:
        for mode in MODES:
            runner_reports[mode] = run_runner_breakdown(args, dirs, mode)
            runner_reports[mode]["fixed_shape_control"] = run_fixed_shape_control(args, dirs, mode)
    return write_latency_summary(args, dirs, trtexec_reports, runner_reports)


def main(argv: list[str] | None = None) -> int:
    report = run_latency_decomposition(parse_args(argv))
    print(report["output_root"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
