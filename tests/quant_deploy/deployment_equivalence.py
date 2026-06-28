from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_lidar_pyramid_onnx import (
    INPUT_NAMES,
    _add_paths,
    _extract_inputs,
    _first_real_sample,
    _input_names_for_export_mode,
    _infer_modality,
    _load_hypes,
    _load_model,
    _prepare_export_tensors,
    _synthetic_sample,
    _tensor_output_names,
    _to_device,
)
from exportable_lidar_pyramid import FixedLidarPyramidExportWrapper
from exportable_lidar_pyramid_dynamic_agent import ExportableLidarPyramidDynamicAgent
from exportable_lidar_pyramid_padded_agent import ExportableLidarPyramidPaddedAgent
from exportable_lidar_pyramid_fixed_k_scatter_plugin import ExportableLidarPyramidFixedKScatterPlugin
from quant_deploy_utils import ensure_quant_deploy_run_dirs, read_json, save_csv, save_json


DEFAULT_DEPLOY_OUTPUT_NAMES = ["cls_preds", "reg_preds", "dir_preds"]


def _as_float_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _empty_accumulator() -> dict[str, Any]:
    return {
        "max_abs_error": 0.0,
        "abs_error_sum": 0.0,
        "reference_abs_sum": 0.0,
        "numel": 0,
        "shape_mismatches": [],
    }


def _update_accumulator(acc: dict[str, Any], reference: Any, candidate: Any, frame_index: int | None = None) -> None:
    ref = _as_float_numpy(reference)
    cand = _as_float_numpy(candidate)
    if ref.shape != cand.shape:
        acc["shape_mismatches"].append(
            {
                "frame_index": frame_index,
                "reference_shape": list(ref.shape),
                "candidate_shape": list(cand.shape),
            }
        )
        return
    diff = np.abs(ref - cand)
    if diff.size:
        acc["max_abs_error"] = max(float(acc["max_abs_error"]), float(diff.max()))
        acc["abs_error_sum"] += float(diff.sum(dtype=np.float64))
        acc["reference_abs_sum"] += float(np.abs(ref).sum(dtype=np.float64))
        acc["numel"] += int(diff.size)


def _finalize_accumulator(acc: dict[str, Any]) -> dict[str, Any]:
    numel = int(acc["numel"])
    mean_abs = float(acc["abs_error_sum"] / numel) if numel else None
    ref_mean_abs = float(acc["reference_abs_sum"] / numel) if numel else None
    return {
        "max_abs_error": float(acc["max_abs_error"]) if numel else None,
        "mean_abs_error": mean_abs,
        "relative_error": float(mean_abs / max(ref_mean_abs or 0.0, 1.0e-12)) if mean_abs is not None else None,
        "numel": numel,
        "shape_mismatches": list(acc["shape_mismatches"]),
    }


def aggregate_output_errors(frames: Iterable[dict[str, Any]], output_names: list[str]) -> dict[str, Any]:
    """Aggregate tensor errors globally and by record_len/agent count."""
    global_acc = {name: _empty_accumulator() for name in output_names}
    grouped_acc: dict[str, dict[str, Any]] = {}
    group_counts: dict[str, int] = {}
    num_frames = 0

    for frame_index, frame in enumerate(frames):
        num_frames += 1
        record_len = int(frame.get("record_len", 0))
        group_key = str(record_len)
        group_counts[group_key] = group_counts.get(group_key, 0) + 1
        grouped_acc.setdefault(group_key, {name: _empty_accumulator() for name in output_names})
        outputs = frame.get("outputs", {})
        for name in output_names:
            pair = outputs.get(name)
            if not pair:
                continue
            reference = pair.get("reference")
            candidate = pair.get("candidate")
            _update_accumulator(global_acc[name], reference, candidate, frame_index=frame_index)
            _update_accumulator(grouped_acc[group_key][name], reference, candidate, frame_index=frame_index)

    report = {
        "num_frames": num_frames,
        "output_names": list(output_names),
        "outputs": {name: _finalize_accumulator(acc) for name, acc in global_acc.items()},
        "by_record_len": {},
    }
    for group_key, outputs in sorted(grouped_acc.items(), key=lambda item: int(item[0])):
        report["by_record_len"][group_key] = {
            "record_len": int(group_key),
            "agent_count": int(group_key),
            "num_frames": group_counts[group_key],
            "outputs": {name: _finalize_accumulator(acc) for name, acc in outputs.items()},
        }
    return report


def compact_error_for_summary(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if report is None:
        return None
    if report.get("success") is False:
        return {
            "success": False,
            "max_abs_error": None,
            "mean_abs_error": None,
            "relative_error": None,
            "error": report.get("error"),
        }
    outputs = report.get("outputs") or {}
    max_abs_values = [item.get("max_abs_error") for item in outputs.values() if item.get("max_abs_error") is not None]
    mean_abs_values = [item.get("mean_abs_error") for item in outputs.values() if item.get("mean_abs_error") is not None]
    relative_values = [item.get("relative_error") for item in outputs.values() if item.get("relative_error") is not None]
    return {
        "success": bool(report.get("success", True)),
        "max_abs_error": max(max_abs_values) if max_abs_values else None,
        "mean_abs_error": max(mean_abs_values) if mean_abs_values else None,
        "relative_error": max(relative_values) if relative_values else None,
    }


def _compact_metric(report: dict[str, Any] | None, key: str) -> float | None:
    compact = compact_error_for_summary(report)
    return compact.get(key) if compact else None


def make_deploy_equivalence_summary_fields(
    export_forward_mode: str,
    num_pyramid_scales: int | None,
    wrapper_report: dict[str, Any] | None,
    trt_report: dict[str, Any] | None,
    special_ops: dict[str, Any] | None,
) -> dict[str, Any]:
    comparisons = (trt_report or {}).get("comparisons") or {}
    sequence_count = int((special_ops or {}).get("sequence_op_count") or 0)
    return {
        "export_forward_mode": export_forward_mode,
        "is_export_specialized_wrapper": export_forward_mode in {"fixed_static", "dynamic_agent_dim", "padded_agent_static", "fixed_k_scatter_plugin"},
        "is_original_forward": export_forward_mode == "original",
        "num_pyramid_scales": num_pyramid_scales,
        "sequence_ops_removed": sequence_count == 0,
        "wrapper_equivalence_num_frames": (wrapper_report or {}).get("num_frames"),
        "wrapper_equivalence_max_abs_error": _compact_metric(wrapper_report, "max_abs_error"),
        "wrapper_equivalence_mean_abs_error": _compact_metric(wrapper_report, "mean_abs_error"),
        "trt_fp32_vs_pytorch_error": compact_error_for_summary(comparisons.get("trt_fp32_vs_pytorch")),
        "trt_fp16_vs_pytorch_error": compact_error_for_summary(comparisons.get("trt_fp16_vs_pytorch")),
    }


def _frame_iterator(hypes: dict[str, Any], device: torch.device, num_frames: int):
    from opencood.data_utils.datasets import build_dataset
    from torch.utils.data import DataLoader

    dataset = build_dataset(hypes, visualize=True, train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=dataset.collate_batch_test)
    produced = 0
    for batch in loader:
        if batch is None:
            continue
        ego = batch["ego"] if isinstance(batch, dict) and "ego" in batch else batch
        yield _to_device(ego, device)
        produced += 1
        if produced >= num_frames:
            break


def _load_model_context(args: argparse.Namespace) -> tuple[dict[str, Any], torch.device, torch.nn.Module, str]:
    _add_paths(args.heal_repo)
    hypes = _load_hypes(args.hypes_yaml, args.heal_repo)
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    model = _load_model(hypes, args.checkpoint, device)
    modality = _infer_modality(model)
    return hypes, device, model, modality


def _record_len_value(sample: dict[str, Any]) -> int:
    record_len = sample["record_len"]
    if torch.is_tensor(record_len):
        return int(record_len.detach().sum().item())
    return int(np.asarray(record_len).sum())


def _outputs_to_dict(output_names: list[str], outputs: tuple[Any, ...]) -> dict[str, Any]:
    return {name: value for name, value in zip(output_names, outputs)}


def _wrapper_for_export_mode(
    model: torch.nn.Module,
    modality: str,
    output_names: list[str],
    export_mode: str,
    max_cav: int,
) -> torch.nn.Module:
    if export_mode == "dynamic_agent_dim":
        return ExportableLidarPyramidDynamicAgent(model, modality, output_names)
    if export_mode == "padded_agent_static":
        return ExportableLidarPyramidPaddedAgent(model, modality, output_names, max_cav=max_cav)
    if export_mode == "fixed_k_scatter_plugin":
        return ExportableLidarPyramidFixedKScatterPlugin(model, modality, output_names, max_cav=max_cav)
    return FixedLidarPyramidExportWrapper(model, modality, output_names)


def _input_names_and_tensors_for_export_mode(
    sample: dict[str, Any],
    modality: str,
    export_mode: str,
    max_cav: int,
) -> tuple[list[str], tuple[torch.Tensor, ...], list[str]]:
    original_tensors, agent_modality_list = _extract_inputs(sample, modality)
    input_names = _input_names_for_export_mode(export_mode)
    tensors = _prepare_export_tensors(original_tensors, export_mode=export_mode, max_cav=max_cav)
    return input_names, tensors, agent_modality_list


def run_wrapper_equivalence(args: argparse.Namespace) -> dict[str, Any]:
    dirs = ensure_quant_deploy_run_dirs(args.output_root)
    log_path = dirs["logs_evaluation"] / "wrapper_equivalence.log"
    try:
        hypes, device, model, modality = _load_model_context(args)
        first_sample = _first_real_sample(hypes, device)
        with torch.no_grad():
            first_raw = model(first_sample)
        output_names = [name for name in DEFAULT_DEPLOY_OUTPUT_NAMES if name in first_raw and torch.is_tensor(first_raw[name])]
        if not output_names:
            output_names = _tensor_output_names(first_raw)
        export_mode = getattr(args, "pyramid_forward_export_mode", "fixed_static")
        wrapper = _wrapper_for_export_mode(model, modality, output_names, export_mode, int(args.max_cav)).to(device).eval()

        frame_reports: list[dict[str, Any]] = []
        for frame_index, sample in enumerate(_frame_iterator(hypes, device, int(args.num_frames))):
            _input_names, tensors, _agent_modality_list = _input_names_and_tensors_for_export_mode(
                sample,
                modality,
                export_mode,
                int(args.max_cav),
            )
            with torch.no_grad():
                raw_output = model(sample)
                wrapper_outputs = _outputs_to_dict(output_names, wrapper(*tensors))
            frame_reports.append(
                {
                    "frame_index": frame_index,
                    "record_len": _record_len_value(sample),
                    "outputs": {
                        name: {"reference": raw_output[name], "candidate": wrapper_outputs[name]}
                        for name in output_names
                    },
                }
            )
        report = aggregate_output_errors(frame_reports, output_names)
        report.update(
            {
                "success": True,
                "comparison": f"original_pytorch_forward_vs_{export_mode}_wrapper",
                "device": str(device),
                "pyramid_forward_export_mode": export_mode,
            }
        )
    except Exception as exc:
        report = {
            "success": False,
            "comparison": f"original_pytorch_forward_vs_{getattr(args, 'pyramid_forward_export_mode', 'fixed_static')}_wrapper",
            "num_frames": 0,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    export_mode = getattr(args, "pyramid_forward_export_mode", "fixed_static")
    suffix = "" if export_mode == "fixed_static" else f"_{export_mode}"
    save_json(report, dirs["evaluation"] / f"wrapper_equivalence{suffix}.json")
    save_json(report, dirs["debug"] / f"wrapper_equivalence{suffix}_debug.json")
    if export_mode == "dynamic_agent_dim":
        save_json(report, dirs["debug"] / "wrapper_equivalence_dynamic_agent_dim.json")
    elif export_mode == "padded_agent_static":
        save_json(report, dirs["debug"] / "wrapper_equivalence_padded_agent_static.json")
    else:
        save_json(report, dirs["evaluation"] / "wrapper_equivalence.json")
        save_json(report, dirs["debug"] / "wrapper_equivalence_debug.json")
    log_path.write_text((report.get("traceback") or report.get("error") or "wrapper equivalence completed") + "\n", encoding="utf-8")
    return report


def _first_verification_sample(args: argparse.Namespace, hypes: dict[str, Any], device: torch.device, model: torch.nn.Module, modality: str) -> tuple[tuple[torch.Tensor, ...], list[str], list[str]]:
    try:
        sample = _first_real_sample(hypes, device)
    except Exception:
        if not getattr(args, "allow_synthetic_fallback", False):
            raise
        sample = _synthetic_sample(model, device, args.max_cav)
    with torch.no_grad():
        raw_output = model(sample)
    output_names = [name for name in DEFAULT_DEPLOY_OUTPUT_NAMES if name in raw_output and torch.is_tensor(raw_output[name])]
    if not output_names:
        output_names = _tensor_output_names(raw_output)
    tensors, agent_modality_list = _extract_inputs(sample, modality)
    return tensors, agent_modality_list, output_names


def _run_onnxruntime(onnx_path: Path, input_names: list[str], tensors: tuple[torch.Tensor, ...]) -> dict[str, np.ndarray]:
    import onnxruntime as ort

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    available = ort.get_available_providers()
    providers = [provider for provider in providers if provider in available] or available
    try:
        session = ort.InferenceSession(str(onnx_path), providers=providers)
    except Exception as exc:
        if "MatMulBnFusion_Gemm" not in str(exc) and "ShapeInferenceError" not in str(exc):
            raise
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(str(onnx_path), sess_options=options, providers=providers)
    inputs = {name: tensor.detach().cpu().numpy() for name, tensor in zip(input_names, tensors)}
    outputs = session.run(None, inputs)
    output_names = [output.name for output in session.get_outputs()]
    return {name: value for name, value in zip(output_names, outputs)}


def _trt_dtype_to_torch(dtype: Any) -> torch.dtype:
    import tensorrt as trt

    if dtype == trt.float16:
        return torch.float16
    if dtype == trt.float32:
        return torch.float32
    if dtype == trt.int32:
        return torch.int32
    if dtype == trt.int64:
        return torch.int64
    if dtype == trt.bool:
        return torch.bool
    return torch.float32


def _tensor_to_trt_input_dtype(tensor: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return tensor.to(device=device, dtype=dtype).contiguous()


def _run_tensorrt(engine_path: Path, tensors_by_name: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    import tensorrt as trt

    if device.type != "cuda":
        raise RuntimeError("TensorRT output equivalence requires a CUDA device.")
    logger = trt.Logger(trt.Logger.ERROR)
    with engine_path.open("rb") as f:
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"failed to deserialize TensorRT engine: {engine_path}")
    context = engine.create_execution_context()
    stream = torch.cuda.current_stream(device=device)
    bindings: dict[str, torch.Tensor] = {}
    output_tensors: dict[str, torch.Tensor] = {}

    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.INPUT:
            dtype = _trt_dtype_to_torch(engine.get_tensor_dtype(name))
            tensor = _tensor_to_trt_input_dtype(tensors_by_name[name], dtype, device)
            context.set_input_shape(name, tuple(tensor.shape))
            bindings[name] = tensor

    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        mode = engine.get_tensor_mode(name)
        if mode == trt.TensorIOMode.OUTPUT:
            shape = tuple(int(dim) for dim in context.get_tensor_shape(name))
            dtype = _trt_dtype_to_torch(engine.get_tensor_dtype(name))
            output = torch.empty(shape, dtype=dtype, device=device)
            bindings[name] = output
            output_tensors[name] = output

    for name, tensor in bindings.items():
        context.set_tensor_address(name, int(tensor.data_ptr()))
    ok = context.execute_async_v3(stream.cuda_stream)
    if not ok:
        raise RuntimeError(f"TensorRT execution failed for engine: {engine_path}")
    stream.synchronize()
    return output_tensors


class TensorRTEngineRunner:
    def __init__(self, engine_path: str | Path, device: torch.device) -> None:
        import tensorrt as trt

        if device.type != "cuda":
            raise RuntimeError("TensorRT execution requires a CUDA device.")
        self.device = device
        self.trt = trt
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.engine_path = str(engine_path)
        with Path(engine_path).open("rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize TensorRT engine: {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream(device=device)
        self.input_buffers: dict[str, torch.Tensor] = {}
        self.output_buffers: dict[str, torch.Tensor] = {}
        self.bound_ptrs: dict[str, int] = {}
        self.last_input_shapes: dict[str, tuple[int, ...]] = {}
        self.last_output_shapes: dict[str, tuple[int, ...]] = {}
        self.allocation_stats: dict[str, Any] = {
            "context_created_per_run": False,
            "stream_created_per_run": False,
            "context_create_count": 1,
            "stream_create_count": 1,
            "run_count": 0,
            "input_buffer_reallocation_count": 0,
            "output_buffer_reallocation_count": 0,
            "set_input_shape_call_count": 0,
            "output_shape_query_call_count": 0,
            "number_of_h2d_copies": 0,
            "number_of_d2d_input_copies": 0,
            "number_of_d2h_copies": 0,
            "total_bytes_h2d": 0,
            "total_bytes_d2d_input": 0,
            "total_bytes_d2h": 0,
        }

    def run(self, tensors_by_name: dict[str, torch.Tensor], *, cast_inputs_to_engine_dtype: bool = True) -> dict[str, torch.Tensor]:
        outputs, _profile = self._run_impl(tensors_by_name, cast_inputs_to_engine_dtype=cast_inputs_to_engine_dtype, collect_profile=False)
        return outputs

    def run_profiled(self, tensors_by_name: dict[str, torch.Tensor], *, cast_inputs_to_engine_dtype: bool = True) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        return self._run_impl(tensors_by_name, cast_inputs_to_engine_dtype=cast_inputs_to_engine_dtype, collect_profile=True)

    def allocation_report(self) -> dict[str, Any]:
        run_count = int(self.allocation_stats["run_count"])
        return {
            **self.allocation_stats,
            "input_buffer_reallocated_per_frame": bool(run_count and self.allocation_stats["input_buffer_reallocation_count"] >= run_count),
            "output_buffer_reallocated_per_frame": bool(run_count and self.allocation_stats["output_buffer_reallocation_count"] >= run_count),
            "set_input_shape_called_per_frame": bool(run_count and self.allocation_stats["set_input_shape_call_count"] >= run_count),
            "output_shape_query_called_per_frame": bool(run_count and self.allocation_stats["output_shape_query_call_count"] >= run_count),
            "input_buffers": {
                name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "numel": int(tensor.numel())}
                for name, tensor in self.input_buffers.items()
            },
            "output_buffers": {
                name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "numel": int(tensor.numel())}
                for name, tensor in self.output_buffers.items()
            },
        }

    def _profile_max_shape(self, name: str, fallback_shape: tuple[int, ...]) -> tuple[int, ...]:
        try:
            shapes = self.engine.get_tensor_profile_shape(name, 0)
            if shapes and len(shapes) == 3:
                max_shape = tuple(int(dim) for dim in shapes[2])
                if all(dim > 0 for dim in max_shape):
                    return max_shape
        except Exception:
            pass
        return fallback_shape

    def _ensure_flat_buffer(self, buffers: dict[str, torch.Tensor], name: str, numel: int, dtype: torch.dtype, stat_key: str) -> torch.Tensor:
        existing = buffers.get(name)
        if existing is None or existing.dtype != dtype or existing.numel() < int(numel):
            buffers[name] = torch.empty((int(numel),), dtype=dtype, device=self.device)
            self.bound_ptrs.pop(name, None)
            self.allocation_stats[stat_key] += 1
        return buffers[name]

    def _bind_address_if_needed(self, name: str, tensor: torch.Tensor) -> None:
        ptr = int(tensor.data_ptr())
        if self.bound_ptrs.get(name) != ptr:
            self.context.set_tensor_address(name, ptr)
            self.bound_ptrs[name] = ptr

    @staticmethod
    def _elapsed_ms(start: float) -> float:
        return (time.perf_counter() - start) * 1000.0

    def _sync_for_profile(self, collect_profile: bool) -> None:
        if collect_profile:
            self.stream.synchronize()

    def _run_impl(
        self,
        tensors_by_name: dict[str, torch.Tensor],
        *,
        cast_inputs_to_engine_dtype: bool,
        collect_profile: bool,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        trt = self.trt
        stream = self.stream
        self.allocation_stats["run_count"] += 1
        profile: dict[str, Any] = {
            "dtype_cast_ms": 0.0,
            "contiguous_ms": 0.0,
            "h2d_copy_ms": 0.0,
            "input_device_copy_ms": 0.0,
            "set_input_shape_ms": 0.0,
            "bind_address_ms": 0.0,
            "execute_async_ms": 0.0,
            "synchronize_ms": 0.0,
            "d2h_copy_ms": 0.0,
            "output_wrap_ms": 0.0,
            "input_buffer_reallocated": False,
            "output_buffer_reallocated": False,
            "set_input_shape_calls": 0,
            "output_shape_query_calls": 0,
            "h2d_copies": 0,
            "d2d_input_copies": 0,
            "d2h_copies": 0,
            "bytes_h2d": 0,
            "bytes_d2d_input": 0,
            "bytes_d2h": 0,
            "input_shapes": {},
            "output_shapes": {},
        }
        total_start = time.perf_counter()
        output_tensors: dict[str, torch.Tensor] = {}

        with torch.cuda.stream(stream):
            for index in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(index)
                if self.engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
                    continue
                source = tensors_by_name[name]
                target_dtype = _trt_dtype_to_torch(self.engine.get_tensor_dtype(name)) if cast_inputs_to_engine_dtype else source.dtype

                tensor = source
                if tensor.dtype != target_dtype:
                    start = time.perf_counter()
                    tensor = tensor.to(dtype=target_dtype)
                    self._sync_for_profile(collect_profile)
                    profile["dtype_cast_ms"] += self._elapsed_ms(start)
                if not tensor.is_contiguous():
                    start = time.perf_counter()
                    tensor = tensor.contiguous()
                    self._sync_for_profile(collect_profile)
                    profile["contiguous_ms"] += self._elapsed_ms(start)

                actual_shape = tuple(int(dim) for dim in tensor.shape)
                profile["input_shapes"][name] = list(actual_shape)
                if self.last_input_shapes.get(name) != actual_shape:
                    start = time.perf_counter()
                    self.context.set_input_shape(name, actual_shape)
                    profile["set_input_shape_ms"] += self._elapsed_ms(start)
                    profile["set_input_shape_calls"] += 1
                    self.allocation_stats["set_input_shape_call_count"] += 1
                    self.last_input_shapes[name] = actual_shape

                max_shape = self._profile_max_shape(name, actual_shape)
                max_numel = int(np.prod(max_shape, dtype=np.int64))
                before = self.allocation_stats["input_buffer_reallocation_count"]
                buffer = self._ensure_flat_buffer(self.input_buffers, name, max(max_numel, int(tensor.numel())), target_dtype, "input_buffer_reallocation_count")
                if self.allocation_stats["input_buffer_reallocation_count"] != before:
                    profile["input_buffer_reallocated"] = True

                view = buffer[: int(tensor.numel())].view(actual_shape)
                start = time.perf_counter()
                if tensor.device != self.device:
                    view.copy_(tensor, non_blocking=True)
                    self._sync_for_profile(collect_profile)
                    profile["h2d_copy_ms"] += self._elapsed_ms(start)
                    profile["h2d_copies"] += 1
                    profile["bytes_h2d"] += int(tensor.numel() * tensor.element_size())
                    self.allocation_stats["number_of_h2d_copies"] += 1
                    self.allocation_stats["total_bytes_h2d"] += int(tensor.numel() * tensor.element_size())
                else:
                    view.copy_(tensor, non_blocking=True)
                    self._sync_for_profile(collect_profile)
                    profile["input_device_copy_ms"] += self._elapsed_ms(start)
                    profile["d2d_input_copies"] += 1
                    profile["bytes_d2d_input"] += int(tensor.numel() * tensor.element_size())
                    self.allocation_stats["number_of_d2d_input_copies"] += 1
                    self.allocation_stats["total_bytes_d2d_input"] += int(tensor.numel() * tensor.element_size())

                start = time.perf_counter()
                self._bind_address_if_needed(name, buffer)
                profile["bind_address_ms"] += self._elapsed_ms(start)

            for index in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(index)
                if self.engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
                    continue
                start = time.perf_counter()
                shape = tuple(int(dim) for dim in self.context.get_tensor_shape(name))
                profile["output_shape_query_calls"] += 1
                self.allocation_stats["output_shape_query_call_count"] += 1
                profile["set_input_shape_ms"] += 0.0
                profile["output_shapes"][name] = list(shape)
                self.last_output_shapes[name] = shape
                profile.setdefault("output_shape_query_ms", 0.0)
                profile["output_shape_query_ms"] += self._elapsed_ms(start)

                dtype = _trt_dtype_to_torch(self.engine.get_tensor_dtype(name))
                numel = int(np.prod(shape, dtype=np.int64))
                before = self.allocation_stats["output_buffer_reallocation_count"]
                buffer = self._ensure_flat_buffer(self.output_buffers, name, numel, dtype, "output_buffer_reallocation_count")
                if self.allocation_stats["output_buffer_reallocation_count"] != before:
                    profile["output_buffer_reallocated"] = True
                start = time.perf_counter()
                self._bind_address_if_needed(name, buffer)
                profile["bind_address_ms"] += self._elapsed_ms(start)
                output_tensors[name] = buffer[:numel].view(shape)

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record(stream)
            ok = self.context.execute_async_v3(stream.cuda_stream)
            end_event.record(stream)
        if not ok:
            raise RuntimeError("TensorRT execution failed.")
        start = time.perf_counter()
        stream.synchronize()
        profile["synchronize_ms"] = self._elapsed_ms(start)
        profile["execute_async_ms"] = float(start_event.elapsed_time(end_event))

        start = time.perf_counter()
        output_tensors = {name: tensor for name, tensor in output_tensors.items()}
        profile["output_wrap_ms"] = self._elapsed_ms(start)
        profile["total_runner_ms"] = self._elapsed_ms(total_start)
        profile["allocation_report"] = self.allocation_report()
        return output_tensors, profile


def _comparison_from_outputs(reference: dict[str, Any], candidate: dict[str, Any], output_names: list[str], label: str) -> dict[str, Any]:
    frames = [
        {
            "record_len": 0,
            "outputs": {
                name: {"reference": reference[name], "candidate": candidate[name]}
                for name in output_names
                if name in reference and name in candidate
            },
        }
    ]
    report = aggregate_output_errors(frames, [name for name in output_names if name in reference and name in candidate])
    report.update({"success": True, "comparison": label})
    missing = [name for name in output_names if name not in candidate]
    if missing:
        report["success"] = False
        report["error"] = f"missing candidate outputs: {missing}"
    return report


def run_trt_output_equivalence(args: argparse.Namespace) -> dict[str, Any]:
    dirs = ensure_quant_deploy_run_dirs(args.output_root)
    onnx_path = Path(args.onnx_path)
    engine_paths = {
        "fp32": Path(args.fp32_engine_path) if getattr(args, "fp32_engine_path", None) else dirs["engine_fp32"] / "lidar_pyramid_fp32.engine",
        "fp16": Path(args.fp16_engine_path) if getattr(args, "fp16_engine_path", None) else dirs["engine_fp16"] / "lidar_pyramid_fp16.engine",
    }
    try:
        hypes, device, model, modality = _load_model_context(args)
        sample = _first_real_sample(hypes, device)
        with torch.no_grad():
            raw_output = model(sample)
        output_names = [name for name in DEFAULT_DEPLOY_OUTPUT_NAMES if name in raw_output and torch.is_tensor(raw_output[name])]
        if not output_names:
            output_names = _tensor_output_names(raw_output)
        export_mode = getattr(args, "pyramid_forward_export_mode", "fixed_static")
        input_names, tensors, _agent_modality_list = _input_names_and_tensors_for_export_mode(sample, modality, export_mode, int(args.max_cav))
        wrapper = _wrapper_for_export_mode(model, modality, output_names, export_mode, int(args.max_cav)).to(device).eval()
        with torch.no_grad():
            pytorch_outputs = _outputs_to_dict(output_names, wrapper(*tensors))
        report: dict[str, Any] = {
            "success": True,
            "output_names": output_names,
            "comparisons": {},
        }
        try:
            ort_outputs = _run_onnxruntime(onnx_path, input_names, tensors)
            report["comparisons"]["onnxruntime_fp32_vs_pytorch"] = _comparison_from_outputs(
                pytorch_outputs, ort_outputs, output_names, "onnxruntime_fp32_vs_pytorch"
            )
        except Exception as exc:
            report["comparisons"]["onnxruntime_fp32_vs_pytorch"] = {
                "success": False,
                "comparison": "onnxruntime_fp32_vs_pytorch",
                "error": str(exc),
            }

        tensors_by_name = {name: tensor for name, tensor in zip(input_names, tensors)}
        for precision, engine_path in engine_paths.items():
            label = f"trt_{precision}_vs_pytorch"
            if not engine_path.exists():
                report["comparisons"][label] = {"success": False, "comparison": label, "error": f"engine file does not exist: {engine_path}"}
                continue
            try:
                trt_outputs = _run_tensorrt(engine_path, tensors_by_name, device)
                report["comparisons"][label] = _comparison_from_outputs(pytorch_outputs, trt_outputs, output_names, label)
            except Exception as exc:
                report["comparisons"][label] = {"success": False, "comparison": label, "error": str(exc)}
        report["success"] = any(item.get("success") for item in report["comparisons"].values())
    except Exception as exc:
        report = {
            "success": False,
            "output_names": [],
            "comparisons": {},
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    export_mode = getattr(args, "pyramid_forward_export_mode", "fixed_static")
    suffix = "" if export_mode == "fixed_static" else f"_{export_mode}"
    save_json(report, dirs["evaluation"] / f"trt_output_equivalence{suffix}.json")
    save_json(report, dirs["debug"] / f"trt_output_equivalence{suffix}_debug.json")
    if export_mode == "fixed_static":
        save_json(report, dirs["evaluation"] / "trt_output_equivalence.json")
        save_json(report, dirs["debug"] / "trt_output_equivalence_debug.json")
    rows = []
    for comparison, item in (report.get("comparisons") or {}).items():
        compact = compact_error_for_summary(item)
        rows.append({"comparison": comparison, **(compact or {})})
    save_csv(rows, dirs["evaluation"] / "trt_output_equivalence.csv")
    return report
