"""Real Fisher and Q/DQ calibration providers."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from ..hashing import canonical_json_hash
from ..proxy.fisher_proxy import FisherStatistics
from .data_provider import build_dataset_and_loader, iter_limited, move_batch_to_device


QDQ_CALIBRATION_SEMANTICS_VERSION = "onnx-bn-fold-fixedk-entropy-v3"


def _tensor_dict_to_cpu(rows: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in rows.items()}


def collect_or_load_fisher_statistics(
    *,
    model: torch.nn.Module,
    adapter: Any,
    model_config_path: str | Path,
    device: torch.device,
    cache_path: str | Path,
    num_batches: int,
) -> FisherStatistics:
    path = Path(cache_path)
    if path.is_file():
        payload = torch.load(path, map_location="cpu")
        return FisherStatistics(
            gradients={key: value for key, value in payload["gradients"].items()},
            fisher_diag={key: value for key, value in payload["fisher_diag"].items()},
            manifest_hash=str(payload.get("manifest_hash", "")),
            statistics_version=str(payload.get("statistics_version", "fisher-diagonal-v1")),
        )
    if int(num_batches) <= 0:
        raise RuntimeError("fisher_statistics_missing:num_batches")
    _dataset, loader = build_dataset_and_loader(adapter, model_config_path, split="train", num_workers=0, visualize=False)
    batches = iter_limited(loader, int(num_batches))
    if not batches:
        raise RuntimeError("fisher_statistics_missing:no_calibration_batches")
    gradients: dict[str, torch.Tensor] = {}
    fisher: dict[str, torch.Tensor] = {}
    model.train(False)
    for batch in batches:
        batch = move_batch_to_device(batch, device)
        model.zero_grad(set_to_none=True)
        output = adapter.forward_for_task(model, batch)
        loss = adapter.compute_task_loss(output, batch)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach()
            gradients.setdefault(name, torch.zeros_like(param.detach(), device=grad.device))
            fisher.setdefault(name, torch.zeros_like(param.detach(), device=grad.device))
            gradients[name] += grad
            fisher[name] += grad.pow(2)
    count = float(len(batches))
    for name in list(gradients):
        gradients[name] = gradients[name] / count
        fisher[name] = fisher[name] / count
    manifest_hash = canonical_json_hash({"split": "train", "num_batches": int(num_batches), "model_config": str(model_config_path)})
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "gradients": _tensor_dict_to_cpu(gradients),
            "fisher_diag": _tensor_dict_to_cpu(fisher),
            "manifest_hash": manifest_hash,
            "statistics_version": "fisher-diagonal-v1",
        },
        path,
    )
    return FisherStatistics(_tensor_dict_to_cpu(gradients), _tensor_dict_to_cpu(fisher), manifest_hash=manifest_hash)


def weight_only_calibration_scales(module_paths: list[str], model: torch.nn.Module) -> dict[str, dict[str, float]]:
    """Fallback scale payload with real per-module weight scales and conservative activation scales."""

    modules = dict(model.named_modules())
    scales: dict[str, dict[str, float]] = {}
    for name in module_paths:
        module = modules.get(name)
        weight = getattr(module, "weight", None)
        if weight is None:
            continue
        weight_amax = float(weight.detach().abs().amax().item())
        scale = max(weight_amax / 127.0, 1.0e-8)
        scales[name] = {
            "activation_input_scale": 1.0,
            "weight_scale": scale,
            "activation_output_scale": 1.0,
            "activation_source": "fallback_static_unit_scale",
            "weight_source": "actual_pruned_weight_absmax_div127",
        }
    return scales


def collect_or_load_qdq_calibration_scales(
    *,
    model: torch.nn.Module,
    adapter: Any,
    model_config_path: str | Path,
    module_paths: list[str],
    device: torch.device,
    cache_path: str | Path,
    num_batches: int,
    onnx_path: str | Path | None = None,
    origin_map: Any | None = None,
    weight_granularity: str = "per_channel",
    activation_calibration_method: str = "entropy",
    histogram_bins: int = 2048,
    fixed_k: int = 29696,
) -> dict[str, dict[str, Any]]:
    path = Path(cache_path)
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return dict(payload["scales"])
    if int(num_batches) <= 0:
        raise RuntimeError("calibration_scales_missing:num_batches")
    _dataset, loader = build_dataset_and_loader(adapter, model_config_path, split="train", num_workers=0, visualize=False)
    batches = [move_batch_to_device(batch, device) for batch in iter_limited(loader, int(num_batches))]
    if not batches:
        raise RuntimeError("calibration_scales_missing:no_calibration_batches")

    def forward_fn(inner_model: torch.nn.Module, batch: Any) -> Any:
        return adapter.forward_for_task(inner_model, batch)

    if onnx_path is not None and origin_map is not None:
        from quantization.config import OnnxExportConfig
        from quantization.export.heal_lidar_pyramid import prepare_signal_maxk_inputs
        from search.integration.trt_compatible_export import build_search_trt_compatible_export_module

        export_config = OnnxExportConfig(fixed_k=int(fixed_k), min_agents=1, opt_agents=2, max_agents=2)
        wrapper = build_search_trt_compatible_export_module(
            model,
            output_names=export_config.output_names,
            fixed_k=export_config.fixed_k,
            modality="m1",
        ).to(device).eval()

        def fixed_k_forward(_inner_model: torch.nn.Module, batch: Any) -> Any:
            ego = batch["ego"] if isinstance(batch, Mapping) and "ego" in batch else batch
            prepared = prepare_signal_maxk_inputs(ego, config=export_config, modality="m1")
            return wrapper(**{name: tensor.to(device) for name, tensor in prepared.items()})

        scales, calibration_details = collect_onnx_bn_fold_aware_qdq_scales(
            model=model,
            batches=batches,
            module_paths=module_paths,
            forward_fn=fixed_k_forward,
            onnx_path=onnx_path,
            origin_map=origin_map,
            weight_granularity=weight_granularity,
            activation_calibration_method=activation_calibration_method,
            histogram_bins=histogram_bins,
        )
    else:
        try:
            from quantization.api import collect_calibration_scales
            from quantization.config import CalibrationConfig
        except ImportError:
            from heal_compress.quantization.api import collect_calibration_scales
            from heal_compress.quantization.config import CalibrationConfig
        result = collect_calibration_scales(
            model,
            batches,
            module_paths=module_paths,
            forward_fn=forward_fn,
            config=CalibrationConfig(frame_count=len(batches), require_observed_scales=True),
        )
        scales = result.scales()
        calibration_details = {"semantics_version": "pytorch-weighted-module-v1"}
    save_calibration_scales(
        path,
        scales,
        {
            "frame_count": len(batches),
            "module_count": len(module_paths),
            "manifest_hash": canonical_json_hash(
                {
                    "split": "train",
                    "frames": len(batches),
                    "modules": sorted(module_paths),
                    "activation_calibration_method": activation_calibration_method,
                    "histogram_bins": int(histogram_bins),
                    "fixed_k": int(fixed_k),
                }
            ),
            "source": "search.collect_onnx_bn_fold_aware_qdq_scales" if onnx_path is not None and origin_map is not None else "quantization.collect_calibration_scales",
            "fixed_k": int(fixed_k),
            **calibration_details,
        },
    )
    return scales


def _first_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, Mapping):
        for item in value.values():
            found = _first_tensor(item)
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _paired_batchnorm_path(model: torch.nn.Module, module_path: str) -> str | None:
    modules = dict(model.named_modules())
    if module_path not in modules or "." not in module_path:
        return None
    parent_path, child_name = module_path.rsplit(".", 1)
    parent = modules.get(parent_path)
    if parent is None:
        return None
    if child_name == "conv" or (child_name.startswith("conv") and child_name[4:].isdigit()):
        sibling = "bn" if child_name == "conv" else f"bn{child_name[4:]}"
        candidate = f"{parent_path}.{sibling}"
        if isinstance(modules.get(candidate), torch.nn.modules.batchnorm._BatchNorm):
            return candidate
    if isinstance(parent, torch.nn.Sequential) and child_name.isdigit():
        candidate = f"{parent_path}.{int(child_name) + 1}"
        if isinstance(modules.get(candidate), torch.nn.modules.batchnorm._BatchNorm):
            return candidate
    return None


def _origin_entries(origin_map: Any) -> list[Any]:
    if isinstance(origin_map, Mapping):
        return list(origin_map.get("entries", []) or [])
    return list(getattr(origin_map, "entries", []) or [])


def _field(row: Any, name: str) -> Any:
    return row.get(name) if isinstance(row, Mapping) else getattr(row, name)


def collect_onnx_bn_fold_aware_qdq_scales(
    *,
    model: torch.nn.Module,
    batches: Iterable[Any],
    module_paths: Sequence[str],
    forward_fn: Any,
    onnx_path: str | Path,
    origin_map: Any,
    weight_granularity: str = "per_tensor",
    activation_calibration_method: str = "entropy",
    histogram_bins: int = 2048,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Collect scales against the final, BatchNorm-folded ONNX compute nodes.

    PyTorch Conv hooks observe pre-BatchNorm weights and outputs.  ONNX export
    normally folds the following BatchNorm into the Conv initializer, so those
    scales are not valid at the explicit Q/DQ insertion points.  Inputs remain
    attached to the Conv, outputs move to the folded BatchNorm output, and
    weights are measured from the actual ONNX initializer.
    """

    import numpy as np
    import onnx
    from onnx import numpy_helper

    requested = [str(value) for value in module_paths]
    batch_rows = list(batches)
    if weight_granularity not in {"per_tensor", "per_channel"}:
        raise RuntimeError(f"unsupported_qdq_weight_granularity:{weight_granularity}")
    if activation_calibration_method not in {"absmax", "entropy"}:
        raise RuntimeError(f"unsupported_activation_calibration_method:{activation_calibration_method}")
    if int(histogram_bins) < 128:
        raise RuntimeError(f"activation_histogram_bins_too_small:{histogram_bins}")
    if len(requested) != len(set(requested)):
        raise RuntimeError("qdq_calibration_module_paths_not_unique")
    modules = dict(model.named_modules())
    missing = [name for name in requested if name not in modules]
    if missing:
        raise RuntimeError(f"qdq_calibration_modules_missing:{missing}")
    entries = {_field(row, "module_path"): row for row in _origin_entries(origin_map)}
    missing_entries = [name for name in requested if name not in entries]
    if missing_entries:
        raise RuntimeError(f"qdq_calibration_origin_entries_missing:{missing_entries}")
    onnx_model = onnx.load(str(onnx_path))
    initializers = {row.name: numpy_helper.to_array(row) for row in onnx_model.graph.initializer}
    node_by_name = {str(row.name): row for row in onnx_model.graph.node}
    consumers: dict[str, list[Any]] = {}
    for node in onnx_model.graph.node:
        for input_name in node.input:
            consumers.setdefault(str(input_name), []).append(node)
    output_paths: dict[str, str] = {}
    for name in requested:
        entry = entries[name]
        node = node_by_name.get(str(_field(entry, "canonical_node_name")))
        paired = _paired_batchnorm_path(model, name)
        onnx_has_bn_consumer = bool(
            node is not None
            and any(str(consumer.op_type) == "BatchNormalization" for output in node.output for consumer in consumers.get(str(output), []))
        )
        output_paths[name] = paired if paired is not None and not onnx_has_bn_consumer else name
    state = {
        name: {
            "input_amax": None,
            "output_amax": None,
            "input_hist": None,
            "output_hist": None,
            "input_count": 0,
            "output_count": 0,
        }
        for name in requested
    }
    phase = {"name": "amax"}
    handles = []

    def observe(name: str, role: str, tensor: torch.Tensor) -> None:
        value = tensor.detach().float().abs()
        row = state[name]
        if phase["name"] == "amax":
            amax = value.amax()
            key = f"{role}_amax"
            row[key] = amax if row[key] is None else torch.maximum(row[key], amax)
            row[f"{role}_count"] += 1
            return
        maximum = float(row[f"{role}_amax"].item())
        histogram = torch.histc(value, bins=int(histogram_bins), min=0.0, max=maximum)
        key = f"{role}_hist"
        row[key] = histogram if row[key] is None else row[key] + histogram

    def input_hook(name: str) -> Any:
        def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            tensor = _first_tensor(inputs)
            if tensor is None:
                raise RuntimeError(f"qdq_calibration_input_tensor_missing:{name}")
            observe(name, "input", tensor)

        return hook

    def output_hook(name: str) -> Any:
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            tensor = _first_tensor(output)
            if tensor is None:
                raise RuntimeError(f"qdq_calibration_output_tensor_missing:{name}")
            observe(name, "output", tensor)

        return hook

    for name in requested:
        handles.append(modules[name].register_forward_pre_hook(input_hook(name)))
        handles.append(modules[output_paths[name]].register_forward_hook(output_hook(name)))
    was_training = bool(model.training)
    frame_count = 0
    model.eval()
    try:
        with torch.inference_mode():
            for batch in batch_rows:
                forward_fn(model, batch)
                frame_count += 1
            if activation_calibration_method == "entropy":
                phase["name"] = "histogram"
                for batch in batch_rows:
                    forward_fn(model, batch)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    if frame_count <= 0:
        raise RuntimeError("qdq_calibration_received_no_frames")
    scales: dict[str, dict[str, Any]] = {}
    weight_scale_sources: dict[str, str] = {}
    weight_axes: dict[str, int | None] = {}
    weight_scale_shapes: dict[str, list[int]] = {}
    activation_thresholds: dict[str, dict[str, Any]] = {}

    def calibrated_amax(row: dict[str, Any], role: str) -> tuple[float, dict[str, Any]]:
        absolute_maximum = float(row[f"{role}_amax"].item())
        if activation_calibration_method == "absmax":
            return absolute_maximum, {
                "method": "absmax",
                "absolute_maximum": absolute_maximum,
                "clipping_threshold": absolute_maximum,
                "selected_bin": int(histogram_bins),
                "clipped_fraction": 0.0,
            }
        import numpy as np
        from modelopt.torch.quantization.calib.histogram import _compute_amax_entropy

        histogram = row[f"{role}_hist"].cpu().numpy().astype(np.int64)
        edges = np.linspace(0.0, absolute_maximum, int(histogram_bins) + 1, dtype=np.float64)
        threshold = float(
            _compute_amax_entropy(
                histogram.copy(),
                edges,
                num_bits=8,
                unsigned=False,
                stride=1,
                start_bin=128,
            ).item()
        )
        selected_bin = min(
            max(int(round(threshold / absolute_maximum * int(histogram_bins))), 128),
            int(histogram_bins),
        )
        return threshold, {
            "method": "entropy",
            "implementation": "modelopt.torch.quantization.calib.histogram._compute_amax_entropy",
            "absolute_maximum": absolute_maximum,
            "clipping_threshold": threshold,
            "selected_bin": selected_bin,
            "clipped_fraction": float(histogram[selected_bin:].sum() / max(histogram.sum(), 1)),
        }

    for name in requested:
        row = state[name]
        if row["input_count"] != frame_count or row["output_count"] != frame_count:
            raise RuntimeError(f"qdq_calibration_observation_count_mismatch:{name}")
        initializer_name = str(_field(entries[name], "weight_initializer"))
        weight = initializers.get(initializer_name)
        if weight is None:
            raise RuntimeError(f"qdq_calibration_onnx_initializer_missing:{name}:{initializer_name}")
        input_amax, input_threshold = calibrated_amax(row, "input")
        output_amax, output_threshold = calibrated_amax(row, "output")
        activation_thresholds[name] = {"input": input_threshold, "output": output_threshold}
        weight_amax = float(np.max(np.abs(weight)))
        values = (input_amax, output_amax, weight_amax)
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise RuntimeError(f"qdq_calibration_nonpositive_or_nonfinite_amax:{name}")
        weight_axis: int | None = None
        weight_scale: float | list[float]
        if weight_granularity == "per_channel":
            if node is None:
                raise RuntimeError(f"qdq_calibration_onnx_node_missing:{name}")
            if str(node.op_type) == "Conv":
                weight_axis = 0
            elif str(node.op_type) == "ConvTranspose":
                # ONNX ConvTranspose weights are [C_in, C_out/group, ...].
                # Axis 1 is the physical output-channel axis for group=1 and
                # the only representable output-channel axis for this layout.
                weight_axis = 1
            elif str(node.op_type) == "MatMul":
                # Linear exported as A @ B uses B=[C_in, C_out].
                weight_axis = 1
            elif str(node.op_type) == "Gemm":
                attributes = {str(attr.name): int(attr.i) for attr in node.attribute if str(attr.name) == "transB"}
                weight_axis = 0 if attributes.get("transB", 0) else 1
            else:
                raise RuntimeError(f"qdq_calibration_per_channel_unsupported_op:{name}:{node.op_type}")
            if weight_axis >= weight.ndim:
                raise RuntimeError(f"qdq_calibration_weight_axis_out_of_range:{name}:{weight_axis}:{list(weight.shape)}")
            reduce_axes = tuple(axis for axis in range(weight.ndim) if axis != weight_axis)
            channel_amax = np.max(np.abs(weight), axis=reduce_axes)
            if not np.all(np.isfinite(channel_amax)) or np.any(channel_amax <= 0.0):
                raise RuntimeError(f"qdq_calibration_nonpositive_per_channel_weight_amax:{name}")
            weight_scale = (channel_amax / 127.0).astype(np.float32).tolist()
        else:
            weight_scale = weight_amax / 127.0
        scales[name] = {
            "activation_input_scale": input_amax / 127.0,
            "weight_scale": weight_scale,
            "weight_axis": weight_axis,
            "weight_granularity": weight_granularity,
            "weight_scale_shape": [len(weight_scale)] if isinstance(weight_scale, list) else [],
            "activation_output_scale": output_amax / 127.0,
            "activation_input_tensor": str(node.input[0]) if node is not None else "",
            "activation_output_tensor": str(node.output[0]) if node is not None else "",
            "activation_scale_source": (
                "fixedK_export_wrapper_NVIDIA_ModelOpt_entropy"
                if activation_calibration_method == "entropy"
                else "fixedK_export_wrapper_absmax"
            ),
            "weight_scale_source": "final_folded_onnx_initializer",
        }
        weight_scale_sources[name] = initializer_name
        weight_axes[name] = weight_axis
        weight_scale_shapes[name] = [len(weight_scale)] if isinstance(weight_scale, list) else []
    return scales, {
        "semantics_version": QDQ_CALIBRATION_SEMANTICS_VERSION,
        "frame_count": frame_count,
        "output_module_paths": output_paths,
        "weight_initializer_names": weight_scale_sources,
        "weight_granularity": weight_granularity,
        "weight_axes": weight_axes,
        "weight_scale_shapes": weight_scale_shapes,
        "activation_calibration_method": activation_calibration_method,
        "histogram_bins": int(histogram_bins),
        "activation_thresholds": activation_thresholds,
        "passes": 2 if activation_calibration_method == "entropy" else 1,
    }


def save_calibration_scales(path: str | Path, scales: dict[str, Any], metadata: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({"metadata": metadata, "scales": scales}, indent=2, sort_keys=True), encoding="utf-8")
