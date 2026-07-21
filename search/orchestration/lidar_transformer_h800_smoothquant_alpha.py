"""Per-model SmoothQuant alpha grid with calibration50/200 stability evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from search.model_families.transformer.smoothquant_profiles import (
    SMOOTHQUANT_ALPHA_GRID,
    choose_alpha,
    selective_smoothquant_config,
)
from search.orchestration.lidar_transformer_h800_inventory import (
    MODEL_SPECS,
    _load,
    _real_batch,
)
from search.orchestration.lidar_transformer_h800_smoothquant import (
    _calibrate_manifest,
    _selected_module_paths,
)
from search.integration.runtime_environment import (
    configure_modelopt_inprocess,
    require_modelopt_cuda_extension,
    runtime_cuda_index_for_physical,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        return next((result for item in value if (result := _tensor(item)) is not None), None)
    if isinstance(value, Mapping):
        return next((result for item in value.values() if (result := _tensor(item)) is not None), None)
    return None


def _capture_modules(model: torch.nn.Module, paths: Iterable[str]) -> tuple[dict[str, torch.Tensor], list[Any]]:
    captured: dict[str, torch.Tensor] = {}
    modules = dict(model.named_modules())
    handles = []
    for path in sorted(set(paths)):
        module = modules.get(path)
        if module is None:
            continue
        def hook(_module: Any, _inputs: Any, output: Any, *, name: str = path) -> None:
            value = _tensor(output)
            if value is not None:
                captured[name] = value.detach().float()
        handles.append(module.register_forward_hook(hook))
    return captured, handles


def _relative_l2(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    left = reference.reshape(-1)
    right = candidate.reshape(-1)
    count = min(left.numel(), right.numel(), 2_000_000)
    if count == 0:
        return 0.0
    left, right = left[:count], right[:count]
    return float(torch.linalg.vector_norm(right - left) / torch.linalg.vector_norm(left).clamp_min(1e-12))


def _cosine(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    left = reference.reshape(-1)
    right = candidate.reshape(-1)
    count = min(left.numel(), right.numel(), 2_000_000)
    if count == 0:
        return 1.0
    left, right = left[:count], right[:count]
    return float(torch.dot(left, right) / (torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)).clamp_min(1e-12))


def _paired_path(path: str) -> str:
    replacements = (
        (".q_proj", ".k_proj"),
        (".q_linears.", ".k_linears."),
    )
    for source, target in replacements:
        if source in path:
            return path.replace(source, target)
    return ""


def _qk_metrics(reference: Mapping[str, torch.Tensor], candidate: Mapping[str, torch.Tensor]) -> tuple[float, float]:
    relative = []
    js_values = []
    for q_path, q_ref in reference.items():
        k_path = _paired_path(q_path)
        if not k_path or k_path not in reference or q_path not in candidate or k_path not in candidate:
            continue
        values = []
        for q_value, k_value in ((q_ref, reference[k_path]), (candidate[q_path], candidate[k_path])):
            q = q_value.reshape(-1, q_value.shape[-1])[:128]
            k = k_value.reshape(-1, k_value.shape[-1])[:128]
            width = min(q.shape[-1], k.shape[-1])
            values.append((q[:, :width] @ k[:, :width].T) / math.sqrt(max(width, 1)))
        qk_ref, qk_actual = values
        relative.append(_relative_l2(qk_ref, qk_actual))
        p = torch.softmax(qk_ref, dim=-1).clamp_min(1e-12)
        q = torch.softmax(qk_actual, dim=-1).clamp_min(1e-12)
        m = 0.5 * (p + q)
        js_values.append(float(0.5 * ((p * (p / m).log()).sum(-1).mean() + (q * (q / m).log()).sum(-1).mean())))
    return (
        float(np.mean(relative)) if relative else 0.0,
        float(np.mean(js_values)) if js_values else 0.0,
    )


def _quantizer_scales(model: torch.nn.Module, selected: Iterable[str]) -> dict[str, Any]:
    modules = dict(model.named_modules())
    rows = {}
    for path in selected:
        quantizer = getattr(modules[path], "input_quantizer", None)
        amax = getattr(quantizer, "_amax", None)
        pre = getattr(quantizer, "pre_quant_scale", None)
        if not torch.is_tensor(amax) or not torch.is_tensor(pre):
            raise RuntimeError(f"alpha_quantizer_state_missing:{path}")
        rows[path] = {
            "input_amax": amax.detach().float().cpu().reshape(-1).tolist(),
            "pre_quant_scale": pre.detach().float().cpu().reshape(-1).tolist(),
        }
    return rows


def _scale_stability(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    deltas = []
    for path in sorted(set(left) & set(right)):
        for key in ("input_amax", "pre_quant_scale"):
            a = np.asarray(left[path][key], dtype=np.float64).reshape(-1)
            b = np.asarray(right[path][key], dtype=np.float64).reshape(-1)
            count = min(a.size, b.size)
            if count:
                deltas.extend((np.abs(a[:count] - b[:count]) / np.maximum(np.abs(b[:count]), 1e-12)).tolist())
    return float(np.median(deltas)) if deltas else float("inf")


def _attention_parents(paths: Iterable[str]) -> tuple[str, ...]:
    result = set()
    for path in paths:
        for marker in (".q_proj", ".q_linears."):
            if marker in path:
                result.add(path.split(marker, 1)[0])
    return tuple(sorted(result))


def _numeric_metrics(
    *, baseline: Any, quantized: Any, batch: Mapping[str, Any],
    selected: tuple[str, ...], inventory: Mapping[str, Any],
) -> dict[str, Any]:
    ffn2 = tuple(
        str(row["module_path"]) for row in inventory["rows"]
        if row.get("canonical_role") == "ffn2" and row.get("module_path")
    )
    parents = _attention_parents(selected)
    paths = tuple(sorted(set(selected) | set(ffn2) | set(parents)))
    reference, ref_handles = _capture_modules(baseline.model, paths)
    candidate, cand_handles = _capture_modules(quantized.model, paths)
    saturation_values: list[float] = []
    saturation_handles = []
    quantized_modules = dict(quantized.model.named_modules())
    for path in selected:
        module = quantized_modules[path]
        quantizer = getattr(module, "input_quantizer", None)
        def pre_hook(_module: Any, inputs: Any, *, owner: Any = quantizer) -> None:
            value = _tensor(inputs)
            amax = getattr(owner, "_amax", None)
            pre = getattr(owner, "pre_quant_scale", None)
            if value is None or not torch.is_tensor(amax):
                return
            observed = value.detach().float()
            if torch.is_tensor(pre):
                observed = observed * pre.detach().float()
            saturation_values.append(
                float((observed.abs() >= amax.detach().float()).float().mean())
            )
        saturation_handles.append(module.register_forward_pre_hook(pre_hook))
    with torch.inference_mode():
        baseline.adapter.forward_for_task(baseline.model, batch)
        quantized.adapter.forward_for_task(quantized.model, batch)
    for handle in (*ref_handles, *cand_handles, *saturation_handles):
        handle.remove()
    shared_projection = [path for path in selected if path in reference and path in candidate]
    projection_l2 = [_relative_l2(reference[path], candidate[path]) for path in shared_projection]
    projection_cosine = [_cosine(reference[path], candidate[path]) for path in shared_projection]
    qk_l2, softmax_js = _qk_metrics(
        {path: reference[path] for path in shared_projection},
        {path: candidate[path] for path in shared_projection},
    )
    ffn_l2 = [
        _relative_l2(reference[path], candidate[path])
        for path in ffn2 if path in reference and path in candidate
    ]
    residual_l2 = [
        _relative_l2(reference[path], candidate[path])
        for path in parents if path in reference and path in candidate
    ]
    finite = all(torch.isfinite(value).all().item() for value in candidate.values())
    return {
        "projection_relative_l2": float(np.mean(projection_l2)) if projection_l2 else 0.0,
        "projection_cosine": float(np.mean(projection_cosine)) if projection_cosine else 1.0,
        "qk_relative_l2": qk_l2,
        "softmax_js": softmax_js,
        "ffn_output_relative_l2": float(np.mean(ffn_l2)) if ffn_l2 else 0.0,
        "residual_update_relative_l2": float(np.mean(residual_l2)) if residual_l2 else 0.0,
        "finite": bool(finite),
        "saturation_ratio": float(np.mean(saturation_values)) if saturation_values else 0.0,
        "projection_count": len(shared_projection),
        "ffn_output_count": len(ffn_l2),
        "residual_scope_count": len(residual_l2),
        "qk_pair_count": sum(bool(_paired_path(path) in shared_projection) for path in shared_projection) // 2,
    }


def _quantize(
    *, model_name: str, inventory: Mapping[str, Any], selected: tuple[str, ...],
    alpha: float, manifest: Path, expected_samples: int, device: torch.device,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    import modelopt.torch.quantization as mtq  # type: ignore
    from search.model_families.transformer.projection_rewrite import (
        split_cobevt_fused_qkv, split_v2xvit_fused_qkv,
    )

    bundle, _ = _load(model_name, device)
    (split_cobevt_fused_qkv if model_name == "lidar_cobevt" else split_v2xvit_fused_qkv)(bundle.model)
    calibration: dict[str, Any] = {}
    def forward_loop(_model: torch.nn.Module) -> None:
        calibration.update(
            _calibrate_manifest(
                bundle=bundle, config_path=str(MODEL_SPECS[model_name]["config"]),
                manifest_path=manifest, device=device, expected_samples=expected_samples,
            )
        )
    quantized = mtq.quantize(
        bundle.model, selective_smoothquant_config(selected, alpha=alpha),
        forward_loop=forward_loop,
    )
    if quantized is not bundle.model:
        bundle.model = quantized
    return bundle, calibration, _quantizer_scales(bundle.model, selected)


def run_grid(
    output_root: Path,
    model_name: str,
    physical_gpu: int,
    destination_section: str = "smoothquant_alpha_sm90",
) -> dict[str, Any]:
    from search.model_families.transformer.projection_rewrite import (
        split_cobevt_fused_qkv, split_v2xvit_fused_qkv,
    )

    toolchain = configure_modelopt_inprocess(
        output_root=output_root,
        cache_namespace=f"alpha_{model_name}_gpu{physical_gpu}",
    )
    cuda_extension = require_modelopt_cuda_extension("int8")
    runtime_gpu = runtime_cuda_index_for_physical(physical_gpu)
    torch.cuda.set_device(runtime_gpu)
    device = torch.device(f"cuda:{runtime_gpu}")
    inventory = json.loads((output_root / "inventory" / model_name / "inventory.json").read_text(encoding="utf-8"))
    selected = _selected_module_paths(inventory, ("q_projection", "k_projection"))
    baseline, _ = _load(model_name, device)
    (split_cobevt_fused_qkv if model_name == "lidar_cobevt" else split_v2xvit_fused_qkv)(baseline.model)
    batch = _real_batch(baseline, str(MODEL_SPECS[model_name]["config"]), device)
    root = output_root / destination_section / model_name
    rows = []
    for alpha in SMOOTHQUANT_ALPHA_GRID:
        scales: dict[int, dict[str, Any]] = {}
        calibrations = {}
        quantized200 = None
        for count in (50, 200):
            manifest = output_root / "evaluation" / "manifests" / model_name / f"calibration{count}.json"
            bundle, calibration, state = _quantize(
                model_name=model_name, inventory=inventory, selected=selected,
                alpha=alpha, manifest=manifest, expected_samples=count, device=device,
            )
            scales[count] = state
            calibrations[count] = calibration
            if count == 200:
                quantized200 = bundle
            else:
                del bundle
                torch.cuda.empty_cache()
        assert quantized200 is not None
        metrics = _numeric_metrics(
            baseline=baseline, quantized=quantized200, batch=batch,
            selected=selected, inventory=inventory,
        )
        metrics["scale_stability_delta"] = _scale_stability(scales[50], scales[200])
        row = {"alpha": float(alpha), **metrics}
        if not row["finite"] or any(not math.isfinite(float(row[key])) for key in (
            "projection_relative_l2", "qk_relative_l2", "softmax_js",
            "ffn_output_relative_l2", "residual_update_relative_l2",
            "saturation_ratio", "scale_stability_delta",
        )):
            raise RuntimeError(f"smoothquant_alpha_nonfinite:{model_name}:{alpha}")
        alpha_dir = root / f"alpha_{alpha:g}"
        _write_json(alpha_dir / "calibration50.json", calibrations[50])
        _write_json(alpha_dir / "calibration200.json", calibrations[200])
        _write_json(alpha_dir / "scale50.json", scales[50])
        _write_json(alpha_dir / "scale200.json", scales[200])
        _write_json(alpha_dir / "numeric_metrics.json", row)
        rows.append(row)
        del quantized200
        torch.cuda.empty_cache()
    selected_row = choose_alpha(rows)
    result = {
        "status": "ok",
        "model": model_name,
        "physical_gpu": physical_gpu,
        "toolchain": toolchain,
        "modelopt_cuda_extension": cuda_extension,
        "profile": "SQ1_QK",
        "alpha_grid": list(SMOOTHQUANT_ALPHA_GRID),
        "rows": rows,
        "selected": selected_row,
        "selection_uses_calibration50_and_200": True,
        "metrics_are_pytorch_fake_quant_screening": True,
        "engine_validation_still_required": True,
    }
    _write_json(root / "alpha_selection.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument(
        "--destination-section",
        choices=("smoothquant_alpha", "smoothquant_alpha_sm90"),
        default="smoothquant_alpha_sm90",
    )
    args = parser.parse_args(argv)
    result = run_grid(
        Path(args.output_root).resolve(),
        args.model,
        args.physical_gpu,
        args.destination_section,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
