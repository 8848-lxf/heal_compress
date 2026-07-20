"""Real-activation ModelOpt SmoothQuant screening for CoBEVT projections."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from search.model_families.lidar_cobevt.attention_dim_pruning import (
    materialize_attention_bottleneck,
    uniform_attention_masks,
)
from search.model_families.lidar_cobevt.minimal_structure_quant_latency import (
    choose_smoothquant_alpha,
    selective_smoothquant_config,
    smoothquant_profiles,
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


def _safe(value: str) -> str:
    return value.replace(".", "__")


def _module_name(role: str) -> str:
    return {
        "q_projection": "q_proj",
        "k_projection": "k_proj",
        "v_projection": "v_proj",
        "output_projection": "out_proj",
    }[str(role)]


class ProjectionBank(nn.Module):
    def __init__(self, records: Mapping[str, Mapping[str, nn.Linear]]) -> None:
        super().__init__()
        self.blocks = nn.ModuleDict(
            {
                _safe(block): nn.ModuleDict(
                    {
                        _module_name(role): nn.Linear(
                            module.in_features,
                            module.out_features,
                            bias=module.bias is not None,
                        )
                        for role, module in roles.items()
                    }
                )
                for block, roles in records.items()
            }
        )
        with torch.no_grad():
            for block, roles in records.items():
                for role, source in roles.items():
                    target = self.blocks[_safe(block)][_module_name(role)]
                    target.weight.copy_(source.weight)
                    if source.bias is not None:
                        target.bias.copy_(source.bias)


def _projection_modules(model: nn.Module) -> dict[str, dict[str, nn.Linear]]:
    result: dict[str, dict[str, nn.Linear]] = {}
    for name, module in model.named_modules():
        if module.__class__.__name__ != "PrunableCobevtAttention":
            continue
        result[name] = {
            "q_projection": module.q_proj,
            "k_projection": module.k_proj,
            "v_projection": module.v_proj,
            "output_projection": module.out_proj,
        }
    if len(result) != 6:
        raise RuntimeError(f"smoothquant_attention_block_count:{len(result)}")
    return result


def capture_projection_inputs(
    *,
    checkpoint: Path,
    config: Path,
    heal_root: Path,
    manifest: Path,
    device: torch.device,
    max_rows_per_frame: int = 512,
) -> tuple[dict[str, dict[str, nn.Linear]], dict[tuple[str, str], list[torch.Tensor]], dict[str, Any]]:
    from search.integration.data_provider import build_dataset_and_loader, move_batch_to_device
    from search.model_families.lidar_cobevt.model_capability import CobevtModelCapability

    bundle = CobevtModelCapability(checkpoint, config, heal_root).load(device=device)
    physical = materialize_attention_bottleneck(
        bundle.model, uniform_attention_masks(bundle.model, d_qk=32, d_v=32)
    )
    if not physical.passed:
        raise RuntimeError(f"smoothquant_s0_materialization_failed:{physical.issues}")
    modules = _projection_modules(bundle.model)
    captures: dict[tuple[str, str], list[torch.Tensor]] = defaultdict(list)
    handles = []
    for block, roles in modules.items():
        for role, module in roles.items():
            def hook(_module: nn.Module, args: tuple[Any, ...], *, key=(block, role)) -> None:
                if not args or not torch.is_tensor(args[0]):
                    raise RuntimeError(f"smoothquant_projection_input_missing:{key}")
                value = args[0].detach().reshape(-1, args[0].shape[-1])
                captures[key].append(value[: int(max_rows_per_frame)].cpu().float())

            handles.append(module.register_forward_pre_hook(hook))
    manifest_payload = json.loads(manifest.read_text())
    selected = set(str(value) for value in manifest_payload["evaluation_frame_ids"])
    dataset, loader = build_dataset_and_loader(
        bundle.adapter, config, split="val", num_workers=8, visualize=True
    )
    resolved = bundle.adapter._absolutize_dataset_paths(dict(bundle.hypes))
    split_ids = [str(value) for value in json.loads(Path(resolved["validate_dir"]).read_text())]
    evaluated: list[str] = []
    try:
        bundle.model.eval()
        with torch.no_grad():
            for index, batch in enumerate(loader):
                if index >= len(split_ids):
                    break
                frame_id = split_ids[index]
                if frame_id not in selected:
                    continue
                bundle.adapter.forward_for_task(
                    bundle.model, move_batch_to_device(batch, device)
                )
                evaluated.append(frame_id)
                if len(evaluated) == len(selected):
                    break
    finally:
        for handle in handles:
            handle.remove()
    if set(evaluated) != selected:
        raise RuntimeError(f"smoothquant_capture_frames_missing:{sorted(selected-set(evaluated))}")
    if any(len(captures[(block, role)]) != len(selected) for block in modules for role in modules[block]):
        raise RuntimeError("smoothquant_projection_capture_incomplete")
    cpu_modules = {
        block: {role: module.cpu() for role, module in roles.items()}
        for block, roles in modules.items()
    }
    return cpu_modules, captures, {
        "evaluated_frame_ids": evaluated,
        "frames": len(evaluated),
        "manifest": str(manifest),
        "manifest_hash": str(manifest_payload["manifest_hash"]),
        "max_rows_per_frame": int(max_rows_per_frame),
        "physical_structure_hash": physical.structure_hash,
    }


def _metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    ref = reference.double().flatten()
    cand = candidate.double().flatten()
    difference = cand - ref
    denominator = max(float(torch.linalg.vector_norm(ref)), torch.finfo(torch.float64).eps)
    cosine = float(torch.nn.functional.cosine_similarity(ref, cand, dim=0))
    return {
        "cosine": cosine,
        "max_abs": float(difference.abs().max()),
        "relative_l2": float(torch.linalg.vector_norm(difference) / denominator),
    }


def run_alpha_sweep(
    *,
    output_dir: Path,
    checkpoint: Path,
    config: Path,
    heal_root: Path,
    manifest: Path,
    device: torch.device,
) -> dict[str, Any]:
    import modelopt.torch.quantization as mtq  # type: ignore

    output_dir.mkdir(parents=True, exist_ok=True)
    modules, captures, capture = capture_projection_inputs(
        checkpoint=checkpoint,
        config=config,
        heal_root=heal_root,
        manifest=manifest,
        device=device,
    )
    _write_json(output_dir / "capture_manifest.json", capture)
    rows: list[dict[str, Any]] = []
    profiles = [row for row in smoothquant_profiles() if row.profile_id != "SQ0"]
    for profile in profiles:
        for alpha in (0.3, 0.5, 0.7):
            bank = ProjectionBank(modules).to(device).eval()
            config_payload = selective_smoothquant_config(
                profile.int8_projection_roles, alpha=alpha
            )

            def forward_loop(model: ProjectionBank) -> None:
                with torch.no_grad():
                    for block, roles in modules.items():
                        for role in profile.int8_projection_roles:
                            module_name = _module_name(role)
                            target = model.blocks[_safe(block)][module_name]
                            for value in captures[(block, role)]:
                                target(value.to(device))

            mtq.quantize(bank, config_payload, forward_loop=forward_loop)
            role_metrics = []
            scale_records = []
            projection_outputs: dict[tuple[str, str, int], tuple[torch.Tensor, torch.Tensor]] = {}
            with torch.no_grad():
                for block, roles in modules.items():
                    for role in profile.int8_projection_roles:
                        module_name = _module_name(role)
                        baseline = roles[role].to(device).eval()
                        candidate = bank.blocks[_safe(block)][module_name]
                        for frame_index, value in enumerate(captures[(block, role)]):
                            reference = baseline(value.to(device))
                            quantized = candidate(value.to(device))
                            projection_outputs[(block, role, frame_index)] = (
                                reference,
                                quantized,
                            )
                            role_metrics.append(
                                {
                                    "block": block,
                                    "frame_index": frame_index,
                                    "role": role,
                                    **_metrics(reference, quantized),
                                }
                            )
                        quantizer = getattr(candidate, "input_quantizer", None)
                        pre_scale = getattr(quantizer, "pre_quant_scale", None)
                        scale_records.append(
                            {
                                "block": block,
                                "role": role,
                                "pre_quant_scale_present": torch.is_tensor(pre_scale),
                                "pre_quant_scale_shape": list(pre_scale.shape) if torch.is_tensor(pre_scale) else [],
                                "activation_axis": getattr(quantizer, "axis", None),
                                "weight_axis": getattr(getattr(candidate, "weight_quantizer", None), "axis", None),
                            }
                        )
            qk_metrics = []
            softmax_js_values = []
            for block in modules:
                for frame_index in range(len(captures[(block, "q_projection")])):
                    q_reference, q_candidate = projection_outputs[
                        (block, "q_projection", frame_index)
                    ]
                    k_reference, k_candidate = projection_outputs[
                        (block, "k_projection", frame_index)
                    ]

                    def per_head(value: torch.Tensor) -> torch.Tensor:
                        return value.reshape(-1, 32, 8, 32).permute(0, 2, 1, 3)

                    q_ref = per_head(q_reference.float())
                    q_cand = per_head(q_candidate.float())
                    k_ref = per_head(k_reference.float())
                    k_cand = per_head(k_candidate.float())
                    score_ref = torch.matmul(
                        q_ref * (32.0**-0.5), k_ref.transpose(-1, -2)
                    )
                    score_cand = torch.matmul(
                        q_cand * (32.0**-0.5), k_cand.transpose(-1, -2)
                    )
                    qk_metrics.append(_metrics(score_ref, score_cand))
                    probability_ref = torch.softmax(score_ref.double(), dim=-1)
                    probability_cand = torch.softmax(score_cand.double(), dim=-1)
                    midpoint = 0.5 * (probability_ref + probability_cand)
                    epsilon = torch.finfo(torch.float64).eps
                    js = 0.5 * (
                        probability_ref
                        * (
                            probability_ref.clamp_min(epsilon).log()
                            - midpoint.clamp_min(epsilon).log()
                        )
                    ).sum(dim=-1) + 0.5 * (
                        probability_cand
                        * (
                            probability_cand.clamp_min(epsilon).log()
                            - midpoint.clamp_min(epsilon).log()
                        )
                    ).sum(dim=-1)
                    softmax_js_values.append(float(js.mean()))
            aggregate = {
                "relative_l2": float(sum(row["relative_l2"] for row in role_metrics) / len(role_metrics)),
                "cosine": float(sum(row["cosine"] for row in role_metrics) / len(role_metrics)),
                "max_abs": float(max(row["max_abs"] for row in role_metrics)),
                "qk_relative_l2": float(
                    sum(row["relative_l2"] for row in qk_metrics) / len(qk_metrics)
                ),
                "qk_cosine": float(
                    sum(row["cosine"] for row in qk_metrics) / len(qk_metrics)
                ),
                "softmax_js": float(
                    sum(softmax_js_values) / len(softmax_js_values)
                ),
            }
            rows.append(
                {
                    "profile_id": profile.profile_id,
                    "alpha": alpha,
                    **aggregate,
                    "pre_quant_scale_complete": all(row["pre_quant_scale_present"] for row in scale_records),
                    "scale_records": scale_records,
                    "role_metrics": role_metrics,
                    "config": config_payload,
                }
            )
            del bank
            torch.cuda.empty_cache()
    selected = {}
    for profile in profiles:
        profile_rows = [row for row in rows if row["profile_id"] == profile.profile_id]
        choice = choose_smoothquant_alpha(
            [
                {
                    "alpha": row["alpha"],
                    "relative_l2": row["relative_l2"],
                    "softmax_js": row["softmax_js"],
                }
                for row in profile_rows
            ]
        )
        selected[profile.profile_id] = next(
            row for row in profile_rows if float(row["alpha"]) == choice["alpha"]
        )
    _write_json(output_dir / "smoothquant_alpha_sweep.json", rows)
    _write_json(output_dir / "smoothquant_selected_alpha.json", selected)
    csv_rows = [
        {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
        for row in rows
    ]
    with (output_dir / "smoothquant_alpha_sweep.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    return {"rows": len(rows), "selected": {key: value["alpha"] for key, value in selected.items()}}


def run_full_model_build(
    *,
    output_dir: Path,
    experiment_root: Path,
    profile_id: str,
    checkpoint: Path,
    config: Path,
    heal_root: Path,
    manifest: Path,
    device: torch.device,
    physical_gpu: int,
    trt_root: Path,
    plugin: Path,
) -> dict[str, Any]:
    import modelopt.torch.quantization as mtq  # type: ignore

    from search.integration.data_provider import build_dataset_and_loader, move_batch_to_device
    from search.orchestration.lidar_cobevt_attention_pruning import run_export_build

    selected_alpha = json.loads(
        (experiment_root / "smoothquant_experiment/smoothquant_selected_alpha.json").read_text()
    )
    profile = next(row for row in smoothquant_profiles() if row.profile_id == profile_id)
    if profile.profile_id == "SQ0":
        raise ValueError("smoothquant_full_build_requires_int8_profile")
    if profile.requires_successful_profile:
        prerequisite = (
            experiment_root
            / "smoothquant_experiment/full_model"
            / profile.requires_successful_profile
            / "full_engine_build_results_f3_rest_fp16_qk_fp32_minimal_island.json"
        )
        if not prerequisite.is_file() or not any(
            row.get("status") == "ok" for row in json.loads(prerequisite.read_text())
        ):
            result = {
                "profile_id": profile.profile_id,
                "status": "blocked_prerequisite_failed",
                "requires_successful_profile": profile.requires_successful_profile,
            }
            _write_json(output_dir / "blocked.json", result)
            return result
    alpha = float(selected_alpha[profile.profile_id]["alpha"])
    source_experiment = json.loads((experiment_root / "experiment_config.json").read_text())
    destination_experiment = copy.deepcopy(source_experiment)
    destination_experiment["candidates"] = [
        row for row in destination_experiment["candidates"] if row["candidate_id"] == "S0"
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "experiment_config.json", destination_experiment)

    manifest_payload = json.loads(manifest.read_text())
    selected_frames = set(str(value) for value in manifest_payload["evaluation_frame_ids"])
    quant_config = selective_smoothquant_config(
        profile.int8_projection_roles, alpha=alpha
    )

    def transform(bundle: Any, _spec: Any, destination: Path) -> Mapping[str, Any]:
        dataset, loader = build_dataset_and_loader(
            bundle.adapter, config, split="val", num_workers=8, visualize=True
        )
        resolved = bundle.adapter._absolutize_dataset_paths(dict(bundle.hypes))
        split_ids = [str(value) for value in json.loads(Path(resolved["validate_dir"]).read_text())]
        calibrated: list[str] = []

        def forward_loop(model: nn.Module) -> None:
            model.eval()
            with torch.no_grad():
                for index, batch in enumerate(loader):
                    if index >= len(split_ids):
                        break
                    frame_id = split_ids[index]
                    if frame_id not in selected_frames:
                        continue
                    bundle.adapter.forward_for_task(
                        model, move_batch_to_device(batch, device)
                    )
                    calibrated.append(frame_id)
                    if len(calibrated) == len(selected_frames):
                        break

        quantized = mtq.quantize(bundle.model, quant_config, forward_loop=forward_loop)
        if quantized is not bundle.model:
            bundle.model = quantized
        if set(calibrated) != selected_frames:
            raise RuntimeError(
                f"smoothquant_full_calibration_frames_missing:{sorted(selected_frames-set(calibrated))}"
            )
        selected_records = []
        for name, module in bundle.model.named_modules():
            if not any(name.endswith(token) for token in ("q_proj", "k_proj", "v_proj", "out_proj")):
                continue
            role = {
                "q_proj": "q_projection",
                "k_proj": "k_projection",
                "v_proj": "v_projection",
                "out_proj": "output_projection",
            }[name.rsplit(".", 1)[-1]]
            if role not in profile.int8_projection_roles:
                continue
            input_quantizer = getattr(module, "input_quantizer", None)
            weight_quantizer = getattr(module, "weight_quantizer", None)
            pre_scale = getattr(input_quantizer, "pre_quant_scale", None)
            input_enabled = getattr(input_quantizer, "is_enabled", False)
            weight_enabled = getattr(weight_quantizer, "is_enabled", False)
            selected_records.append(
                {
                    "module": name,
                    "role": role,
                    "module_class": module.__class__.__name__,
                    "pre_quant_scale_present": torch.is_tensor(pre_scale),
                    "pre_quant_scale_shape": list(pre_scale.shape) if torch.is_tensor(pre_scale) else [],
                    "input_quantizer_axis": getattr(input_quantizer, "axis", None),
                    "weight_quantizer_axis": getattr(weight_quantizer, "axis", None),
                    "input_quantizer_enabled": bool(
                        input_enabled() if callable(input_enabled) else input_enabled
                    ),
                    "weight_quantizer_enabled": bool(
                        weight_enabled() if callable(weight_enabled) else weight_enabled
                    ),
                }
            )
        expected = 6 * len(profile.int8_projection_roles)
        if len(selected_records) != expected:
            raise RuntimeError(
                f"smoothquant_selected_projection_count:{len(selected_records)}:{expected}"
            )
        report = {
            "alpha": alpha,
            "calibration_manifest": str(manifest),
            "calibration_manifest_hash": manifest_payload["manifest_hash"],
            "calibrated_frames": calibrated,
            "modelopt_config": quant_config,
            "profile_id": profile.profile_id,
            "selected_projection_records": selected_records,
        }
        _write_json(destination / "smoothquant_transform_report.json", report)
        return report

    result = run_export_build(
        output_dir=output_dir,
        checkpoint=checkpoint,
        config=config,
        heal_root=heal_root,
        device=device,
        physical_gpu=physical_gpu,
        trt_root=trt_root,
        plugin_path=plugin,
        candidate_ids=("S0",),
        attention_boundary_profile_name="F3_rest_fp16_qk_fp32_minimal_island",
        pre_export_transform=transform,
    )
    _write_json(output_dir / "smoothquant_full_build_summary.json", result)
    return {"profile_id": profile_id, **result}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("alpha-sweep", "full-build"), default="alpha-sweep")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--heal-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--experiment-root", default="")
    parser.add_argument("--profile-id", choices=("SQ1", "SQ2", "SQ3"), default="SQ1")
    parser.add_argument("--physical-gpu", type=int, default=3)
    parser.add_argument("--trt-root", default="")
    parser.add_argument("--plugin", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.phase == "alpha-sweep":
        result = run_alpha_sweep(
            output_dir=Path(args.output_dir),
            checkpoint=Path(args.checkpoint),
            config=Path(args.config),
            heal_root=Path(args.heal_root),
            manifest=Path(args.manifest),
            device=torch.device(args.device),
        )
    else:
        if not args.experiment_root or not args.trt_root or not args.plugin:
            raise ValueError("smoothquant_full_build_paths_required")
        result = run_full_model_build(
            output_dir=Path(args.output_dir),
            experiment_root=Path(args.experiment_root),
            profile_id=args.profile_id,
            checkpoint=Path(args.checkpoint),
            config=Path(args.config),
            heal_root=Path(args.heal_root),
            manifest=Path(args.manifest),
            device=torch.device(args.device),
            physical_gpu=args.physical_gpu,
            trt_root=Path(args.trt_root),
            plugin=Path(args.plugin),
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
