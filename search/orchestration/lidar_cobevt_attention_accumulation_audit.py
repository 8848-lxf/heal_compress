"""Reproducible CoBEVT Attention multiplication/accumulation audit."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from search.model_families.lidar_cobevt.attention_accumulation import (
    accumulation_profile_manifest,
    dynamic_quantize_api_verdict,
    run_fp16_numerical_matrix,
    write_tp_attention_contract,
)


DEFAULT_CHECKPOINT = Path(
    "/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/"
    "LiDAROnly/lidar_cobevt/net_epoch_bestval_at19.pth"
)
DEFAULT_CONFIG = DEFAULT_CHECKPOINT.with_name("config.yaml")
DEFAULT_HEAL_ROOT = Path("/home/lixingfeng/UniAD_examine/HEAL")
DEFAULT_TRT_ROOT = Path(
    "/home/lixingfeng/UniAD_examine/HEAL/prune_model/TensorRT-10.9_x86_cu118"
)
DEFAULT_SOURCE_OUTPUT = Path(
    "/data/lxf/heal_data/outputs/cobevt_head_dim_trt_capability_20260719_122046"
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _command(command: list[str], *, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=dict(env or os.environ),
        check=False,
    )
    return {
        "command": command,
        "returncode": int(completed.returncode),
        "output": completed.stdout,
    }


def audit_tensorrt_10_9(
    *, trt_root: str | Path = DEFAULT_TRT_ROOT, header_root: str | Path | None = None
) -> dict[str, Any]:
    root = Path(trt_root).expanduser().resolve()
    headers = Path(header_root).expanduser().resolve() if header_root else root / "include"
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join(
        [str(root / "lib"), str(root / "targets/x86_64-linux-gnu/lib"), env.get("LD_LIBRARY_PATH", "")]
    )
    probe = [
        sys.executable,
        "-c",
        (
            "import inspect, json, tensorrt as trt; "
            "n=trt.Builder(trt.Logger(trt.Logger.ERROR)).create_network(0); "
            "print(json.dumps({'version':trt.__version__,"
            "'dynamic_quantize_class':hasattr(trt,'IDynamicQuantizeLayer'),"
            "'attention_class':hasattr(trt,'IAttention'),"
            "'add_dynamic_quantize':hasattr(type(n),'add_dynamic_quantize'),"
            "'matrix_multiply_precision':hasattr(trt.IMatrixMultiplyLayer,'precision'),"
            "'normalization_compute_precision':hasattr(trt.INormalizationLayer,'compute_precision'),"
            "'network_methods':[x for x in dir(n) if 'quant' in x.lower()]},sort_keys=True))"
        ),
    ]
    python_probe = _command(probe, env=env)
    api: dict[str, Any] = {}
    if python_probe["returncode"] == 0:
        try:
            api = json.loads(python_probe["output"].strip().splitlines()[-1])
        except json.JSONDecodeError:
            api = {"parse_error": True}
    header_text = ""
    header_paths: list[str] = []
    if headers.is_dir():
        for path in sorted(headers.rglob("*.h")):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if "IDynamicQuantizeLayer" in text or "addDynamicQuantize" in text:
                header_paths.append(str(path))
                header_text += text + "\n"
    dynamic = dynamic_quantize_api_verdict(
        python_api_present=bool(api.get("dynamic_quantize_class", False)),
        cpp_api_present=bool(header_paths),
        allowed_output_types=("FP4",) if "OutputType" in header_text else (),
        allowed_scale_types=("FP8",) if "ScaleType" in header_text else (),
        block_sizes=(16,) if "blockSize" in header_text or "block_size" in header_text else (),
    )
    return {
        "trt_root": str(root),
        "python_probe": python_probe,
        "python_api": api,
        "header_paths": header_paths,
        "dynamic_quantize_verdict": dynamic,
        "installed_version": str(api.get("version", "unknown")),
        "accumulator_api": {
            "separate_matrix_multiply_accumulator": False,
            "evidence": "IMatrixMultiplyLayer exposes precision but no accumulator property",
        },
    }


def audit_modelopt_029(*, source_root: str | Path | None = None) -> dict[str, Any]:
    try:
        import modelopt  # type: ignore

        imported = {
            "import_success": True,
            "version": str(getattr(modelopt, "__version__", "unknown")),
            "path": str(getattr(modelopt, "__file__", "")),
        }
    except Exception as exc:  # noqa: BLE001
        imported = {
            "import_success": False,
            "failure": f"{type(exc).__name__}: {exc}",
        }
    root = Path(source_root).expanduser().resolve() if source_root else None
    source_matches: list[str] = []
    if root and root.is_dir():
        for path in root.rglob("*.py"):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if re.search(r"smooth.?quant|SmoothQuant|pre_quant_scale|activation.?smoothing", text, re.I):
                source_matches.append(str(path))
    source_recipe_available = bool(source_matches)
    installed_callable_recipe_available = bool(
        imported.get("import_success") and source_recipe_available
    )
    return {
        "requested_version": "0.29.0",
        "imported_package": imported,
        "source_root": str(root) if root else "",
        "smoothquant_source_matches": source_matches,
        "source_recipe_available": source_recipe_available,
        "installed_callable_recipe_available": installed_callable_recipe_available,
        # Compatibility field: this names the recipe that can actually be called
        # in the active environment, not merely source files found elsewhere.
        "smoothquant_recipe_available": installed_callable_recipe_available,
        "note": (
            "An editable distribution pointing at a missing source tree is not treated as "
            "an installed callable ModelOpt recipe."
        ),
    }


def _attention_rows(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    return [
        (name, module)
        for name, module in model.named_modules()
        if name.startswith("fusion_net")
        and module.__class__.__name__ == "Attention"
        and hasattr(module, "to_qkv")
    ]


def capture_real_qk(
    *,
    checkpoint: str | Path,
    config: str | Path,
    heal_root: str | Path,
    manifest: str | Path,
    output_dir: str | Path,
    device: str,
    max_frames: int = 10,
    max_groups: int = 8,
) -> dict[str, Any]:
    """Capture real model Q/K and FP16 projection Q/K; no random replacement."""

    from search.integration.data_provider import build_dataset_and_loader, move_batch_to_device
    from search.model_families.lidar_cobevt.model_capability import CobevtModelCapability

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
    selected = set(str(value) for value in manifest_payload.get("evaluation_frame_ids", []))
    if not selected:
        raise RuntimeError("capture_manifest_has_no_evaluation_frames")
    bundle = CobevtModelCapability(checkpoint, config, heal_root).load(device=device)
    model = bundle.model
    module_rows = _attention_rows(model)
    if not module_rows:
        raise RuntimeError("cobevt_attention_modules_missing_for_capture")
    captures: dict[tuple[str, str], dict[str, Any]] = {}
    contexts: dict[str, dict[str, Any]] = {}
    current_frame_id = {"value": ""}

    def make_attention_pre(name: str):
        def pre(_module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]):
            context = contexts.setdefault(name, {})
            context.update(
                {
                    "frame_id": str(current_frame_id["value"]),
                    "mask": kwargs.get("mask") if torch.is_tensor(kwargs.get("mask")) else (
                    args[1] if len(args) > 1 and torch.is_tensor(args[1]) else None
                ),
                }
            )
        return pre

    def make_qkv_pre(name: str):
        def pre(_module: torch.nn.Module, args: tuple[Any, ...], _kwargs: dict[str, Any]):
            if args and torch.is_tensor(args[0]):
                contexts.setdefault(name, {})["x"] = args[0].detach()
        return pre

    def make_post(name: str, module: torch.nn.Module):
        def post(_module: torch.nn.Module, _args: tuple[Any, ...], output_value: Any):
            context = contexts.get(name)
            if context is None or not torch.is_tensor(output_value):
                return
            qkv = output_value.detach()
            if qkv.ndim < 2 or qkv.shape[-1] % 3:
                return
            width = qkv.shape[-1] // 3
            q, k, _v = qkv.split(width, dim=-1)
            x = context["x"]
            weight = module.to_qkv.weight.detach()
            bias = module.to_qkv.bias.detach() if module.to_qkv.bias is not None else None
            with torch.no_grad():
                qkv_half = torch.nn.functional.linear(
                    x.to(dtype=torch.float16),
                    weight.to(dtype=torch.float16),
                    bias.to(dtype=torch.float16) if bias is not None else None,
                )
            qh, kh, _vh = qkv_half.split(width, dim=-1)
            groups = min(int(max_groups), int(q.shape[0]))
            heads = int(module.heads)
            if width % heads:
                raise RuntimeError(f"captured_qk_width_not_head_divisible:{name}:{width}")
            dim = width // heads

            def per_head(value: torch.Tensor) -> torch.Tensor:
                return value[:groups].reshape(groups, value.shape[1], heads, dim).permute(0, 2, 1, 3)

            frame_id = str(context.get("frame_id", ""))
            record = {
                "attention_type": name,
                "frame_id": frame_id,
                "q_fp32": per_head(q).cpu().float(),
                "k_fp32": per_head(k).cpu().float(),
                "q_fp16_projection": per_head(qh).cpu().half(),
                "k_fp16_projection": per_head(kh).cpu().half(),
                "scale": float(getattr(module.fn if hasattr(module, "fn") else module, "scale", width ** -0.5)),
                "mask_available": torch.is_tensor(context.get("mask")),
            }
            captures[(frame_id, name)] = record
        return post

    handles = []
    for name, module in module_rows:
        handles.append(module.register_forward_pre_hook(make_attention_pre(name), with_kwargs=True))
        handles.append(module.to_qkv.register_forward_pre_hook(make_qkv_pre(name), with_kwargs=True))
        handles.append(module.to_qkv.register_forward_hook(make_post(name, module)))
    dataset, loader = build_dataset_and_loader(
        bundle.adapter, config, split="val", num_workers=0, visualize=True
    )
    resolved_hypes = bundle.adapter._absolutize_dataset_paths(dict(bundle.hypes))
    split_path = Path(str(resolved_hypes["validate_dir"]))
    split_ids = [str(value) for value in json.loads(split_path.read_text(encoding="utf-8"))]
    evaluated = 0
    try:
        with torch.no_grad():
            for index, batch in enumerate(loader):
                if index >= len(split_ids):
                    break
                frame_id = split_ids[index]
                if frame_id not in selected:
                    continue
                current_frame_id["value"] = frame_id
                bundle.adapter.forward_for_task(model, move_batch_to_device(batch, torch.device(device)))
                evaluated += 1
                if evaluated >= int(max_frames):
                    break
    finally:
        for handle in handles:
            handle.remove()
    paths = []
    for key, record in sorted(captures.items()):
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{key[0]}__{key[1]}")
        path = output / f"{safe}.pt"
        torch.save(record, path)
        paths.append(str(path))
    return {
        "capture_manifest": str(Path(manifest).resolve()),
        "capture_manifest_hash": str(manifest_payload.get("manifest_hash", "")),
        "checkpoint_sha256": _sha256(checkpoint),
        "config_sha256": _sha256(config),
        "device": str(device),
        "attention_module_count": len(module_rows),
        "captured_attention_types": sorted({name for _, name in captures}),
        "unavailable_requested_attention_types": [
            "HGTCavAttention",
            "8x8_window",
            "16x16_window",
        ],
        "frames_evaluated": evaluated,
        "capture_count": len(paths),
        "capture_paths": paths,
        "mask_available_count": sum(
            bool(torch.load(path, map_location="cpu").get("mask_available", False))
            for path in paths
        ),
    }


def build_audit_manifest(
    *,
    output_dir: str | Path,
    checkpoint: str | Path,
    config: str | Path,
    trt_root: str | Path,
    gpu: int,
    source_output: str | Path,
) -> dict[str, Any]:
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "config": str(Path(config).resolve()),
        "config_sha256": _sha256(config),
        "trt_root": str(Path(trt_root).resolve()),
        "physical_gpu": int(gpu),
        "source_output": str(Path(source_output).resolve()),
        "fixed_k": 29696,
        "profiles": accumulation_profile_manifest(),
    }
    _write_json(destination / "run_manifest.json", payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("manifest", "tp", "api", "capture", "numerical"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--heal-root", default=str(DEFAULT_HEAL_ROOT))
    parser.add_argument("--trt-root", default=str(DEFAULT_TRT_ROOT))
    parser.add_argument("--source-output", default=str(DEFAULT_SOURCE_OUTPUT))
    parser.add_argument("--modelopt-source-root")
    parser.add_argument("--manifest")
    parser.add_argument("--capture-dir")
    parser.add_argument("--gpu", type=int, default=5)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    destination = Path(args.output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if args.phase == "manifest":
        result = build_audit_manifest(
            output_dir=destination,
            checkpoint=args.checkpoint,
            config=args.config,
            trt_root=args.trt_root,
            gpu=args.gpu,
            source_output=args.source_output,
        )
    elif args.phase == "tp":
        result = write_tp_attention_contract(destination / "tp_attention_pruning_contract.md")
        _write_json(destination / "tp_attention_pruning_contract.json", result)
    elif args.phase == "api":
        result = {
            "tensorrt": audit_tensorrt_10_9(trt_root=args.trt_root),
            "modelopt": audit_modelopt_029(source_root=args.modelopt_source_root),
        }
        _write_json(destination / "trt_10_9_dynamic_quant_api_audit.json", result["tensorrt"])
        _write_json(destination / "modelopt_0_29_smoothquant_audit.json", result["modelopt"])
        (destination / "trt_10_9_dynamic_quant_api_audit.md").write_text(
            "# TensorRT 10.9 dynamic quantization audit\n\n"
            + json.dumps(result["tensorrt"], indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        (destination / "modelopt_0_29_smoothquant_audit.md").write_text(
            "# ModelOpt 0.29 SmoothQuant audit\n\n"
            + json.dumps(result["modelopt"], indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    elif args.phase == "capture":
        if not args.manifest or not args.capture_dir:
            raise ValueError("capture_requires_manifest_and_capture_dir")
        result = capture_real_qk(
            checkpoint=args.checkpoint,
            config=args.config,
            heal_root=args.heal_root,
            manifest=args.manifest,
            output_dir=args.capture_dir,
            device=f"cuda:{int(args.gpu)}",
        )
        _write_json(destination / "capture_manifest.json", result)
    else:
        if not args.capture_dir:
            raise ValueError("numerical_requires_capture_dir")
        paths = sorted(Path(args.capture_dir).glob("*.pt"))
        rows = run_fp16_numerical_matrix(paths, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        _write_json(destination / "fp16_accumulation_matrix.json", rows)
        import csv

        if rows:
            with (destination / "fp16_accumulation_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        result = {"rows": len(rows), "capture_paths": [str(path) for path in paths]}
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
