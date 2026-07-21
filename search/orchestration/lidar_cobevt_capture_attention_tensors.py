"""Capture real tensors from all six CoBEVT Attention blocks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

import torch
from einops import rearrange

from search.model_families.lidar_cobevt.attention_numerical_boundary import (
    derive_attention_tensors,
    tensor_content_hash,
    tensor_statistics,
    validate_capture_inventory,
)


DEFAULT_CHECKPOINT = Path(
    "/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/"
    "LiDAROnly/lidar_cobevt/net_epoch_bestval_at19.pth"
)
DEFAULT_CONFIG = DEFAULT_CHECKPOINT.with_name("config.yaml")
DEFAULT_HEAL_ROOT = Path("/home/lixingfeng/UniAD_examine/HEAL")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _attention_modules(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    rows = [
        (name, module)
        for name, module in model.named_modules()
        if re.fullmatch(
            r"fusion_net\.layers\.[0-2]\.(?:window|grid)_attention\.fn", name
        )
        and module.__class__.__name__ == "Attention"
    ]
    if len(rows) != 6:
        raise RuntimeError(f"expected_six_attention_modules:{[name for name, _ in rows]}")
    return rows


def capture_attention_tensors(
    *,
    checkpoint: str | Path,
    config: str | Path,
    heal_root: str | Path,
    manifest: str | Path,
    output_dir: str | Path,
    device: str,
    num_workers: int = 8,
    max_groups_per_block: int = 8,
) -> dict[str, Any]:
    from search.integration.data_provider import build_dataset_and_loader, move_batch_to_device
    from search.model_families.lidar_cobevt.model_capability import CobevtModelCapability

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    config_path = Path(config).expanduser().resolve()
    manifest_path = Path(manifest).expanduser().resolve()
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = tuple(str(value) for value in manifest_payload.get("evaluation_frame_ids", ()))
    if len(selected) != 10:
        raise ValueError(f"capture_requires_smoke10_manifest:{len(selected)}")
    selected_set = set(selected)
    bundle = CobevtModelCapability(
        checkpoint_path, config_path, Path(heal_root).expanduser().resolve()
    ).load(device=device)
    model = bundle.model.eval()
    modules = _attention_modules(model)
    current_frame = {"value": ""}
    contexts: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []
    tensor_hashes: dict[str, dict[str, str]] = {}

    def attention_pre(name: str):
        def hook(
            _module: torch.nn.Module,
            args: tuple[Any, ...],
            kwargs: Mapping[str, Any],
        ) -> None:
            mask = kwargs.get("mask", args[1] if len(args) > 1 else None)
            contexts[name] = {
                "frame_id": current_frame["value"],
                "mask": mask.detach() if torch.is_tensor(mask) else None,
            }

        return hook

    def qkv_post(name: str, attention: torch.nn.Module):
        def hook(
            _module: torch.nn.Module,
            _args: tuple[Any, ...],
            qkv_output: torch.Tensor,
        ) -> None:
            context = contexts.get(name)
            if context is None:
                raise RuntimeError(f"attention_capture_context_missing:{name}")
            groups = min(int(max_groups_per_block), int(qkv_output.shape[0]))
            qkv = qkv_output[:groups].detach().float()
            raw_mask = context.get("mask")
            mask = None
            if torch.is_tensor(raw_mask):
                mask = rearrange(
                    raw_mask,
                    "b x y w1 w2 e l -> (b x y) e (l w1 w2)",
                )[:groups].unsqueeze(1).bool()
            bias = attention.relative_position_bias_table(
                attention.relative_position_index
            ).permute(2, 0, 1)
            derived = derive_attention_tensors(
                qkv=qkv,
                heads=int(attention.heads),
                scale=float(attention.scale),
                relative_bias=bias,
                attention_mask=mask,
            )
            av = derived["av"]
            batch, heads, tokens, dim = av.shape
            out_input = av.permute(0, 2, 1, 3).reshape(batch, tokens, heads * dim)
            out_projection = attention.to_out[0](out_input)
            derived["out_projection_input"] = out_input
            derived["residual_update"] = out_projection
            frame_id = str(context["frame_id"])
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
            capture_path = destination / "tensors" / f"{frame_id}__{safe_name}.pt"
            capture_path.parent.mkdir(parents=True, exist_ok=True)
            cpu_tensors = {key: value.detach().cpu() for key, value in derived.items()}
            torch.save(
                {
                    "frame_id": frame_id,
                    "module_name": name,
                    "heads": int(attention.heads),
                    "head_dim": int(derived["q"].shape[-1]),
                    "scale": float(attention.scale),
                    "groups_captured": groups,
                    "mask_available": mask is not None,
                    "tensors": cpu_tensors,
                },
                capture_path,
            )
            hashes = {key: tensor_content_hash(value) for key, value in cpu_tensors.items()}
            tensor_hashes[str(capture_path)] = hashes
            for tensor_name, value in cpu_tensors.items():
                stats_rows.append(
                    {
                        "frame_id": frame_id,
                        "module_name": name,
                        "tensor_name": tensor_name,
                        **tensor_statistics(value),
                    }
                )
            records.append(
                {
                    "frame_id": frame_id,
                    "module_name": name,
                    "capture_path": str(capture_path),
                    "groups_captured": groups,
                    "mask_available": mask is not None,
                    "heads": int(attention.heads),
                    "head_dim": int(derived["q"].shape[-1]),
                    "token_length": int(derived["q"].shape[-2]),
                }
            )

        return hook

    handles = []
    for name, module in modules:
        handles.append(module.register_forward_pre_hook(attention_pre(name), with_kwargs=True))
        handles.append(module.to_qkv.register_forward_hook(qkv_post(name, module)))
    dataset, loader = build_dataset_and_loader(
        bundle.adapter,
        config_path,
        split="val",
        num_workers=int(num_workers),
        visualize=True,
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
                if frame_id not in selected_set:
                    continue
                current_frame["value"] = frame_id
                bundle.adapter.forward_for_task(
                    model, move_batch_to_device(batch, torch.device(device))
                )
                evaluated += 1
                if evaluated == len(selected):
                    break
    finally:
        for handle in handles:
            handle.remove()
    inventory = validate_capture_inventory(records, expected_frames=len(selected))
    fields = sorted({key for row in stats_rows for key in row})
    with (destination / "per_frame_tensor_statistics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(stats_rows)
    _write_json(destination / "tensor_hashes.json", tensor_hashes)
    manifest_result = {
        **inventory,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256(manifest_path),
        "source_manifest_hash": str(manifest_payload.get("manifest_hash", "")),
        "device": device,
        "num_workers": int(num_workers),
        "max_groups_per_block": int(max_groups_per_block),
        "evaluated_frames": evaluated,
        "skipped_frames": len(selected) - evaluated,
        "records": records,
    }
    _write_json(destination / "capture_manifest.json", manifest_result)
    return manifest_result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--heal-root", default=str(DEFAULT_HEAL_ROOT))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-groups-per-block", type=int, default=8)
    args = parser.parse_args()
    result = capture_attention_tensors(
        checkpoint=args.checkpoint,
        config=args.config,
        heal_root=args.heal_root,
        manifest=args.manifest,
        output_dir=args.output_dir,
        device=args.device,
        num_workers=args.num_workers,
        max_groups_per_block=args.max_groups_per_block,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
