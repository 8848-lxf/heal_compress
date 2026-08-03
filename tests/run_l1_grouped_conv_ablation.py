#!/usr/bin/env python3
"""One-command L1 grouped-conv pruning ablation runner."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent
_UNIAD = _ROOT.parent
if str(_UNIAD) not in sys.path:
    sys.path.insert(0, str(_UNIAD))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from heal_compress.utils.gpu_select import GPUInfo, GPUSelectionError, resolve_gpu
from heal_compress.utils.io_utils import ensure_dir, save_json, save_text


DEFAULT_CHECKPOINT = "${MODEL_ROOT}/lidar_pyramid/net_epoch_bestval_at17.pth"
DEFAULT_EXCLUDE_GPU_IDS = [5, 6, 7]
DEFAULT_RATIOS = [0.25, 0.50, 0.75]
DEFAULT_MODES = ["shared_local_mean", "independent_group_topk"]


@dataclass(frozen=True)
class ExperimentConfig:
    model_name: str
    ratio: float
    ratio_tag: str
    mode: str


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def ratio_tag(ratio: float) -> str:
    return f"p{int(round(float(ratio) * 100.0))}"


def build_experiment_configs(ratios: Sequence[float], modes: Sequence[str]) -> list[ExperimentConfig]:
    configs: list[ExperimentConfig] = []
    for mode in modes:
        for ratio in ratios:
            tag = ratio_tag(float(ratio))
            configs.append(
                ExperimentConfig(
                    model_name=f"{mode}_{tag}",
                    ratio=float(ratio),
                    ratio_tag=tag,
                    mode=str(mode),
                )
            )
    return configs


def create_experiment_root(
    *,
    output_root: str | Path,
    experiment_name: str,
    timestamp: str | None = None,
    overwrite: bool = False,
) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    base = root / f"{experiment_name}_{stamp}"
    if overwrite:
        if base.exists():
            shutil.rmtree(base)
        base.mkdir(parents=True, exist_ok=False)
        return base
    if not base.exists():
        base.mkdir(parents=True, exist_ok=False)
        return base
    for idx in range(1, 10000):
        candidate = root / f"{experiment_name}_{stamp}_run{idx:03d}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
    raise RuntimeError(f"could not create unique experiment root for {base}")


def build_prune_command(
    cfg: ExperimentConfig,
    *,
    checkpoint: str,
    device: str,
    output_dir: str | Path,
) -> list[str]:
    return [
        sys.executable,
        "tests/test_general_pruner.py",
        "--checkpoint",
        checkpoint,
        "--prune-ratio",
        f"{cfg.ratio:.2f}",
        "--importance-mode",
        "l1_norm",
        "--selection-mode",
        "constrained_global",
        "--group-conv-selection-mode",
        cfg.mode,
        "--group-conv-align",
        "8",
        "--group-conv-prune-mode",
        "keep_groups",
        "--align",
        "16",
        "--protect-residual-add",
        "true",
        "--allow-remove-groups",
        "false",
        "--device",
        device,
        "--output-dir",
        str(output_dir),
    ]


def build_eval_command(args: argparse.Namespace, *, experiment_root: Path, device: str, selected_gpu_id: str | int) -> list[str]:
    return [
        sys.executable,
        "tests/evaluate_l1_grouped_conv_ablation.py",
        "--checkpoint",
        args.checkpoint,
        "--experiment-root",
        str(experiment_root),
        "--gpu-id",
        str(selected_gpu_id),
        "--device",
        device,
        "--max-frames",
        str(args.max_frames),
        "--warmup-frames",
        str(args.warmup_frames),
        "--rounds",
        str(args.rounds),
        "--run-eval",
        str(args.run_eval).lower(),
        "--run-latency",
        str(args.run_latency).lower(),
        "--output-dir",
        str(experiment_root / "eval"),
    ]


def _quote_command(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in cmd)


def write_commands(path: str | Path, commands: Sequence[Sequence[str]]) -> None:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", "", f"cd {_ROOT}", ""]
    for cmd in commands:
        lines.append(_quote_command(cmd))
    save_text("\n".join(lines) + "\n", path)


def _run_command(cmd: list[str], *, fail_fast: bool) -> int:
    result = subprocess.run(cmd, cwd=_ROOT, text=True)
    if result.returncode != 0 and fail_fast:
        raise RuntimeError(f"command failed rc={result.returncode}: {_quote_command(cmd)}")
    return int(result.returncode)


def _load_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _target_unreachable(target: float, actual: float) -> tuple[bool, str]:
    if actual + 1e-6 >= target:
        return False, ""
    return True, (
        "target prune ratio was not reached under keep_groups grouped-conv constraints, "
        "8-aligned per-group channel floors, residual-add protection, and disabled remove_groups"
    )


def _manifest_from_prune_output(
    *,
    cfg: ExperimentConfig,
    checkpoint: str,
    output_dir: Path,
    gpu_id_arg: str,
    selected_gpu_id: str | int | None,
    device: str,
    returncode: int,
) -> dict[str, Any]:
    pruning = _load_json(output_dir / "pruning_summary.json", {}) or {}
    selection = _load_json(output_dir / "selection_summary.json", {}) or {}
    legality = _load_json(output_dir / "legality_check_report.json", {}) or {}
    actual = float(pruning.get("actual_prune_ratio", selection.get("actual_prune_ratio", 0.0)) or 0.0)
    unreachable, unreachable_reason = _target_unreachable(cfg.ratio, actual)
    status = "success" if returncode == 0 else "prune_failed"
    failure_reason = "" if returncode == 0 else "subprocess_failed"
    structure_legal = bool(pruning.get("structure_legal", legality.get("legal", False)))
    forward_ok = bool(pruning.get("forward_sanity_check", False))
    if returncode == 0 and not structure_legal:
        status = "structure_illegal"
        failure_reason = "structure_illegal"
    elif returncode == 0 and not forward_ok:
        status = "forward_sanity_failed"
        failure_reason = "forward_sanity_failed"
    manifest = {
        "model_name": cfg.model_name,
        "checkpoint_type": "pruned",
        "source_checkpoint": checkpoint,
        "checkpoint_path": str(output_dir / "pruned_model.pth"),
        "pruned_model_path": str(output_dir / "pruned_model.pth"),
        "target_prune_ratio": cfg.ratio,
        "actual_prune_ratio": actual,
        "importance_mode": "l1_norm",
        "selection_mode": "constrained_global",
        "group_conv_selection_mode": cfg.mode,
        "group_conv_prune_mode": "keep_groups",
        "group_conv_align": 8,
        "align": 16,
        "protect_residual_add": True,
        "allow_remove_groups": False,
        "gpu_id": gpu_id_arg,
        "selected_gpu_id": selected_gpu_id,
        "device": device,
        "structure_legal": structure_legal,
        "forward_sanity_check": forward_ok,
        "params": pruning.get("pruned_params", ""),
        "pruned_params": pruning.get("pruned_params", ""),
        "params_removed": (
            int(pruning.get("original_params", 0) or 0) - int(pruning.get("pruned_params", 0) or 0)
            if pruning
            else ""
        ),
        "params_removed_ratio": actual,
        "coupled_channels_before": pruning.get("total_coupled_channels_before", ""),
        "coupled_channels_after": pruning.get("total_coupled_channels_after", ""),
        "coupled_channels_removed": pruning.get("pruned_coupled_channels", ""),
        "num_dependency_scopes": selection.get("num_dependency_scopes", pruning.get("num_total_groups", "")),
        "num_coupled_channel_units": selection.get("num_coupled_channel_units", ""),
        "num_atomic_prune_units": selection.get("num_atomic_prune_units", ""),
        "num_concrete_pruning_groups": selection.get("num_concrete_pruning_groups", ""),
        "num_grouped_conv_align_violations": selection.get("num_grouped_conv_align_violations", ""),
        "target_unreachable": unreachable,
        "unreachable_reason": unreachable_reason,
        "status": status,
        "failure_reason": failure_reason,
        "returncode": returncode,
    }
    return manifest


def write_baseline_manifest(
    *,
    experiment_root: Path,
    checkpoint: str,
    gpu_id_arg: str,
    selected_gpu_id: str | int | None,
    device: str,
) -> dict[str, Any]:
    out = experiment_root / "baseline_original"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model_name": "baseline_original",
        "checkpoint_type": "original",
        "source_checkpoint": checkpoint,
        "checkpoint_path": checkpoint,
        "target_prune_ratio": 0.0,
        "actual_prune_ratio": 0.0,
        "importance_mode": "not_available",
        "selection_mode": "not_available",
        "group_conv_selection_mode": "not_available",
        "group_conv_prune_mode": "not_available",
        "group_conv_align": 8,
        "align": 16,
        "protect_residual_add": True,
        "allow_remove_groups": False,
        "gpu_id": gpu_id_arg,
        "selected_gpu_id": selected_gpu_id,
        "device": device,
        "structure_legal": True,
        "forward_sanity_check": True,
        "status": "success",
        "failure_reason": "",
    }
    save_json(manifest, out / "model_manifest.json")
    return manifest


def _write_prune_manifest(*args: Any, **kwargs: Any) -> dict[str, Any]:
    manifest = _manifest_from_prune_output(*args, **kwargs)
    output_dir = Path(kwargs["output_dir"])
    save_json(manifest, output_dir / "model_manifest.json")
    return manifest


def _gpu_to_manifest(selected_gpu: GPUInfo | None, queried_gpus: list[GPUInfo], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "gpu_id_arg": args.gpu_id,
        "selected_gpu_id": selected_gpu.index if selected_gpu else None,
        "selected_gpu": selected_gpu.to_dict() if selected_gpu else None,
        "candidate_gpus": [gpu.to_dict() for gpu in queried_gpus],
        "auto_excluded_gpu_ids": list(args.exclude_gpu_ids) if str(args.gpu_id).lower() == "auto" else [],
        "min_free_memory_mb": args.min_free_memory_mb,
        "max_gpu_util": args.max_gpu_util,
    }


def build_run_manifest(
    *,
    args: argparse.Namespace,
    experiment_root: Path,
    device: str,
    selected_gpu: GPUInfo | None,
    queried_gpus: list[GPUInfo],
) -> dict[str, Any]:
    return {
        "experiment_name": args.experiment_name,
        "experiment_root": str(experiment_root),
        "checkpoint": args.checkpoint,
        **_gpu_to_manifest(selected_gpu, queried_gpus, args),
        "ratios": [float(v) for v in args.ratios],
        "modes": list(args.modes),
        "importance_mode": "l1_norm",
        "selection_mode": "constrained_global",
        "group_conv_prune_mode": "keep_groups",
        "group_conv_align": 8,
        "align": 16,
        "protect_residual_add": True,
        "allow_remove_groups": False,
        "run_prune": bool(args.run_prune),
        "run_eval": bool(args.run_eval),
        "run_latency": bool(args.run_latency),
        "max_frames": args.max_frames,
        "warmup_frames": args.warmup_frames,
        "rounds": args.rounds,
        "device": device,
    }


def run_ablation(args: argparse.Namespace) -> dict[str, Any]:
    experiment_root = create_experiment_root(
        output_root=args.output_root,
        experiment_name=args.experiment_name,
        overwrite=args.overwrite,
    )
    device, selected_gpu, queried_gpus = resolve_gpu(
        args.gpu_id,
        exclude_gpu_ids=args.exclude_gpu_ids,
        min_free_memory_mb=args.min_free_memory_mb,
        max_gpu_util=args.max_gpu_util,
    )
    selected_gpu_id: str | int | None = selected_gpu.index if selected_gpu else ("cpu" if device == "cpu" else None)
    commands: list[list[str]] = []
    manifests: list[dict[str, Any]] = []
    manifests.append(
        write_baseline_manifest(
            experiment_root=experiment_root,
            checkpoint=args.checkpoint,
            gpu_id_arg=args.gpu_id,
            selected_gpu_id=selected_gpu_id,
            device=device,
        )
    )

    configs = build_experiment_configs(args.ratios, args.modes)
    for cfg in configs:
        out = experiment_root / cfg.model_name
        cmd = build_prune_command(cfg, checkpoint=args.checkpoint, device=device, output_dir=out)
        commands.append(cmd)
        rc = 0
        if args.run_prune:
            rc = _run_command(
                cmd,
                fail_fast=args.fail_fast,
            )
        else:
            out.mkdir(parents=True, exist_ok=True)
        manifest = _write_prune_manifest(
            cfg=cfg,
            checkpoint=args.checkpoint,
            output_dir=out,
            gpu_id_arg=args.gpu_id,
            selected_gpu_id=selected_gpu_id,
            device=device,
            returncode=rc,
        )
        manifests.append(manifest)

    eval_cmd: list[str] | None = None
    if args.run_eval or args.run_latency:
        eval_cmd = build_eval_command(args, experiment_root=experiment_root, device=device, selected_gpu_id=selected_gpu_id or args.gpu_id)
        commands.append(eval_cmd)
        _run_command(
            eval_cmd,
            fail_fast=args.fail_fast,
        )

    write_commands(experiment_root / "commands.sh", commands)
    manifest = build_run_manifest(
        args=args,
        experiment_root=experiment_root,
        device=device,
        selected_gpu=selected_gpu,
        queried_gpus=queried_gpus,
    )
    manifest["models"] = manifests
    manifest["commands"] = [_quote_command(cmd) for cmd in commands]
    save_json(manifest, experiment_root / "run_manifest.json")
    return {"experiment_root": str(experiment_root), "manifest": manifest, "commands": commands}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run L1 grouped-conv pruning ablation")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--gpu-id", default="auto")
    parser.add_argument("--exclude-gpu-ids", nargs="*", type=int, default=list(DEFAULT_EXCLUDE_GPU_IDS))
    parser.add_argument("--min-free-memory-mb", type=int, default=8000)
    parser.add_argument("--max-gpu-util", type=int, default=30)
    parser.add_argument("--output-root", default=str(_THIS_DIR / "outputs"))
    parser.add_argument("--experiment-name", default="l1_grouped_conv_ablation")
    parser.add_argument("--ratios", nargs="+", type=float, default=list(DEFAULT_RATIOS))
    parser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES), choices=DEFAULT_MODES)
    parser.add_argument("--run-prune", type=str2bool, default=True)
    parser.add_argument("--run-eval", type=str2bool, default=True)
    parser.add_argument("--run-latency", type=str2bool, default=True)
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--warmup-frames", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--overwrite", type=str2bool, default=False)
    parser.add_argument("--fail-fast", type=str2bool, default=False)
    return parser.parse_args(argv)


if __name__ == "__main__":
    try:
        result = run_ablation(parse_args())
        print(json.dumps({"experiment_root": result["experiment_root"]}, ensure_ascii=False, indent=2))
    except GPUSelectionError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False, indent=2), file=sys.stderr)
        raise SystemExit(2) from exc
