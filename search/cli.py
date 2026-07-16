"""CLI entrypoint for two-stage joint search."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from .orchestration.lidar_pyramid_search import LidarPyramidTwoStageSearch
from .orchestration.two_stage_search import TwoStageSearchRunner


def _load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"config does not exist: {config_path}")
    if config_path.suffix.lower() == ".json":
        return dict(json.loads(config_path.read_text(encoding="utf-8")))
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load YAML configs") from exc
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return dict(data)


def _dump_yaml(path: Path, payload: dict[str, Any]) -> None:
    try:
        import yaml

        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    except ImportError:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _gpu_report() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=10,
        )
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": f"{type(exc).__name__}: {exc}", "gpus": []}
    gpus = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 6:
            gpus.append(
                {
                    "index": parts[0],
                    "name": parts[1],
                    "memory_used_mib": parts[2],
                    "memory_total_mib": parts[3],
                    "utilization_gpu_pct": parts[4],
                    "compute_capability": parts[5],
                }
            )
    return {"available": completed.returncode == 0, "gpus": gpus, "stderr": completed.stderr.strip()}


def _select_gpu(requested: str, config: dict[str, Any], report: dict[str, Any]) -> str:
    if requested and requested != "auto":
        return requested
    excluded = {str(value) for value in config.get("runtime", {}).get("exclude_gpu_ids", [])}
    candidates = [gpu for gpu in report.get("gpus", []) if str(gpu.get("index")) not in excluded]
    if not candidates:
        return "cpu"
    candidates.sort(key=lambda row: (int(row.get("memory_used_mib") or 10**9), int(row.get("utilization_gpu_pct") or 10**9)))
    return f"cuda:{candidates[0]['index']}"


def _default_space(config: dict[str, Any]) -> tuple[list[str], list[str], set[str]]:
    space = config.get("search_space", {})
    pruning = [str(value) for value in space.get("pruning_unit_ids", [])]
    precision = [str(value) for value in space.get("precision_layer_ids", [])]
    protected = {str(value) for value in space.get("protected_pruning_unit_ids", [])}
    candidate = config.get("candidate_config") or {}
    if isinstance(candidate, dict):
        pruning.extend(str(value) for value in (candidate.get("pruning_genes") or {}).keys())
        precision.extend(str(value) for value in (candidate.get("precision_genes") or {}).keys())
    pruning = sorted(set(pruning))
    precision = sorted(set(precision))
    if not pruning:
        pruning = [f"dryrun_scope.stage{i // 8}.unit{i:03d}" for i in range(32)]
        protected = {pruning[0]}
    if not precision:
        precision = [f"dryrun_model.stage{i // 4}.layer{i:02d}" for i in range(16)]
    return pruning, precision, protected


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-stage structured pruning + mixed precision GA search.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-root", default="tests/outputs")
    parser.add_argument("--gpu-id", default=None)
    parser.add_argument("--search-method", choices=("ga", "greedy"), default=None)
    parser.add_argument("--exclude-gpu-ids", default=None)
    parser.add_argument("--resume", nargs="?", const="auto", default=None)
    parser.add_argument("--outer-rounds", type=int, default=None)
    parser.add_argument("--initial-population-size", type=int, default=None)
    parser.add_argument("--population-size", type=int, default=None)
    parser.add_argument("--offspring-size", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--topk-real", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--warmup-frames", type=int, default=None)
    parser.add_argument("--latency-rounds", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stage1-only", action="store_true")
    parser.add_argument("--stage2-only", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--candidate-config", action="append", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    effective_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parse_args(effective_argv)
    config = _load_config(args.config)
    search_cfg = dict(config.get("search", {}))
    eval_cfg = dict(config.get("evaluation", {}))
    stage2_cfg = dict(config.get("stage2", {}))
    runtime_cfg = dict(config.get("runtime", {}))
    model_cfg = dict(config.get("model", {}))
    if args.outer_rounds is not None:
        search_cfg["outer_rounds"] = args.outer_rounds
    if args.initial_population_size is not None:
        search_cfg["initial_population_size"] = args.initial_population_size
    if args.population_size is not None:
        search_cfg["population_size"] = args.population_size
    if args.offspring_size is not None:
        search_cfg["offspring_size"] = args.offspring_size
    if args.generations is not None:
        search_cfg["generations_per_round"] = args.generations
    if args.search_method is not None:
        search_cfg["method"] = args.search_method
    if args.topk_real is not None:
        search_cfg["topk_real"] = args.topk_real
    if args.num_frames is not None:
        eval_cfg["num_frames"] = args.num_frames
        stage2_cfg["num_frames"] = args.num_frames
    if args.warmup_frames is not None:
        eval_cfg["warmup_frames"] = args.warmup_frames
        stage2_cfg["warmup_frames"] = args.warmup_frames
    if args.latency_rounds is not None:
        eval_cfg["rounds"] = args.latency_rounds
        stage2_cfg["latency_rounds"] = args.latency_rounds
    if args.candidate_config:
        config["candidate_config"] = [_load_config(path) for path in args.candidate_config]
    if args.skip_baselines:
        baseline_cfg = dict(config.get("baselines", {}))
        baseline_cfg["build_before_search"] = False
        config["baselines"] = baseline_cfg
    if args.gpu_id is not None:
        runtime_cfg["gpu_id"] = str(args.gpu_id)
    if args.exclude_gpu_ids:
        runtime_cfg["exclude_gpu_ids"] = [int(value) for value in str(args.exclude_gpu_ids).split(",") if value.strip()]
    if args.checkpoint:
        model_cfg["checkpoint"] = args.checkpoint
    config["search"] = search_cfg
    config["evaluation"] = eval_cfg
    if stage2_cfg:
        config["stage2"] = stage2_cfg
    config["runtime"] = runtime_cfg
    if model_cfg:
        config["model"] = model_cfg

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint or model_cfg.get("checkpoint")
    if not checkpoint:
        raise RuntimeError("checkpoint_required")

    if not args.dry_run:
        resume_path = None if args.resume in (None, "auto") else args.resume
        runner = LidarPyramidTwoStageSearch(
            config=config,
            checkpoint=checkpoint,
            output_root=output_root,
            resume=resume_path,
        )
        result = runner.run(
            stage1_only=bool(args.stage1_only),
            stage2_only=bool(args.stage2_only),
            baseline_only=bool(args.baseline_only),
            candidate_config=args.candidate_config,
        )
        run_dir = Path(result["run_dir"])
        _dump_yaml(run_dir / "resolved_config.yaml", config)
        command_path = run_dir / "commands.sh"
        with command_path.open("a" if command_path.exists() else "w", encoding="utf-8") as handle:
            handle.write(
                "python -m search.cli "
                + " ".join(shlex.quote(value) for value in effective_argv)
                + "\n"
            )
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0

    gpu_report = _gpu_report()
    selected_gpu = _select_gpu(
        str(runtime_cfg.get("gpu_id", "auto")),
        config,
        gpu_report,
    )
    pruning_ids, precision_ids, protected_ids = _default_space(config)
    _dump_yaml(output_root / "resolved_config.yaml", config)
    (output_root / "environment.json").write_text(
        json.dumps(
            {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "cwd": os.getcwd(),
                "checkpoint": checkpoint,
                "selected_gpu": selected_gpu,
                "gpu_report": gpu_report,
                "conda_prefix": os.environ.get("CONDA_PREFIX", ""),
                "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    runner = TwoStageSearchRunner(
        output_root=output_root,
        pruning_unit_ids=pruning_ids,
        precision_layer_ids=precision_ids,
        protected_pruning_unit_ids=protected_ids,
        random_seed=int(search_cfg.get("seed", 42)),
        tensorrt_version="10.9",
        gpu_compute_capability=str(selected_gpu),
        builder_flags={"precision_constraints": "obey", "fp16": True, "int8": True},
    )
    result = runner.run(
        outer_rounds=int(search_cfg.get("outer_rounds", 1)),
        population_size=int(search_cfg.get("population_size", 8)),
        generations=int(search_cfg.get("generations_per_round", 2)),
        topk_real=int(search_cfg.get("topk_real", 1)),
        dry_run=bool(args.dry_run),
        stage1_only=bool(args.stage1_only),
    )
    run_dir = Path(result["run_dir"])
    if (output_root / "resolved_config.yaml").exists():
        (run_dir / "resolved_config.yaml").write_text((output_root / "resolved_config.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    if (output_root / "environment.json").exists():
        (run_dir / "environment.json").write_text((output_root / "environment.json").read_text(encoding="utf-8"), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
