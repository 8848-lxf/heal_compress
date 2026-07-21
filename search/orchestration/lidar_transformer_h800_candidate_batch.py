"""Crash-isolated sequential builder for H800 Transformer quant profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from search.model_families.transformer.fp8_profiles import FP8_PROFILE_ROLES
from search.model_families.transformer.smoothquant_profiles import smoothquant_profiles


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def run_batch(
    *, output_root: Path, model: str, family: str, profiles: tuple[str, ...],
    alpha: float, physical_gpu: int, plugin: Path,
) -> list[dict[str, Any]]:
    known = (
        {row.profile_id for row in smoothquant_profiles()}
        if family == "smoothquant"
        else set(FP8_PROFILE_ROLES)
    )
    unknown = sorted(set(profiles) - known)
    if unknown:
        raise ValueError(f"unknown_{family}_profiles:{unknown}")
    section = "smoothquant_sm90" if family == "smoothquant" else "fp8"
    module = (
        "search.orchestration.lidar_transformer_h800_smoothquant"
        if family == "smoothquant"
        else "search.orchestration.lidar_transformer_h800_fp8"
    )
    rows = []
    for profile in profiles:
        command = [
            sys.executable, "-m", module,
            "--output-root", str(output_root),
            "--model", model,
            "--profile", profile,
            "--physical-gpu", str(physical_gpu),
            "--plugin", str(plugin),
        ]
        if family == "smoothquant":
            command.extend(
                ["--alpha", str(alpha), "--destination-section", section]
            )
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        profile_dir = output_root / section / model / profile
        failure_dir = output_root / "failures" / family / model / profile
        failure_dir.mkdir(parents=True, exist_ok=True)
        (failure_dir / "subprocess.log").write_text(completed.stdout, encoding="utf-8")
        acceptance = profile_dir / "baseline_result.json"
        if completed.returncode == 0 and acceptance.is_file():
            result = json.loads(acceptance.read_text(encoding="utf-8"))
            row = {
                "profile": profile,
                "status": result.get("status", "unknown"),
                "returncode": completed.returncode,
                "accepted_artifact": str(acceptance),
            }
        else:
            reason = f"builder_subprocess_returncode:{completed.returncode}"
            failure = {
                "status": "engine_build_failed",
                "model": model,
                "profile": profile,
                "family": family,
                "physical_gpu": physical_gpu,
                "failure_reason": reason,
                "subprocess_log": str(failure_dir / "subprocess.log"),
                "crash_isolated": True,
            }
            _write_json(failure_dir / "failure.json", failure)
            if not acceptance.is_file():
                _write_json(acceptance, failure)
            row = {"profile": profile, "status": "engine_build_failed", "returncode": completed.returncode}
        rows.append(row)
    _write_json(output_root / "failures" / family / model / "batch_summary.json", rows)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model", choices=("lidar_cobevt", "lidar_v2xvit"), required=True)
    parser.add_argument("--family", choices=("smoothquant", "fp8"), required=True)
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--plugin", required=True)
    args = parser.parse_args(argv)
    rows = run_batch(
        output_root=Path(args.output_root).resolve(), model=args.model,
        family=args.family,
        profiles=tuple(value.strip() for value in args.profiles.split(",") if value.strip()),
        alpha=args.alpha, physical_gpu=args.physical_gpu,
        plugin=Path(args.plugin).resolve(),
    )
    print(json.dumps(rows, sort_keys=True))
    return 0 if all(row["status"] == "ok" for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
