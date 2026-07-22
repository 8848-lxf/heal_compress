"""Persistent same-GPU formal-latency matrix after all fixed500 evaluations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from search.model_families.transformer.dh_candidate_grid import dense_head_dimension_grid
from search.orchestration.lidar_transformer_h800_latency import audit_isolation


PROFILES = ("P32", "P16", "P8")


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _groups(output_root: Path) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for model in ("lidar_cobevt", "lidar_v2xvit"):
        families = _read(
            output_root / "inventory" / f"{model.removeprefix('lidar_')}_attention_families.json"
        )
        if not isinstance(families, list):
            raise RuntimeError(f"formal_latency_inventory_missing:{model}")
        for family in families:
            family_id = str(family["family_id"])
            d0 = int(family["original_d_h"])
            widths = [
                row.d_h
                for row in dense_head_dimension_grid(
                    d0,
                    heads=int(family["heads"]),
                    low_width_extension=d0 <= 16,
                )
            ]
            for profile in PROFILES:
                accepted: list[int] = []
                for d_h in widths:
                    directory = (
                        output_root / "engines" / model / family_id
                        / f"dh_{d_h:03d}" / profile
                    )
                    build = _read(directory / "baseline_result.json") or {}
                    evaluation = _read(
                        directory / "evaluation" / "fixed500" / "evaluation_acceptance.json"
                    ) or {}
                    alignment = _read(directory / "engine_alignment_audit.json") or {}
                    build_exact = (
                        build.get("status") == "ok"
                        and int(build.get("requested_realized_conflict_count", -1)) == 0
                        and alignment.get("padding_status")
                        in {"EXACT_ALIGNED", "EXACT_NONALIGNED", "INTERNAL_PADDED"}
                        and not bool(alignment.get("fallback_hint"))
                    )
                    if build_exact and evaluation.get("status") != "ok":
                        raise RuntimeError(
                            f"formal_latency_fixed500_missing_for_exact_build:"
                            f"{model}:{family_id}:{profile}:d_h={d_h}"
                        )
                    if build_exact and evaluation.get("status") == "ok":
                        accepted.append(d_h)
                if d0 not in accepted:
                    raise RuntimeError(
                        f"formal_latency_same_profile_d0_not_accepted:"
                        f"{model}:{family_id}:{profile}"
                    )
                groups.append(
                    {
                        "model": model,
                        "family": family_id,
                        "profile": profile,
                        "original_d_h": d0,
                        "widths": accepted,
                    }
                )
    return groups


def _complete(output_root: Path, group: dict[str, Any]) -> bool:
    path = (
        output_root / "latency" / group["model"] / group["family"]
        / group["profile"] / "formal_latency.json"
    )
    rows = _read(path)
    if not isinstance(rows, list):
        return False
    observed = {int(row["d_h"]) for row in rows if not row.get("baseline_replay")}
    observed.update(
        int(row["d_h"])
        for row in rows
        if row.get("baseline_replay") and int(row.get("replay_index", 0)) == 0
    )
    return set(int(value) for value in group["widths"]) <= observed


def _wait_for_isolated_gpu(
    *,
    gpu_candidates: tuple[int, ...],
    isolation_seconds: int,
    poll_seconds: int,
    fixed_gpu: int | None,
) -> tuple[int, dict[str, Any]]:
    candidates = (fixed_gpu,) if fixed_gpu is not None else gpu_candidates
    while True:
        for gpu in candidates:
            audit = audit_isolation(int(gpu), int(isolation_seconds))
            if audit["isolated"]:
                return int(gpu), audit
        time.sleep(max(1, int(poll_seconds)))


def run(
    *,
    output_root: Path,
    plugin: Path,
    gpu_candidates: tuple[int, ...],
    isolation_seconds: int = 300,
    poll_seconds: int = 60,
    warmup: int = 200,
    iterations: int = 2000,
    repeats: int = 5,
) -> dict[str, Any]:
    groups = _groups(output_root)
    reports = output_root / "reports"
    choice_path = reports / "formal_latency_gpu_choice.json"
    choice = _read(choice_path) or {}
    fixed_gpu = int(choice["physical_gpu"]) if "physical_gpu" in choice else None
    progress_path = reports / "formal_latency_matrix_progress.json"
    progress = _read(progress_path) or []
    group_index = 0
    while group_index < len(groups):
        group = groups[group_index]
        if _complete(output_root, group):
            group_index += 1
            continue
        gpu, audit = _wait_for_isolated_gpu(
            gpu_candidates=gpu_candidates,
            isolation_seconds=isolation_seconds,
            poll_seconds=poll_seconds,
            fixed_gpu=fixed_gpu,
        )
        if fixed_gpu is None:
            fixed_gpu = gpu
            choice = {
                "physical_gpu": gpu,
                "selection_rule": "first candidate with continuous external-process-free isolation",
                "required_isolation_seconds": isolation_seconds,
                "initial_isolation_audit": audit,
                "same_gpu_for_all_groups": True,
            }
            _write(choice_path, choice)
        widths = ",".join(str(value) for value in group["widths"])
        command = [
            sys.executable,
            "-m",
            "search.orchestration.lidar_transformer_dh_latency",
            "--output-root",
            str(output_root),
            "--model",
            str(group["model"]),
            "--family",
            str(group["family"]),
            "--profile",
            str(group["profile"]),
            "--widths",
            widths,
            "--original-dh",
            str(group["original_d_h"]),
            "--physical-gpu",
            str(fixed_gpu),
            "--plugin",
            str(plugin),
            "--isolation-seconds",
            str(isolation_seconds),
            "--warmup",
            str(warmup),
            "--iterations",
            str(iterations),
            "--repeats",
            str(repeats),
        ]
        log = reports / (
            f"formal_latency_{group['model']}_{group['family']}_{group['profile']}.log"
        )
        with log.open("a", encoding="utf-8") as handle:
            completed = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[2],
                env=dict(os.environ),
                text=True,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        attempt = {
            **group,
            "physical_gpu": fixed_gpu,
            "returncode": completed.returncode,
            "complete": _complete(output_root, group),
            "command": command,
        }
        progress.append(attempt)
        _write(progress_path, progress)
        if completed.returncode != 0 or not attempt["complete"]:
            # An external process may have appeared after the isolation gate.
            # Keep the same GPU for comparability and retry only after it again
            # satisfies the complete isolation window.
            time.sleep(max(1, int(poll_seconds)))
            continue
        group_index += 1
    result = {
        "status": "ok",
        "physical_gpu": fixed_gpu,
        "groups": len(groups),
        "all_groups_complete": all(_complete(output_root, group) for group in groups),
    }
    _write(reports / "formal_latency_matrix_result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--physical-gpus", default="0,2,4,7,1,3,5,6")
    parser.add_argument("--isolation-seconds", type=int, default=300)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)
    result = run(
        output_root=Path(args.output_root).resolve(),
        plugin=Path(args.plugin).resolve(),
        gpu_candidates=tuple(int(value) for value in args.physical_gpus.split(",") if value),
        isolation_seconds=args.isolation_seconds,
        poll_seconds=args.poll_seconds,
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["all_groups_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
