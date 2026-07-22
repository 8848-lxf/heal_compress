"""Resumable dense build/evaluation matrix for all families, widths and profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback
from typing import Any

from search.model_families.transformer.dh_candidate_grid import dense_head_dimension_grid
from search.orchestration.lidar_transformer_dh_build import (
    build_float_profile,
    build_int8_profile,
    prepare_structure,
)
from search.orchestration.lidar_transformer_dh_evaluate import evaluate


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _families(root: Path, model: str) -> list[dict[str, Any]]:
    path = root / "inventory" / f"{model.removeprefix('lidar_')}_attention_families.json"
    return json.loads(path.read_text(encoding="utf-8"))


def run(
    *, output_root: Path, model: str, physical_gpu: int, plugin: Path,
    family_filter: str = "", width_filter: tuple[int, ...] = (),
    low_width_extension: bool = False,
    profiles: tuple[str, ...] = ("P32", "P16", "P8"),
    protocols: tuple[str, ...] = ("smoke10", "fixed50", "fixed500"),
) -> list[dict[str, Any]]:
    # Family-scoped workers may run concurrently on separate GPUs.  Keep their
    # resumable journals independent so one process cannot overwrite another
    # process' completed rows.
    progress_scope = family_filter or "all_families"
    if width_filter:
        progress_scope += "__widths_" + "_".join(f"{value:03d}" for value in sorted(set(width_filter), reverse=True))
    checkpoint = output_root / "reports" / f"{model}_{progress_scope}_matrix_progress.json"
    rows = json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.is_file() else []
    completed = {
        (str(row.get("family")), int(row.get("d_h", -1)), str(row.get("profile")), str(row.get("phase")))
        for row in rows if row.get("status") == "ok"
    }
    for family in _families(output_root, model):
        family_id = str(family["family_id"])
        if family_filter and family_id != family_filter:
            continue
        widths = [
            row.d_h
            for row in dense_head_dimension_grid(
                int(family["original_d_h"]),
                heads=int(family["heads"]),
                low_width_extension=low_width_extension,
            )
        ]
        if width_filter:
            widths = [value for value in widths if value in set(width_filter)]
        for d_h in widths:
            structure_key = (family_id, d_h, "structure", "structure")
            structure_result_path = output_root / "structures" / model / family_id / f"dh_{d_h:03d}" / "structure_result.json"
            if structure_result_path.is_file() and json.loads(structure_result_path.read_text(encoding="utf-8")).get("status") == "ok":
                completed.add(structure_key)
            if structure_key not in completed:
                try:
                    result = prepare_structure(
                        output_root=output_root, model_name=model, family_id=family_id,
                        d_h=d_h, physical_gpu=physical_gpu,
                    )
                    rows.append({"model": model, "family": family_id, "d_h": d_h, "profile": "structure", "phase": "structure", "status": result["status"]})
                    completed.add(structure_key)
                except Exception as exc:
                    rows.append({"model": model, "family": family_id, "d_h": d_h, "profile": "structure", "phase": "structure", "status": "failed", "error": repr(exc), "traceback": traceback.format_exc()})
                    _write(checkpoint, rows)
                    continue
                _write(checkpoint, rows)
            for profile in profiles:
                build_key = (family_id, d_h, profile, "build")
                engine_dir = output_root / "engines" / model / family_id / f"dh_{d_h:03d}" / profile
                build_result_path = engine_dir / "baseline_result.json"
                if build_result_path.is_file():
                    existing_build = json.loads(build_result_path.read_text(encoding="utf-8"))
                    if existing_build.get("status") == "ok" and int(existing_build.get("requested_realized_conflict_count", -1)) == 0:
                        completed.add(build_key)
                if build_key not in completed:
                    try:
                        result = (
                            build_int8_profile(
                                output_root=output_root, model_name=model, family_id=family_id,
                                d_h=d_h, physical_gpu=physical_gpu, plugin_path=plugin,
                            ) if profile == "P8" else build_float_profile(
                                output_root=output_root, model_name=model, family_id=family_id,
                                d_h=d_h, profile_id=profile, physical_gpu=physical_gpu,
                                plugin_path=plugin,
                            )
                        )
                        rows.append({"model": model, "family": family_id, "d_h": d_h, "profile": profile, "phase": "build", "status": result["status"]})
                        if result["status"] == "ok":
                            completed.add(build_key)
                    except Exception as exc:
                        rows.append({"model": model, "family": family_id, "d_h": d_h, "profile": profile, "phase": "build", "status": "failed", "error": repr(exc), "traceback": traceback.format_exc()})
                    _write(checkpoint, rows)
                if build_key not in completed:
                    continue
                for protocol in protocols:
                    phase = f"evaluate_{protocol}"
                    evaluation_key = (family_id, d_h, profile, phase)
                    evaluation_path = engine_dir / "evaluation" / protocol / "evaluation_acceptance.json"
                    if evaluation_path.is_file() and json.loads(evaluation_path.read_text(encoding="utf-8")).get("status") == "ok":
                        completed.add(evaluation_key)
                    if evaluation_key in completed:
                        continue
                    try:
                        result = evaluate(
                            output_root=output_root, model_name=model, family_id=family_id,
                            d_h=d_h, profile=profile, protocol=protocol,
                            physical_gpu=physical_gpu, plugin=plugin,
                        )
                        rows.append({"model": model, "family": family_id, "d_h": d_h, "profile": profile, "phase": phase, "status": result["status"], "mAP": result.get("mAP")})
                        if result["status"] == "ok":
                            completed.add(evaluation_key)
                    except Exception as exc:
                        rows.append({"model": model, "family": family_id, "d_h": d_h, "profile": profile, "phase": phase, "status": "failed", "error": repr(exc), "traceback": traceback.format_exc()})
                    _write(checkpoint, rows)
                    if evaluation_key not in completed:
                        break
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model", choices=("lidar_cobevt", "lidar_v2xvit"), required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--family", default="")
    parser.add_argument("--widths", default="")
    parser.add_argument("--profiles", default="P32,P16,P8")
    parser.add_argument("--protocols", default="smoke10,fixed50,fixed500")
    parser.add_argument("--low-width-extension", action="store_true")
    args = parser.parse_args(argv)
    rows = run(
        output_root=Path(args.output_root).resolve(), model=args.model,
        physical_gpu=args.physical_gpu, plugin=Path(args.plugin).resolve(),
        family_filter=args.family,
        width_filter=tuple(int(value) for value in args.widths.split(",") if value),
        low_width_extension=args.low_width_extension,
        profiles=tuple(value for value in args.profiles.split(",") if value),
        protocols=tuple(value for value in args.protocols.split(",") if value),
    )
    print(json.dumps({"rows": len(rows), "failures": sum(row["status"] != "ok" for row in rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
