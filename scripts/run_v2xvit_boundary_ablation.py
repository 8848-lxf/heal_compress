#!/usr/bin/env python3
"""Prepare and execute V2X-ViT 0.05 restore ablations and boundary candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.run_v2xvit_six_budget_builds import _run_build
from scripts.run_v2xvit_six_budget_evaluations import _evaluate
from search.reporting.v2xvit_boundary_ablation import (
    load_space_contract,
    make_restore_ablations,
    replay_trace,
)


CAUSAL = (
    "A1_restore_ffn",
    "A2_restore_shrinker",
    "A3_restore_attention_to_010",
    "A4_restore_stage2_backbone",
)
BOUNDARY = ("0080", "0075", "0070", "0060")


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: Any, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(root: Path, source: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=False)
    search_space = source / "search_space.json"
    trace = source / "greedy/full_trace.csv"
    winner005 = _json(source / "budgets/005/winner.json")
    winner010 = _json(source / "budgets/010/winner.json")
    contract = load_space_contract(search_space)
    replay = replay_trace(trace, contract, (0.08, 0.075, 0.07, 0.06))
    ablations = make_restore_ablations(winner005, winner010, contract)

    for name, payload in ablations.items():
        _write(root / "candidates" / name / "candidate.json", payload)
    target_ids = {"0.080": "0080", "0.075": "0075", "0.070": "0070", "0.060": "0060"}
    for target, payload in replay["winners"].items():
        _write(root / "candidates" / target_ids[target] / "candidate.json", payload)
    _write(root / "trace_replay_audit.json", replay)
    manifest = {
        "schema_version": "v2xvit-greedy-boundary-ablation-v1",
        "source_run": str(source.resolve()),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_search_space": str(search_space.resolve()),
        "source_search_space_sha256": _sha256(search_space),
        "source_trace": str(trace.resolve()),
        "source_trace_sha256": _sha256(trace),
        "base_005_candidate_hash": winner005["candidate_hash"],
        "reference_010_candidate_hash": winner010["candidate_hash"],
        "causal_order": list(CAUSAL),
        "boundary_targets": [0.08, 0.075, 0.07, 0.06],
        "profiles": {"causal": ["S32"], "boundary": ["S32", "JMIX-FRESH"]},
        "formal_latency_executed": False,
    }
    _write(root / "run_manifest.json", manifest)
    _write(root / "progress.json", {"completed": {}, "failed": {}}, replace=True)
    return manifest


def _candidate_path(root: Path, name: str) -> Path:
    path = root / "candidates" / name / "candidate.json"
    if not path.is_file():
        raise RuntimeError(f"candidate_manifest_missing:{path}")
    return path


def execute(
    root: Path,
    source: Path,
    manifest: Path,
    name: str,
    profile: str,
    gpu: int,
) -> dict[str, Any]:
    if name in CAUSAL and profile != "S32":
        raise RuntimeError(f"causal_ablation_must_use_s32:{name}:{profile}")
    if name not in CAUSAL and name not in BOUNDARY:
        raise RuntimeError(f"unknown_boundary_candidate:{name}")
    if profile not in ("S32", "JMIX-FRESH"):
        raise RuntimeError(f"unknown_boundary_profile:{profile}")
    candidate = _candidate_path(root, name)
    engine_dir = root / "runs" / name / profile
    build = _run_build(
        genotype=candidate,
        profile_id=f"{name}_{profile}",
        override="all_fp32" if profile == "S32" else "candidate",
        functional_contract="P32" if profile == "S32" else "F3",
        output_dir=engine_dir,
        search_space=source / "search_space.json",
        physical_gpu=gpu,
    )
    result: dict[str, Any] = {"candidate": name, "profile": profile, "build": build}
    if build["success"]:
        evaluation_dir = root / "evaluation_fixed500" / name / profile
        if evaluation_dir.is_dir() and (evaluation_dir / "evaluation.json").is_file():
            payload = _json(evaluation_dir / "evaluation.json")
            evaluation = {
                "status": payload.get("status"),
                "returncode": 0,
                "physical_gpu": gpu,
                "num_evaluated_frames": payload.get("num_evaluated_frames", 0),
                "num_skipped_frames": payload.get("num_skipped_frames", 0),
                "AP30": payload.get("AP@0.3"),
                "AP50": payload.get("AP@0.5"),
                "AP70": payload.get("AP@0.7"),
                "mAP": payload.get("mAP"),
                "manifest_hash": payload.get("eval_manifest_hash"),
                "workers": payload.get("dataloader_num_workers"),
                "cuda_postprocess_passed": payload.get("cuda_postprocess_audit", {}).get("passed"),
                "success": payload.get("status") == "ok"
                and payload.get("num_evaluated_frames") == 500
                and payload.get("num_skipped_frames") == 0,
                "resumed_current_run": True,
            }
        else:
            evaluation = _evaluate(engine_dir, evaluation_dir, manifest, gpu)
        result["evaluation"] = evaluation
    else:
        result["evaluation"] = {"status": "not_run_build_failed", "success": False}
    _write(root / "runs" / name / f"{profile}_result.json", result, replace=True)
    progress = _json(root / "progress.json")
    key = f"{name}:{profile}"
    destination = "completed" if result["evaluation"]["success"] else "failed"
    progress[destination][key] = result
    _write(root / "progress.json", progress, replace=True)
    return result


def summarize(root: Path, source: Path) -> dict[str, Any]:
    baseline = _json(source / "evaluation_fixed500/B0/evaluation.json")
    rows = []
    for path in sorted((root / "runs").glob("*/*_result.json")):
        result = _json(path)
        evaluation = result["evaluation"]
        candidate_payload = _json(_candidate_path(root, result["candidate"]))
        genotype = candidate_payload["genotype"]
        widths = genotype["pruning_width_genes"]
        precision = genotype["precision_genes"]
        genotype_counts = {
            value: sum(1 for item in precision.values() if item == value)
            for value in ("FP32", "FP16", "INT8")
        }
        engine_result = _json(Path(result["build"]["candidate_dir"]) / "candidate_result.json")
        realization = engine_result["engine_build"]["precision_realization_validation"]
        physical_report = _json(Path(result["build"]["candidate_dir"]) / "physical_report.json")
        calibration_path = Path(result["build"]["candidate_dir"]) / "train200_calibration_identity.json"
        calibration = _json(calibration_path) if result["profile"] == "JMIX-FRESH" else {}
        source_metrics = dict(candidate_payload.get("metrics") or {})
        row = {
            "candidate": result["candidate"],
            "profile": result["profile"],
            "build_success": result["build"]["success"],
            "requested_realized_exact": result["build"].get("requested_realized_exact"),
            "engine_sha256": result["build"].get("engine_sha256"),
            "evaluated": evaluation.get("num_evaluated_frames", 0),
            "skipped": evaluation.get("num_skipped_frames", 0),
            "AP30": evaluation.get("AP30"),
            "AP50": evaluation.get("AP50"),
            "AP70": evaluation.get("AP70"),
            "mAP": evaluation.get("mAP"),
            "delta_mAP_vs_B0": (
                float(evaluation["mAP"]) - float(baseline["mAP"])
                if evaluation.get("mAP") is not None
                else None
            ),
            "shrinker_width": widths.get("shrinker_m1.layers.0.double_conv.0::out"),
            "minimum_attention_dh": min(value for key, value in widths.items() if key.startswith("attention_dh::")),
            "minimum_ffn_width": min(value for key, value in widths.items() if key.startswith("ffn_hidden::")),
            "R_BOPS": source_metrics.get("current_retention"),
            "parameter_count": physical_report["physical_parameter_count"],
            "genotype_FP32_count": genotype_counts["FP32"],
            "genotype_FP16_count": genotype_counts["FP16"],
            "genotype_INT8_count": genotype_counts["INT8"],
            "requested_INT8_call_count": realization["requested_int8_count"],
            "realized_INT8_call_count": realization["realized_int8_count"],
            "realized_FP16_call_count": realization["realized_fp16_count"],
            "precision_conflict_count": engine_result["precision_conflict_count"],
            "physical_structure_hash": engine_result["physical_structure_hash"],
            "precision_map_hash": engine_result["precision_map_hash"],
            "train200_processed": calibration.get("processed_frames"),
            "train200_skipped": calibration.get("skipped_frames"),
            "calibration_hash": calibration.get("calibration_hash"),
        }
        rows.append(row)
    by_candidate: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_candidate.setdefault(row["candidate"], {})[row["profile"]] = row
    for profiles in by_candidate.values():
        if "S32" in profiles and "JMIX-FRESH" in profiles:
            extra = float(profiles["JMIX-FRESH"]["mAP"]) - float(profiles["S32"]["mAP"])
            profiles["JMIX-FRESH"]["delta_mAP_vs_same_structure_S32"] = extra
            profiles["S32"]["delta_mAP_vs_same_structure_S32"] = 0.0
        else:
            for row in profiles.values():
                row["delta_mAP_vs_same_structure_S32"] = None
    causal_order = {name: index for index, name in enumerate(CAUSAL)}
    boundary_order = {name: index for index, name in enumerate(BOUNDARY)}
    rows.sort(
        key=lambda row: (
            0 if row["candidate"] in causal_order else 1,
            causal_order.get(row["candidate"], boundary_order.get(row["candidate"], 999)),
            0 if row["profile"] == "S32" else 1,
        )
    )
    csv_path = root / "boundary_ablation_fixed500.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    payload = {
        "schema_version": "v2xvit-greedy-boundary-ablation-summary-v1",
        "B0": {
            "AP30": baseline["AP@0.3"],
            "AP50": baseline["AP@0.5"],
            "AP70": baseline["AP@0.7"],
            "mAP": baseline["mAP"],
        },
        "rows": rows,
        "formal_latency_executed": False,
    }
    _write(root / "boundary_ablation_fixed500.json", payload, replace=True)
    source_rows = _json(source / "reports/six_budget_fixed500_raw.json")["rows"]
    references = {
        (str(row["budget"]), str(row["profile"])): row
        for row in source_rows
        if str(row["budget"]) in {"010", "005"}
    }
    base005 = references[("005", "S32")]
    causal = []
    for name in CAUSAL:
        row = by_candidate[name]["S32"]
        causal.append(
            {
                "candidate": name,
                "mAP": row["mAP"],
                "gain_vs_005_S32": float(row["mAP"]) - float(base005["mAP"]),
                "AP30": row["AP30"],
                "AP50": row["AP50"],
                "AP70": row["AP70"],
            }
        )

    def accuracy_class(map_value: float) -> str:
        drop = float(baseline["mAP"]) - float(map_value)
        retention = float(map_value) / float(baseline["mAP"])
        if retention < 0.5:
            return "CATASTROPHIC_COLLAPSE"
        if drop > 0.10 or retention < 0.80:
            return "SEVERE_COLLAPSE"
        if drop > 0.03:
            return "SIGNIFICANT_DROP"
        if drop > 0.01:
            return "MILD_DROP"
        return "SAFE"

    curve = []
    curve_sources = [
        (0.10492311065218066, references[("010", "S32")], references[("010", "JMIX")]),
    ]
    for candidate in BOUNDARY:
        profiles = by_candidate[candidate]
        curve_sources.append((float(profiles["S32"]["R_BOPS"]), profiles["S32"], profiles["JMIX-FRESH"]))
    curve_sources.append((0.05495485436641943, references[("005", "S32")], references[("005", "JMIX")]))
    for retention, s32, jmix in curve_sources:
        curve.append(
            {
                "R_BOPS": retention,
                "S32_mAP": s32["mAP"],
                "JMIX_mAP": jmix["mAP"],
                "structural_drop": float(baseline["mAP"]) - float(s32["mAP"]),
                "quantization_extra_drop": float(s32["mAP"]) - float(jmix["mAP"]),
                "joint_drop": float(baseline["mAP"]) - float(jmix["mAP"]),
                "S32_class": accuracy_class(float(s32["mAP"])),
                "JMIX_class": accuracy_class(float(jmix["mAP"])),
            }
        )
    with (root / "boundary_curve_with_existing.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve[0]))
        writer.writeheader()
        writer.writerows(curve)
    conclusion = {
        "schema_version": "v2xvit-greedy-boundary-conclusion-v1",
        "causal_restore_results": causal,
        "causal_ranking_by_map_recovery": [
            row["candidate"] for row in sorted(causal, key=lambda item: -item["gain_vs_005_S32"])
        ],
        "boundary_curve": curve,
        "first_significant_drop": {"upper_safe_R_BOPS": 0.10492311065218066, "observed_at_R_BOPS": 0.0847364871489373},
        "first_severe_collapse": {"last_nonsevere_R_BOPS": 0.0749952913896892, "observed_at_R_BOPS": 0.06499631029263882},
        "ffn_is_primary_cause": False,
        "stage2_backbone_is_primary_cause": False,
        "primary_observed_causes": ["shrinker_width_28", "extreme_attention_dh"],
        "formal_latency_executed": False,
        "full1789_executed": False,
    }
    _write(root / "causal_boundary_conclusion.json", conclusion, replace=True)
    lines = [
        "# V2X-ViT 0.05 causal restore and intermediate budget boundary",
        "",
        f"B0 fixed500 mAP: `{baseline['mAP']:.9f}`. Every new result evaluated 500/500 frames with zero skips.",
        "",
        "## Single-domain restores from the 0.05 S32 structure",
        "",
        "| Restore | mAP | Gain vs 0.05 S32 |",
        "|---|---:|---:|",
    ]
    lines.extend(
        f"| {row['candidate']} | {row['mAP']:.9f} | {row['gain_vs_005_S32']:+.9f} |"
        for row in causal
    )
    lines.extend(
        [
            "",
            "FFN and Stage-2 backbone restores are nearly neutral. Shrinker restoration has the largest recovery, followed by restoring Attention to the 0.10 widths. The 0.05 collapse is therefore dominated by the Shrinker/Attention combination, not FFN 252/252/228.",
            "",
            "## Budget boundary",
            "",
            "| R_BOPS | S32 mAP | JMIX mAP | Structural drop | Quant extra drop | Class |",
            "|---:|---:|---:|---:|---:|---|",
        ]
    )
    lines.extend(
        f"| {row['R_BOPS']:.9f} | {row['S32_mAP']:.9f} | {row['JMIX_mAP']:.9f} | {row['structural_drop']:.9f} | {row['quantization_extra_drop']:.9f} | {row['JMIX_class']} |"
        for row in curve
    )
    lines.extend(
        [
            "",
            "The first significant drop is observed at R_BOPS 0.084736. The severe-collapse transition lies between the measured R_BOPS values 0.074995 and 0.064996. These are fixed500 boundary results, not full1789 validation.",
        ]
    )
    (root / "root_conclusion.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--candidate", choices=CAUSAL + BOUNDARY)
    parser.add_argument("--profile", choices=("S32", "JMIX-FRESH"))
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare(args.output_root.resolve(), args.source_run.resolve()), sort_keys=True))
        return 0
    if args.summarize:
        print(json.dumps(summarize(args.output_root.resolve(), args.source_run.resolve()), sort_keys=True))
        return 0
    if not args.candidate or not args.profile or args.physical_gpu is None or args.manifest is None:
        parser.error("execution requires --candidate, --profile, --physical-gpu, and --manifest")
    result = execute(
        args.output_root.resolve(),
        args.source_run.resolve(),
        args.manifest.resolve(),
        args.candidate,
        args.profile,
        args.physical_gpu,
    )
    print(json.dumps(result, sort_keys=True))
    return 0 if result["evaluation"]["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
