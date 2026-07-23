"""Small, deterministic Phase-B joint-family d_h validation matrix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
from typing import Any, Mapping

from search.model_families.transformer.dh_candidate_grid import dense_head_dimension_grid
from search.orchestration.lidar_transformer_dh_joint import joint_id, run_joint
from search.orchestration.lidar_transformer_dh_recovery import (
    ExperimentState,
    require_phase_a_certificate,
)


PROFILES = ("P32", "P16", "P8")
LABELS = (
    "B0_BASELINE",
    "B1_CONSERVATIVE_ALIGNED",
    "B2_CONSERVATIVE_NONALIGNED",
    "B3_MODERATE_NONALIGNED",
    "B4_BOUNDARY_ALIGNED",
    "B5_BOUNDARY_NONALIGNED",
    "B6_AGGRESSIVE_DIAGNOSTIC",
)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def _choose_family_widths(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Backward-compatible primitive used by the original Phase-B tests."""
    safe = [row for row in rows if bool(row.get("safe_all_profiles"))]
    if not safe:
        raise RuntimeError("phase_b_family_has_no_accuracy_safe_width")
    aligned = [row for row in safe if int(row["d_h"]) % 8 == 0]
    nonaligned = [row for row in safe if int(row["d_h"]) % 4 != 0]
    aligned_row = min(aligned or safe, key=lambda row: int(row["d_h"]))
    nonaligned_row = min(nonaligned or safe, key=lambda row: int(row["d_h"]))
    latency_row = max(
        safe,
        key=lambda row: (
            float(row.get("provisional_speedup_median") or 0.0),
            -int(row["d_h"]),
        ),
    )
    return {
        "aligned_safe": int(aligned_row["d_h"]),
        "nonaligned_safe": int(nonaligned_row["d_h"]),
        "provisional_latency": int(latency_row["d_h"]),
        "safe_widths": [int(row["d_h"]) for row in safe],
    }


def _family_evidence(output_root: Path, model: str, family: Mapping[str, Any]) -> dict[str, Any]:
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
    baseline_by_profile: dict[str, float] = {}
    for profile in PROFILES:
        baseline = _read(
            output_root
            / "engines"
            / model
            / family_id
            / f"dh_{d0:03d}"
            / profile
            / "evaluation"
            / "fixed500"
            / "evaluation_acceptance.json"
        )
        if not baseline or baseline.get("status") != "ok":
            raise RuntimeError(f"phase_b_missing_baseline:{model}:{family_id}:{profile}")
        baseline_by_profile[profile] = float(baseline["mAP"])
    rows: list[dict[str, Any]] = []
    with (output_root / "dh_alignment_full_matrix.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        matrix = [
            row
            for row in csv.DictReader(handle)
            if row["model"] == model and row["attention_family"] == family_id
        ]
    for width in widths:
        profiles: dict[str, Any] = {}
        safe_all = True
        for profile in PROFILES:
            directory = output_root / "engines" / model / family_id / f"dh_{width:03d}" / profile
            build = _read(directory / "baseline_result.json") or {}
            evaluation = _read(directory / "evaluation" / "fixed500" / "evaluation_acceptance.json") or {}
            alignment = _read(directory / "engine_alignment_audit.json") or {}
            exact = (
                build.get("status") == "ok"
                and int(build.get("requested_realized_conflict_count", -1)) == 0
                and not bool(alignment.get("fallback_hint"))
                and evaluation.get("status") == "ok"
                and int(evaluation.get("evaluated", -1)) == 500
                and int(evaluation.get("skipped", -1)) == 0
            )
            delta = (
                float(evaluation["mAP"]) - baseline_by_profile[profile]
                if evaluation.get("mAP") is not None
                else None
            )
            safe = exact and delta is not None and abs(delta) <= 0.003
            safe_all &= safe
            profiles[profile] = {
                "exact": exact,
                "safe": safe,
                "mAP": evaluation.get("mAP"),
                "delta_mAP": delta,
                "engine_sha256": build.get("engine_sha256"),
            }
        structure = _read(
            output_root / "structures" / model / family_id / f"dh_{width:03d}" / "structure_result.json"
        ) or {}
        rows.append(
            {
                "d_h": width,
                "safe_all_profiles": safe_all,
                "profiles": profiles,
                "physical_parameter_count": structure.get("physical_parameter_count"),
                "weighted_macs": structure.get("weighted_macs"),
                "bops_by_profile": {
                    profile: float(
                        next(
                            row
                            for row in matrix
                            if int(row["d_h"]) == width and row["profile"] == profile
                        )["BOPS"]
                    )
                    for profile in PROFILES
                },
            }
        )
    continuous: list[int] = []
    for row in rows:
        if not row["safe_all_profiles"]:
            break
        continuous.append(int(row["d_h"]))
    if not continuous or continuous[0] != d0:
        raise RuntimeError(f"phase_b_no_contiguous_safe_prefix:{model}:{family_id}")
    return {
        "family_id": family_id,
        "heads": int(family["heads"]),
        "original_d_h": d0,
        "continuous_safe_widths": continuous,
        "continuous_safe_lower_bound": continuous[-1],
        "widths": rows,
    }


def _nearest(values: list[int], target: float, predicate: Any) -> int:
    accepted = [value for value in values if predicate(value)]
    if not accepted:
        raise RuntimeError(f"phase_b_width_class_unavailable:{target}")
    return min(accepted, key=lambda value: (abs(value - target), -value))


def _targets_for_family(evidence: Mapping[str, Any]) -> dict[str, int]:
    d0 = int(evidence["original_d_h"])
    lower = int(evidence["continuous_safe_lower_bound"])
    safe = [int(value) for value in evidence["continuous_safe_widths"]]
    midpoint = (d0 + lower) / 2.0
    return {
        "B0_BASELINE": d0,
        "B1_CONSERVATIVE_ALIGNED": _nearest(safe, d0 - 4, lambda value: value % 4 == 0),
        "B2_CONSERVATIVE_NONALIGNED": _nearest(safe, d0 - 1, lambda value: value % 4 != 0),
        "B3_MODERATE_NONALIGNED": _nearest(safe, midpoint, lambda value: value % 4 != 0),
        "B4_BOUNDARY_ALIGNED": _nearest(safe, lower, lambda value: value % 4 == 0),
        "B5_BOUNDARY_NONALIGNED": _nearest(
            safe, lower + 2, lambda value: value % 4 != 0 and value > lower
        ),
        "B6_AGGRESSIVE_DIAGNOSTIC": lower,
    }


def select_phase_b_candidates(output_root: Path, model: str) -> dict[str, Any]:
    require_phase_a_certificate(output_root)
    families = _read(
        output_root / "inventory" / f"{model.removeprefix('lidar_')}_attention_families.json"
    )
    if not isinstance(families, list) or len(families) < 2:
        raise RuntimeError(f"phase_b_requires_multiple_families:{model}")
    evidence = {
        str(family["family_id"]): _family_evidence(output_root, model, family)
        for family in families
    }
    family_targets = {
        family_id: _targets_for_family(value) for family_id, value in evidence.items()
    }
    candidates: list[dict[str, Any]] = []
    signatures: set[tuple[tuple[str, int], ...]] = set()
    for label in LABELS:
        targets = {
            family_id: int(values[label]) for family_id, values in family_targets.items()
        }
        signature = tuple(sorted(targets.items()))
        if signature in signatures:
            raise RuntimeError(f"phase_b_duplicate_candidate:{model}:{label}:{signature}")
        signatures.add(signature)
        per_profile_delta: dict[str, dict[str, float]] = {}
        for profile in PROFILES:
            per_profile_delta[profile] = {
                family_id: float(
                    next(
                        row
                        for row in evidence[family_id]["widths"]
                        if int(row["d_h"]) == width
                    )["profiles"][profile]["delta_mAP"]
                )
                for family_id, width in targets.items()
            }
        original_parameters = int(
            next(iter(evidence.values()))["widths"][0]["physical_parameter_count"]
        )
        parameter_drop = sum(
            original_parameters
            - int(
                next(
                    row
                    for row in evidence[family_id]["widths"]
                    if int(row["d_h"]) == width
                )["physical_parameter_count"]
            )
            for family_id, width in targets.items()
        )
        bops_reduction: dict[str, float] = {}
        for profile in PROFILES:
            first_family = next(iter(evidence))
            baseline_bops = float(evidence[first_family]["widths"][0]["bops_by_profile"][profile])
            bops_drop = sum(
                float(evidence[family_id]["widths"][0]["bops_by_profile"][profile])
                - float(
                    next(
                        row
                        for row in evidence[family_id]["widths"]
                        if int(row["d_h"]) == width
                    )["bops_by_profile"][profile]
                )
                for family_id, width in targets.items()
            )
            bops_reduction[profile] = bops_drop / baseline_bops
        candidates.append(
            {
                "candidate_id": label,
                "joint_id": joint_id(targets),
                "targets": targets,
                "alignment_class_by_family": {
                    family_id: (
                        "multiple_of_8"
                        if width % 8 == 0
                        else "multiple_of_4"
                        if width % 4 == 0
                        else "non4"
                    )
                    for family_id, width in targets.items()
                },
                "phase_a_single_family_delta_by_profile": per_profile_delta,
                "delta_predicted_additive_by_profile": {
                    profile: sum(per_profile_delta[profile].values()) for profile in PROFILES
                },
                "predicted_parameter_reduction": parameter_drop / original_parameters,
                "predicted_bops_reduction_by_profile": bops_reduction,
                "contains_odd_width": any(width % 2 for width in targets.values()),
                "contains_non4_width": any(width % 4 for width in targets.values()),
                "contains_non8_width": any(width % 8 for width in targets.values()),
                "selection_reason": {
                    "B0_BASELINE": "fresh joint unpruned same-profile baseline",
                    "B1_CONSERVATIVE_ALIGNED": "light, multiple-of-4 joint control",
                    "B2_CONSERVATIVE_NONALIGNED": "light non-4-aligned joint candidate",
                    "B3_MODERATE_NONALIGNED": "middle of contiguous all-profile SAFE intervals",
                    "B4_BOUNDARY_ALIGNED": "aligned controls nearest contiguous SAFE lower bounds",
                    "B5_BOUNDARY_NONALIGNED": "non-4 controls near contiguous SAFE lower bounds",
                    "B6_AGGRESSIVE_DIAGNOSTIC": "all families at contiguous SAFE lower bounds",
                }[label],
                "diagnostic": label == "B6_AGGRESSIVE_DIAGNOSTIC",
                "fresh_structure_required": True,
                "fresh_engine_required": True,
                "fresh_p8_calibration_required": True,
            }
        )
    result = {
        "schema_version": "h800-transformer-dh-phase-b-selection-v2",
        "model": model,
        "phase_a_certificate": str(output_root / "phase_a_completion_certificate.json"),
        "family_evidence": evidence,
        "joint_candidates": candidates,
        "cartesian_product_used": False,
    }
    _write(output_root / "reports" / f"{model}_phase_b_selection.json", result)
    _consolidate_selection(output_root)
    return result


def _consolidate_selection(output_root: Path) -> None:
    selections = []
    rows: list[dict[str, Any]] = []
    for model in ("lidar_cobevt", "lidar_v2xvit"):
        payload = _read(output_root / "reports" / f"{model}_phase_b_selection.json")
        if not payload:
            continue
        selections.append(payload)
        for candidate in payload["joint_candidates"]:
            rows.append({"model": model, **candidate})
    _write(output_root / "phase_b_selection.json", {"models": selections})
    _write_csv(output_root / "phase_b_candidate_manifest.csv", rows)


def run(
    *,
    output_root: Path,
    model: str,
    physical_gpu: int,
    plugin: Path,
    profiles: tuple[str, ...] = PROFILES,
    protocols: tuple[str, ...] = ("smoke10", "fixed50", "fixed500"),
) -> dict[str, Any]:
    selection = select_phase_b_candidates(output_root, model)
    state = ExperimentState(output_root)
    results: list[dict[str, Any]] = []
    state.update_phase("phase_b", status="running", model=model)
    for candidate in selection["joint_candidates"]:
        targets = {str(name): int(width) for name, width in candidate["targets"].items()}
        artifact = output_root / "engines" / model / "joint" / joint_id(targets)
        with state.attempt(
            phase="phase_b",
            candidate_id=f"{model}-{candidate['candidate_id']}",
            artifact_output_path=artifact,
        ):
            result = run_joint(
                output_root=output_root,
                model_name=model,
                targets=targets,
                physical_gpu=physical_gpu,
                plugin=plugin,
                profiles=profiles,
                protocols=protocols,
            )
        results.append({**candidate, "result": result})
        _write(
            output_root / "reports" / f"{model}_phase_b_result.json",
            {"selection": selection, "results": results},
        )
    payload = {"selection": selection, "results": results}
    state.update_phase("phase_b", status="complete", model=model, candidates=len(results))
    summarize_phase_b(output_root)
    return payload


def summarize_phase_b(output_root: Path) -> dict[str, Any]:
    joint_rows: list[dict[str, Any]] = []
    interaction_rows: list[dict[str, Any]] = []
    complete = True
    for model in ("lidar_cobevt", "lidar_v2xvit"):
        payload = _read(output_root / "reports" / f"{model}_phase_b_result.json")
        if not payload:
            complete = False
            continue
        results = payload.get("results", [])
        baseline_candidate = next(
            (row for row in results if row["candidate_id"] == "B0_BASELINE"), None
        )
        if baseline_candidate is None:
            complete = False
            continue
        baseline_fixed = {
            str(row["profile"]): row
            for row in baseline_candidate["result"]["evaluations"]
            if row.get("protocol") == "fixed500"
        }
        for candidate in results:
            structure = candidate["result"]["structure"]
            builds = {str(row["profile"]): row for row in candidate["result"]["builds"]}
            fixed = {
                str(row["profile"]): row
                for row in candidate["result"]["evaluations"]
                if row.get("protocol") == "fixed500"
            }
            for profile in PROFILES:
                build = builds.get(profile, {})
                evaluation = fixed.get(profile, {})
                baseline = baseline_fixed.get(profile, {})
                accepted = (
                    build.get("status") == "ok"
                    and int(build.get("requested_realized_conflict_count", -1)) == 0
                    and evaluation.get("status") == "ok"
                    and int(evaluation.get("evaluated", -1)) == 500
                    and int(evaluation.get("skipped", -1)) == 0
                    and baseline.get("status") == "ok"
                )
                complete &= accepted
                delta_joint = (
                    float(evaluation["mAP"]) - float(baseline["mAP"]) if accepted else None
                )
                predicted = float(candidate["delta_predicted_additive_by_profile"][profile])
                interaction = delta_joint - predicted if delta_joint is not None else None
                classification = (
                    "UNAVAILABLE"
                    if delta_joint is None
                    else "SAFE"
                    if abs(delta_joint) <= 0.003
                    else "UNSAFE"
                    if delta_joint < -0.010
                    else "BORDERLINE"
                    if delta_joint < -0.003
                    else "POSITIVE_OUTLIER"
                )
                row = {
                    "model": model,
                    "candidate_id": candidate["candidate_id"],
                    "joint_id": candidate["joint_id"],
                    "targets": candidate["targets"],
                    "profile": profile,
                    "diagnostic": candidate["diagnostic"],
                    "structure_hash": structure.get("structure_hash"),
                    "state_dict_shape_hash": structure.get("state_dict_shape_hash"),
                    "onnx_sha256": structure.get("onnx_sha256"),
                    "engine_sha256": build.get("engine_sha256"),
                    "build_status": build.get("status"),
                    "requested_realized_conflict_count": build.get(
                        "requested_realized_conflict_count"
                    ),
                    "fixed500_status": evaluation.get("status"),
                    "evaluated": evaluation.get("evaluated"),
                    "skipped": evaluation.get("skipped"),
                    "AP30": evaluation.get("AP@0.3"),
                    "AP50": evaluation.get("AP@0.5"),
                    "AP70": evaluation.get("AP@0.7"),
                    "mAP": evaluation.get("mAP"),
                    "baseline_mAP": baseline.get("mAP"),
                    "delta_predicted_additive": predicted,
                    "delta_joint": delta_joint,
                    "interaction_joint": interaction,
                    "accuracy_status": classification,
                    "scale_hash": evaluation.get("scale_hash"),
                    "manifest_hash": evaluation.get("manifest_hash"),
                }
                joint_rows.append(row)
                interaction_rows.append(
                    {
                        key: row[key]
                        for key in (
                            "model",
                            "candidate_id",
                            "joint_id",
                            "targets",
                            "profile",
                            "delta_predicted_additive",
                            "delta_joint",
                            "interaction_joint",
                            "accuracy_status",
                        )
                    }
                )
    _write_csv(output_root / "phase_b_joint_fixed500.csv", joint_rows)
    _write_csv(output_root / "phase_b_interaction_matrix.csv", interaction_rows)
    summary = {
        "schema_version": "h800-transformer-dh-phase-b-accuracy-v1",
        "fixed500_complete": complete and len(joint_rows) == 42,
        "expected_rows": 42,
        "accepted_rows": sum(row["fixed500_status"] == "ok" for row in joint_rows),
        "safe_rows": sum(row["accuracy_status"] == "SAFE" for row in joint_rows),
        "borderline_rows": sum(row["accuracy_status"] == "BORDERLINE" for row in joint_rows),
        "unsafe_rows": sum(row["accuracy_status"] == "UNSAFE" for row in joint_rows),
        "interaction_range": [
            min((row["interaction_joint"] for row in joint_rows if row["interaction_joint"] is not None), default=None),
            max((row["interaction_joint"] for row in joint_rows if row["interaction_joint"] is not None), default=None),
        ],
        "full1789_executed": False,
    }
    _write(output_root / "phase_b_accuracy_boundary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model", choices=("lidar_cobevt", "lidar_v2xvit"), required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--profiles", default="P32,P16,P8")
    parser.add_argument("--protocols", default="smoke10,fixed50,fixed500")
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.output_root).resolve()
    if args.select_only:
        result = select_phase_b_candidates(root, args.model)
        print(json.dumps({"joint_candidates": len(result["joint_candidates"])}, sort_keys=True))
        return 0
    result = run(
        output_root=root,
        model=args.model,
        physical_gpu=args.physical_gpu,
        plugin=Path(args.plugin).resolve(),
        profiles=tuple(value for value in args.profiles.split(",") if value),
        protocols=tuple(value for value in args.protocols.split(",") if value),
    )
    failures = sum(
        row.get("status") != "ok"
        for candidate in result["results"]
        for row in (*candidate["result"]["builds"], *candidate["result"]["evaluations"])
    )
    print(json.dumps({"joint_candidates": len(result["results"]), "failures": failures}, sort_keys=True))
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
