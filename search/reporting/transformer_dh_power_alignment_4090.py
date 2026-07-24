"""Compact evidence reporter for the RTX 4090 Transformer alignment run."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


_COMPACT_FIELDS = (
    "model",
    "family",
    "candidate_id",
    "structure_kind",
    "target_d_h_by_family",
    "d_h",
    "original_d_h",
    "heads",
    "projection_width",
    "reduction_ratio",
    "exact_power_of_two",
    "divisible_by_4",
    "divisible_by_8",
    "divisible_by_16",
    "divisible_by_32",
    "divisible_by_64",
    "projection_divisible_by_4",
    "projection_divisible_by_8",
    "projection_divisible_by_16",
    "projection_divisible_by_32",
    "projection_divisible_by_64",
    "profile",
    "structure_hash",
    "structure_signature",
    "onnx_sha256",
    "engine_sha256",
    "scale_hash",
    "requested_realized_conflict_count",
    "evaluated",
    "skipped",
    "AP30",
    "AP50",
    "AP70",
    "mAP",
    "delta_structure",
    "delta_structure_p32",
    "delta_precision_base",
    "delta_precision_candidate",
    "delta_total",
    "interaction",
    "p50_ms",
    "p90_ms",
    "p95_ms",
    "p99_ms",
    "structure_speedup",
    "precision_speedup",
    "total_speedup",
    "neighbor_control_speedup",
    "neighbor_control_advantage",
    "baseline_replay_drift_ratio",
    "repeat_p50_cv",
    "required_latency_reduction",
    "observed_latency_reduction",
    "latency_beneficial",
    "tactic_signature",
    "fusion_signature",
    "kernel_count",
    "cast_count",
    "reformat_count",
    "accuracy_class",
    "latency_class",
    "build_repeat_stable",
    "search_space_candidate",
    "diagnostic_only",
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    if not fields:
        fields = ["status"]
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


def accuracy_class(
    delta_map: float,
    *,
    evaluated: int,
    skipped: int,
    finite: bool,
) -> str:
    if int(evaluated) != 500 or int(skipped) != 0 or not bool(finite):
        return "INVALID_EVALUATION"
    delta = float(delta_map)
    if delta < -0.010:
        return "UNSAFE"
    if delta < -0.003:
        return "BORDERLINE"
    return "SAFE"


def precision_interaction(
    *,
    baseline_p32_map: float,
    baseline_profile_map: float,
    candidate_p32_map: float,
    candidate_profile_map: float,
) -> dict[str, float]:
    baseline_p32 = float(baseline_p32_map)
    baseline_profile = float(baseline_profile_map)
    candidate_p32 = float(candidate_p32_map)
    candidate_profile = float(candidate_profile_map)
    delta_structure_p32 = candidate_p32 - baseline_p32
    delta_structure = candidate_profile - baseline_profile
    delta_precision_base = baseline_profile - baseline_p32
    delta_precision_candidate = candidate_profile - candidate_p32
    return {
        "delta_structure": delta_structure,
        "delta_structure_p32": delta_structure_p32,
        "delta_precision_base": delta_precision_base,
        "delta_precision_candidate": delta_precision_candidate,
        "delta_total": candidate_profile - baseline_p32,
        "interaction": delta_precision_candidate - delta_precision_base,
    }


def build_repeat_stability(
    reduction_ratios: Sequence[float], *, required_reduction: float
) -> dict[str, Any]:
    values = tuple(float(value) for value in reduction_ratios)
    if len(values) != 3:
        raise ValueError("three_independent_builds_required")
    threshold = float(required_reduction)
    passing = sum(value > threshold for value in values)
    no_slowdown = min(values) >= 0.0
    stable = passing == 3 or (passing >= 2 and no_slowdown)
    return {
        "build_repeat_stable": stable,
        "passing_builds": passing,
        "required_reduction_ratio": threshold,
        "minimum_reduction_ratio": min(values),
        "maximum_reduction_ratio": max(values),
        "reductions": list(values),
    }


def result_cardinality(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Separate report aliases from physical structures and phenotypes."""

    structures = {str(row["structure_signature"]) for row in rows}
    phenotypes = {
        (
            str(row["structure_signature"]),
            str(row["profile"]),
            str(row["engine_sha256"]),
        )
        for row in rows
    }
    return {
        "result_alias_rows": len(rows),
        "unique_physical_structures": len(structures),
        "unique_phenotypes": len(phenotypes),
    }


def summarize_build_repeat_evidence(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate_id: str,
    profile: str,
    role: str | None = None,
) -> dict[str, Any]:
    """Normalize three independent full-engine build measurements."""

    selected = [
        row
        for row in rows
        if not bool(row.get("baseline_replay", False))
        and bool(row.get("formal", False))
        and (row.get("profile") in (None, profile))
        and (
            str(row.get("candidate_id", "")) == candidate_id
            or (role is not None and str(row.get("role", "")) == role)
        )
    ]
    hashes = [str(row.get("engine_sha256", "")) for row in selected]
    if len(selected) != 3 or len(set(hashes)) != 3 or any(not value for value in hashes):
        raise RuntimeError("three_unique_fresh_engines_required")
    thresholds = {float(row["required_reduction"]) for row in selected}
    if len(thresholds) != 1:
        raise RuntimeError("build_repeat_threshold_mismatch")
    reductions = [float(row["latency_reduction"]) for row in selected]
    stability = build_repeat_stability(
        reductions,
        required_reduction=thresholds.pop(),
    )
    return {
        "candidate_id": candidate_id,
        "profile": profile,
        "independent_engine_count": 3,
        "engine_sha256s": hashes,
        "p50_ms": [float(row["p50_ms"]) for row in selected],
        **stability,
    }


def apply_search_admission(
    rows: Sequence[Mapping[str, Any]],
    build_repeat_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Apply the five evidence gates without promoting untested aliases."""

    from search.model_families.transformer.dh_power_alignment_4090 import (
        search_candidate_gate,
    )

    result = [dict(row) for row in rows]
    repeat_index = {
        (str(row["candidate_id"]), str(row["profile"])): row
        for row in build_repeat_rows
    }
    joint_rows = [row for row in result if row.get("structure_kind") == "joint"]
    for row in result:
        repeat = repeat_index.get((str(row["candidate_id"]), str(row["profile"])))
        row["build_repeat_stable"] = (
            bool(repeat["build_repeat_stable"]) if repeat is not None else None
        )
        joint_supported = False
        if row.get("structure_kind") == "single_family" and row.get("d_h") is not None:
            joint_supported = any(
                str(joint.get("model")) == str(row.get("model"))
                and str(joint.get("profile")) == str(row.get("profile"))
                and joint.get("accuracy_class") in {"SAFE", "BORDERLINE"}
                and bool(joint.get("latency_beneficial", False))
                and int(joint.get("target_d_h_by_family", {}).get(str(row["family"]), -1))
                == int(row["d_h"])
                for joint in joint_rows
            )
        row["joint_supported"] = joint_supported
        gate = search_candidate_gate(
            fixed500_acceptable=row.get("accuracy_class") == "SAFE",
            same_profile_latency=bool(row.get("latency_beneficial", False)),
            neighbor_advantage_passed=bool(
                row.get("neighbor_control_advantage", False)
            ),
            build_repeat_stable=bool(row.get("build_repeat_stable", False)),
            joint_supported=joint_supported,
        )
        row.update(gate)
    return result


def compact_evidence_record(row: Mapping[str, Any]) -> dict[str, Any]:
    required = (
        "candidate_id",
        "structure_hash",
        "structure_signature",
        "onnx_sha256",
        "engine_sha256",
        "requested_realized_conflict_count",
        "evaluated",
        "skipped",
    )
    missing = [key for key in required if row.get(key) in (None, "")]
    if missing:
        raise RuntimeError(f"compact_evidence_missing:{','.join(missing)}")
    if int(row["requested_realized_conflict_count"]) != 0:
        raise RuntimeError("compact_evidence_precision_conflict")
    return {key: row[key] for key in _COMPACT_FIELDS if key in row}


def _interaction_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    index = {
        (str(row.get("model")), str(row.get("structure_hash")), str(row.get("profile"))): row
        for row in rows
        if row.get("mAP") is not None
    }
    baselines = {
        (str(row.get("model")), str(row.get("profile"))): row
        for row in rows
        if row.get("structure_kind") == "baseline" and row.get("mAP") is not None
    }
    result: list[dict[str, Any]] = []
    for row in rows:
        profile = str(row.get("profile"))
        if profile == "P32" or row.get("mAP") is None:
            continue
        model = str(row.get("model"))
        structure_hash = str(row.get("structure_hash"))
        candidate_p32 = index.get((model, structure_hash, "P32"))
        baseline_p32 = baselines.get((model, "P32"))
        baseline_profile = baselines.get((model, profile))
        if not candidate_p32 or not baseline_p32 or not baseline_profile:
            continue
        metrics = precision_interaction(
            baseline_p32_map=float(baseline_p32["mAP"]),
            baseline_profile_map=float(baseline_profile["mAP"]),
            candidate_p32_map=float(candidate_p32["mAP"]),
            candidate_profile_map=float(row["mAP"]),
        )
        result.append(
            {
                "model": model,
                "family": row.get("family"),
                "candidate_id": row.get("candidate_id"),
                "structure_hash": structure_hash,
                "profile": profile,
                **metrics,
            }
        )
    return result


def _contract(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": "transformer-dh-power-alignment-4090-contract-v1",
        "execution_platform": "RTX4090_SM89",
        "models": {},
        "ga_executed": False,
        "greedy_executed": False,
        "full1789_executed": False,
    }
    for model in sorted({str(row.get("model")) for row in rows if row.get("model")}):
        selected = [row for row in rows if str(row.get("model")) == model]
        profiles: dict[str, dict[str, list[Any]]] = {}
        for profile in ("P32", "P16", "P8"):
            profile_rows = [row for row in selected if row.get("profile") == profile]
            profiles[profile] = {
                "allowed": [row.get("candidate_id") for row in profile_rows if row.get("search_space_candidate") is True],
                "experimental": [
                    row.get("candidate_id")
                    for row in profile_rows
                    if row.get("search_space_candidate") is not True
                    and row.get("accuracy_class") in {"SAFE", "BORDERLINE"}
                ],
                "rejected": [
                    row.get("candidate_id")
                    for row in profile_rows
                    if row.get("accuracy_class") in {"UNSAFE", "INVALID_EVALUATION"}
                ],
            }
        single = [row for row in selected if row.get("structure_kind") == "single_family"]
        joint = [row for row in selected if row.get("structure_kind") == "joint"]
        result["models"][model] = {
            "exact_power_widths_build_supported": sorted(
                {int(row["d_h"]) for row in single if row.get("exact_power_of_two") is True}
            ),
            "exact_power_widths_accuracy_safe": sorted(
                {
                    int(row["d_h"])
                    for row in single
                    if row.get("exact_power_of_two") is True
                    and row.get("accuracy_class") == "SAFE"
                }
            ),
            "exact_power_widths_same_profile_latency_beneficial": sorted(
                {
                    int(row["d_h"])
                    for row in single
                    if row.get("exact_power_of_two") is True
                    and row.get("latency_class") in {
                        "HIGH_ALIGNMENT_BENEFICIAL",
                        "ALIGNMENT_ADVANTAGE",
                    }
                }
            ),
            "multiple_of_8_latency_beneficial": sorted(
                {
                    int(row["d_h"])
                    for row in single
                    if row.get("divisible_by_8") is True
                    and row.get("latency_class") in {
                        "HIGH_ALIGNMENT_BENEFICIAL",
                        "ALIGNMENT_ADVANTAGE",
                    }
                }
            ),
            "multiple_of_16_latency_beneficial": sorted(
                {
                    int(row["d_h"])
                    for row in single
                    if row.get("divisible_by_16") is True
                    and row.get("latency_class") in {
                        "HIGH_ALIGNMENT_BENEFICIAL",
                        "ALIGNMENT_ADVANTAGE",
                    }
                }
            ),
            "multiple_of_32_latency_beneficial": sorted(
                {
                    int(row["d_h"])
                    for row in single
                    if row.get("divisible_by_32") is True
                    and row.get("latency_class") in {
                        "HIGH_ALIGNMENT_BENEFICIAL",
                        "ALIGNMENT_ADVANTAGE",
                    }
                }
            ),
            "joint_high_alignment_combinations": sorted(
                {
                    str(row["candidate_id"])
                    for row in joint
                    if row.get("accuracy_class") in {"SAFE", "BORDERLINE"}
                    and row.get("latency_class") in {
                        "HIGH_ALIGNMENT_BENEFICIAL",
                        "ALIGNMENT_ADVANTAGE",
                    }
                }
            ),
            "search_space_candidates": sorted(
                {
                    str(row["candidate_id"])
                    for row in selected
                    if row.get("search_space_candidate") is True
                }
            ),
            "profiles": profiles,
        }
    return result


def write_power_alignment_reports(
    output_root: Path,
    *,
    candidate_rows: Sequence[Mapping[str, Any]],
    single_family_rows: Sequence[Mapping[str, Any]],
    joint_rows: Sequence[Mapping[str, Any]],
    build_repeat_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    root = Path(output_root).resolve()
    single = [dict(row) for row in single_family_rows]
    joint = [dict(row) for row in joint_rows]
    all_rows = [*single, *joint]
    fixed_single = [row for row in single if int(row.get("evaluated", 0)) == 500]
    fixed_joint = [row for row in joint if int(row.get("evaluated", 0)) == 500]
    latency_single = [row for row in single if row.get("p50_ms") is not None]
    latency_joint = [row for row in joint if row.get("p50_ms") is not None]
    interactions = _interaction_rows(all_rows)

    outputs = {
        "power_alignment_candidate_manifest.csv": candidate_rows,
        "power_alignment_single_family_fixed500.csv": fixed_single,
        "power_alignment_single_family_latency.csv": latency_single,
        "power_alignment_joint_fixed500.csv": fixed_joint,
        "power_alignment_joint_latency.csv": latency_joint,
        "power_alignment_same_profile_speedup.csv": [row for row in all_rows if row.get("structure_speedup") is not None],
        "power_alignment_fp32_reference_speedup.csv": [row for row in all_rows if row.get("total_speedup") is not None],
        "power_alignment_neighbor_controls.csv": [row for row in all_rows if row.get("neighbor_control_speedup") is not None],
        "power_alignment_build_repeat_stability.csv": build_repeat_rows,
        "power_alignment_tactic_transitions.csv": [row for row in all_rows if row.get("tactic_signature")],
        "power_alignment_precision_interaction.csv": interactions,
    }
    for name, rows in outputs.items():
        _write_csv(root / name, list(rows))

    accuracy_boundary = {
        status: [row.get("candidate_id") for row in all_rows if row.get("accuracy_class") == status]
        for status in ("SAFE", "BORDERLINE", "UNSAFE", "INVALID_EVALUATION")
    }
    latency_boundary = {
        status: [row.get("candidate_id") for row in all_rows if row.get("latency_class") == status]
        for status in ("HIGH_ALIGNMENT_BENEFICIAL", "ALIGNMENT_ADVANTAGE", "TACTIC_UNSTABLE", "NO_BENEFIT")
    }
    contract = _contract(all_rows)
    _write_json(root / "power_alignment_accuracy_boundary.json", accuracy_boundary)
    _write_json(root / "power_alignment_latency_boundary.json", latency_boundary)
    _write_json(root / "power_alignment_contract.json", contract)
    summary = {
        "candidate_rows": len(candidate_rows),
        "single_family_rows": len(single),
        "joint_rows": len(joint),
        "single_family_fixed500_rows": len(fixed_single),
        "joint_fixed500_rows": len(fixed_joint),
        "formal_latency_rows": len(latency_single) + len(latency_joint),
        "search_space_candidates": sum(row.get("search_space_candidate") is True for row in all_rows),
        **result_cardinality(all_rows),
    }
    _write_json(root / "reports" / "report_summary.json", summary)
    allowed = sorted(
        (
            str(row["candidate_id"]),
            str(row["profile"]),
            float(row["mAP"]),
            float(row["p50_ms"]),
            float(row["structure_speedup"]),
        )
        for row in all_rows
        if row.get("search_space_candidate") is True
    )
    safe_beneficial = sorted(
        (
            str(row["model"]),
            str(row["candidate_id"]),
            str(row["profile"]),
            float(row["mAP"]),
            float(row["p50_ms"]),
            float(row["structure_speedup"]),
            bool(row.get("neighbor_control_advantage", False)),
        )
        for row in single
        if row.get("accuracy_class") == "SAFE"
        and row.get("latency_beneficial") is True
    )
    lines = [
        "# RTX 4090 Transformer d_h power-alignment conclusion",
        "",
        f"- candidate manifest rows: `{summary['candidate_rows']}`",
        f"- result alias rows: `{summary['result_alias_rows']}`",
        f"- unique physical structures: `{summary['unique_physical_structures']}`",
        f"- unique structure/precision phenotypes: `{summary['unique_phenotypes']}`",
        f"- fixed500 rows: `{summary['single_family_fixed500_rows'] + summary['joint_fixed500_rows']}`",
        f"- formal latency rows: `{summary['formal_latency_rows']}`",
        f"- independent fresh-build evidence rows: `{len(build_repeat_rows)}`",
        f"- SEARCH_SPACE_CANDIDATE rows: `{summary['search_space_candidates']}`",
        "- execution platform: `RTX4090_SM89`",
        "- P8 means SmoothQuant INT8 Q/K projection, not FP8.",
        "- GA/Greedy/full1789 were not executed by this experiment.",
        "",
        "## Allowed search candidates",
        "",
    ]
    if allowed:
        lines.extend(
            [
                "| candidate | profile | fixed500 mAP | p50 ms | same-profile speedup |",
                "|---|---:|---:|---:|---:|",
                *(
                    f"| {candidate} | {profile} | {map_value:.9f} | {p50:.6f} | {speedup:.6f}x |"
                    for candidate, profile, map_value, p50, speedup in allowed
                ),
            ]
        )
    else:
        lines.append("No candidate passed all five admission gates.")
    lines.extend(
        [
            "",
            "## Safe same-profile speedups",
            "",
            "| model | candidate | profile | fixed500 mAP | p50 ms | speedup | neighbor advantage |",
            "|---|---|---:|---:|---:|---:|---:|",
            *(
                f"| {model} | {candidate} | {profile} | {map_value:.9f} | {p50:.6f} | {speedup:.6f}x | {neighbor} |"
                for model, candidate, profile, map_value, p50, speedup, neighbor in safe_beneficial
            ),
            "",
            "## Interpretation",
            "",
            "- Physical d_h reduction can produce real same-profile full-engine speedup, but alignment is not a universal cause.",
            "- A width is an alignment advantage only when it also beats the available target-4 and target+4 controls.",
            "- Rows without three independent fresh builds remain experimental even when fixed500 and latency are favorable.",
            "- Detailed evidence and rejected profiles are recorded in `power_alignment_contract.json`.",
        ]
    )
    (root / "root_conclusion_power_alignment.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return {"summary": summary, "contract": contract}


__all__ = [
    "accuracy_class",
    "apply_search_admission",
    "build_repeat_stability",
    "compact_evidence_record",
    "precision_interaction",
    "result_cardinality",
    "summarize_build_repeat_evidence",
    "write_power_alignment_reports",
]
