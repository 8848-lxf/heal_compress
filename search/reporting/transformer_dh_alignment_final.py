"""Final evidence report and deployment contract for the H800 d_h experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

from search.orchestration.lidar_transformer_dh_phase_b import (
    PROFILES,
    _family_evidence,
    select_phase_b_candidates,
    summarize_phase_b,
)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _bool(value: Any) -> bool:
    return value is True or str(value).lower() in {"true", "1", "yes"}


def _csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size <= 2:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _git(workdir: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=workdir, text=True).strip()


def _phase_a_widths(output_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for model in ("lidar_cobevt", "lidar_v2xvit"):
        families = _read(
            output_root / "inventory" / f"{model.removeprefix('lidar_')}_attention_families.json"
        )
        result[model] = {}
        for family in families:
            evidence = _family_evidence(output_root, model, family)
            safe = [
                int(row["d_h"]) for row in evidence["widths"] if row["safe_all_profiles"]
            ]
            tested = [int(row["d_h"]) for row in evidence["widths"]]
            result[model][str(family["family_id"])] = {
                "build_supported_widths": tested,
                "accuracy_safe_all_profiles": safe,
                "contiguous_accuracy_safe_widths": evidence["continuous_safe_widths"],
                "accuracy_rejected_widths": [value for value in tested if value not in safe],
            }
    return result


def _latency_summary(rows: list[dict[str, Any]], *, phase: str) -> dict[str, Any]:
    canonical = [
        row
        for row in rows
        if not (
            _bool(row.get("baseline_replay"))
            and int(row.get("replay_index", 0)) == max(
                (
                    int(value.get("replay_index", 0))
                    for value in rows
                    if value.get("model") == row.get("model")
                    and value.get("profile") == row.get("profile")
                    and (
                        phase == "phase_b"
                        or value.get("attention_family") == row.get("attention_family")
                    )
                ),
                default=0,
            )
        )
    ]
    beneficial = [row for row in canonical if _bool(row.get("latency_beneficial"))]
    if phase == "phase_a":
        nonaligned = [row for row in canonical if row.get("alignment_class") == "non4"]
        strict = [
            row
            for row in nonaligned
            if _bool(row.get("latency_beneficial"))
            and _bool(row.get("beneficial_vs_control4"))
            and _bool(row.get("beneficial_vs_control8"))
        ]
        identity = lambda row: {
            "model": row["model"],
            "attention_family": row["attention_family"],
            "profile": row["profile"],
            "d_h": row["d_h"],
            "p50_ms": row["p50_ms"],
            "speedup": row["speedup_vs_same_profile_baseline"],
            "control4": row.get("control_4_d_h"),
            "speedup_vs_control4": row.get("speedup_vs_control4"),
            "control8": row.get("control_8_d_h"),
            "speedup_vs_control8": row.get("speedup_vs_control8"),
        }
    else:
        nonaligned = [
            row
            for row in canonical
            if any(int(width) % 4 for width in row.get("targets", {}).values())
        ]
        strict = [
            row
            for row in nonaligned
            if _bool(row.get("latency_beneficial"))
            and _bool(row.get("beneficial_vs_aligned_control"))
        ]
        identity = lambda row: {
            "model": row["model"],
            "profile": row["profile"],
            "candidate_id": row["candidate_id"],
            "targets": row["targets"],
            "p50_ms": row["p50_ms"],
            "speedup": row["speedup_vs_same_profile_baseline"],
            "alignment_control_id": row.get("alignment_control_id"),
            "speedup_vs_aligned_control": row.get("speedup_vs_aligned_control"),
        }
    return {
        "rows_including_replay": len(rows),
        "canonical_candidate_rows": len(canonical),
        "beneficial_vs_original_rows": len(beneficial),
        "nonaligned_rows": len(nonaligned),
        "strict_nonaligned_beneficial_rows": len(strict),
        "strict_nonaligned_beneficial": [identity(row) for row in strict],
        "gpu_uuids": sorted({str(row.get("gpu_uuid")) for row in rows}),
        "protocols": sorted(
            {
                (
                    int(row.get("warmup", 0)),
                    int(row.get("iterations", 0)),
                    int(row.get("repeats", 0)),
                )
                for row in rows
            }
        ),
    }


def _width_contract(
    widths: dict[str, Any],
    phase_a_latency: dict[str, Any],
    phase_b_latency: dict[str, Any],
    phase_b_rows: list[dict[str, str]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    single = phase_a_latency["strict_nonaligned_beneficial"]
    joint = phase_b_latency["strict_nonaligned_beneficial"]
    for model in ("lidar_cobevt", "lidar_v2xvit"):
        allowed_profiles = [row for row in joint if row["model"] == model]
        result[model] = {
            "families": {},
            "allowed_joint_profiles": allowed_profiles,
            "experimental_joint_profiles": [
                {
                    "candidate_id": row["candidate_id"],
                    "profile": row["profile"],
                    "targets": json.loads(row["targets"]),
                    "accuracy_status": row["accuracy_status"],
                }
                for row in phase_b_rows
                if row["model"] == model
                and row["accuracy_status"] == "SAFE"
                and not any(
                    value["candidate_id"] == row["candidate_id"]
                    and value["profile"] == row["profile"]
                    for value in allowed_profiles
                )
            ],
            "rejected_joint_profiles": [
                {
                    "candidate_id": row["candidate_id"],
                    "profile": row["profile"],
                    "targets": json.loads(row["targets"]),
                    "accuracy_status": row["accuracy_status"],
                }
                for row in phase_b_rows
                if row["model"] == model
                and row["accuracy_status"] in {"BORDERLINE", "UNSAFE"}
            ],
        }
        for family, value in widths[model].items():
            supported = sorted(
                {
                    int(row["d_h"])
                    for row in single
                    if row["model"] == model and row["attention_family"] == family
                },
                reverse=True,
            )
            allowed = sorted(
                {
                    int(profile["targets"][family])
                    for profile in allowed_profiles
                    if family in profile["targets"]
                },
                reverse=True,
            )
            safe = value["accuracy_safe_all_profiles"]
            result[model]["families"][family] = {
                **value,
                "single_family_latency_beneficial_widths": supported,
                "allowed_widths": allowed,
                "experimental_widths": [item for item in safe if item not in allowed],
                "rejected_widths": value["accuracy_rejected_widths"],
            }
    return result


def finalize(output_root: Path) -> dict[str, Any]:
    certificate = _read(output_root / "phase_a_completion_certificate.json")
    micro = _read(output_root / "microbenchmark" / "final_microbenchmark_acceptance.json")
    if not certificate or certificate.get("status") != "accepted_from_existing_artifacts":
        raise RuntimeError("final_report_requires_phase_a_certificate")
    if not micro or micro.get("status") != "accepted":
        raise RuntimeError("final_report_requires_microbenchmark_acceptance")
    # Refresh selection metadata after code fixes without executing candidates.
    for model in ("lidar_cobevt", "lidar_v2xvit"):
        selection = select_phase_b_candidates(output_root, model)
        result_path = output_root / "reports" / f"{model}_phase_b_result.json"
        result = _read(result_path)
        if not result:
            raise RuntimeError(f"final_report_phase_b_result_missing:{model}")
        result["selection"] = selection
        selection_by_id = {row["candidate_id"]: row for row in selection["joint_candidates"]}
        for row in result["results"]:
            preserved_result = row["result"]
            candidate_id = next(
                key
                for key, value in selection_by_id.items()
                if value["joint_id"] == preserved_result["structure"]["joint_id"]
            )
            row.clear()
            row.update(selection_by_id[candidate_id])
            row["result"] = preserved_result
        _write(result_path, result)
    phase_b_accuracy = summarize_phase_b(output_root)
    if not phase_b_accuracy.get("fixed500_complete"):
        raise RuntimeError("final_report_phase_b_fixed500_incomplete")
    phase_a_rows = _read(
        output_root / "formal-latency" / "phase_a" / "formal_latency.json"
    )
    phase_b_latency_rows = _read(
        output_root / "formal-latency" / "phase_b" / "formal_latency.json"
    )
    if not isinstance(phase_a_rows, list) or not phase_a_rows:
        raise RuntimeError("final_report_phase_a_formal_latency_missing")
    if not isinstance(phase_b_latency_rows, list) or not phase_b_latency_rows:
        raise RuntimeError("final_report_phase_b_formal_latency_missing")
    phase_a_latency = _latency_summary(phase_a_rows, phase="phase_a")
    phase_b_latency = _latency_summary(phase_b_latency_rows, phase="phase_b")
    phase_b_rows = _csv(output_root / "phase_b_joint_fixed500.csv")
    widths = _phase_a_widths(output_root)
    decisions = _width_contract(widths, phase_a_latency, phase_b_latency, phase_b_rows)
    workdir = Path(__file__).resolve().parents[2]
    head = _git(workdir, "rev-parse", "HEAD")
    branch = _git(workdir, "branch", "--show-current")
    formal_remote = _git(workdir, "rev-parse", "origin/feature/heal-unified-search-h800")
    formal_unchanged = formal_remote == "139d2c351889405c66fef05e420995663c91f08e"
    engine_hashes = [row["engine_sha256"] for row in phase_b_rows]
    structure_hashes = {row["structure_hash"] for row in phase_b_rows}
    p8_interactions = [
        float(row["interaction_joint"])
        for row in phase_b_rows
        if row["profile"] == "P8" and row["interaction_joint"]
    ]
    search_ready_by_model = {
        model: any(row["model"] == model for row in phase_b_latency["strict_nonaligned_beneficial"])
        for model in ("lidar_cobevt", "lidar_v2xvit")
    }
    contract = {
        "schema_version": "h800-transformer-dh-alignment-contract-v2",
        "evidence_branch": branch,
        "evidence_head": head,
        "formal_search_branch_modified": not formal_unchanged,
        "platform": {
            "gpu": "H800",
            "sm": "SM90",
            "tensorrt": "10.9.0.34",
            "portable": False,
        },
        "precision_contracts": {
            "P32": "Transformer attention contract with QK F32A32O32; not strict whole-engine FP32",
            "P16": "F3: FP16 projections/AV/Out/FFN with Q/K Cast to FP32 QK",
            "P8": "SQ1 Q/K projection INT8, DQ to FP32 QK; not native INT8 QK",
        },
        "phase_a": {
            "certificate": str(output_root / "phase_a_completion_certificate.json"),
            "complete": True,
            "formal_latency_complete": True,
            "formal_latency": phase_a_latency,
        },
        "phase_b": {
            "joint_candidates": 14,
            "structures": len(structure_hashes),
            "engines": len(engine_hashes),
            "unique_engine_hashes": len(set(engine_hashes)),
            "fixed500_complete": True,
            "accuracy": phase_b_accuracy,
            "formal_latency_complete": True,
            "formal_latency": phase_b_latency,
        },
        "cobevt": decisions["lidar_cobevt"],
        "v2xvit": decisions["lidar_v2xvit"],
        "alignment_conclusion": {
            "multiple_of_8_required_for_build": False,
            "multiple_of_4_required_for_build": False,
            "arbitrary_integer_accuracy_supported": False,
            "arbitrary_integer_single_family_latency_supported": (
                phase_a_latency["strict_nonaligned_beneficial_rows"] > 0
            ),
            "arbitrary_integer_joint_latency_supported": (
                phase_b_latency["strict_nonaligned_beneficial_rows"] > 0
            ),
            "search_ready_by_model": search_ready_by_model,
            "search_ready": all(search_ready_by_model.values()),
        },
        "limitations": {
            "full1789_executed": False,
            "ga_executed": False,
            "greedy_executed": False,
            "attempt_lineage_complete": False,
            "cross_platform_portability": False,
        },
    }
    _write(output_root / "transformer_dh_alignment_contract.json", contract)
    _write(output_root / "reports" / "transformer_dh_alignment_contract.json", contract)
    summary = {
        "schema_version": "h800-transformer-dh-final-report-v2",
        "experiment_branch": branch,
        "head": head,
        "formal_search_branch_head": formal_remote,
        "formal_search_branch_unchanged": formal_unchanged,
        "phase_a_certificate": certificate,
        "microbenchmark": micro,
        "phase_a_formal_latency": phase_a_latency,
        "phase_b": {
            "candidate_count": 14,
            "structure_count": len(structure_hashes),
            "engine_count": len(engine_hashes),
            "unique_engine_hashes": len(set(engine_hashes)),
            "fixed500": phase_b_accuracy,
            "p8_interaction_range": [min(p8_interactions), max(p8_interactions)] if p8_interactions else None,
            "formal_latency": phase_b_latency,
        },
        "decision": contract["alignment_conclusion"],
        "full1789_executed": False,
        "formal_search_code_migrated": False,
        "pyramid_modified": False,
        "ga_or_greedy_executed": False,
    }
    _write(output_root / "reports" / "report_summary.json", summary)
    _write_reports(output_root, summary, contract, phase_b_rows)
    return summary


def _write_reports(
    output_root: Path,
    summary: Mapping[str, Any],
    contract: Mapping[str, Any],
    phase_b_rows: list[dict[str, str]],
) -> None:
    reports = output_root / "reports"
    old_root = output_root / "root_conclusion.md"
    superseded = reports / "root_conclusion_superseded_phase_a_only.md"
    if old_root.is_file() and not superseded.is_file():
        superseded.write_text(old_root.read_text(encoding="utf-8"), encoding="utf-8")
    certificate = summary["phase_a_certificate"]
    micro = summary["microbenchmark"]
    phase_a = summary["phase_a_formal_latency"]
    phase_b = summary["phase_b"]
    selections = _read(output_root / "phase_b_selection.json")["models"]
    selection_rows = [
        {
            "model": model["model"],
            "candidate_id": candidate["candidate_id"],
            "targets": candidate["targets"],
            "diagnostic": candidate["diagnostic"],
        }
        for model in selections
        for candidate in model["joint_candidates"]
    ]
    material_negative = [
        row
        for row in phase_b_rows
        if row.get("interaction_joint")
        and float(row["interaction_joint"]) < -0.003
    ]
    sq1_negative = [row for row in material_negative if row["profile"] == "P8"]
    (reports / "phase_a_completion_summary.md").write_text(
        "# Phase-A completion\n\n"
        f"Existing artifacts were accepted: {certificate['structures']['accepted']}/110 structures, "
        f"{certificate['engines']['accepted']}/330 engines, and "
        f"{certificate['fixed500']['accepted']}/330 fixed500 rows.\n\n"
        "All fixed500 rows contain 500 evaluated frames and zero skips; precision conflicts and "
        "fallbacks are zero. Attempt lineage remains incomplete because old retries did not record "
        "attempt IDs.\n",
        encoding="utf-8",
    )
    (reports / "phase_a_formal_latency.md").write_text(
        "# Phase-A selected formal latency\n\n"
        f"Canonical selected rows: {phase_a['canonical_candidate_rows']}; non-aligned rows: "
        f"{phase_a['nonaligned_rows']}; non-aligned rows faster than the same-profile original, "
        f"nearest 4-aligned control, and nearest 8-aligned control: "
        f"{phase_a['strict_nonaligned_beneficial_rows']}.\n\n"
        f"Strict beneficial candidates:\n\n```json\n{json.dumps(phase_a['strict_nonaligned_beneficial'], indent=2)}\n```\n",
        encoding="utf-8",
    )
    (reports / "phase_b_joint_validation.md").write_text(
        "# Phase-B joint validation\n\n"
        f"Fourteen fresh joint structures and {phase_b['engine_count']} three-profile engines were "
        f"validated. fixed500 accepted {phase_b['fixed500']['accepted_rows']}/42 rows: "
        f"SAFE={phase_b['fixed500']['safe_rows']}, "
        f"BORDERLINE={phase_b['fixed500']['borderline_rows']}, "
        f"UNSAFE={phase_b['fixed500']['unsafe_rows']}.\n\n"
        f"Joint interaction range: {phase_b['fixed500']['interaction_range']}. Additive Phase-A "
        "deltas are predictions only; classifications use measured joint mAP. "
        f"Material negative interactions below -0.003: {len(material_negative)}, including "
        f"SQ1: {len(sq1_negative)}.\n\n"
        f"Candidate matrix:\n\n```json\n{json.dumps(selection_rows, indent=2)}\n```\n",
        encoding="utf-8",
    )
    (reports / "phase_b_formal_latency.md").write_text(
        "# Phase-B formal latency\n\n"
        f"Canonical joint rows: {phase_b['formal_latency']['canonical_candidate_rows']}; "
        f"strict non-aligned beneficial rows: "
        f"{phase_b['formal_latency']['strict_nonaligned_beneficial_rows']}.\n\n"
        f"Strict beneficial candidates:\n\n```json\n{json.dumps(phase_b['formal_latency']['strict_nonaligned_beneficial'], indent=2)}\n```\n",
        encoding="utf-8",
    )
    final_lines = [
        "# Final H800 Transformer d_h alignment conclusion",
        "",
        f"- Branch: `{summary['experiment_branch']}` at `{summary['head']}`.",
        f"- Formal search branch unchanged: `{summary['formal_search_branch_unchanged']}`.",
        "- Phase-A accepted from existing artifacts; no Phase-A structure/build/fixed500 was rerun.",
        "- Attempt lineage is incomplete because the historical interrupted retries did not preserve attempt IDs.",
        f"- Microbenchmark accepted/superseded: {micro['accepted_rows']}/{micro['superseded_invalid_rows']}; all accepted output hashes are complete.",
        f"- Phase-A formal selected canonical rows: {phase_a['canonical_candidate_rows']}.",
        f"- Phase-A strict non-aligned full-engine beneficial rows: {phase_a['strict_nonaligned_beneficial_rows']}.",
        f"- Phase-A strict candidates and aligned-control comparisons: `{json.dumps(phase_a['strict_nonaligned_beneficial'], sort_keys=True)}`.",
        f"- Phase-B candidate matrix: `{json.dumps(selection_rows, sort_keys=True)}`.",
        f"- Phase-B fresh structures/engines: {phase_b['structure_count']}/{phase_b['engine_count']}.",
        f"- Phase-B fixed500 SAFE/BORDERLINE/UNSAFE: {phase_b['fixed500']['safe_rows']}/{phase_b['fixed500']['borderline_rows']}/{phase_b['fixed500']['unsafe_rows']}.",
        f"- Phase-B joint interaction range: {phase_b['fixed500']['interaction_range']}.",
        f"- Material negative joint interactions below -0.003: {len(material_negative)}; SQ1 subset: {len(sq1_negative)}.",
        f"- SQ1 joint interaction range: {phase_b['p8_interaction_range']}.",
        f"- Phase-B strict non-aligned full-engine beneficial rows: {phase_b['formal_latency']['strict_nonaligned_beneficial_rows']}.",
        f"- Phase-B strict candidates and aligned-control comparisons: `{json.dumps(phase_b['formal_latency']['strict_nonaligned_beneficial'], sort_keys=True)}`.",
        f"- CoBEVT search-ready: {contract['alignment_conclusion']['search_ready_by_model']['lidar_cobevt']}.",
        f"- V2XViT search-ready: {contract['alignment_conclusion']['search_ready_by_model']['lidar_v2xvit']}.",
        f"- Overall search-ready: {contract['alignment_conclusion']['search_ready']}.",
        f"- CoBEVT allowed/experimental/rejected evidence: `{json.dumps(contract['cobevt'], sort_keys=True)}`.",
        f"- V2XViT allowed/experimental/rejected evidence: `{json.dumps(contract['v2xvit'], sort_keys=True)}`.",
        "- P32 is a Transformer attention FP32 contract, not strict whole-engine FP32.",
        "- P16 keeps QK F32A32O32 and is not strict full-Attention FP16.",
        "- P8 is Q/K projection SQ INT8 followed by DQ to FP32 QK; it is not native INT8 QK.",
        "- full1789 was not executed; the accuracy evidence is fixed500 only.",
        "- No GA, Greedy, Pyramid modification, formal-search migration, merge, or cherry-pick occurred.",
        "- Any later migration must be a separate review of generic acceptance, scheduler-state, joint-hash, cache-isolation, and latency-control modules.",
        "",
        "Allowed, experimental, and rejected widths/combinations are frozen in `transformer_dh_alignment_contract.json`.",
    ]
    final_text = "\n".join(final_lines) + "\n"
    (reports / "final_dh_alignment_conclusion.md").write_text(final_text, encoding="utf-8")
    old_root.write_text(final_text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args(argv)
    result = finalize(Path(args.output_root).resolve())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
