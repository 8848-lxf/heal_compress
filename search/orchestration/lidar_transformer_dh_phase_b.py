"""Evidence-gated Phase-B joint-family candidates after complete Phase-A sweeps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any, Mapping

from search.model_families.transformer.dh_candidate_grid import dense_head_dimension_grid
from search.orchestration.lidar_transformer_dh_joint import run_joint


PROFILES = ("P32", "P16", "P8")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _choose_family_widths(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
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


def select_phase_b_candidates(output_root: Path, model: str) -> dict[str, Any]:
    families = _read(
        output_root / "inventory" / f"{model.removeprefix('lidar_')}_attention_families.json"
    )
    if not isinstance(families, list) or len(families) < 2:
        raise RuntimeError(f"phase_b_requires_multiple_families:{model}")
    family_evidence: dict[str, Any] = {}
    selections: dict[str, dict[str, int]] = {
        "aligned_safe": {},
        "nonaligned_safe": {},
        "provisional_latency": {},
    }
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
        baselines: dict[str, dict[str, Any]] = {}
        for profile in PROFILES:
            path = (
                output_root / "engines" / model / family_id / f"dh_{d0:03d}"
                / profile / "evaluation" / "fixed500" / "evaluation_acceptance.json"
            )
            baselines[profile] = _read(path)
            if baselines[profile].get("status") != "ok":
                raise RuntimeError(
                    f"phase_b_missing_same_profile_d0_fixed500:{model}:{family_id}:{profile}"
                )
        width_rows: list[dict[str, Any]] = []
        for d_h in widths:
            profiles: dict[str, Any] = {}
            safe = True
            speedups: list[float] = []
            for profile in PROFILES:
                directory = (
                    output_root / "engines" / model / family_id / f"dh_{d_h:03d}" / profile
                )
                build = _read(directory / "baseline_result.json")
                evaluation = _read(
                    directory / "evaluation" / "fixed500" / "evaluation_acceptance.json"
                )
                alignment = _read(directory / "engine_alignment_audit.json")
                exact = (
                    build.get("status") == "ok"
                    and int(build.get("requested_realized_conflict_count", -1)) == 0
                    and alignment.get("padding_status")
                    in {"EXACT_ALIGNED", "EXACT_NONALIGNED", "INTERNAL_PADDED"}
                    and not bool(alignment.get("fallback_hint"))
                    and evaluation.get("status") == "ok"
                )
                baseline_map = float(baselines[profile]["mAP"])
                candidate_map = evaluation.get("mAP")
                delta = float(candidate_map) - baseline_map if candidate_map is not None else None
                safe &= exact and delta is not None and abs(delta) <= 0.003
                baseline_p50 = baselines[profile].get("forward_p50_ms")
                candidate_p50 = evaluation.get("forward_p50_ms")
                if baseline_p50 and candidate_p50:
                    speedups.append(float(baseline_p50) / float(candidate_p50))
                profiles[profile] = {
                    "exact": exact,
                    "fixed500_mAP": candidate_map,
                    "delta_mAP_structure": delta,
                    "forward_p50_ms_selection_only": candidate_p50,
                }
            width_rows.append(
                {
                    "d_h": d_h,
                    "exact_all_profiles": all(
                        bool(value["exact"]) for value in profiles.values()
                    ),
                    "safe_all_profiles": safe,
                    "provisional_speedup_median": statistics.median(speedups) if speedups else None,
                    "profiles": profiles,
                }
            )
        chosen = _choose_family_widths(width_rows)
        family_evidence[family_id] = {
            "original_d_h": d0,
            "widths": width_rows,
            "selection": chosen,
        }
        for label in selections:
            selections[label][family_id] = int(chosen[label])
    joint_candidates: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, int], ...]] = set()
    for label, targets in selections.items():
        signature = tuple(sorted((name, int(width)) for name, width in targets.items()))
        if signature in seen:
            continue
        seen.add(signature)
        joint_candidates.append(
            {
                "selection_label": label,
                "targets": dict(signature),
                "formal_latency_used_for_selection": False,
                "selection_latency_source": "fixed500_forward_p50_selection_only",
            }
        )
    result = {
        "model": model,
        "phase_a_complete": True,
        "accuracy_gate": "all P32/P16/P8 have abs(same-profile fixed500 delta_mAP)<=0.003",
        "family_evidence": family_evidence,
        "joint_candidates": joint_candidates,
    }
    _write(output_root / "reports" / f"{model}_phase_b_selection.json", result)
    return result


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
    results: list[dict[str, Any]] = []
    for candidate in selection["joint_candidates"]:
        targets = {str(name): int(width) for name, width in candidate["targets"].items()}
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
    payload = {"selection": selection, "results": results}
    _write(output_root / "reports" / f"{model}_phase_b_result.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--model", choices=("lidar_cobevt", "lidar_v2xvit"), required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--profiles", default="P32,P16,P8")
    parser.add_argument("--protocols", default="smoke10,fixed50,fixed500")
    args = parser.parse_args(argv)
    result = run(
        output_root=Path(args.output_root).resolve(),
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
