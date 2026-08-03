"""Fail-closed Stage-2 accuracy admission relative to a real greedy anchor."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


MAX_ACCURACY_TOLERANCE = 0.005


@dataclass(frozen=True)
class GreedyAnchorGatePolicy:
    tolerance: float = MAX_ACCURACY_TOLERANCE

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.tolerance) <= MAX_ACCURACY_TOLERANCE:
            raise ValueError(
                "greedy_anchor_accuracy_tolerance_must_be_in_closed_interval_0_0_005"
            )


def _target_key(target: float) -> str:
    return f"{float(target):.6f}"


def build_greedy_anchor_manifest(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    anchors: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        for target in row.get("budgets", ()):
            key = _target_key(float(target))
            candidate = {
                "candidate_hash": str(row.get("candidate_hash", "")),
                "mAP": float(row["mAP"]),
                "p50_ms": float(
                    row.get("p50_ms", row.get("forward_p50_ms", float("inf")))
                ),
                "engine_path": str(row.get("engine_path", "")),
                "artifact_dir": str(row.get("artifact_dir", "")),
            }
            incumbent = anchors.get(key)
            if incumbent is None or (
                -candidate["mAP"], candidate["p50_ms"], candidate["candidate_hash"]
            ) < (
                -float(incumbent["mAP"]),
                float(incumbent["p50_ms"]),
                str(incumbent["candidate_hash"]),
            ):
                anchors[key] = candidate
    return {
        "schema_version": "heal-greedy-stage2-anchor-v1",
        "accuracy_tolerance_max": MAX_ACCURACY_TOLERANCE,
        "anchors": anchors,
    }


def load_greedy_anchor(
    manifest_path: str | Path,
    *,
    target: float,
) -> dict[str, Any]:
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"greedy_anchor_manifest_missing:{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    anchors = dict(payload.get("anchors", {}) or {})
    key = _target_key(target)
    if key not in anchors:
        raise KeyError(f"greedy_anchor_target_missing:{key}:{path}")
    anchor = dict(anchors[key])
    if not math.isfinite(float(anchor.get("mAP", float("nan")))):
        raise ValueError(f"greedy_anchor_map_invalid:{key}:{path}")
    return anchor


def apply_greedy_anchor_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    anchor: Mapping[str, Any],
    policy: GreedyAnchorGatePolicy | None = None,
) -> list[dict[str, Any]]:
    config = policy or GreedyAnchorGatePolicy()
    anchor_map = float(anchor["mAP"])
    threshold = anchor_map - float(config.tolerance)
    result: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        candidate_map = float(row.get("mAP", float("-inf")))
        passed = math.isfinite(candidate_map) and candidate_map >= threshold
        row.update(
            {
                "greedy_anchor_candidate_hash": str(anchor.get("candidate_hash", "")),
                "greedy_anchor_mAP": anchor_map,
                "greedy_anchor_accuracy_tolerance": float(config.tolerance),
                "greedy_anchor_accuracy_threshold": threshold,
                "accuracy_gate_passed": passed,
                "stage2_eligible": bool(
                    passed and str(row.get("status", "")) == "ok"
                ),
            }
        )
        if not passed:
            row["accuracy_gate_rejection"] = (
                f"candidate_mAP_{candidate_map:.9f}_below_greedy_anchor_minus_"
                f"{float(config.tolerance):.6f}_{threshold:.9f}"
            )
        result.append(row)
    return result


def gate_rows_from_search_config(
    rows: Sequence[Mapping[str, Any]],
    *,
    search_config: Mapping[str, Any],
    round_index: int,
) -> list[dict[str, Any]]:
    stage2 = dict(search_config.get("stage2", {}) or {})
    gate = dict(stage2.get("greedy_anchor_accuracy_gate", {}) or {})
    if not bool(gate.get("enabled", False)):
        return [dict(row) for row in rows]
    targets = [
        float(value)
        for value in dict(search_config.get("search", {}) or {}).get(
            "bops_targets", ()
        )
    ]
    if not targets or not 0 <= int(round_index) < len(targets):
        raise RuntimeError(f"greedy_anchor_target_unavailable_for_round:{round_index}")
    manifest = gate.get("anchor_manifest")
    if not manifest:
        raise RuntimeError("greedy_anchor_manifest_required_for_stage2_gate")
    anchor = load_greedy_anchor(manifest, target=targets[int(round_index)])
    return apply_greedy_anchor_gate(
        rows,
        anchor=anchor,
        policy=GreedyAnchorGatePolicy(float(gate.get("tolerance", 0.005))),
    )


__all__ = [
    "MAX_ACCURACY_TOLERANCE",
    "GreedyAnchorGatePolicy",
    "apply_greedy_anchor_gate",
    "build_greedy_anchor_manifest",
    "gate_rows_from_search_config",
    "load_greedy_anchor",
]
