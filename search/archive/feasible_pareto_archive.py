"""Feasibility-first archive with structure and precision diversity."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping, Sequence


class FeasibleParetoArchive:
    def __init__(self) -> None:
        self._records: dict[str, dict[str, Any]] = {}
        self._rejections: Counter[str] = Counter()

    @property
    def records(self) -> list[dict[str, Any]]:
        return [dict(self._records[key]) for key in sorted(self._records)]

    @property
    def rejection_counts(self) -> dict[str, int]:
        return dict(sorted(self._rejections.items()))

    @staticmethod
    def _failure_reason(row: Mapping[str, Any], active_budget: float) -> str:
        required_true = (
            "structure_legal",
            "precision_legal",
            "finite_joint_proxy",
        )
        for field in required_true:
            if not bool(row.get(field, False)):
                return f"{field}_false"
        if int(row.get("missing_mapping", 0)) != 0:
            return "missing_mapping_nonzero"
        bops = float(row.get("R_BOPS", float("inf")))
        if not math.isfinite(bops):
            return "R_BOPS_nonfinite"
        if bops > float(active_budget) + 1.0e-12:
            return "R_BOPS_above_active_budget"
        for field in ("S_task", "R_param"):
            if not math.isfinite(float(row.get(field, float("nan")))):
                return f"{field}_nonfinite"
        latency_proxy = row.get("latency_proxy_value", row.get("latency_proxy_ms"))
        if not math.isfinite(float(latency_proxy)):
            return "latency_proxy_nonfinite"
        return ""

    def add(self, row: Mapping[str, Any], *, active_budget: float) -> bool:
        payload = dict(row)
        phenotype_hash = str(payload.get("phenotype_hash", ""))
        if not phenotype_hash:
            self._rejections["phenotype_hash_missing"] += 1
            return False
        if phenotype_hash in self._records:
            self._rejections["duplicate_phenotype_hash"] += 1
            return False
        reason = self._failure_reason(payload, active_budget)
        if reason:
            self._rejections[reason] += 1
            return False
        self._records[phenotype_hash] = payload
        return True

    def select_stage2_candidates(self, count: int) -> list[dict[str, Any]]:
        """Prefer one candidate per structure before structure reuse."""

        target = max(0, int(count))
        ordered = sorted(
            self.records,
            key=lambda row: (
                float(row["R_BOPS"]),
                -float(row["S_task"]),
                float(row["R_param"]),
                float(row.get("latency_proxy_value", row.get("latency_proxy_ms"))),
                str(row["phenotype_hash"]),
            ),
        )
        selected: list[dict[str, Any]] = []
        used_structures: set[str] = set()
        used_precisions: set[str] = set()
        for prefer_new_precision in (True, False):
            for row in ordered:
                if row in selected:
                    continue
                structure = str(row.get("structure_hash", ""))
                precision = str(row.get("precision_hash", ""))
                if structure in used_structures:
                    continue
                if prefer_new_precision and precision in used_precisions:
                    continue
                selected.append(row)
                used_structures.add(structure)
                used_precisions.add(precision)
                if len(selected) >= target:
                    return selected
        for row in ordered:
            if row not in selected:
                selected.append(row)
                if len(selected) >= target:
                    break
        return selected

    def summary(self) -> dict[str, Any]:
        rows = self.records
        return {
            "feasible_phenotype_count": len(rows),
            "unique_structure_count": len(
                {str(row.get("structure_hash", "")) for row in rows}
            ),
            "unique_precision_count": len(
                {str(row.get("precision_hash", "")) for row in rows}
            ),
            "rejection_counts": self.rejection_counts,
        }
