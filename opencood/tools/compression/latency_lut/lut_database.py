from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .interpolation import linear_interpolate, nearest_channel_record
from .key_builder import BOUNDARY_BLOCK_TYPE
from .schema import LatencyLUTKey, LatencyRecord, read_record_jsonl, write_jsonl, write_records_csv


@dataclass
class LatencyEstimateItem:
    key: LatencyLUTKey
    latency_ms: float
    uncertainty_ms: float
    match_type: str
    matched_key_hash: str | None = None
    record: LatencyRecord | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key.to_dict(),
            "latency_ms": float(self.latency_ms),
            "uncertainty_ms": float(self.uncertainty_ms),
            "match_type": self.match_type,
            "matched_key_hash": self.matched_key_hash,
            "note": self.note,
        }


class LatencyLUTDatabase:
    def __init__(
        self,
        records: list[LatencyRecord] | None = None,
        *,
        default_latency_ms: float = 1.0,
        default_uncertainty_ms: float = 0.5,
        default_boundary_penalty_ms: float = 0.02,
        allow_linear_interpolation: bool = True,
    ) -> None:
        self.records_by_hash: dict[str, LatencyRecord] = {}
        self.default_latency_ms = float(default_latency_ms)
        self.default_uncertainty_ms = float(default_uncertainty_ms)
        self.default_boundary_penalty_ms = float(default_boundary_penalty_ms)
        self.allow_linear_interpolation = bool(allow_linear_interpolation)
        for record in records or []:
            self.add_record(record)

    def add_record(self, record: LatencyRecord) -> None:
        if record.status not in {"success", "ok"}:
            return
        self.records_by_hash[record.key_hash] = record

    @classmethod
    def from_jsonl(cls, path: str | Path, **kwargs: Any) -> "LatencyLUTDatabase":
        return cls(read_record_jsonl(path), **kwargs)

    def to_jsonl(self, path: str | Path) -> None:
        write_jsonl(list(self.records_by_hash.values()), path)

    def to_csv(self, path: str | Path) -> None:
        write_records_csv(list(self.records_by_hash.values()), path)

    def lookup_exact(self, key: LatencyLUTKey) -> LatencyEstimateItem | None:
        if key.block_type == BOUNDARY_BLOCK_TYPE and key.src_precision == key.dst_precision:
            return LatencyEstimateItem(key, 0.0, 0.0, "exact", note="same_precision_boundary")
        record = self.records_by_hash.get(key.stable_hash())
        if record is None:
            return None
        return LatencyEstimateItem(
            key=key,
            latency_ms=float(record.latency_p50_ms),
            uncertainty_ms=float(record.latency_std_ms),
            match_type="exact",
            matched_key_hash=record.key_hash,
            record=record,
        )

    def _compatible_records(self, key: LatencyLUTKey) -> list[LatencyRecord]:
        return [
            record for record in self.records_by_hash.values()
            if record.key.comparable_tuple() == key.comparable_tuple()
        ]

    def _same_family_records(self, key: LatencyLUTKey) -> list[LatencyRecord]:
        return [
            record for record in self.records_by_hash.values()
            if record.key.deploy_mode == key.deploy_mode
            and record.key.fixed_K == key.fixed_K
            and record.key.module_name == key.module_name
            and record.key.block_name == key.block_name
            and record.key.block_type == key.block_type
            and record.key.H == key.H
            and record.key.W == key.W
            and record.key.kernel_size == key.kernel_size
            and record.key.stride == key.stride
            and record.key.precision_profile == key.precision_profile
            and record.key.plugin_flag == key.plugin_flag
            and record.key.plugin_name == key.plugin_name
        ]

    def lookup_nearest(self, key: LatencyLUTKey) -> LatencyEstimateItem:
        exact = self.lookup_exact(key)
        if exact is not None:
            return exact
        if key.block_type == BOUNDARY_BLOCK_TYPE:
            candidates = [
                record for record in self.records_by_hash.values()
                if record.key.block_type == BOUNDARY_BLOCK_TYPE
                and record.key.deploy_mode == key.deploy_mode
                and record.key.fixed_K == key.fixed_K
                and record.key.src_precision == key.src_precision
                and record.key.dst_precision == key.dst_precision
                and record.key.H == key.H
                and record.key.W == key.W
            ]
            nearest = nearest_channel_record(key, candidates)
            if nearest is not None:
                dist = abs((key.C_in or 0) - (nearest.key.C_in or 0))
                return LatencyEstimateItem(
                    key=key,
                    latency_ms=float(nearest.latency_p50_ms),
                    uncertainty_ms=float(nearest.latency_std_ms) + 0.001 * float(dist),
                    match_type="nearest",
                    matched_key_hash=nearest.key_hash,
                    record=nearest,
                    note="nearest_boundary_record",
                )
            return LatencyEstimateItem(
                key=key,
                latency_ms=0.0 if key.src_precision == key.dst_precision else self.default_boundary_penalty_ms,
                uncertainty_ms=0.0 if key.src_precision == key.dst_precision else self.default_boundary_penalty_ms,
                match_type="default",
                note="default_boundary_penalty",
            )
        if key.plugin_flag or key.block_type == "plugin":
            return LatencyEstimateItem(
                key=key,
                latency_ms=self.default_latency_ms,
                uncertainty_ms=max(10.0, self.default_uncertainty_ms * 20.0),
                match_type="unavailable",
                note="plugin_lut_record_missing",
            )
        if key.precision_profile == "TRT_INT8_QDQ":
            return LatencyEstimateItem(
                key=key,
                latency_ms=self.default_latency_ms,
                uncertainty_ms=max(10.0, self.default_uncertainty_ms * 20.0),
                match_type="unavailable",
                note="int8_qdq_lut_record_missing",
            )
        candidates = self._same_family_records(key)
        nearest = nearest_channel_record(key, candidates)
        if nearest is None:
            return LatencyEstimateItem(
                key=key,
                latency_ms=self.default_latency_ms,
                uncertainty_ms=self.default_uncertainty_ms,
                match_type="default",
                note="no_compatible_lut_record",
            )
        dist = abs((key.C_in or 0) - (nearest.key.C_in or 0)) + abs((key.C_out or 0) - (nearest.key.C_out or 0))
        uncertainty = float(nearest.latency_std_ms) + 0.001 * float(dist)
        return LatencyEstimateItem(
            key=key,
            latency_ms=float(nearest.latency_p50_ms),
            uncertainty_ms=uncertainty,
            match_type="nearest",
            matched_key_hash=nearest.key_hash,
            record=nearest,
        )

    def interpolate(self, key: LatencyLUTKey) -> LatencyEstimateItem:
        exact = self.lookup_exact(key)
        if exact is not None:
            return exact
        if key.block_type == BOUNDARY_BLOCK_TYPE or not self.allow_linear_interpolation:
            return self.lookup_nearest(key)
        candidates = sorted(
            self._same_family_records(key),
            key=lambda rec: ((rec.key.C_out or 0) + (rec.key.C_in or 0)),
        )
        target = float((key.C_in or 0) + (key.C_out or 0))
        lower = [rec for rec in candidates if float((rec.key.C_in or 0) + (rec.key.C_out or 0)) <= target]
        upper = [rec for rec in candidates if float((rec.key.C_in or 0) + (rec.key.C_out or 0)) >= target]
        if not lower or not upper:
            return self.lookup_nearest(key)
        low = lower[-1]
        high = upper[0]
        if low.key_hash == high.key_hash:
            return self.lookup_nearest(key)
        low_x = float((low.key.C_in or 0) + (low.key.C_out or 0))
        high_x = float((high.key.C_in or 0) + (high.key.C_out or 0))
        latency = linear_interpolate(target, low_x, low.latency_p50_ms, high_x, high.latency_p50_ms)
        uncertainty = max(float(low.latency_std_ms), float(high.latency_std_ms), abs(float(high.latency_p50_ms) - float(low.latency_p50_ms)) * 0.1)
        return LatencyEstimateItem(
            key=key,
            latency_ms=latency,
            uncertainty_ms=uncertainty,
            match_type="interpolate",
            matched_key_hash=f"{low.key_hash},{high.key_hash}",
        )

    def estimate_key(self, key: LatencyLUTKey, *, prefer_interpolation: bool = True) -> LatencyEstimateItem:
        exact = self.lookup_exact(key)
        if exact is not None:
            return exact
        if prefer_interpolation:
            return self.interpolate(key)
        return self.lookup_nearest(key)
