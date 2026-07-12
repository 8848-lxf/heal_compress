"""Full validation manifest and baseline comparison helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..hashing import canonical_json_hash


def write_full_validation_manifest(
    path: str | Path,
    *,
    frame_ids: Sequence[str],
    dataset_split: str,
    dataset_config_hash: str,
    postprocess_config: Mapping[str, Any],
    skip_policy: str,
    fixed_k: int = 29696,
) -> dict[str, Any]:
    """Persist a deterministic complete-validation manifest."""

    rows = [str(frame_id) for frame_id in frame_ids]
    payload = {
        "schema_version": "search-full-val-manifest-v1",
        "dataset_split": str(dataset_split),
        "total_manifest_frames": len(rows),
        "frame_ids": rows,
        "frame_order": "as_provided",
        "dataset_config_hash": str(dataset_config_hash),
        "postprocess_config": dict(postprocess_config),
        "fixed_k": int(fixed_k),
        "skip_policy": str(skip_policy),
    }
    payload["manifest_hash"] = canonical_json_hash(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def compute_common_evaluated_subset(results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Return the frame-id intersection evaluated by every baseline/candidate."""

    evaluated_sets = {
        str(name): {str(frame_id) for frame_id in payload.get("evaluated_frame_ids", [])}
        for name, payload in results.items()
    }
    common = sorted(set.intersection(*evaluated_sets.values())) if evaluated_sets else []
    skip_reason_counts: dict[str, int] = {}
    for payload in results.values():
        for reason, count in dict(payload.get("skip_reason_counts", {}) or {}).items():
            skip_reason_counts[str(reason)] = skip_reason_counts.get(str(reason), 0) + int(count)
    return {
        "models": sorted(str(name) for name in results),
        "common_frame_ids": common,
        "common_subset_frames": len(common),
        "skip_sets_comparable": bool(results) and all("evaluated_frame_ids" in payload for payload in results.values()),
        "per_model_evaluated_frames": {name: len(values) for name, values in sorted(evaluated_sets.items())},
        "skip_reason_counts": skip_reason_counts,
    }
