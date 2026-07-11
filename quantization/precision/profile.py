"""Deterministic formal FP16/INT8 precision profiles."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence

from ..config import PrecisionProfileConfig
from ..exceptions import CanonicalMappingError
from ..types import PrecisionAssignment, PrecisionProfileResult


def _descriptor(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return {"module_path": value}
    if isinstance(value, Mapping):
        row = dict(value)
    elif hasattr(value, "to_dict"):
        row = dict(value.to_dict())
    else:
        row = dict(vars(value))
    path = str(row.get("module_path") or row.get("canonical_module_name") or row.get("name") or "")
    if not path:
        raise CanonicalMappingError("precision profile module has no stable path")
    row["module_path"] = path
    return row


def generate_precision_profile(
    modules: Sequence[str | Mapping[str, Any] | Any],
    *,
    profile_id: str = "profile_000",
    config: PrecisionProfileConfig | None = None,
) -> PrecisionProfileResult:
    """Generate profile_000/001/002/003 as 0/20/50/80% INT8 requests."""

    policy = config or PrecisionProfileConfig()
    strict_precision = {
        "strict_fp32": "fp32",
        "strict_fp16": "fp16",
        "strict_int8": "int8",
    }.get(str(profile_id))
    ratio = (1.0 if strict_precision == "int8" else 0.0) if strict_precision else policy.ratio_for(profile_id)
    rows = sorted((_descriptor(value) for value in modules), key=lambda row: row["module_path"])
    names = [str(row["module_path"]) for row in rows]
    if len(names) != len(set(names)):
        raise CanonicalMappingError("precision profile contains duplicate module paths")
    eligible = [
        row
        for row in rows
        if not any(pattern.lower() in str(row["module_path"]).lower() for pattern in policy.protected_fp16_patterns)
        and not bool(row.get("protected_fp16", False))
    ]
    target = 0 if ratio <= 0.0 or not rows else max(1, int(math.floor(len(rows) * ratio + 0.5)))
    target = min(target, len(eligible))
    ranked = sorted(
        eligible,
        key=lambda row: (
            hashlib.sha256(f"{policy.seed}|{profile_id}|{row['module_path']}".encode("utf-8")).hexdigest(),
            row["module_path"],
        ),
    )
    selected = {str(row["module_path"]) for row in ranked[:target]}
    assignments: list[PrecisionAssignment] = []
    for ordering, row in enumerate(rows):
        module_path = str(row["module_path"])
        protected = module_path not in {str(item["module_path"]) for item in eligible}
        if strict_precision in {"fp32", "fp16"}:
            requested_precision = strict_precision
            protected_precision = strict_precision if protected else ""
        elif strict_precision == "int8":
            requested_precision = "fp16" if protected else "int8"
            protected_precision = "fp16" if protected else ""
        else:
            requested_precision = "int8" if module_path in selected else "fp16"
            protected_precision = "fp16" if protected else ""
        assignments.append(
            PrecisionAssignment(
                module_path=module_path,
                precision_group=str(row.get("precision_group") or row.get("precision_group_id") or f"pg::{module_path}"),
                requested_precision=requested_precision,
                protected_precision=protected_precision,
                ordering=ordering,
            )
        )
    return PrecisionProfileResult(
        profile_id=str(profile_id),
        assignments=assignments,
        requested_int8_count=sum(row.requested_precision == "int8" for row in assignments),
        requested_int8_ratio=float(ratio),
        policy_version=(f"strict-weighted-layer-v1::{profile_id}" if strict_precision else policy.policy_version),
    )
