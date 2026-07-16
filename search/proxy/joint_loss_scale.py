"""Calibrate and freeze the linear joint-Taylor loss scale."""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..hashing import canonical_json_hash


MAPPING = "linear_fixed_scale"
FORMULA = "J1=-0.8*(L_joint/L_scale)+0.2*R_prune"
QUANTILE = 0.90
QUANTILE_METHOD = "nearest_rank"


def joint_loss_scale_hash(payload: Mapping[str, Any]) -> str:
    data = dict(payload)
    data.pop("joint_loss_scale_hash", None)
    return canonical_json_hash(data)


def calibrate_joint_loss_scale(
    rows: Sequence[Mapping[str, Any]],
    *,
    code_commit: str = "",
    created_at: str | None = None,
) -> dict[str, Any]:
    unique: dict[str, float] = {}
    for row in rows:
        identity = str(row.get("phenotype_hash", "")).strip()
        loss = float(row.get("L_joint_raw", float("nan")))
        if not identity or not math.isfinite(loss) or loss <= 0.0:
            continue
        previous = unique.get(identity)
        if previous is not None and not math.isclose(
            previous, loss, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError(f"joint_loss_scale_duplicate_loss_conflict:{identity}")
        unique.setdefault(identity, loss)
    if not unique:
        raise ValueError("joint_loss_scale_positive_finite_pool_required")

    ordered_losses = sorted(unique.values())
    index = math.ceil(QUANTILE * len(ordered_losses)) - 1
    payload: dict[str, Any] = {
        "mapping": MAPPING,
        "formula": FORMULA,
        "quantile": QUANTILE,
        "quantile_method": QUANTILE_METHOD,
        "value": float(ordered_losses[index]),
        "member_count": len(unique),
        "members": [
            {"phenotype_hash": identity, "L_joint_raw": float(unique[identity])}
            for identity in sorted(unique)
        ],
        "code_commit": str(code_commit),
        "created_at": created_at
        or datetime.now(timezone.utc).astimezone().isoformat(),
    }
    payload["joint_loss_scale_hash"] = joint_loss_scale_hash(payload)
    return payload


def _validate_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(payload)
    if data.get("mapping") != MAPPING or data.get("formula") != FORMULA:
        raise RuntimeError("joint_loss_scale_contract_invalid")
    value = float(data.get("value", float("nan")))
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError("joint_loss_scale_value_invalid")
    members = list(data.get("members", ()))
    if int(data.get("member_count", -1)) != len(members) or not members:
        raise RuntimeError("joint_loss_scale_members_invalid")
    expected = str(data.get("joint_loss_scale_hash", ""))
    if not expected or expected != joint_loss_scale_hash(data):
        raise RuntimeError("joint_loss_scale_hash_mismatch")
    return data


def load_joint_loss_scale(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise RuntimeError(f"joint_loss_scale_missing:{source}")
    if os.stat(source).st_mode & 0o222:
        raise RuntimeError("joint_loss_scale_must_be_read_only")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"joint_loss_scale_unreadable:{source}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("joint_loss_scale_contract_invalid")
    return _validate_payload(payload)


def write_joint_loss_scale(
    path: str | Path, payload: Mapping[str, Any]
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload)
    expected = str(data.get("joint_loss_scale_hash", ""))
    if not expected:
        data["joint_loss_scale_hash"] = joint_loss_scale_hash(data)
    _validate_payload(data)

    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True),
        encoding="utf-8",
    )
    os.replace(temporary, target)
    os.chmod(target, 0o444)
    load_joint_loss_scale(target)
    return target
