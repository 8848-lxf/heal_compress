"""Caller-fed calibration sample writer retained as a compatibility entry."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from ..artifacts.io import atomic_write_json, file_sha256
from ..config import CalibrationConfig


def dump_calibration_npz(
    samples: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    config: CalibrationConfig | None = None,
) -> dict[str, Any]:
    """Write caller-prepared NumPy-compatible inputs and a train manifest."""

    import numpy as np

    policy = config or CalibrationConfig()
    if policy.split != "train":
        raise ValueError("formal INT8 calibration split must be train")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, sample in enumerate(samples):
        path = output / f"sample_{index:06d}.npz"
        np.savez(path, **{str(name): value for name, value in sample.items()})
        rows.append({"sample_index": index, "path": str(path), "sha256": file_sha256(path)})
        if len(rows) >= int(policy.frame_count):
            break
    manifest = {
        "calibration_schema_version": policy.schema_version,
        "calibration_split": policy.split,
        "requested_frame_count": int(policy.frame_count),
        "num_samples": len(rows),
        "samples": rows,
    }
    atomic_write_json(output / "manifest.json", manifest)
    return manifest


def main() -> int:
    raise SystemExit("Use dump_calibration_npz with caller-prepared samples.")


if __name__ == "__main__":
    raise SystemExit(main())
