#!/usr/bin/env python3
"""Scan real DAIR validation batches and freeze a per-model fixed-K audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
from typing import Any

import torch
from torch.utils.data import DataLoader


REPO = Path(__file__).resolve().parents[1]
for path in (REPO.parent, REPO, Path("/home/lixingfeng/UniAD_examine/HEAL")):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return int(ordered[round((len(ordered) - 1) * fraction)])


def _worker_init(_worker_id: int) -> None:
    torch.set_num_threads(1)


def _scan(
    config: Path,
    manifest: dict[str, Any],
    workers: int,
    heal_root: Path,
) -> dict[str, Any]:
    from opencood.data_utils.datasets import build_dataset
    from opencood.hypes_yaml import yaml_utils

    hypes = yaml_utils.load_yaml(str(config))
    for key in ("data_dir", "root_dir", "validate_dir", "test_dir"):
        value = hypes.get(key)
        if isinstance(value, str) and not Path(value).is_absolute():
            hypes[key] = str(heal_root / value)
    dataset = build_dataset(hypes, visualize=True, train=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        collate_fn=dataset.collate_batch_test,
        pin_memory=False,
        prefetch_factor=2 if workers else None,
        persistent_workers=bool(workers),
        worker_init_fn=_worker_init if workers else None,
    )
    split_ids = [str(value) for value in json.loads(Path(hypes["validate_dir"]).read_text())]
    wanted = [str(value) for value in manifest["evaluation_frame_ids"]]
    if wanted != split_ids[: len(wanted)]:
        raise RuntimeError(f"fixed_k_manifest_not_exact_validation_order:{config}")
    counts: list[int] = []
    agents: list[int] = []
    maximum_rows: list[dict[str, Any]] = []
    for index, batch in enumerate(loader):
        if index >= len(wanted):
            break
        if batch is None:
            raise RuntimeError(f"fixed_k_empty_batch:{config}:{wanted[index]}")
        ego = batch["ego"]
        count = int(ego["inputs_m1"]["voxel_features"].shape[0])
        agent_count = int(ego["record_len"][0].item())
        counts.append(count)
        agents.append(agent_count)
        if not maximum_rows or count > int(maximum_rows[0]["voxel_count"]):
            maximum_rows = [
                {"frame_id": wanted[index], "voxel_count": count, "agent_count": agent_count}
            ]
        elif count == int(maximum_rows[0]["voxel_count"]):
            maximum_rows.append(
                {"frame_id": wanted[index], "voxel_count": count, "agent_count": agent_count}
            )
    if len(counts) != len(wanted):
        raise RuntimeError(f"fixed_k_scan_incomplete:{config}:{len(counts)}/{len(wanted)}")
    return {
        "config_path": str(config.resolve()),
        "config_sha256": _sha256(config),
        "evaluated_frames": len(counts),
        "fixed_k_required": max(counts),
        "max_rows": maximum_rows,
        "voxel_count_min": min(counts),
        "voxel_count_mean": statistics.mean(counts),
        "voxel_count_p50": _percentile(counts, 0.50),
        "voxel_count_p90": _percentile(counts, 0.90),
        "voxel_count_p99": _percentile(counts, 0.99),
        "max_agents": max(agents),
        "agent_count_histogram": {
            str(value): agents.count(value) for value in sorted(set(agents))
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-root",
        type=Path,
        default=Path(
            "/home/lixingfeng/UniAD_examine/Auto_Search/original_models/"
            "dairv2s/LiDAROnly"
        ),
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--heal-root",
        type=Path,
        default=Path("/home/lixingfeng/UniAD_examine/HEAL"),
    )
    parser.add_argument("--models", nargs="*", default=[])
    args = parser.parse_args()
    if os.environ.get("CONDA_DEFAULT_ENV") != "univ2x-opt":
        raise RuntimeError("fixed_k_audit_requires_univ2x_opt")
    if args.output.exists():
        raise RuntimeError(f"refusing_to_overwrite_fixed_k_audit:{args.output}")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected = set(str(value) for value in args.models)
    configs = sorted(args.models_root.glob("*/config.yaml"))
    if selected:
        configs = [path for path in configs if path.parent.name in selected]
    rows = []
    for config in configs:
        print(json.dumps({"status": "scanning", "model": config.parent.name}), flush=True)
        rows.append(
            {
                "model_name": config.parent.name,
                **_scan(config, manifest, args.workers, args.heal_root.resolve()),
            }
        )
    payload = {
        "schema_version": "heal-dair-lidar-full-validation-fixed-k-audit-v1",
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": _sha256(args.manifest),
        "models": rows,
        "shared_safe_fixed_k": max(int(row["fixed_k_required"]) for row in rows),
        "all_scans_complete": all(
            int(row["evaluated_frames"]) == len(manifest["evaluation_frame_ids"])
            for row in rows
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(args.output), "models": len(rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
