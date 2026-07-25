#!/usr/bin/env python3
"""Audit the old 0.05 calibration and freeze the new train200 contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from quantization.types import stable_json_hash


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> int:
    old = args.old_root / "engines/JMIX-FRESH/calibration_manifest.json"
    payload = json.loads(old.read_text(encoding="utf-8"))
    required = {
        "algorithm": payload.get("algorithm"),
        "requested_frames": payload.get("requested_frames"),
        "processed_frames": payload.get("processed_frames"),
        "skipped_frames": payload.get("skipped_frames"),
        "manifest_hash": payload.get("manifest_hash"),
        "checkpoint_hash": payload.get("checkpoint_hash") or payload.get("checkpoint_sha256"),
        "physical_hash": payload.get("physical_hash") or payload.get("physical_structure_hash"),
        "state_dict_shape_hash": payload.get("state_dict_shape_hash"),
        "precision_map_hash": payload.get("precision_map_hash"),
        "onnx_hash": payload.get("onnx_hash"),
        "calibration_algorithm_config_hash": payload.get("calibration_algorithm_config_hash"),
        "cache_hash": payload.get("cache_hash"),
        "scale_hash": payload.get("scale_hash"),
    }
    failures = []
    if int(payload.get("frame_count", -1)) != 200:
        failures.append(f"frame_count={payload.get('frame_count')}!=200")
    for key, value in required.items():
        if value in (None, ""):
            failures.append(f"missing:{key}")
    report = {
        "schema_version": "old005-train200-calibration-audit-v1",
        "source": str(old),
        "source_sha256": _sha(old),
        "old_manifest": payload,
        "required_contract_fields": required,
        "failures": failures,
        "old_train200_calibration_verified": not failures,
        "conclusion": "old JMIX-FRESH used four-frame max-absolute module statistics; train200 EntropyCalibration2 provenance is not established" if failures else "verified",
    }
    train_manifest = json.loads(args.train200_manifest.read_text(encoding="utf-8"))
    algorithm_config = {
        "algorithm": "ModelOptEntropyKL2048To128",
        "formal_equivalence": "project_current_entropy_KL_calibrator",
        "histogram_bins": 2048,
        "quantized_bins": 128,
        "passes": 2,
        "requested_frames": 200,
        "processed_frames": 200,
        "skipped_frames": 0,
        "dataset_split": "train",
        "weight_update": False,
    }
    contract = {
        "schema_version": "v2xvit-train200-contract-plan-v1",
        **algorithm_config,
        "manifest_path": str(args.train200_manifest.resolve()),
        "manifest_hash": train_manifest["manifest_hash"],
        "manifest_file_sha256": _sha(args.train200_manifest),
        "required_candidate_hashes": [
            "checkpoint_hash", "physical_hash", "state_dict_shape_hash", "precision_map_hash",
            "onnx_hash", "calibration_algorithm_config_hash", "cache_hash", "scale_hash",
        ],
        "calibration_algorithm_config_hash": stable_json_hash(algorithm_config),
        "reuse_policy": "exact_dependency_key_only",
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "old005_calibration_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_root / "train200_contract.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_root / "train200_contract.md").write_text(
        "# V2X-ViT train200 calibration contract\n\n"
        "The old 0.05 artifact is not train200-verified. New INT8 deployment uses the frozen train200 manifest, two-pass entropy/KL statistics, exactly 200 processed and zero skipped, with reuse bound to all structure, precision, ONNX, checkpoint and algorithm hashes.\n",
        encoding="utf-8",
    )
    print(json.dumps({"old_train200_calibration_verified": report["old_train200_calibration_verified"], "failure_count": len(failures), "train200_manifest_hash": train_manifest["manifest_hash"]}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--train200-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
