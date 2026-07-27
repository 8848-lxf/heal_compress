#!/usr/bin/env python3
"""Repeat the strict B0 and six-budget Greedy/GA engines on full1789.

The runner is deliberately read-only with respect to the completed searches.
Every evaluation is written below a new output root and can be resumed without
touching either the main six-budget root or the frozen-domain R=0.05 root.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from search.model_family.evaluation import evaluate_v2xvit_engine_modelopt


LABELS = ("030", "025", "020", "015", "010", "005")
METRICS = ("AP@0.3", "AP@0.5", "AP@0.7", "mAP", "forward_p50_ms")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _gpu_uuid(index: int) -> str:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = {
        int(parts[0].strip()): parts[1].strip()
        for line in output.splitlines()
        if len(parts := line.split(",", 1)) == 2
    }
    if int(index) not in rows:
        raise RuntimeError(f"physical_gpu_missing:{index}")
    return rows[int(index)]


def _source_root(label: str, main_root: Path, frozen005_root: Path) -> Path:
    return frozen005_root if label == "005" else main_root


def _candidate_dir(root: Path, label: str, candidate_hash: str) -> Path:
    return root / f"ga/stage2_cache/budget_{label}/{candidate_hash}"


def _engine_acceptance(candidate_dir: Path) -> dict[str, Any]:
    path = candidate_dir / "JMIX-FRESH/engine_build_acceptance.json"
    if not path.is_file():
        raise RuntimeError(f"engine_acceptance_missing:{path}")
    acceptance = _read(path)
    precision = dict(acceptance.get("precision_realization_validation") or {})
    structure = dict(acceptance.get("engine_structure_validation") or {})
    if acceptance.get("status") != "ok" or not precision.get("passed") or not structure.get("passed"):
        raise RuntimeError(f"engine_acceptance_failed:{path}")
    if precision.get("mismatches") or int(precision.get("unresolved_layer_count", 0)):
        raise RuntimeError(f"engine_precision_not_exact:{path}")
    return acceptance


def _control_record(
    *,
    label: str,
    method: str,
    candidate: Mapping[str, Any],
    source_root: Path,
) -> dict[str, Any]:
    candidate_hash = str(candidate["complete_phenotype_hash"])
    candidate_dir = _candidate_dir(source_root, label, candidate_hash)
    engine = Path(str(candidate["metadata"]["engine_path"])).resolve()
    expected = candidate_dir / "JMIX-FRESH/candidate.plan"
    if engine != expected.resolve() or not engine.is_file():
        raise RuntimeError(
            f"candidate_engine_identity_mismatch:{label}:{method}:{engine}:{expected}"
        )
    engine_hash = _sha256(engine)
    recorded_hash = str(candidate["metadata"].get("engine_sha256") or "")
    if recorded_hash and engine_hash != recorded_hash:
        raise RuntimeError(
            f"candidate_engine_sha_mismatch:{label}:{method}:{engine_hash}:{recorded_hash}"
        )
    acceptance = _engine_acceptance(candidate_dir)
    physical = _read(candidate_dir / "physical_report.json")
    precision_genes = dict(candidate["genotype"]["precision_genes"])
    requested_counts = {
        state: sum(value == state for value in precision_genes.values())
        for state in ("FP32", "FP16", "INT8")
    }
    realized = dict(acceptance["precision_realization_validation"])
    return {
        "id": f"budget_{label}/{method}",
        "budget_label": label,
        "budget": int(label) / 100.0,
        "method": method,
        "candidate_hash": candidate_hash,
        "physical_structure_hash": str(candidate["metadata"]["physical_structure_hash"]),
        "engine_path": str(engine),
        "engine_sha256": engine_hash,
        "engine_size_bytes": engine.stat().st_size,
        "requested_realized_exact": True,
        "mutable_precision_counts": requested_counts,
        "engine_policy_counts": {
            "FP16": int(realized.get("realized_fp16_count", 0)),
            "INT8": int(realized.get("realized_int8_count", 0)),
            "requested_INT8": int(realized.get("requested_int8_count", 0)),
        },
        "physical_parameter_count": int(physical["physical_parameter_count"]),
        "original_parameter_count": int(physical["original_parameter_count"]),
        "parameter_retention": (
            int(physical["physical_parameter_count"])
            / int(physical["original_parameter_count"])
        ),
        "parameter_prune_rate": 1.0
        - int(physical["physical_parameter_count"])
        / int(physical["original_parameter_count"]),
        "source_root": str(source_root),
        "source_summary": str(
            source_root / f"ga/budget_{label}/seed_0/budget_summary.json"
        ),
    }


def discover_controls(
    *,
    main_root: Path,
    frozen005_root: Path,
    b0_engine: Path,
) -> list[dict[str, Any]]:
    if not b0_engine.is_file():
        raise RuntimeError(f"b0_engine_missing:{b0_engine}")
    controls: list[dict[str, Any]] = [
        {
            "id": "B0",
            "budget_label": "B0",
            "budget": 1.0,
            "method": "strict_FP32",
            "candidate_hash": "B0-strict",
            "physical_structure_hash": "original",
            "engine_path": str(b0_engine.resolve()),
            "engine_sha256": _sha256(b0_engine),
            "engine_size_bytes": b0_engine.stat().st_size,
            "requested_realized_exact": True,
            "mutable_precision_counts": {"FP32": 53, "FP16": 0, "INT8": 0},
            "physical_parameter_count": 13_453_197,
            "original_parameter_count": 13_453_197,
            "parameter_retention": 1.0,
            "parameter_prune_rate": 0.0,
        }
    ]
    for label in LABELS:
        root = _source_root(label, main_root, frozen005_root)
        summary_path = root / f"ga/budget_{label}/seed_0/budget_summary.json"
        if not summary_path.is_file():
            raise RuntimeError(f"budget_summary_missing:{label}:{summary_path}")
        summary = _read(summary_path)
        if int(summary.get("completed_evolution_generations", -1)) != 10:
            raise RuntimeError(f"budget_not_gen10_complete:{label}")
        controls.append(
            _control_record(
                label=label,
                method="Greedy",
                candidate=summary["greedy_anchor"],
                source_root=root,
            )
        )
        controls.append(
            _control_record(
                label=label,
                method="GA-final",
                candidate=summary["final_winner"],
                source_root=root,
            )
        )
    if len(controls) != 13:
        raise RuntimeError(f"unexpected_control_count:{len(controls)}")
    return controls


def summarize_repetitions(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    if len(materialized) != 3:
        raise RuntimeError(f"repeat_count_mismatch:{len(materialized)}")
    result: dict[str, Any] = {"repetitions": materialized}
    for metric in METRICS:
        values = [float(row[metric]) for row in materialized]
        result[f"{metric}_mean"] = statistics.fmean(values)
        result[f"{metric}_std"] = statistics.stdev(values)
        result[f"{metric}_min"] = min(values)
        result[f"{metric}_max"] = max(values)
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    for name in ("provenance", "evaluation_full1789", "latency", "reports", "logs"):
        (output_root / name).mkdir(parents=True, exist_ok=True)
    manifest = _read(args.full_manifest.resolve())
    evaluation_ids = tuple(str(value) for value in manifest["evaluation_frame_ids"])
    warmup_ids = tuple(str(value) for value in manifest["warmup_frame_ids"])
    if len(evaluation_ids) != 1789 or len(set(evaluation_ids)) != 1789:
        raise RuntimeError(f"full1789_manifest_invalid:{len(evaluation_ids)}")
    if len(warmup_ids) < 200:
        raise RuntimeError(f"full1789_warmup_insufficient:{len(warmup_ids)}")
    request = _read(args.request_json.resolve())
    controls = discover_controls(
        main_root=args.main_root.resolve(),
        frozen005_root=args.frozen005_root.resolve(),
        b0_engine=args.b0_engine.resolve(),
    )
    gpu_uuid = _gpu_uuid(args.physical_gpu)
    provenance = {
        "protocol": "v2xvit-sixbudget-greedy-ga-full1789-repeat3-v1",
        "physical_gpu": int(args.physical_gpu),
        "gpu_uuid": gpu_uuid,
        "manifest_path": str(args.full_manifest.resolve()),
        "manifest_hash": manifest.get("manifest_hash"),
        "evaluation_frame_count": len(evaluation_ids),
        "warmup_frame_count": len(warmup_ids),
        "repetitions": 3,
        "control_count": len(controls),
        "controls": controls,
    }
    _write(output_root / "provenance/input_inventory.json", provenance)
    if args.inventory_only:
        print(json.dumps({"status": "inventory_complete", **provenance}, sort_keys=True))
        return 0
    all_results: dict[str, list[dict[str, Any]]] = {}
    progress_path = output_root / "reports/progress.json"
    for repeat in range(1, 4):
        for control in controls:
            control_id = str(control["id"])
            destination = output_root / "evaluation_full1789" / f"repeat_{repeat}" / control_id
            result_path = destination / "evaluation.json"
            if result_path.is_file():
                result = _read(result_path)
            else:
                result = evaluate_v2xvit_engine_modelopt(
                    engine_path=control["engine_path"],
                    model_config=request["model_config"],
                    heal_root=request["heal_root"],
                    output_dir=destination,
                    tensorrt_root=args.tensorrt_root.resolve(),
                    plugin_path=request["plugin_path"],
                    eval_manifest_path=args.full_manifest.resolve(),
                    physical_gpu_id=int(args.physical_gpu),
                    fixed_k=int(request["fixed_k"]),
                    max_agents=int(request["max_agents"]),
                    num_frames=1789,
                    warmup_frames=200,
                    latency_rounds=1,
                    dataloader_num_workers=8,
                )
            if not (
                result.get("status") == "ok"
                and int(result.get("num_evaluated_frames", -1)) == 1789
                and int(result.get("num_skipped_frames", -1)) == 0
                and str(result.get("eval_manifest_hash")) == str(manifest.get("manifest_hash"))
            ):
                raise RuntimeError(
                    f"full1789_gate_failed:repeat={repeat}:control={control_id}:"
                    f"status={result.get('status')}:evaluated={result.get('num_evaluated_frames')}:"
                    f"skipped={result.get('num_skipped_frames')}"
                )
            row = {
                "repeat": repeat,
                "AP@0.3": float(result["AP@0.3"]),
                "AP@0.5": float(result["AP@0.5"]),
                "AP@0.7": float(result["AP@0.7"]),
                "mAP": float(result["mAP"]),
                "forward_p50_ms": float(result["forward_p50_ms"]),
                "evaluated": 1789,
                "skipped": 0,
                "manifest_hash": str(result["eval_manifest_hash"]),
            }
            all_results.setdefault(control_id, [])
            existing = {int(item["repeat"]): item for item in all_results[control_id]}
            existing[repeat] = row
            all_results[control_id] = [existing[key] for key in sorted(existing)]
            _write(progress_path, {
                "status": "running",
                "gpu_uuid": gpu_uuid,
                "completed_evaluations": sum(len(value) for value in all_results.values()),
                "total_evaluations": 39,
                "last_control": control_id,
                "last_repeat": repeat,
                "results": all_results,
            })
            print(json.dumps({"event": "full1789_complete", "control": control_id,
                              "repeat": repeat, "mAP": row["mAP"]}, sort_keys=True), flush=True)
    summaries = {key: summarize_repetitions(value) for key, value in all_results.items()}
    baseline = summaries["B0"]
    baseline_map = float(baseline["mAP_mean"])
    baseline_p50 = float(baseline["forward_p50_ms_mean"])
    baseline_engine_size = int(controls[0]["engine_size_bytes"])
    inventory = {str(row["id"]): row for row in controls}
    table = []
    for control_id, summary in summaries.items():
        control = inventory[control_id]
        table.append({
            "budget": control["budget"],
            "method": control["method"],
            "candidate_hash": control["candidate_hash"],
            "AP30_mean": summary["AP@0.3_mean"],
            "AP30_std": summary["AP@0.3_std"],
            "AP50_mean": summary["AP@0.5_mean"],
            "AP50_std": summary["AP@0.5_std"],
            "AP70_mean": summary["AP@0.7_mean"],
            "AP70_std": summary["AP@0.7_std"],
            "mAP_mean": summary["mAP_mean"],
            "mAP_std": summary["mAP_std"],
            "mAP_drop_vs_B0": baseline_map - float(summary["mAP_mean"]),
            "mAP_retention_vs_B0": float(summary["mAP_mean"]) / baseline_map,
            "forward_p50_ms_mean": summary["forward_p50_ms_mean"],
            "forward_p50_ms_std": summary["forward_p50_ms_std"],
            "speedup_vs_B0_fullval_p50": baseline_p50 / float(summary["forward_p50_ms_mean"]),
            "parameter_count": control["physical_parameter_count"],
            "parameter_retention": control["parameter_retention"],
            "parameter_prune_rate": control["parameter_prune_rate"],
            "engine_size_bytes": control["engine_size_bytes"],
            "engine_file_compression_vs_B0": baseline_engine_size / int(control["engine_size_bytes"]),
            "mutable_FP32": control["mutable_precision_counts"]["FP32"],
            "mutable_FP16": control["mutable_precision_counts"]["FP16"],
            "mutable_INT8": control["mutable_precision_counts"]["INT8"],
            "requested_realized_exact": control["requested_realized_exact"],
            "evaluated_per_repeat": 1789,
            "skipped_total": 0,
        })
    table.sort(key=lambda row: (-float(row["budget"]), str(row["method"])))
    _write_csv(output_root / "reports/full1789_repeat3_summary.csv", table)
    _write(output_root / "reports/full1789_repeat3_summary.json", {
        "status": "complete",
        "protocol": provenance,
        "controls": inventory,
        "summaries": summaries,
        "table": table,
    })
    _write(progress_path, {
        "status": "complete",
        "gpu_uuid": gpu_uuid,
        "completed_evaluations": 39,
        "total_evaluations": 39,
    })
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--frozen005-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--b0-engine", type=Path, required=True)
    parser.add_argument("--request-json", type=Path, required=True)
    parser.add_argument("--full-manifest", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument(
        "--tensorrt-root",
        type=Path,
        default=Path("/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118"),
    )
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
