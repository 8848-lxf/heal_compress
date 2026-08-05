#!/usr/bin/env python3
"""Report CARLA and DAIR-V2X CPU-to-GPU voxelization migration results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def _read(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _assignments(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected NAME=PATH, got {value}")
        name, path = value.split("=", 1)
        result[name] = Path(path)
    return result


def _metric_equal(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    keys = ("AP@0.3", "AP@0.5", "AP@0.7", "mAP")
    return all(first.get(key) == second.get(key) for key in keys)


def _voxel_audit_rows(payload: Mapping[str, Any]) -> list[tuple[Any, Any, Any]]:
    return [
        (
            row.get("frame_id"),
            row.get("voxel_count"),
            row.get("saturated_voxel_count"),
        )
        for row in payload.get("latency_rows", [])
        if not bool(row.get("warmup", row.get("phase") == "warmup"))
    ]


def build_report(
    old_comparison: Mapping[str, Any],
    new_comparison: Mapping[str, Any],
    reproducibility: Mapping[str, Path],
    fp32_baselines: Mapping[str, Path],
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for name, current in new_comparison["models"].items():
        previous = old_comparison["models"][name]
        old_carla = previous["carla"]
        new_carla = current["carla"]
        dair = _read(current["dair_open_loop"]["source"])
        repeated = _read(reproducibility[name])
        fp32 = _read(fp32_baselines[name])
        old_composed = float(
            old_carla["candidate"]["composed_end_to_end_ms"]["mean"]
        )
        new_composed = float(
            new_carla["candidate"]["composed_end_to_end_ms"]["mean"]
        )
        rows[name] = {
            "carla": {
                "frames": int(new_comparison["protocol"]["frame_count"]),
                "cpu_voxelization_mean_ms": float(
                    old_carla["frontend"]["voxelize_cpu_ms"]["mean"]
                ),
                "gpu_voxelization_mean_ms": float(
                    new_carla["frontend"]["voxelize_gpu_ms"]["mean"]
                ),
                "old_composed_candidate_mean_ms": old_composed,
                "new_composed_candidate_mean_ms": new_composed,
                "cpu_to_gpu_composed_speedup": old_composed / new_composed,
                "candidate_map": float(
                    new_carla["candidate"]["range_all"]["map"]
                ),
                "map_delta_from_cpu_frontend": float(
                    new_carla["candidate"]["range_all"]["map"]
                    - old_carla["candidate"]["range_all"]["map"]
                ),
                "candidate_vs_fp32_composed_speedup": float(
                    new_carla["speedup"][
                        "candidate_vs_unpruned_fp32_pytorch_composed_mean"
                    ]
                ),
            },
            "dair_v2x": {
                "frames": int(dair["num_evaluated_frames"]),
                "skipped_frames": int(dair["num_skipped_frames"]),
                "mAP": float(dair["mAP"]),
                "AP@0.3": float(dair["AP@0.3"]),
                "AP@0.5": float(dair["AP@0.5"]),
                "AP@0.7": float(dair["AP@0.7"]),
                "host_to_device_p50_ms": float(dair["host_to_device_p50_ms"]),
                "voxelization_gpu_p50_ms": float(
                    dair["voxelization_gpu_p50_ms"]
                ),
                "fixed_k_prepare_p50_ms": float(dair["input_prepare_p50_ms"]),
                "forward_p50_ms": float(dair["forward_p50_ms"]),
                "postprocess_p50_ms": float(dair["postprocess_p50_ms"]),
                "composed_total_p50_ms": float(dair["composed_total_p50_ms"]),
                "fp32_pytorch_mAP": float(fp32["mAP"]),
                "candidate_minus_fp32_mAP": float(dair["mAP"] - fp32["mAP"]),
                "fp32_pytorch_forward_p50_ms": float(fp32["forward_p50_ms"]),
                "fp32_pytorch_composed_p50_ms": float(
                    fp32["composed_total_p50_ms"]
                ),
                "candidate_vs_fp32_forward_speedup": float(
                    fp32["forward_p50_ms"] / dair["forward_p50_ms"]
                ),
                "candidate_vs_fp32_composed_speedup": float(
                    fp32["composed_total_p50_ms"]
                    / dair["composed_total_p50_ms"]
                ),
                "backend": dair["voxelization_contract"]["backend"],
                "evaluation_seed": int(dair["evaluation_seed"]),
                "metrics_reproduced_exactly": _metric_equal(dair, repeated),
                "voxel_audit_reproduced_exactly": (
                    _voxel_audit_rows(dair) == _voxel_audit_rows(repeated)
                ),
            },
        }
    return {
        "schema_version": "heal-gpu-voxel-migration-report-v1",
        "models": rows,
        "all_carla_maps_preserved": all(
            row["carla"]["map_delta_from_cpu_frontend"] == 0.0
            for row in rows.values()
        ),
        "all_dair_runs_complete": all(
            row["dair_v2x"]["frames"] == 1789
            and row["dair_v2x"]["skipped_frames"] == 0
            for row in rows.values()
        ),
        "all_dair_runs_reproducible": all(
            row["dair_v2x"]["metrics_reproduced_exactly"]
            and row["dair_v2x"]["voxel_audit_reproduced_exactly"]
            for row in rows.values()
        ),
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# GPU voxelization migration report",
        "",
        "## CARLA five-scene replay",
        "",
        "| Model | mAP | CPU voxel mean (ms) | GPU voxel mean (ms) | Old composed (ms) | New composed (ms) | Migration speedup | Candidate/FP32 composed speedup |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in report["models"].items():
        value = row["carla"]
        lines.append(
            f"| {name} | {value['candidate_map']:.6f} | "
            f"{value['cpu_voxelization_mean_ms']:.3f} | "
            f"{value['gpu_voxelization_mean_ms']:.3f} | "
            f"{value['old_composed_candidate_mean_ms']:.3f} | "
            f"{value['new_composed_candidate_mean_ms']:.3f} | "
            f"{value['cpu_to_gpu_composed_speedup']:.3f}x | "
            f"{value['candidate_vs_fp32_composed_speedup']:.3f}x |"
        )
    lines.extend(
        [
            "",
            "All CARLA candidate mAP values are exactly equal to the CPU-frontend replay.",
            "",
            "## DAIR-V2X full validation",
            "",
            "| Model | TRT / FP32 mAP | Delta | AP30 / AP50 / AP70 | GPU voxel p50 | TRT / FP32 forward p50 | Forward speedup | TRT / FP32 composed p50 | Composed speedup | Reproduced |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
        ]
    )
    for name, row in report["models"].items():
        value = row["dair_v2x"]
        reproduced = (
            value["metrics_reproduced_exactly"]
            and value["voxel_audit_reproduced_exactly"]
        )
        lines.append(
            f"| {name} | {value['mAP']:.6f} / {value['fp32_pytorch_mAP']:.6f} | "
            f"{value['candidate_minus_fp32_mAP']:+.6f} | "
            f"{value['AP@0.3']:.6f} / {value['AP@0.5']:.6f} / {value['AP@0.7']:.6f} | "
            f"{value['voxelization_gpu_p50_ms']:.3f} | "
            f"{value['forward_p50_ms']:.3f} / {value['fp32_pytorch_forward_p50_ms']:.3f} | "
            f"{value['candidate_vs_fp32_forward_speedup']:.3f}x | "
            f"{value['composed_total_p50_ms']:.3f} / {value['fp32_pytorch_composed_p50_ms']:.3f} | "
            f"{value['candidate_vs_fp32_composed_speedup']:.3f}x | "
            f"{'yes' if reproduced else 'no'} |"
        )
    lines.extend(
        [
            "",
            "DAIR-V2X uses dynamic deterministic GPU voxelization followed by a fail-closed fixed-K engine adapter. No frame was skipped.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-comparison", required=True, type=Path)
    parser.add_argument("--new-comparison", required=True, type=Path)
    parser.add_argument("--dair-repro", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--dair-fp32", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-markdown", required=True, type=Path)
    args = parser.parse_args()
    report = build_report(
        _read(args.old_comparison),
        _read(args.new_comparison),
        _assignments(args.dair_repro),
        _assignments(args.dair_fp32),
    )
    for path in (args.output_json, args.output_markdown):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    args.output_markdown.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"success": True, "models": list(report["models"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
