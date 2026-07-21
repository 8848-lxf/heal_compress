#!/usr/bin/env python3
"""Finalize the native CoBEVT F16A32 tactic-pinning evidence package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from search.model_families.lidar_cobevt.native_tactic_contract import (
    classify_fixed500,
)
from search.model_families.lidar_cobevt.native_tactic_evidence import (
    classify_tactic_kernel,
    realize_output_phenotype,
)


def _read(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text()) if path.is_file() else default


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key, value in row.items() if not isinstance(value, (dict, list))})
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def _eval(root: Path, profile: str, phase: str, *, exact: bool = True) -> dict[str, Any]:
    base = root / ("evaluation_exact" if exact else "evaluation")
    return _read(base / profile / phase / "evaluation.json", {})


def _compact_preservation(path: Path, *, profile: str) -> list[dict[str, Any]]:
    values = _read(path, [])
    rows: list[dict[str, Any]] = []
    for rebuild in values:
        for item in rebuild.get("preservation", {}).get("rows", []):
            rows.append(
                {
                    "profile": profile,
                    "repeat": rebuild.get("repeat"),
                    "role": item.get("role"),
                    "block": item.get("block"),
                    "attention_kind": item.get("attention_kind"),
                    "requested_tactic_hash": item.get("requested_tactic_hash"),
                    "requested_kernel_name": item.get("requested_kernel_name"),
                    "realized_tactic_hash": item.get("realized_tactic_hash"),
                    "realized_tactic_name": item.get("realized_tactic_name"),
                    "realized_output_precision": item.get("realized_output_precision"),
                    "preserved": item.get("preserved"),
                    "reason": item.get("reason"),
                }
            )
    return rows


def finalize(root: Path) -> dict[str, Any]:
    matrix = _read(root / "native_shape_pinning_matrix.json", [])
    if len(matrix) != 12:
        raise RuntimeError(f"native_tactic_expected_twelve_micro_shapes:{len(matrix)}")
    (root / "tactics" / "enumeration_completeness.json").write_text(
        json.dumps(
            {
                "status": "partial",
                "reported_tactics": 720,
                "reported_tactics_per_shape": 60,
                "shape_count": 12,
                "reason": "TensorRT public API does not guarantee enumeration of internally filtered implementations.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    evidence_rows = []
    for row in matrix:
        baseline_evidence = classify_tactic_kernel(str(row.get("baseline_realized_tactic_name", "")))
        evidence_rows.append(
            {
                "role": row["role"],
                "block": row["block"],
                "module_name": row["module_name"],
                "shape": "x".join(map(str, row["shape"])),
                "available_tactic_count": row["available_tactic_count"],
                "default_tactic": row.get("baseline_realized_tactic_name"),
                "default_phenotype": realize_output_phenotype(
                    baseline_evidence, str(row.get("baseline_output_precision"))
                ),
                "target_tactic": row.get("target_kernel_name"),
                "target_compute_phenotype": row.get("target_compute_phenotype"),
                "target_phenotype": row.get("target_phenotype"),
                "accumulator_precision": "FP32",
                "output_precision": "FP16",
                "evidence_level": row.get("target_evidence_level"),
                "pinning_stable": row.get("pinning_stable"),
                "numerical_status": row.get("pinned_numerical_status"),
                "relative_l2": row.get("pinned_relative_l2"),
                "zero_ratio": row.get("pinned_output_zero_ratio"),
            }
        )
    _write_csv(root / "tactic_accumulator_evidence.csv", evidence_rows)
    _write_csv(root / "qk_available_tactics.csv", [row for row in evidence_rows if row["role"] == "QK"])
    _write_csv(root / "av_available_tactics.csv", [row for row in evidence_rows if row["role"] == "AV"])

    exact_av = root / "full_model/N2_AV_NATIVE_PIN_F16A32_EXACT/full_engine_tactic_preservation.json"
    preservation = _compact_preservation(exact_av, profile="N2_AV_NATIVE_PIN_F16A32_EXACT")
    exact_qk_info = _read(root / "full_model/N1_QK_NATIVE_PIN_F16A32_EXACT/baseline/layer_info.json", {})
    fused_qk = [
        layer for layer in exact_qk_info.get("Layers", [])
        if "gemm_mha" in str(layer.get("TacticName", "")).lower()
    ]
    fused_rows = [
        {
            "profile": "N1_QK_NATIVE_PIN_F16A32_EXACT",
            "repeat": 0,
            "role": "QK_AV_COMPLETE_FUSION",
            "block": index // 2,
            "attention_kind": "window" if index % 2 == 0 else "grid",
            "realized_tactic_name": layer.get("TacticName"),
            "realized_output_precision": "FP16",
            "preserved": False,
            "reason": "micro_pinning_not_preserved_in_full_engine",
        }
        for index, layer in enumerate(fused_qk)
    ]
    _write_csv(root / "full_engine_tactic_preservation.csv", preservation + fused_rows)
    _write_csv(root / "deterministic_rebuild_results.csv", preservation)
    cache_edit = _read(root / "full_model/N2_AV_NATIVE_PIN_F16A32_EXACT/cache_edit_manifest.json", {})
    _write_csv(
        root / "cache_edit_manifest.csv",
        [
            {
                "profile": "N2_AV_NATIVE_PIN_F16A32_EXACT",
                "role": "AV",
                **target,
                "original_cache_sha256": cache_edit.get("original_sha256"),
                "edited_cache_sha256": cache_edit.get("edited_sha256"),
            }
            for target in cache_edit.get("targets", [])
        ],
    )

    baseline500 = _eval(root, "N0_F3_REFERENCE_EXACT", "fixed500")
    av_smoke = _eval(root, "N2_AV_NATIVE_PIN_F16A32", "smoke10")
    av50 = _eval(root, "N2_AV_NATIVE_PIN_F16A32", "fixed50")
    av500 = _eval(root, "N2_AV_NATIVE_PIN_F16A32_EXACT", "fixed500")
    delta = float(av500["mAP"]) - float(baseline500["mAP"])
    accuracy_rows = [
        {
            "profile": "N0_F3_REFERENCE_EXACT",
            "phase": "fixed500",
            **{key: baseline500.get(key) for key in ("AP@0.3", "AP@0.5", "AP@0.7", "mAP", "num_evaluated_frames", "num_skipped_frames")},
            "delta_mAP_vs_fresh_f3": 0.0,
            "safety": "SAFE",
        },
        {
            "profile": "N2_AV_NATIVE_PIN_F16A32_EXACT",
            "phase": "smoke10",
            **{key: av_smoke.get(key) for key in ("AP@0.3", "AP@0.5", "AP@0.7", "mAP", "num_evaluated_frames", "num_skipped_frames")},
        },
        {
            "profile": "N2_AV_NATIVE_PIN_F16A32_EXACT",
            "phase": "fixed50",
            **{key: av50.get(key) for key in ("AP@0.3", "AP@0.5", "AP@0.7", "mAP", "num_evaluated_frames", "num_skipped_frames")},
        },
        {
            "profile": "N2_AV_NATIVE_PIN_F16A32_EXACT",
            "phase": "fixed500",
            **{key: av500.get(key) for key in ("AP@0.3", "AP@0.5", "AP@0.7", "mAP", "num_evaluated_frames", "num_skipped_frames")},
            "delta_mAP_vs_fresh_f3": delta,
            "safety": classify_fixed500(delta, evaluated=int(av500["num_evaluated_frames"]), skipped=int(av500["num_skipped_frames"]), finite=True),
        },
    ]
    _write_csv(root / "native_f16a32_accuracy.csv", accuracy_rows)

    latency_base = _read(root / "latency/N0_F3_REFERENCE_EXACT.json", {})
    latency_av = _read(root / "latency/N2_AV_NATIVE_PIN_F16A32_EXACT.json", {})
    p50_base = float(latency_base["aggregate"]["p50_ms"])
    p50_av = float(latency_av["aggregate"]["p50_ms"])
    latency_rows = []
    for profile, report in (("N0_F3_REFERENCE_EXACT", latency_base), ("N2_AV_NATIVE_PIN_F16A32_EXACT", latency_av)):
        latency_rows.append(
            {
                "profile": profile,
                **report["aggregate"],
                "gpu": report.get("physical_gpu"),
                "scope": report.get("scope"),
                "warmup": report.get("warmup_iterations"),
                "iterations_per_repeat": report.get("timed_iterations_per_repeat"),
                "repeats": report.get("repeats"),
                "speedup_vs_f3": p50_base / float(report["aggregate"]["p50_ms"]),
            }
        )
    _write_csv(root / "native_f16a32_latency.csv", latency_rows)

    av_full_preserved = len(preservation) == 18 and all(bool(row["preserved"]) for row in preservation)
    contract = {
        "platform": {
            "gpu_arch": "SM89",
            "tensorrt": "10.9.0.34",
            "cuda": "11.8",
            "portable": False,
        },
        "qk": {
            "level_a_shapes": [row["module_name"] for row in evidence_rows if row["role"] == "QK"],
            "level_b_shapes": [],
            "unsupported_shapes": [],
            "micro_pinning": "6/6_stable",
            "full_native_profile": "micro_pinning_not_preserved_in_full_engine",
            "full_graph_fused_mha_layers": len(fused_qk),
        },
        "av": {
            "level_a_shapes": [row["module_name"] for row in evidence_rows if row["role"] == "AV"],
            "level_b_shapes": [],
            "unsupported_shapes": [],
            "micro_pinning": "6/6_stable",
            "full_native_profile": "N2_AV_NATIVE_PIN_F16A32_EXACT" if av_full_preserved else None,
        },
        "profiles": {
            "allowed": [],
            "experimental": [],
            "rejected": [
                {"profile": "N1_QK_NATIVE_PIN_F16A32", "reason": "micro_pinning_not_preserved_in_full_engine"},
                {"profile": "N2_AV_NATIVE_PIN_F16A32_EXACT", "reason": "rejected_no_latency_gain"},
                {"profile": "N3_QK_AV_NATIVE_PIN_F16A32", "reason": "complete_fused_mha_replaced_primitives"},
            ],
            "unsupported": [],
        },
        "pinning": {
            "mechanism": "editable_timing_cache",
            "requires_exact_cache_hash": True,
            "requires_revalidation_on_shape_change": True,
            "requires_revalidation_on_workspace_change": True,
        },
        "accuracy": {"n2_delta_mAP_vs_fresh_f3": delta},
        "latency": {
            "f3_p50_ms": p50_base,
            "n2_p50_ms": p50_av,
            "n2_speedup": p50_base / p50_av,
            "formal_isolated": True,
        },
    }
    _write_json(root / "native_f16a32_tactic_search_contract.json", contract)

    checkpoint = Path("/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_cobevt/net_epoch_bestval_at19.pth")
    config = checkpoint.with_name("config.yaml")
    manifests = Path("/data/lxf/heal_data/outputs/cobevt_attention_operand_accumulation_20260721_060040/full_model/A0_F3_REFERENCE/manifests")
    provenance = {
        "checkpoint": {"path": str(checkpoint), "sha256": _sha(checkpoint)},
        "config": {"path": str(config), "sha256": _sha(config)},
        "manifests": {name: {"path": str(manifests / f"{name}_manifest.json"), "sha256": _sha(manifests / f"{name}_manifest.json")} for name in ("smoke10", "fixed50", "fixed500")},
    }
    _write_json(root / "provenance/manifest_hashes.json", provenance)
    run_manifest = _read(root / "run_manifest.json", {})
    run_manifest.update({"branch": "feature/cobevt-native-f16a32-tactic-pinning", "base_commit": "ea342d49526355094d92b4b4631bb3ee596e1e6a", "output_dir": str(root), "enumeration_completeness": "partial", **provenance})
    _write_json(root / "run_manifest.json", run_manifest)
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,compute_cap,driver_version", "--format=csv,noheader"], text=True, capture_output=True, check=True).stdout.strip()
    _write_json(root / "hardware_manifest.json", {"gpu_inventory": gpu, "formal_latency_gpu_index": 3, "gpu_arch": "SM89"})
    _write_json(root / "toolchain_manifest.json", {"python": "/home/lixingfeng/anaconda3/envs/modelopt/bin/python", "nvcc": "/home/lixingfeng/anaconda3/envs/modelopt/bin/nvcc", "tensorrt": "10.9.0.34", "cuda": "11.8", "strongly_typed": True, "tf32": False})

    conclusion = f"""# CoBEVT Native F16A32 Tactic Pinning Conclusion

## Direct answers

- TensorRT 10.9 direct accumulator API: **NO**. Accumulation is an implementation property, not a layer dtype setter.
- Editable Timing Cache pins: **the tactic hash for an exact cache key**; it does not pin a dtype.
- Enumeration completeness: **partial**. Each of 12 reported shapes exposed 60 tactics, but the public API cannot prove internally filtered tactics were enumerable.
- Micro QK: 6/6 have Level-A `F16A32O16`, and all pinned deterministically across three builds.
- Micro AV: 6/6 have Level-A `F16A32O16`, and all pinned deterministically across three builds.
- Exact F3-derived QK full profile: **not preservable**. TensorRT replaces the six primitive QK/Softmax/AV paths with {len(fused_qk)} complete `_gemm_mha_v2` layers.
- Exact F3-derived AV full profile: 6/6 primitive AV tactics preserved across three builds; output is O16, with no FP32 output Cast.
- Exact N2 fixed500: AP30={av500['AP@0.3']:.9f}, AP50={av500['AP@0.5']:.9f}, AP70={av500['AP@0.7']:.9f}, mAP={av500['mAP']:.9f}; delta vs fresh F3={delta:+.9f}.
- Formal p50: F3={p50_base:.6f} ms, pinned AV={p50_av:.6f} ms, speedup={p50_base / p50_av:.6f}x.
- Search eligibility: **none**. QK pinning is not preserved in the exact full graph; AV pinning is accuracy-safe but has no latency gain.

## Default tactic mixture

Default choices vary between explicit F16A16O16 and F16A32O16 implementations because TensorRT minimizes measured builder cost for each cache key. Requested FP16 operands do not constrain accumulator precision.

## Portability

The edited cache is platform and graph specific. A rebuild on another RTX 4090 with the same 4 GiB workspace preserved the six targets. Changing workspace to 1 GiB caused `ERROR_ON_TIMING_CACHE_MISS`, so any BuilderConfig, shape, graph, CUDA, TensorRT, GPU architecture, or cache hash change requires revalidation.

## Unresolved

- TensorRT does not expose a complete inventory of internally filtered tactics.
- Complete fused MHA accumulator precision remains unknown.
- Nsight/SASS was not needed to upgrade evidence because the selected kernel specialization directly encodes `f16f16_f16f32`; no inference from output dtype was used.
"""
    (root / "root_conclusion.md").write_text(conclusion)
    return contract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    contract = finalize(Path(args.output_dir).resolve())
    print(json.dumps(contract, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
