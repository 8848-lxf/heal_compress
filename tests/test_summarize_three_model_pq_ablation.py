from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.summarize_three_model_pq_ablation import (
    ALL_AGGREGATED_METRICS,
    BUDGETS,
    METHODS,
    VARIANTS,
    aggregate_three_models,
    write_outputs,
)


MANIFEST_HASH = "shared-manifest"
FRAME_HASH = "shared-frame-order"


def _json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _aggregate_row(method: str, variant: str, budget: float | None) -> dict:
    pruned = variant in {"prune_quant", "prune_only"}
    row = {
        "assigned_method": method,
        "variant": variant,
        "budget": budget,
        "actual_bops": 1.0 if budget is None else budget + 0.001,
        "repeat_count": 5,
        "engine_sha256": f"engine-{method}-{variant}-{budget}",
        "frame_order_hash": FRAME_HASH,
        "parameter_count_base": None if variant in {"fp32", "prune_quant"} else 100,
        "parameter_count_pruned": None if variant in {"fp32", "prune_quant"} else (80 if pruned else 100),
        "parameter_reduction": None if variant == "prune_quant" else (0.2 if pruned else 0.0),
        "int8_count": 1 if variant in {"prune_quant", "quant_only"} else 0,
        "fp16_count": 2,
        "fp32_count": 3,
    }
    for metric in ALL_AGGREGATED_METRICS:
        row[f"{metric}_across_runs_mean"] = 1.0
        row[f"{metric}_across_runs_std"] = 0.0
    row["speedup_vs_same_gpu_fp32_across_runs_mean"] = 1.25
    row["speedup_vs_same_gpu_fp32_across_runs_std"] = 0.0
    return row


def _aggregate_rows() -> list[dict]:
    rows = []
    for method in METHODS:
        rows.append(_aggregate_row(method, "fp32", None))
        for budget in BUDGETS:
            for variant in VARIANTS:
                rows.append(_aggregate_row(method, variant, budget))
    return rows


def _repeat_rows(*, family: bool) -> list[dict]:
    rows = []
    for repeat in range(5):
        for method in METHODS:
            identities = [("fp32", None)] + [
                (variant, budget) for budget in BUDGETS for variant in VARIANTS
            ]
            for variant, budget in identities:
                row = {
                    "assigned_method": method,
                    "variant": variant,
                    "budget": budget,
                    "repeat_index": repeat,
                    "engine_sha256": f"engine-{method}-{variant}-{budget}",
                    "frame_order_hash": FRAME_HASH,
                }
                for metric in ALL_AGGREGATED_METRICS:
                    row[metric] = 1.0
                row["speedup_vs_same_gpu_fp32"] = 1.25
                if family:
                    row["num_evaluated_frames"] = 1789
                    row["num_skipped_frames"] = 0
                else:
                    row["evaluated_frames"] = 1789
                    row["skipped_frames"] = 0
                rows.append(row)
    return rows


def _protocol(eval_manifest: Path, *, family: bool) -> dict:
    content = {
        "manifest_hash": MANIFEST_HASH,
        "reset_after_warmup": True,
        "warmup_frame_ids": [f"warmup-{index}" for index in range(200)],
        "evaluation_frame_ids": [f"frame-{index}" for index in range(1789)],
    }
    eval_manifest.write_text(json.dumps(content), encoding="utf-8")
    digest = hashlib.sha256(eval_manifest.read_bytes()).hexdigest()
    return {
        "num_frames": 1789,
        "warmup_frames": 200,
        "latency_rounds": 3,
        "dataloader_num_workers": 8,
        "cuda_postprocess": True,
        "fixed_k": 29696,
        "reset_after_warmup": True,
        "eval_manifest_hash": MANIFEST_HASH,
        "eval_manifest_file_sha256": digest,
        ("eval_manifest" if family else "eval_manifest_path"): str(eval_manifest),
    }


def _pyramid_root(root: Path) -> None:
    root.mkdir(parents=True)
    _json(
        root / "split_gpu_five_repeat_report.json",
        {
            "passed": True,
            "repeat_count": 5,
            "five_repeat_mean_results": _aggregate_rows(),
        },
    )
    rows = _repeat_rows(family=False)
    with (root / "split_gpu_repeat_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _json(
        root / "split_gpu_evaluation_manifest.json",
        {"protocol": _protocol(root / "eval_manifest.json", family=False)},
    )


def _family_root(root: Path, family_id: str) -> None:
    root.mkdir(parents=True)
    _json(
        root / "family_fair_evaluation_report.json",
        {
            "status": "complete",
            "family_id": family_id,
            "repeat_count": 5,
            "repeat_results": _repeat_rows(family=True),
            "five_repeat_mean_std": _aggregate_rows(),
        },
    )
    _json(
        root / "family_fair_evaluation_manifest.json",
        {
            "family_id": family_id,
            "protocol": _protocol(root / "eval_manifest.json", family=True),
        },
    )


def _roots(tmp_path: Path) -> tuple[Path, Path, Path]:
    pyramid = tmp_path / "pyramid"
    fcooper = tmp_path / "fcooper"
    disco = tmp_path / "disco"
    _pyramid_root(pyramid)
    _family_root(fcooper, "heal_lidar_fcooper")
    _family_root(disco, "heal_lidar_disco")
    return pyramid, fcooper, disco


def test_unified_three_model_summary_adapts_both_schemas(tmp_path: Path) -> None:
    pyramid, fcooper, disco = _roots(tmp_path)

    payload = aggregate_three_models(
        pyramid_root=pyramid, fcooper_root=fcooper, disco_root=disco
    )
    output = tmp_path / "summary"
    write_outputs(payload, output)

    assert payload["status"] == "accepted"
    assert payload["row_count"] == 114
    assert payload["shared_eval_manifest_hash"] == MANIFEST_HASH
    assert payload["shared_frame_order_hash"] == FRAME_HASH
    assert {row["model"] for row in payload["rows"]} == {
        "lidar_pyramid",
        "lidar_fcooper",
        "lidar_disco",
    }
    pq = next(
        row
        for row in payload["rows"]
        if row["model"] == "lidar_pyramid"
        and row["method"] == "ga"
        and row["budget"] == 0.30
        and row["variant"] == "prune_quant"
    )
    assert pq["parameter_count_base"] == 100
    assert pq["parameter_count_effective"] == 80
    assert pq["parameter_pruning_rate"] == pytest.approx(0.2)
    assert pq["evaluated_frames_total"] == 8945
    assert (output / "three_model_pq_ablation_summary.csv").is_file()
    assert (output / "three_model_pq_ablation_summary.json").is_file()
    assert (output / "three_model_pq_ablation_summary.md").is_file()


def test_unified_summary_rejects_cross_model_manifest_mismatch(tmp_path: Path) -> None:
    pyramid, fcooper, disco = _roots(tmp_path)
    path = disco / "family_fair_evaluation_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["protocol"]["eval_manifest_hash"] = "different"
    eval_path = Path(payload["protocol"]["eval_manifest"])
    eval_payload = json.loads(eval_path.read_text(encoding="utf-8"))
    eval_payload["manifest_hash"] = "different"
    eval_path.write_text(json.dumps(eval_payload), encoding="utf-8")
    payload["protocol"]["eval_manifest_file_sha256"] = hashlib.sha256(
        eval_path.read_bytes()
    ).hexdigest()
    _json(path, payload)

    with pytest.raises(RuntimeError, match="unified_eval_manifest_mismatch"):
        aggregate_three_models(
            pyramid_root=pyramid, fcooper_root=fcooper, disco_root=disco
        )


def test_unified_summary_rejects_any_skipped_frame(tmp_path: Path) -> None:
    pyramid, fcooper, disco = _roots(tmp_path)
    path = fcooper / "family_fair_evaluation_report.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["repeat_results"][0]["num_skipped_frames"] = 1
    _json(path, payload)

    with pytest.raises(RuntimeError, match="unified_skipped_frames"):
        aggregate_three_models(
            pyramid_root=pyramid, fcooper_root=fcooper, disco_root=disco
        )
