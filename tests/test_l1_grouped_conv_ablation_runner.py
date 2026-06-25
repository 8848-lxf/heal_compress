from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from heal_compress.utils.gpu_select import GPUInfo, GPUSelectionError, resolve_gpu, select_gpu
from evaluate_l1_grouped_conv_ablation import (
    SUMMARY_COLUMNS,
    build_summary_rows_from_manifests,
    discover_model_manifests,
    write_summary_outputs,
)
from test_prune_and_eval import _hydrate_grouped_independent_replay
from run_l1_grouped_conv_ablation import (
    DEFAULT_EXCLUDE_GPU_IDS,
    build_experiment_configs,
    build_prune_command,
    create_experiment_root,
    parse_args,
    write_commands,
)


def test_runner_generates_six_experiment_configs_with_fixed_names():
    configs = build_experiment_configs(
        ratios=[0.25, 0.50, 0.75],
        modes=["shared_local_mean", "independent_group_topk"],
    )

    assert [cfg.model_name for cfg in configs] == [
        "shared_local_mean_p25",
        "shared_local_mean_p50",
        "shared_local_mean_p75",
        "independent_group_topk_p25",
        "independent_group_topk_p50",
        "independent_group_topk_p75",
    ]


def test_default_gpu_arg_is_auto_and_excludes_5_6_7():
    args = parse_args([])

    assert args.gpu_id == "auto"
    assert args.exclude_gpu_ids == DEFAULT_EXCLUDE_GPU_IDS == [5, 6, 7]


def test_explicit_gpu_id_bypasses_default_exclusion():
    selected = select_gpu(
        "5",
        [
            GPUInfo(index=5, memory_total_mb=40000, memory_used_mb=39000, memory_free_mb=1000, utilization_gpu=99),
        ],
        exclude_gpu_ids=[5, 6, 7],
        min_free_memory_mb=8000,
        max_gpu_util=30,
    )

    assert selected.index == 5


def test_resolve_explicit_gpu_id_does_not_require_nvidia_smi():
    device, selected, queried = resolve_gpu(
        "5",
        exclude_gpu_ids=[5, 6, 7],
        min_free_memory_mb=8000,
        max_gpu_util=30,
    )

    assert device == "cuda:5"
    assert selected is not None and selected.index == 5
    assert queried == []


def test_auto_gpu_selection_filters_excluded_and_resource_limits():
    gpus = [
        GPUInfo(index=0, memory_total_mb=24000, memory_used_mb=12000, memory_free_mb=12000, utilization_gpu=50),
        GPUInfo(index=1, memory_total_mb=24000, memory_used_mb=4000, memory_free_mb=20000, utilization_gpu=10),
        GPUInfo(index=5, memory_total_mb=80000, memory_used_mb=0, memory_free_mb=80000, utilization_gpu=0),
    ]

    selected = select_gpu(
        "auto",
        gpus,
        exclude_gpu_ids=[5, 6, 7],
        min_free_memory_mb=8000,
        max_gpu_util=30,
    )

    assert selected.index == 1


def test_auto_gpu_selection_reports_no_available_gpu():
    with pytest.raises(GPUSelectionError) as exc:
        select_gpu(
            "auto",
            [GPUInfo(index=0, memory_total_mb=24000, memory_used_mb=23000, memory_free_mb=1000, utilization_gpu=80)],
            exclude_gpu_ids=[5, 6, 7],
            min_free_memory_mb=8000,
            max_gpu_util=30,
        )

    error = exc.value.to_dict()
    assert error["error"] == "no_available_gpu"
    assert error["required_min_free_memory_mb"] == 8000
    assert error["required_max_gpu_util"] == 30
    assert error["candidate_gpus"][0]["index"] == 0


def test_create_experiment_root_uses_timestamp_and_does_not_overwrite(tmp_path):
    first = create_experiment_root(
        output_root=tmp_path,
        experiment_name="l1_grouped_conv_ablation",
        timestamp="20260625_153012",
        overwrite=False,
    )
    second = create_experiment_root(
        output_root=tmp_path,
        experiment_name="l1_grouped_conv_ablation",
        timestamp="20260625_153012",
        overwrite=False,
    )

    assert first.name == "l1_grouped_conv_ablation_20260625_153012"
    assert second.name == "l1_grouped_conv_ablation_20260625_153012_run001"
    assert first.is_dir()
    assert second.is_dir()


def test_pruning_command_contains_required_fixed_flags(tmp_path):
    cfg = build_experiment_configs([0.25], ["shared_local_mean"])[0]
    cmd = build_prune_command(
        cfg,
        checkpoint="/ckpt.pth",
        device="cuda:2",
        output_dir=tmp_path / cfg.model_name,
    )
    joined = " ".join(cmd)

    assert "--importance-mode l1_norm" in joined
    assert "--selection-mode constrained_global" in joined
    assert "--group-conv-selection-mode shared_local_mean" in joined
    assert "--group-conv-align 8" in joined
    assert "--align 16" in joined
    assert "--protect-residual-add true" in joined
    assert "--allow-remove-groups false" in joined
    assert "--device cuda:2" in joined


def test_write_commands_records_reproducible_shell(tmp_path):
    commands_path = tmp_path / "commands.sh"
    write_commands(commands_path, [["python", "tests/test_general_pruner.py", "--device", "cuda:0"]])

    text = commands_path.read_text(encoding="utf-8")
    assert "python tests/test_general_pruner.py --device cuda:0" in text


def _write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_evaluator_discovers_original_plus_six_manifests(tmp_path):
    _write_manifest(tmp_path / "baseline_original" / "model_manifest.json", {"model_name": "baseline_original"})
    for cfg in build_experiment_configs([0.25, 0.50, 0.75], ["shared_local_mean", "independent_group_topk"]):
        _write_manifest(tmp_path / cfg.model_name / "model_manifest.json", {"model_name": cfg.model_name})

    manifests = discover_model_manifests(tmp_path)

    assert [m["model_name"] for m in manifests] == [
        "baseline_original",
        "shared_local_mean_p25",
        "shared_local_mean_p50",
        "shared_local_mean_p75",
        "independent_group_topk_p25",
        "independent_group_topk_p50",
        "independent_group_topk_p75",
    ]


def test_failed_model_stays_in_summary_with_failure_reason(tmp_path):
    manifests = [
        {
            "model_name": "baseline_original",
            "checkpoint_path": "/original.pth",
            "checkpoint_type": "original",
            "status": "success",
            "params": 100,
        },
        {
            "model_name": "shared_local_mean_p25",
            "checkpoint_path": "/missing.pth",
            "checkpoint_type": "pruned",
            "status": "prune_failed",
            "failure_reason": "subprocess_failed",
            "structure_legal": False,
            "forward_sanity_check": False,
        },
    ]

    rows = build_summary_rows_from_manifests(manifests, eval_results={})

    failed = rows[1]
    assert failed["model_name"] == "shared_local_mean_p25"
    assert failed["eval_status"] == "skipped"
    assert failed["failure_reason"] == "subprocess_failed"


def test_missing_eval_outputs_are_reported_as_failed():
    manifests = [
        {
            "model_name": "independent_group_topk_p25",
            "checkpoint_path": "/pruned.pth",
            "checkpoint_type": "pruned",
            "status": "success",
            "structure_legal": True,
            "forward_sanity_check": True,
        }
    ]

    rows = build_summary_rows_from_manifests(manifests, eval_results={})

    assert rows[0]["eval_status"] == "failed"
    assert rows[0]["failure_reason"] == "evaluation_outputs_missing"
    assert rows[0]["AP_drop_vs_original"] == "not_available"


def test_summary_csv_fields_are_stable(tmp_path):
    rows = build_summary_rows_from_manifests(
        [{"model_name": "baseline_original", "checkpoint_type": "original", "status": "success"}],
        eval_results={},
    )
    write_summary_outputs(rows, tmp_path)

    with (tmp_path / "ablation_eval_summary.csv").open(newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))

    assert header == SUMMARY_COLUMNS
    assert (tmp_path / "ablation_eval_summary.json").is_file()
    assert (tmp_path / "ablation_report.md").is_file()


def test_grouped_independent_replay_hydrates_keep_map_from_metadata():
    replay = [
        {
            "layer": "backbone.grouped",
            "direction": "out",
            "axis": "grouped_independent_keep",
            "before": 16,
            "after": 8,
            "groups": 4,
        }
    ]
    metadata = {
        "applied": [
            {
                "operations": [
                    {
                        "layer": "backbone.grouped",
                        "axis": "grouped_independent_keep",
                        "group_keep_map": {
                            "0": [0, 1],
                            "1": [1, 2],
                            "2": [0, 3],
                            "3": [2, 3],
                        },
                        "per_group_after": 2,
                    }
                ]
            }
        ]
    }

    hydrated = _hydrate_grouped_independent_replay(replay, metadata)

    assert hydrated[0]["group_keep_map"]["1"] == [1, 2]
    assert hydrated[0]["per_group_after"] == 2
