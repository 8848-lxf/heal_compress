from __future__ import annotations

from pathlib import Path

from opencood.tools.compression.latency_lut.schema import LatencyLUTKey, LatencyRecord, write_jsonl
from tools.latency_lut.build_full_engine_calibration_samples import build_named_candidate_preset, parse_args, run


def _record() -> LatencyRecord:
    key = LatencyLUTKey(
        deploy_mode="single_engine_maxK",
        fixed_K=29696,
        module_name="backbone",
        block_name="stage1",
        block_type="conv_block",
        H=16,
        W=16,
        C_in=16,
        C_out=16,
        kernel_size=3,
        stride=1,
        padding=1,
        batch_size=1,
        precision_profile="TRT_FP16",
        weight_precision="FP16",
        activation_precision="FP16",
        compute_precision="FP16",
    )
    return LatencyRecord(
        key=key,
        latency_p50_ms=0.1,
        latency_p90_ms=0.11,
        latency_p95_ms=0.12,
        latency_p99_ms=0.13,
        latency_mean_ms=0.105,
        latency_std_ms=0.01,
        num_warmup=1,
        num_repeat=1,
        timing_method="synthetic",
        status="success",
    )


def test_full_engine_sample_builder_does_not_fake_without_command_template(tmp_path: Path):
    lut = tmp_path / "lut.jsonl"
    output = tmp_path / "full_engine_samples.jsonl"
    report = tmp_path / "report.json"
    write_jsonl([_record()], lut)

    stats = run(
        parse_args(
            [
                "--lut",
                str(lut),
                "--output",
                str(output),
                "--report",
                str(report),
                "--num-samples",
                "2",
            ]
        )
    )

    assert stats["status"] == "skipped_no_engine_builder"
    assert stats["num_samples_written"] == 0
    assert not output.exists() or output.read_text(encoding="utf-8") == ""


def test_named_candidate_preset_contains_baseline_and_light_prune():
    candidates = build_named_candidate_preset(num_samples=10)
    ids = [item["candidate_id"] for item in candidates]

    assert ids[:3] == ["baseline_like_fp16", "baseline_like_fp32", "light_prune_fp16"]
    assert candidates[0]["pruning"]["enabled"] is False
    assert candidates[0]["precision_config"]["default"] == "FP16"
    assert candidates[2]["pruning"]["source"] == "pruning_tool"
    assert candidates[2]["pruning"]["target_keep_ratio"] == 0.875
