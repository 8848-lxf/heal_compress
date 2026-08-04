from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from search.candidate import CandidateGenotype, CandidatePhenotype, PrecisionDecision
from search.orchestration.two_stage_search import TwoStageSearchRunner
from search.proxy.bops_proxy import BOPSProxy
from search.stage2.objective import Stage2ObjectiveConfig, compute_stage2_score


def test_stage2_failure_has_infinite_score() -> None:
    score = compute_stage2_score(
        {"status": "engine_build_failed"},
        baseline={"mAP": 0.5, "forward_mean_ms": 10.0},
        config=Stage2ObjectiveConfig(),
    )
    assert score["F2"] == float("inf")


def test_stage2_objective_uses_map_drop_and_latency_ratio() -> None:
    score = compute_stage2_score(
        {"status": "ok", "mAP": 0.45, "forward_mean_ms": 5.0},
        baseline={"mAP": 0.5, "forward_mean_ms": 10.0},
        config=Stage2ObjectiveConfig(eta_map=1.0, eta_latency=1.0, latency_metric="forward_mean_ms"),
    )
    assert score["F2"] == 0.6


def test_bops_proxy_uses_matching_activation_bits() -> None:
    proxy = BOPSProxy(layer_ops={"a": 10, "b": 10, "c": 10})
    phenotype = CandidatePhenotype(
        pruned_unit_ids=[],
        precision_profile={
            "a": PrecisionDecision("FP32", "FP32", ""),
            "b": PrecisionDecision("FP16", "FP16", ""),
            "c": PrecisionDecision("INT8", "INT8", ""),
        },
    )

    expected = (10 * 32 * 32 + 10 * 16 * 16 + 10 * 8 * 8) / (30 * 32 * 32)
    assert proxy.evaluate(phenotype) == expected


def test_dry_run_does_not_call_stage2(tmp_path: Path) -> None:
    calls = {"stage2": 0}

    def stage2(_phenotype: CandidatePhenotype) -> dict[str, object]:
        calls["stage2"] += 1
        return {"status": "ok"}

    runner = TwoStageSearchRunner(
        output_root=tmp_path,
        pruning_unit_ids=["u1", "u2"],
        precision_layer_ids=["m1"],
        stage2_evaluator=stage2,
        random_seed=1,
    )
    result = runner.run(outer_rounds=1, population_size=4, generations=1, topk_real=1, dry_run=True)

    assert result["dry_run"] is True
    assert calls["stage2"] == 0
    assert (Path(result["run_dir"]) / "run_manifest.json").is_file()


def test_real_mode_requires_stage2_evaluator(tmp_path: Path) -> None:
    runner = TwoStageSearchRunner(
        output_root=tmp_path,
        pruning_unit_ids=["u1"],
        precision_layer_ids=["m1"],
        random_seed=1,
    )

    try:
        runner.run(outer_rounds=1, population_size=2, generations=1, topk_real=1, dry_run=False)
    except RuntimeError as exc:
        assert "real_stage2_evaluator_required" in str(exc)
    else:
        raise AssertionError("non-dry-run search must fail without a real Stage-2 evaluator")


def test_search_code_does_not_modify_original_model() -> None:
    genotype = CandidateGenotype({"u1": 0}, {"m1": "INT8"})
    phenotype = CandidatePhenotype(
        pruned_unit_ids=["u1"],
        precision_profile={"m1": PrecisionDecision("INT8", "INT8", "")},
    )
    assert genotype.pruning_genes["u1"] == 0
    assert phenotype.pruned_unit_ids == ["u1"]


def test_resume_reuses_real_eval_cache(tmp_path: Path) -> None:
    calls = {"stage2": 0}

    def stage2(_phenotype: CandidatePhenotype) -> dict[str, object]:
        calls["stage2"] += 1
        return {
            "status": "ok",
            "mAP": 0.5,
            "AP@0.3": 0.5,
            "AP@0.5": 0.5,
            "AP@0.7": 0.5,
            "forward_mean_ms": 1.0,
        }

    runner = TwoStageSearchRunner(
        output_root=tmp_path,
        pruning_unit_ids=["u1"],
        precision_layer_ids=["m1"],
        stage2_evaluator=stage2,
        random_seed=0,
    )
    baseline = {"mAP": 0.5, "forward_mean_ms": 1.0}
    runner.run(outer_rounds=1, population_size=2, generations=1, topk_real=1, dry_run=False, baseline=baseline)
    runner.run(outer_rounds=1, population_size=2, generations=1, topk_real=1, dry_run=False, baseline=baseline)

    assert calls["stage2"] == 1


def test_real_evaluator_reuses_single_fp32_accuracy_and_latency_reference(
    tmp_path: Path,
) -> None:
    from search.stage2.lidar_pyramid_real_evaluator import LidarPyramidRealEvaluator

    evaluator = object.__new__(LidarPyramidRealEvaluator)
    evaluator.objective_config = Stage2ObjectiveConfig(latency_metric="forward_p50_ms")
    calls: list[str] = []

    def fake_baseline(precision: str, *, full_validation: bool = False):
        calls.append(precision)
        if precision == "strict_fp32":
            return {"status": "ok", "mAP": 0.75, "forward_p50_ms": 4.0}
        if precision == "strict_fp16":
            return {"status": "ok", "mAP": 0.70, "forward_p50_ms": 2.0}
        raise AssertionError(precision)

    evaluator.evaluate_original_baseline = fake_baseline  # type: ignore[method-assign]
    evaluator.run_dir = tmp_path / "nonexistent"

    baseline = evaluator._stage2_reference_baseline()

    assert calls == ["strict_fp32"]
    assert baseline["mAP"] == 0.75
    assert baseline["forward_p50_ms"] == 4.0
    assert baseline["accuracy_reference"] == "original_strict_fp32"
    assert baseline["latency_reference"] == "original_strict_fp32"
