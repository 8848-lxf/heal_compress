from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_outer_round_bops_target_schedule_uses_explicit_v2_targets() -> None:
    from search.proxy.objective import bops_target_for_outer_round

    schedule = {"targets": [0.230, 0.213, 0.197, 0.180]}

    assert [bops_target_for_outer_round(idx, 4, schedule) for idx in range(4)] == pytest.approx(
        [0.230, 0.213, 0.197, 0.180]
    )
    assert bops_target_for_outer_round(0, 1, schedule) == pytest.approx(0.230)


def test_proxy_objective_does_not_normalize_fisher_or_sqnr_terms_for_v2_contract() -> None:
    from search.candidate import CandidatePhenotype, PrecisionDecision
    from search.proxy.normalization import NormalizationStats
    from search.proxy.objective import ProxyObjective, ProxyObjectiveConfig

    class ConstantProxy:
        def __init__(self, value: float) -> None:
            self.value = value

        def evaluate(self, _phenotype: CandidatePhenotype) -> float:
            return self.value

    class Size:
        def evaluate_breakdown(self, _phenotype: CandidatePhenotype) -> dict[str, float]:
            return {"R_size_vs_fp32": 0.50}

    class Bops:
        def evaluate_breakdown(self, _phenotype: CandidatePhenotype) -> dict[str, float]:
            return {"R_bops_vs_fp32": 0.30}

    objective = ProxyObjective(
        fisher=ConstantProxy(0.10),
        sqnr=ConstantProxy(0.20),
        size=Size(),
        bops=Bops(),
        normalization=NormalizationStats(medians={"L_fisher": 100.0, "L_sqnr": 100.0}),
        config=ProxyObjectiveConfig(
            alpha_fisher=0.55,
            beta_sqnr=0.25,
            gamma_size=0.05,
            delta_bops=0.15,
            bops_threshold=0.25,
            bops_penalty_formula="squared_relative_excess",
        ),
    )

    metrics = objective.evaluate(CandidatePhenotype(precision_profile={"conv": PrecisionDecision("FP16", "FP16")}))

    expected_penalty = max(0.0, 0.30 / 0.25 - 1.0) ** 2
    expected = 0.55 * 0.10 + 0.25 * 0.20 + 0.05 * 0.50 + 0.15 * expected_penalty
    assert metrics["F1"] == pytest.approx(expected)
    assert metrics["normalization"]["applied_to_objective"] is False


def test_stage2_admission_filters_repaired_bops_budget_and_control_candidates() -> None:
    from search.candidate import CandidatePhenotype, PrecisionDecision
    from search.stage2.admission import Stage2AdmissionPolicy

    policy = Stage2AdmissionPolicy(bops_target=0.230)
    pruned = CandidatePhenotype(pruned_unit_ids=["u0"], precision_profile={"conv": PrecisionDecision("FP16", "FP16")})
    int8_only = CandidatePhenotype(
        precision_profile={"conv": PrecisionDecision("INT8", "INT8")},
        metadata={"stage1_legalized_group_profile": {"pg0": "INT8"}},
    )
    control = CandidatePhenotype(
        precision_profile={"conv": PrecisionDecision("FP16", "FP16")},
        metadata={"stage1_legalized_group_profile": {"pg0": "FP16"}},
    )

    assert policy.check_repaired_candidate(pruned, {"R_bops": 0.235}).accepted is True
    assert policy.check_repaired_candidate(int8_only, {"R_bops_vs_fp32": 0.230}).accepted is True
    assert policy.check_repaired_candidate(pruned, {"R_bops": 0.236}).accepted is False
    rejected = policy.check_repaired_candidate(control, {"R_bops": 0.100})
    assert rejected.accepted is False
    assert rejected.reason == "control_only_repaired_candidate"


def test_deployment_signature_uses_physical_hash_and_realized_profile_hash() -> None:
    from search.stage2.admission import DeploymentSignatureRegistry, deployment_signature, realized_precision_profile_hash

    profile_a = {"conv1": "FP16", "conv2": "INT8"}
    profile_b = {"conv1": "FP16", "conv2": "FP16"}
    hash_a = realized_precision_profile_hash(profile_a)
    hash_b = realized_precision_profile_hash(profile_b)

    assert hash_a != hash_b
    assert deployment_signature("physical-a", hash_a) == deployment_signature("physical-a", hash_a)
    assert deployment_signature("physical-a", hash_a) != deployment_signature("physical-a", hash_b)

    registry = DeploymentSignatureRegistry()
    first = registry.register("physical-a", profile_a)
    duplicate = registry.register("physical-a", dict(reversed(list(profile_a.items()))))
    different_profile = registry.register("physical-a", profile_b)

    assert first.accepted is True
    assert duplicate.accepted is False
    assert duplicate.reason == "duplicate_deployment_signature"
    assert different_profile.accepted is True


def test_realized_bops_uses_physical_layer_shapes_and_actual_realized_precision() -> None:
    from search.proxy.runtime_shape_profiler import RuntimeLayerShape
    from search.stage2.realized_bops import compute_realized_bops

    model = torch.nn.Sequential()
    model.add_module("conv", torch.nn.Conv2d(4, 4, kernel_size=1, bias=False))
    original_shape = RuntimeLayerShape(
        module_path="conv",
        module_type="Conv2d",
        call_index=0,
        input_shape=(1, 4, 2, 2),
        output_shape=(1, 8, 2, 2),
        c_in=4,
        c_out=8,
        h_out=2,
        w_out=2,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        groups=1,
        weight_shape=(8, 4, 1, 1),
        macs=128,
        precision_group_id="pg0",
    )

    report = compute_realized_bops(
        model,
        runtime_shapes=[original_shape],
        realized_precision_profile={"conv": "INT8"},
    )

    expected_macs = 2 * 2 * 1 * 1 * 4 * 4
    expected_bops = expected_macs * 8 * 8
    expected_fp32 = 128 * 32 * 32
    assert report["weighted_layer_count"] == 1
    assert report["realized_int8_layer_count"] == 1
    assert report["R_BOPS_realized"] == pytest.approx(expected_bops / expected_fp32)
    assert report["layers"][0]["C_out_after"] == 4


def test_round_winner_selection_excludes_control_only_and_over_budget_candidates(tmp_path: Path) -> None:
    from search.stage2.round_results import write_round_stage2_results

    round_dir = tmp_path / "run" / "round_000"
    hashes = ["control", "over_budget", "eligible"]
    (round_dir / "stage2").mkdir(parents=True)
    (round_dir / "repaired_top5_manifest.json").write_text(
        json.dumps(
            {
                "candidates": [
                    {"candidate_rank": 0, "repaired_phenotype_hash": "control", "repaired_F1": 0.1},
                    {"candidate_rank": 1, "repaired_phenotype_hash": "over_budget", "repaired_F1": 0.2},
                    {"candidate_rank": 2, "repaired_phenotype_hash": "eligible", "repaired_F1": 0.3},
                ]
            }
        ),
        encoding="utf-8",
    )
    scores = {
        "control": {"status": "ok", "F2": 0.01, "control_only": True, "R_BOPS_realized": 0.1},
        "over_budget": {"status": "ok", "F2": 0.02, "control_only": False, "R_BOPS_realized": 0.236},
        "eligible": {
            "status": "ok",
            "F2": 0.50,
            "control_only": False,
            "R_BOPS_realized": 0.230,
            "deployment_signature": "sig-eligible",
        },
    }
    for candidate_hash in hashes:
        candidate_dir = round_dir / "stage2" / candidate_hash
        candidate_dir.mkdir(parents=True)
        (candidate_dir / "stage2_score.json").write_text(json.dumps(scores[candidate_hash]), encoding="utf-8")
        for artifact in ("pruned_checkpoint.pth", "pruned_fp32.onnx", "pruned_qdq.onnx", "engine.plan", "evaluation_300.json"):
            (candidate_dir / artifact).write_bytes(b"x")

    result = write_round_stage2_results(tmp_path / "run", round_index=0, bops_target=0.230)

    assert result["winner"]["candidate_hash"] == "eligible"
    persisted = json.loads((round_dir / "stage2_top5_results.json").read_text(encoding="utf-8"))
    rejected = {row["candidate_hash"]: row["winner_pool_status"] for row in persisted["candidates"]}
    assert rejected["control"] == "control_only"
    assert rejected["over_budget"] == "realized_bops_over_budget"


def test_round_winner_selection_excludes_seen_deployment_signature(tmp_path: Path) -> None:
    from search.stage2.round_results import write_round_stage2_results

    round_dir = tmp_path / "run" / "round_001"
    round_dir.mkdir(parents=True)
    (round_dir / "repaired_top5_manifest.json").write_text(
        json.dumps(
            {
                "candidates": [
                    {"candidate_rank": 0, "repaired_phenotype_hash": "duplicate", "repaired_F1": 0.1},
                    {"candidate_rank": 1, "repaired_phenotype_hash": "fresh", "repaired_F1": 0.2},
                ]
            }
        ),
        encoding="utf-8",
    )
    scores = {
        "duplicate": {"status": "ok", "F2": 0.01, "control_only": False, "R_BOPS_realized": 0.180, "deployment_signature": "sig-old"},
        "fresh": {"status": "ok", "F2": 0.02, "control_only": False, "R_BOPS_realized": 0.181, "deployment_signature": "sig-new"},
    }
    for candidate_hash, score in scores.items():
        candidate_dir = round_dir / "stage2" / candidate_hash
        candidate_dir.mkdir(parents=True)
        (candidate_dir / "stage2_score.json").write_text(json.dumps(score), encoding="utf-8")
        for artifact in ("pruned_checkpoint.pth", "pruned_fp32.onnx", "pruned_qdq.onnx", "engine.plan", "evaluation_300.json"):
            (candidate_dir / artifact).write_bytes(b"x")

    result = write_round_stage2_results(
        tmp_path / "run",
        round_index=1,
        bops_target=0.180,
        seen_deployment_signatures={"sig-old"},
    )

    assert result["winner"]["candidate_hash"] == "fresh"
    assert result["accepted_deployment_signatures"] == ["sig-new"]
    persisted = json.loads((round_dir / "stage2_top5_results.json").read_text(encoding="utf-8"))
    statuses = {row["candidate_hash"]: row["winner_pool_status"] for row in persisted["candidates"]}
    assert statuses["duplicate"] == "duplicate_deployment_signature"


def test_orchestration_legacy_round_best_writer_does_not_overwrite_gated_stage2_winner(tmp_path: Path) -> None:
    from search.orchestration.lidar_pyramid_search import _write_legacy_round_best_candidate_if_missing

    round_dir = tmp_path / "round_000"
    round_dir.mkdir()
    (round_dir / "stage2_top5_results.json").write_text(
        json.dumps(
            {
                "winner": {
                    "candidate_hash": "eligible",
                    "F2": 0.50,
                    "winner_pool_status": "eligible",
                }
            }
        ),
        encoding="utf-8",
    )
    (round_dir / "round_best_candidate.json").write_text(
        json.dumps({"candidate_hash": "eligible", "F2": 0.50, "winner_pool_status": "eligible"}),
        encoding="utf-8",
    )

    _write_legacy_round_best_candidate_if_missing(
        round_dir,
        evaluated_rows=[
            {"candidate_hash": "control", "F2": 0.01, "control_only": True},
            {"candidate_hash": "eligible", "F2": 0.50, "control_only": False},
        ],
        selected_hashes={"control", "eligible"},
    )

    persisted = json.loads((round_dir / "round_best_candidate.json").read_text(encoding="utf-8"))
    assert persisted["candidate_hash"] == "eligible"


def test_final_selection_requires_budget_compression_unique_signature_and_full_eval(tmp_path: Path) -> None:
    from search.stage2.final_selection import select_final_winner_from_full_validation

    rows = [
        {
            "candidate_hash": "fast-control",
            "F2_full": 0.01,
            "R_BOPS_realized": 0.10,
            "pruned_unit_count": 0,
            "realized_int8_layer_count": 0,
            "num_evaluated_frames": 1789,
            "num_skipped_frames": 0,
            "deployment_signature": "sig-control",
        },
        {
            "candidate_hash": "over-budget",
            "F2_full": 0.02,
            "R_BOPS_realized": 0.19,
            "pruned_unit_count": 10,
            "realized_int8_layer_count": 0,
            "num_evaluated_frames": 1789,
            "num_skipped_frames": 0,
            "deployment_signature": "sig-over",
        },
        {
            "candidate_hash": "eligible",
            "F2_full": 0.50,
            "R_BOPS_realized": 0.18,
            "pruned_unit_count": 10,
            "realized_int8_layer_count": 0,
            "num_evaluated_frames": 1789,
            "num_skipped_frames": 0,
            "deployment_signature": "sig-eligible",
        },
    ]

    selected = select_final_winner_from_full_validation(rows, budget=0.185, expected_frames=1789)

    assert selected["status"] == "ok"
    assert selected["winner"]["candidate_hash"] == "eligible"
    assert selected["eligible_count"] == 1


def test_final_selection_reports_no_winner_when_budget_has_no_eligible_candidate() -> None:
    from search.stage2.final_selection import select_final_winner_from_full_validation

    selected = select_final_winner_from_full_validation(
        [
            {
                "candidate_hash": "over-budget",
                "F2_full": 0.02,
                "R_BOPS_realized": 0.19,
                "pruned_unit_count": 10,
                "realized_int8_layer_count": 0,
                "num_evaluated_frames": 1789,
                "num_skipped_frames": 0,
                "deployment_signature": "sig-over",
            }
        ],
        budget=0.185,
        expected_frames=1789,
    )

    assert selected["status"] == "no_final_candidate_meets_budget"
    assert selected["winner"] is None


def test_full_validation_request_requires_warmup_reset_and_exact_frame_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import search.integration.evaluation_provider as provider

    captured: dict[str, object] = {}

    def fake_run(cmd, text, stdout, stderr, env, check):  # noqa: ANN001
        request_index = cmd.index("--request") + 1
        request_path = Path(cmd[request_index])
        captured.update(json.loads(request_path.read_text(encoding="utf-8")))
        output_path = Path(captured["output_path"])
        output_path.write_text(
            json.dumps(
                {
                    "status": "ok",
                    "warmup_executions": 200,
                    "num_evaluated_frames": 1789,
                    "latency_measured_frames": 1789,
                    "num_skipped_frames": 0,
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(provider.subprocess, "run", fake_run)

    result = provider.evaluate_engine_modelopt(
        engine_path="engine.plan",
        checkpoint="model.pth",
        model_config="config.yaml",
        heal_root="/heal",
        device="cuda:0",
        output_dir=tmp_path,
        tensorrt_root="/trt",
        plugin_path=None,
        num_frames=1789,
        warmup_frames=200,
        latency_rounds=3,
        full_validation=True,
        reset_after_warmup=True,
        fail_on_skips=True,
    )

    assert result["status"] == "ok"
    assert captured["full_validation"] is True
    assert captured["reset_after_warmup"] is True
    assert captured["fail_on_skips"] is True
    assert captured["num_frames"] == 1789
    assert captured["warmup_frames"] == 200


def test_stage2_score_persists_deployment_signature_and_realized_bops(tmp_path: Path) -> None:
    from search.cache.real_eval_cache import RealEvalCache
    from search.candidate import CandidatePhenotype, PrecisionDecision
    from search.stage2.lidar_pyramid_real_evaluator import LidarPyramidRealEvaluator
    from search.stage2.objective import Stage2ObjectiveConfig

    evaluator = object.__new__(LidarPyramidRealEvaluator)
    evaluator.objective_config = Stage2ObjectiveConfig(eta_map=0.8, eta_latency=0.2, latency_metric="forward_p50_ms", tau_ap=0.02)
    evaluator.num_frames = 300
    evaluator.warmup_frames = 30
    evaluator.latency_rounds = 3
    evaluator.context = SimpleNamespace(
        eval_manifest_hash="manifest",
        physical_gpu_id=0,
        tensorrt=SimpleNamespace(to_dict=lambda: {}),
    )
    evaluator.real_cache = RealEvalCache(tmp_path / "real_eval_archive.jsonl")
    evaluator._stage2_reference_baseline = lambda: {"mAP": 0.8, "forward_p50_ms": 2.0}  # type: ignore[method-assign]
    evaluator._stage1_manifest_record = lambda _candidate_hash: {}  # type: ignore[method-assign]
    evaluator._load_existing_deployment_evaluation = lambda _destination: None  # type: ignore[method-assign]
    evaluator._deploy_and_evaluate = lambda **_kwargs: {  # type: ignore[method-assign]
        "status": "ok",
        "evaluation": {"status": "ok", "mAP": 0.79, "forward_p50_ms": 2.2},
        "physical_hash": "physical",
        "deployment_hash": "deploy",
        "deployment_signature": "signature",
        "realized_precision_profile_hash": "profile-hash",
        "R_BOPS_realized": 0.18,
        "realized_int8_layer_count": 1,
        "control_only": False,
        "eval_hash": "eval",
        "engine_hash": "engine",
        "engine_path": "engine.plan",
    }

    result = evaluator.evaluate_candidate(
        CandidatePhenotype(precision_profile={"conv": PrecisionDecision("INT8", "INT8")}),
        output_dir=tmp_path / "candidate",
        candidate_hash="candidate",
    )

    assert result["deployment_signature"] == "signature"
    assert result["realized_precision_profile_hash"] == "profile-hash"
    assert result["R_BOPS_realized"] == pytest.approx(0.18)
    assert result["realized_int8_layer_count"] == 1
    persisted = json.loads((tmp_path / "candidate" / "stage2_score.json").read_text(encoding="utf-8"))
    assert persisted["deployment_signature"] == "signature"


def test_stage2_only_output_round_is_inferred_from_candidate_config_path(tmp_path: Path) -> None:
    from search.orchestration.lidar_pyramid_search import _stage2_round_dir_for_candidate_config

    run_dir = tmp_path / "run"
    candidate_config = run_dir / "round_002" / "stage2_candidate_configs" / "candidate_00_hash.json"

    assert _stage2_round_dir_for_candidate_config(run_dir, candidate_config) == run_dir / "round_002"
    assert _stage2_round_dir_for_candidate_config(run_dir, None) == run_dir / "round_000"
