from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_ga_config_supports_initial_population_and_offspring_size() -> None:
    from search.ga.engine import GAConfig

    config = GAConfig(initial_population_size=1024, population_size=512, offspring_size=512)

    assert config.initial_population_size == 1024
    assert config.population_size == 512
    assert config.offspring_size == 512


def test_large_population_yaml_contains_required_counts() -> None:
    import yaml

    path = Path("search/configs/lidar_pyramid_joint_search_large_population.yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert data["search"]["initial_population_size"] == 1024
    assert data["search"]["population_size"] == 512
    assert data["search"]["offspring_size"] == 512
    assert data["pruning"]["grouped_conv"]["position_mode"] == "independent_group_topk"


def test_ga_evaluates_full_initial_population_before_active_population() -> None:
    from search.candidate import CandidateGenotype
    from search.canonicalization import SearchSpaceSpec
    from search.ga.engine import GAConfig, GeneticSearchEngine

    calls: list[int] = []
    space = SearchSpaceSpec(pruning_unit_ids=["a"], precision_layer_ids=["m"], default_precision="FP16")
    engine = GeneticSearchEngine(
        space,
        GAConfig(
            initial_population_size=8,
            population_size=4,
            offspring_size=4,
            num_generations=2,
            random_seed=3,
        ),
    )

    def evaluator(candidate: CandidateGenotype, generation: int) -> dict[str, float]:
        calls.append(generation)
        return {"F1": float(candidate.pruning_genes.get("a", 1))}

    engine.run(evaluator)

    assert calls.count(0) == 8
    assert calls.count(1) == 4


def test_4090_readiness_config_uses_fresh_matched_e67_entropy_gate() -> None:
    import yaml

    path = Path("search/configs/lidar_pyramid_4090_explicit_qdq_readiness.yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert data["runtime"]["tensorrt_root"] == "/home/lixingfeng/UniAD_examine/HEAL/prune_model/TensorRT-10.9_x86_cu118"
    assert data["runtime"]["gpu_id"] == "4"
    assert data["runtime"]["plugin_boundary_dtype"] == "fp16"
    assert data["runtime"]["allow_foreign_gpu_processes"] is False
    assert data["runtime"]["max_gpu_utilization_pct"] == 20
    assert data["baselines"]["precisions"] == ["strict_fp16", "matched_legacy_int8"]
    assert data["proxy"]["quant_activation_calibration_backend"] == "tensorrt_entropy_calibration2"
    assert data["proxy"]["quant_calibration_batches"] == 200
    assert data["proxy"]["quant_calibration_force_rebuild"] is True
    assert data["stage2"]["num_frames"] == 200
    assert data["stage2"]["reset_after_warmup"] is True

    fp32_path = Path(
        "search/configs/lidar_pyramid_4090_strongly_typed_e67_fp32_readiness.yaml"
    )
    fp32 = yaml.safe_load(fp32_path.read_text(encoding="utf-8"))
    assert fp32["runtime"]["gpu_id"] == "4"
    assert fp32["runtime"]["plugin_boundary_dtype"] == "fp32"
    assert fp32["baselines"]["precisions"] == ["matched_legacy_int8"]
    assert fp32["proxy"]["quant_calibration_npz_manifest"] == data["proxy"]["quant_calibration_npz_manifest"]
    assert fp32["stage2"] == data["stage2"]


def test_4090_stage_a_config_is_per_generation_top5_at_fixed_budget() -> None:
    import yaml

    path = Path("search/configs/lidar_pyramid_4090_ga_stage_a.yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert data["search"]["initial_population_size"] == 1024
    assert data["search"]["population_size"] == 512
    assert data["search"]["offspring_size"] == 512
    assert data["search"]["generations_per_round"] == 5
    assert data["search"]["per_generation_stage2"] is True
    assert data["search"]["topk_stage2"] == 5
    assert data["search"]["target_bops_retention"] == 0.21
    assert data["runtime"]["gpu_id"] == "5"
    assert data["runtime"]["allow_foreign_gpu_processes"] is False
    assert data["runtime"]["max_gpu_utilization_pct"] == 20
    assert data["runtime"]["plugin_boundary_dtype"] == "fp32"
    assert data["stage2_parallel"]["enabled"] is True
    assert data["stage2_parallel"]["gpu_ids"] == [4, 5, 6, 7]
    assert data["stage2_parallel"]["allow_controller_process_on_stage1_gpu"] is True
    assert data["search"]["bops_tolerance"] == 0.005
    assert data["stage2"]["num_frames"] == 300
    assert data["stage2"]["target_bops_retention"] == 0.21
    assert data["budget_final"]["num_frames"] == 500
    assert data["budget_final"]["evaluation_offset"] == 300


def test_4090_multigpu_stage2_smoke_uses_full_stage1_and_real_top5() -> None:
    import yaml

    path = Path(
        "search/configs/lidar_pyramid_4090_ga_stage2_multigpu_smoke.yaml"
    )
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert data["runtime"]["gpu_id"] == "5"
    assert data["runtime"]["plugin_boundary_dtype"] == "fp32"
    assert data["baselines"]["build_before_search"] is False
    assert data["search"]["initial_population_size"] == 1024
    assert data["search"]["population_size"] == 512
    assert data["search"]["offspring_size"] == 512
    assert data["search"]["generations_per_round"] == 1
    assert data["search"]["per_generation_stage2"] is True
    assert data["search"]["topk_stage2"] == 5
    assert data["search"]["target_bops_retention"] == 0.21
    assert data["search"]["bops_tolerance"] == 0.005
    assert data["stage2"]["num_frames"] == 10
    assert data["stage2"]["warmup_frames"] == 10
    assert data["stage2"]["reset_after_warmup"] is True
    assert data["stage2"]["max_map_drop"] == 0.75
    assert data["stage2_parallel"]["enabled"] is True
    assert data["stage2_parallel"]["gpu_ids"] == [4, 5, 6, 7]
    assert data["budget_final"] == {}
    assert data["cache"]["fresh_run"] is True
    assert data["cache"]["reuse_external_cache"] is False
