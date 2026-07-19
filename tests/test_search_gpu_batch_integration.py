from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_ga_batch_path_does_not_call_scalar_evaluator() -> None:
    from search.canonicalization import SearchSpaceSpec
    from search.ga.engine import GAConfig, GeneticSearchEngine
    from search.stage1.proxy_evaluator import BatchProxyResult

    space = SearchSpaceSpec(pruning_unit_ids=["a"], precision_layer_ids=["m"], default_precision="FP16")
    engine = GeneticSearchEngine(
        space,
        GAConfig(initial_population_size=16, population_size=8, offspring_size=8, num_generations=2, random_seed=5),
    )
    scalar_calls = 0
    batch_calls = 0
    batch_sizes: list[int] = []

    def scalar(_candidate, _generation):
        nonlocal scalar_calls
        scalar_calls += 1
        return {"F1": 999.0}

    def batch(candidates, generation):
        nonlocal batch_calls
        batch_calls += 1
        batch_sizes.append(len(candidates))
        return BatchProxyResult(
            metrics=[
                {"F1": float(index), "generation": generation, "proxy_backend": "cuda_batched"}
                for index, _candidate in enumerate(candidates)
            ],
            stats={"proxy_backend": "cuda_batched", "gpu_batch_count": 1},
        )

    engine.run(scalar, batch_evaluator=batch)

    assert scalar_calls == 0
    assert batch_calls == 2
    assert batch_sizes == [16, 8]


def test_stage1_batch_evaluator_scores_cache_misses_once() -> None:
    from search.candidate import CandidateGenotype
    from search.canonicalization import SearchSpaceSpec
    from search.stage1.proxy_evaluator import BatchProxyResult, Stage1ProxyEvaluator

    class FakeBatchScorer:
        backend = "cuda_batched"
        device = "cuda:0"
        batch_size = 128
        gpu_batch_count = 0

        def evaluate_batch(self, phenotypes, *, generation: int, outer_round: int):
            self.gpu_batch_count += 1
            return BatchProxyResult(
                metrics=[
                    {
                        "F1": float(index),
                        "L_fisher": 0.0,
                        "L_sqnr": 0.0,
                        "R_size_vs_fp16_deploy": 1.0,
                        "R_bops_vs_fp16_deploy": 1.0,
                        "generation": generation,
                        "outer_round": outer_round,
                    }
                    for index, _phenotype in enumerate(phenotypes)
                ],
                stats={"gpu_batch_count": self.gpu_batch_count},
            )

    space = SearchSpaceSpec(pruning_unit_ids=["a"], precision_layer_ids=["m"], default_precision="FP16")
    scorer = FakeBatchScorer()
    evaluator = Stage1ProxyEvaluator(space, batch_scorer=scorer, proxy_backend="cuda_batched", proxy_device="cuda:0", proxy_batch_size=128)
    candidates = [
        CandidateGenotype(pruning_genes={"a": 1}, precision_genes={"m": "FP16"}),
        CandidateGenotype(pruning_genes={"a": 1}, precision_genes={"m": "FP16"}),
        CandidateGenotype(pruning_genes={"a": 0}, precision_genes={"m": "FP16"}),
    ]

    result = evaluator.evaluate_batch(candidates, generation=0, outer_round=0)

    assert evaluator.scalar_evaluate_call_count == 0
    assert evaluator.batch_evaluate_call_count == 1
    assert evaluator.cache_miss_count == 2
    assert evaluator.unique_phenotype_count == 2
    assert evaluator.last_batch_stats["gpu_batch_count"] == 1
    assert result.stats["unique_phenotype_count"] == 2
    assert result.stats["gpu_batch_count"] == 1
    assert len(result.metrics) == 3


def test_stage1_batch_evaluator_rejects_missing_gpu_scorer() -> None:
    from search.candidate import CandidateGenotype
    from search.canonicalization import SearchSpaceSpec
    from search.stage1.proxy_evaluator import Stage1ProxyEvaluator

    space = SearchSpaceSpec(pruning_unit_ids=["a"], precision_layer_ids=["m"], default_precision="FP16")
    evaluator = Stage1ProxyEvaluator(space, batch_scorer=None, proxy_backend="cuda_batched", proxy_device="cuda:0", proxy_batch_size=128)

    try:
        evaluator.evaluate_batch([CandidateGenotype(pruning_genes={"a": 1}, precision_genes={"m": "FP16"})], generation=0)
    except RuntimeError as exc:
        assert "gpu_proxy_required_but_not_active" in str(exc)
    else:
        raise AssertionError("expected gpu proxy guard")
    assert evaluator.scalar_evaluate_call_count == 0


def test_stage1_batch_cache_hits_use_current_generation(tmp_path) -> None:
    from search.cache.proxy_cache import ProxyCache
    from search.candidate import CandidateGenotype
    from search.canonicalization import SearchSpaceSpec
    from search.stage1.proxy_evaluator import BatchProxyResult, Stage1ProxyEvaluator

    class FakeBatchScorer:
        def evaluate_batch(self, phenotypes, *, generation: int, outer_round: int):
            return BatchProxyResult(metrics=[{"F1": 1.0, "generation": generation, "outer_round": outer_round} for _ in phenotypes])

    space = SearchSpaceSpec(pruning_unit_ids=["a"], precision_layer_ids=["m"], default_precision="FP16")
    evaluator = Stage1ProxyEvaluator(
        space,
        cache=ProxyCache(tmp_path / "proxy.jsonl"),
        batch_scorer=FakeBatchScorer(),
        proxy_backend="cuda_batched",
        proxy_device="cuda:0",
        proxy_batch_size=128,
    )
    candidate = CandidateGenotype(pruning_genes={"a": 1}, precision_genes={"m": "FP16"})

    evaluator.evaluate_batch([candidate], generation=0, outer_round=0)
    cached = evaluator.evaluate_batch([candidate], generation=4, outer_round=0)

    assert cached.metrics[0]["cache_hit"] is True
    assert cached.metrics[0]["generation"] == 4


def test_stage1_proxy_cache_persists_hash_not_redundant_phenotype(
    tmp_path,
) -> None:
    from search.cache.proxy_cache import ProxyCache
    from search.candidate import CandidateGenotype
    from search.canonicalization import SearchSpaceSpec
    from search.stage1.proxy_evaluator import BatchProxyResult, Stage1ProxyEvaluator

    class FakeBatchScorer:
        def evaluate_batch(self, phenotypes, *, generation: int, outer_round: int):
            return BatchProxyResult(
                metrics=[{"F1": 1.0} for _phenotype in phenotypes]
            )

    cache_path = tmp_path / "proxy.jsonl"
    space = SearchSpaceSpec(
        pruning_unit_ids=["a"],
        precision_layer_ids=["m"],
        default_precision="FP16",
    )
    evaluator = Stage1ProxyEvaluator(
        space,
        cache=ProxyCache(cache_path),
        batch_scorer=FakeBatchScorer(),
        proxy_backend="cuda_batched",
        proxy_device="cuda:0",
        proxy_batch_size=128,
    )
    candidate = CandidateGenotype(
        pruning_genes={"a": 1},
        precision_genes={"m": "FP16"},
        meta={"group_mask": {f"unit_{index:05d}": 1 for index in range(1000)}},
    )

    first = evaluator.evaluate_batch([candidate], generation=0).metrics[0]
    cached = evaluator.evaluate_batch([candidate], generation=1).metrics[0]
    disk_row = json.loads(cache_path.read_text(encoding="utf-8").splitlines()[0])

    assert "phenotype" not in disk_row
    assert len(disk_row["phenotype_archive_hash"]) == 64
    assert disk_row["deployment_candidate_hash"] == first["candidate_hash"]
    assert cached["phenotype"] == first["phenotype"]
    assert cached["phenotype"]["metadata"]["group_mask"] == {
        f"unit_{index:05d}": 1 for index in range(1000)
    }


def test_lidar_large_population_requires_cuda_batched_backend() -> None:
    from search.orchestration.lidar_pyramid_search import _require_gpu_proxy_if_needed

    try:
        _require_gpu_proxy_if_needed(
            proxy_cfg={"device": "cuda:0", "batch_size": 128},
            search_cfg={"initial_population_size": 1024},
            actual_backend="scalar_cpu",
        )
    except RuntimeError as exc:
        assert "gpu_proxy_required_but_not_active" in str(exc)
    else:
        raise AssertionError("expected gpu_proxy_required_but_not_active")


def test_torch_batched_proxy_matches_scalar_reference_on_small_candidates() -> None:
    import pytest
    import torch

    from search.candidate import CandidateGenotype
    from search.canonicalization import SearchSpaceSpec, canonicalize_candidate
    from search.proxy.gpu_batch_proxy import TorchBatchedProxyScorer
    from search.proxy.objective import ProxyObjective, ProxyObjectiveConfig
    from search.proxy.runtime_shape_profiler import RuntimeLayerShape
    from search.proxy.size_proxy import SizeProxy
    from search.proxy.bops_proxy import BOPSProxy
    from search.proxy.sqnr_proxy import SQNRProxy
    from search.proxy.fisher_proxy import FisherStatistics, FisherTaylorProxy
    from search.proxy.parameter_slice_resolver import ParameterSlice

    model = torch.nn.Sequential()
    model.add_module("conv", torch.nn.Conv2d(2, 2, kernel_size=1, bias=False))
    with torch.no_grad():
        model.conv.weight[:] = torch.tensor([[[[1.0]], [[2.0]]], [[[3.0]], [[4.0]]]])
    stats = FisherStatistics(
        gradients={"conv.weight": torch.full_like(model.conv.weight, 0.1)},
        fisher_diag={"conv.weight": torch.full_like(model.conv.weight, 0.01)},
    )
    unit_slices = {
        "u0": [ParameterSlice("conv.weight", "conv", 0, (0,), "prune_weight_slice")],
        "u1": [ParameterSlice("conv.weight", "conv", 0, (1,), "prune_weight_slice")],
    }
    runtime_shapes = [
        RuntimeLayerShape(
            module_path="conv",
            call_index=0,
            module_type="Conv2d",
            input_shape=(1, 2, 4, 4),
            output_shape=(1, 2, 4, 4),
            c_in=2,
            c_out=2,
            h_out=4,
            w_out=4,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
            weight_shape=(2, 2, 1, 1),
            precision_group_id="conv",
            macs=64.0,
        )
    ]
    space = SearchSpaceSpec(pruning_unit_ids=["u0", "u1"], precision_layer_ids=["conv"], default_precision="FP16")
    scalar = ProxyObjective(
        fisher=FisherTaylorProxy(model, statistics=stats, unit_to_parameter_names=unit_slices),
        sqnr=SQNRProxy(model, unit_to_parameter_slices=unit_slices),
        size=SizeProxy(model, unit_to_parameter_slices=unit_slices),
        bops=BOPSProxy(model, unit_to_parameter_slices=unit_slices, runtime_shapes=runtime_shapes),
        config=ProxyObjectiveConfig(bops_threshold=None),
    )
    scorer = TorchBatchedProxyScorer.from_components(
        model=model,
        space=space,
        unit_to_parameter_slices=unit_slices,
        fisher_statistics=stats,
        runtime_shapes=runtime_shapes,
        normalization=scalar.normalization,
        config=scalar.config,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        batch_size=8,
    )
    genotypes = [
        CandidateGenotype({"u0": 1, "u1": 1}, {"conv": "FP16"}),
        CandidateGenotype({"u0": 0, "u1": 1}, {"conv": "FP16"}),
        CandidateGenotype({"u0": 1, "u1": 0}, {"conv": "INT8"}),
    ]
    phenotypes = [canonicalize_candidate(row, space) for row in genotypes]
    batched = scorer.evaluate_batch(phenotypes, generation=0, outer_round=0).metrics

    for phenotype, row in zip(phenotypes, batched):
        ref = scalar.evaluate(phenotype)
        for key in (
            "L_fisher",
            "L_sqnr",
            "L_quant_incremental",
            "L_prune_x_quant_prior",
            "L_MAC_weighted",
            "R_MAC",
            "int8_macs_share_full",
            "R_size_vs_fp16_deploy",
            "R_bops_vs_fp16_deploy",
            "F1",
        ):
            assert row[key] == pytest.approx(ref[key], rel=1e-5, abs=1e-6)


def test_torch_batched_proxy_size_matches_scalar_when_layer_has_bias() -> None:
    import pytest
    import torch

    from search.candidate import CandidateGenotype
    from search.canonicalization import SearchSpaceSpec, canonicalize_candidate
    from search.proxy.bops_proxy import BOPSProxy
    from search.proxy.fisher_proxy import FisherStatistics, FisherTaylorProxy
    from search.proxy.gpu_batch_proxy import TorchBatchedProxyScorer
    from search.proxy.objective import ProxyObjective, ProxyObjectiveConfig
    from search.proxy.parameter_slice_resolver import ParameterSlice
    from search.proxy.runtime_shape_profiler import RuntimeLayerShape
    from search.proxy.size_proxy import SizeProxy
    from search.proxy.sqnr_proxy import SQNRProxy

    model = torch.nn.Sequential()
    model.add_module("conv", torch.nn.Conv2d(2, 2, kernel_size=1, bias=True))
    stats = FisherStatistics(
        gradients={"conv.weight": torch.full_like(model.conv.weight, 0.1), "conv.bias": torch.full_like(model.conv.bias, 0.1)},
        fisher_diag={"conv.weight": torch.full_like(model.conv.weight, 0.01), "conv.bias": torch.full_like(model.conv.bias, 0.01)},
    )
    unit_slices = {
        "u0": [
            ParameterSlice("conv.weight", "conv", 0, (0,), "prune_weight_slice"),
            ParameterSlice("conv.bias", "conv", 0, (0,), "prune_bias_slice"),
        ]
    }
    runtime_shapes = [
        RuntimeLayerShape(
            module_path="conv",
            call_index=0,
            module_type="Conv2d",
            input_shape=(1, 2, 4, 4),
            output_shape=(1, 2, 4, 4),
            c_in=2,
            c_out=2,
            h_out=4,
            w_out=4,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
            weight_shape=tuple(model.conv.weight.shape),
            precision_group_id="conv",
            macs=64.0,
        )
    ]
    space = SearchSpaceSpec(pruning_unit_ids=["u0"], precision_layer_ids=["conv"], default_precision="FP16")
    scalar = ProxyObjective(
        fisher=FisherTaylorProxy(model, statistics=stats, unit_to_parameter_names=unit_slices),
        sqnr=SQNRProxy(model, unit_to_parameter_slices=unit_slices),
        size=SizeProxy(model, unit_to_parameter_slices=unit_slices),
        bops=BOPSProxy(model, unit_to_parameter_slices=unit_slices, runtime_shapes=runtime_shapes),
        config=ProxyObjectiveConfig(bops_threshold=None),
    )
    scorer = TorchBatchedProxyScorer.from_components(
        model=model,
        space=space,
        unit_to_parameter_slices=unit_slices,
        fisher_statistics=stats,
        runtime_shapes=runtime_shapes,
        normalization=scalar.normalization,
        config=scalar.config,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        batch_size=8,
    )
    phenotype = canonicalize_candidate(CandidateGenotype({"u0": 1}, {"conv": "FP16"}), space)

    batched = scorer.evaluate_batch([phenotype], generation=0, outer_round=0).metrics[0]
    ref = scalar.evaluate(phenotype)

    assert batched["R_size_vs_fp16_deploy"] == pytest.approx(ref["R_size_vs_fp16_deploy"], rel=1e-5, abs=1e-6)


def test_torch_batched_proxy_builds_with_cpu_fisher_stats_and_cuda_model() -> None:
    import pytest
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for mixed-device Fisher stat coverage")

    from search.canonicalization import SearchSpaceSpec
    from search.proxy.fisher_proxy import FisherStatistics
    from search.proxy.gpu_batch_proxy import TorchBatchedProxyScorer
    from search.proxy.normalization import NormalizationStats
    from search.proxy.objective import ProxyObjectiveConfig
    from search.proxy.parameter_slice_resolver import ParameterSlice
    from search.proxy.runtime_shape_profiler import RuntimeLayerShape

    model = torch.nn.Sequential()
    model.add_module("conv", torch.nn.Conv2d(2, 2, kernel_size=1, bias=False))
    model = model.cuda()
    cpu_weight_shape = tuple(model.conv.weight.shape)
    stats = FisherStatistics(
        gradients={"conv.weight": torch.full(cpu_weight_shape, 0.1)},
        fisher_diag={"conv.weight": torch.full(cpu_weight_shape, 0.01)},
    )
    scorer = TorchBatchedProxyScorer.from_components(
        model=model,
        space=SearchSpaceSpec(pruning_unit_ids=["u0"], precision_layer_ids=["conv"], default_precision="FP16"),
        unit_to_parameter_slices={"u0": [ParameterSlice("conv.weight", "conv", 0, (0,), "prune_weight_slice")]},
        fisher_statistics=stats,
        runtime_shapes=[
            RuntimeLayerShape(
                module_path="conv",
                call_index=0,
                module_type="Conv2d",
                input_shape=(1, 2, 4, 4),
                output_shape=(1, 2, 4, 4),
                c_in=2,
                c_out=2,
                h_out=4,
                w_out=4,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                dilation=(1, 1),
                groups=1,
                weight_shape=cpu_weight_shape,
                precision_group_id="conv",
                macs=64.0,
            )
        ],
        normalization=NormalizationStats(),
        config=ProxyObjectiveConfig(bops_threshold=None),
        device="cuda:0",
        batch_size=8,
    )

    assert scorer.action_fisher_cost.device.type == "cuda"
    assert torch.isfinite(scorer.action_fisher_cost).all()


def test_lidar_runner_ga_uses_proxy_backend_from_evaluator(tmp_path) -> None:
    import json
    from types import SimpleNamespace

    from search.canonicalization import SearchSpaceSpec
    from search.orchestration.lidar_pyramid_search import LidarPyramidTwoStageSearch
    from search.stage1.proxy_evaluator import BatchProxyResult

    class FakeProxy:
        proxy_backend = "cuda_batched"
        cache_miss_count = 0
        cache_hit_count = 0
        gpu_batch_count = 0
        scalar_evaluate_call_count = 0
        batch_evaluate_call_count = 0

        def evaluate(self, *_args, **_kwargs):
            self.scalar_evaluate_call_count += 1
            raise AssertionError("scalar evaluator must not be called")

        def evaluate_batch(self, genotypes, *, generation: int, outer_round: int):
            self.batch_evaluate_call_count += 1
            self.gpu_batch_count += 1
            self.cache_miss_count += len(genotypes)
            return BatchProxyResult(
                metrics=[
                    {
                        "F1": float(index),
                        "L_fisher": 0.0,
                        "L_sqnr": 0.0,
                        "R_size": 1.0,
                        "R_size_vs_fp16_deploy": 1.0,
                        "R_bops": 1.0,
                        "R_bops_vs_fp16_deploy": 1.0,
                        "int8_macs_ratio": 0.0,
                        "generation": generation,
                        "outer_round": outer_round,
                    }
                    for index, _genotype in enumerate(genotypes)
                ],
                stats={"proxy_backend": "cuda_batched", "gpu_batch_count": 1},
            )

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_manifest.json").write_text("{}", encoding="utf-8")
    # Keep this routing test's finite genotype space larger than the number of
    # globally deduplicated candidates requested across both generations.
    space = SearchSpaceSpec(
        pruning_unit_ids=["a", "b", "c", "d"],
        precision_layer_ids=["m"],
        default_precision="FP16",
    )
    context = SimpleNamespace(search_space=space, pruning_action_catalog=SimpleNamespace(actions=[]))
    proxy = FakeProxy()
    runner = LidarPyramidTwoStageSearch(config={}, checkpoint=tmp_path / "model.pth", output_root=tmp_path)

    runner._run_ga(
        context,
        proxy,
        real_evaluator=SimpleNamespace(),
        run_dir=run_dir,
        search_cfg={
            "initial_population_size": 16,
            "population_size": 8,
            "offspring_size": 8,
            "generations_per_round": 2,
            "topk_real": 1,
        },
        stage1_only=True,
    )

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert proxy.scalar_evaluate_call_count == 0
    assert proxy.batch_evaluate_call_count == 2
    assert manifest["scalar_evaluate_call_count"] == 0
    assert manifest["gpu_batch_count"] == 2
