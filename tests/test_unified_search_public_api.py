from __future__ import annotations

import json
from pathlib import Path

import pytest

from search.candidate import CandidateGenotype
from search.canonicalization import SearchSpaceSpec
from search.ga.strict_stage12_v3 import StrictGAConfig
from search.pruning_space.local_domains import LocalPruningDomain
from search.quantization_space.types import QuantizationSearchGroup
from search.stage2.generation_results import write_generation_stage2_results
from search.stage2.greedy_anchor_gate import (
    GreedyAnchorGatePolicy,
    apply_greedy_anchor_gate,
    build_greedy_anchor_manifest,
    load_greedy_anchor,
)
from search.unified.artifacts import BestEnginePublisher
from search.unified.config import PROJECT_ROOT, load_search_config
from search.unified.families import registered_families
from search.unified.formal import build_strict_stage1
from search.unified.runner import UnifiedSearchRunner
from search.unified.stage1 import Stage1TaylorEvaluator, Stage1TaylorPolicy


def test_all_four_family_templates_resolve_and_dry_run(tmp_path: Path) -> None:
    families = registered_families()
    assert {row.family_id for row in families} == {
        "lidar_pyramid",
        "heal_lidar_disco",
        "heal_lidar_fcooper",
        "heal_lidar_v2xvit",
    }
    for family in families:
        config = load_search_config(
            PROJECT_ROOT / family.config_template,
            allow_unresolved=True,
        )
        result = UnifiedSearchRunner(
            config,
            output_root=tmp_path / family.family_id,
        ).run(dry_run=True)
        assert result["status"] == "dry_run_ok"
        assert result["family_id"] == family.family_id
        assert result["full_search_executed"] is False
        assert result["greedy_beam_recovery"]["beam_width"] == 8
        assert config.payload["search"]["protocol"] == "strict_stage12_v3"
        assert config.payload["search"]["population_size"] == 64
        assert config.payload["search"]["generations_per_round"] == 10


def test_all_four_families_dispatch_the_unified_backend_contract(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def backend(config, output_root):
        calls.append(config.family.family_id)
        engine = output_root / f"{config.family.family_id}.plan"
        engine.write_bytes(config.family.family_id.encode("ascii"))
        return {"status": "ok", "best": {"engine_path": str(engine)}}

    for family in registered_families():
        config = load_search_config(
            PROJECT_ROOT / family.config_template,
            allow_unresolved=True,
        )
        result = UnifiedSearchRunner(
            config,
            output_root=tmp_path / family.family_id,
            backends={family.runner_kind: backend},
        ).run()
        assert result["status"] == "ok"
        assert result["best_engine_publication"]["status"] == "published"
    assert calls == [row.family_id for row in registered_families()]


def test_v2xvit_default_backend_is_a_real_registered_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from search.orchestration.v2xvit_formal_search import V2XViTFormalSearch

    config = load_search_config(
        PROJECT_ROOT / "search/configs/unified/lidar_v2xvit_ga.yaml",
        allow_unresolved=True,
    )

    def fake_run(instance: V2XViTFormalSearch) -> dict[str, object]:
        engine = instance.output_root / "v2xvit.plan"
        engine.write_bytes(b"formal-v2xvit-executor")
        return {"status": "ok", "best": {"engine_path": str(engine)}}

    monkeypatch.setattr(V2XViTFormalSearch, "run", fake_run)
    result = UnifiedSearchRunner(
        config,
        output_root=tmp_path / "v2xvit",
    ).run()
    assert result["status"] == "ok"
    assert result["best_engine_publication"]["status"] == "published"


def test_stage1_activation_taylor_is_explicitly_switchable() -> None:
    disabled = Stage1TaylorEvaluator(
        structural_proxy=lambda _value: {"J_struct": 1.0},
        weight_quantization_proxy=lambda _value: {"J_WQ": 2.0},
        activation_quantization_proxy=lambda _value: {"J_AQ": 3.0},
        policy=Stage1TaylorPolicy(include_activation_taylor=False),
    )
    enabled = Stage1TaylorEvaluator(
        structural_proxy=lambda _value: {"J_struct": 1.0},
        weight_quantization_proxy=lambda _value: {"J_WQ": 2.0},
        activation_quantization_proxy=lambda _value: {"J_AQ": 3.0},
        policy=Stage1TaylorPolicy(include_activation_taylor=True),
    )
    assert disabled(None)["J_total"] == 3.0
    assert disabled(None)["activation_taylor_included"] is False
    assert enabled(None)["J_total"] == 6.0
    assert enabled(None)["activation_taylor_included"] is True
    with pytest.raises(ValueError, match="proxy_missing"):
        Stage1TaylorEvaluator(
            structural_proxy=lambda _value: 1.0,
            weight_quantization_proxy=lambda _value: 2.0,
            activation_quantization_proxy=None,
            policy=Stage1TaylorPolicy(include_activation_taylor=True),
        )


def test_greedy_anchor_gate_is_capped_at_point_005(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="0_0_005"):
        GreedyAnchorGatePolicy(tolerance=0.005001)
    manifest = build_greedy_anchor_manifest(
        [
            {
                "candidate_hash": "greedy",
                "budgets": [0.1],
                "mAP": 0.60,
                "forward_p50_ms": 10.0,
            }
        ]
    )
    path = tmp_path / "anchors.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    anchor = load_greedy_anchor(path, target=0.1)
    rows = apply_greedy_anchor_gate(
        [
            {"candidate_hash": "pass", "status": "ok", "mAP": 0.595},
            {"candidate_hash": "fail", "status": "ok", "mAP": 0.5949},
        ],
        anchor=anchor,
    )
    assert rows[0]["accuracy_gate_passed"] is True
    assert rows[1]["accuracy_gate_passed"] is False


def test_generation_winner_rejects_accuracy_gate_failure(tmp_path: Path) -> None:
    report = write_generation_stage2_results(
        tmp_path,
        round_index=0,
        generation_index=0,
        bops_target=0.1,
        selection_report={"selected_count": 2},
        candidate_rows=[
            {
                "candidate_hash": "fast_but_unsafe",
                "status": "ok",
                "F2": 0.1,
                "num_evaluated_frames": 500,
                "num_skipped_frames": 0,
                "accuracy_gate_passed": False,
                "accuracy_gate_rejection": "below_anchor_minus_0.005",
            },
            {
                "candidate_hash": "safe",
                "status": "ok",
                "F2": 0.2,
                "num_evaluated_frames": 500,
                "num_skipped_frames": 0,
                "accuracy_gate_passed": True,
            },
        ],
        expected_screening_frames=500,
    )
    assert report["winner"]["candidate_hash"] == "safe"
    assert report["failures"][0]["candidate_hash"] == "fast_but_unsafe"


def test_best_engine_is_published_to_isolated_directory(tmp_path: Path) -> None:
    source = tmp_path / "candidate.plan"
    source.write_bytes(b"small-test-engine")
    publication = BestEnginePublisher(tmp_path / "best").publish(
        {"best": {"engine_path": str(source)}},
        family_id="lidar_pyramid",
    )
    assert publication.status == "published"
    assert Path(publication.destination).read_bytes() == source.read_bytes()


def _strict_space() -> SearchSpaceSpec:
    domain = LocalPruningDomain(
        domain_id="cnn::a",
        root_module_path="a",
        root_axis="out",
        scope_id="a",
        kind="dense",
        original_width=8,
        total_original_width=8,
        ordered_unit_ids=("u0",),
        legal_widths=(4, 8),
        width_to_pruned_unit_ids={4: ("u0",), 8: ()},
        unit_root_indices={"u0": (0,)},
    )
    precision = QuantizationSearchGroup(
        group_id="precision::a",
        module_paths=("a",),
        canonical_node_ids=("n0",),
        allowed_precisions=("FP32", "FP16", "INT8"),
        protected=False,
        protection_reason="",
        ordering=0,
        parameter_count=8,
        baseline_macs=8,
    )
    return SearchSpaceSpec(
        pruning_unit_ids=["u0"],
        precision_layer_ids=["a"],
        quantization_groups=(precision,),
        pruning_domains=(domain,),
        default_precision="FP32",
    )


def test_strict_stage1_switch_controls_real_activation_cache_calls() -> None:
    class Structure:
        def pruning_action_breakdown(self, _left, _right):
            return {"delta_J_prune": 1.0}

    class Weight:
        def weight_quantization_action_breakdown(self, _left, _right):
            return {"delta_J_WQ": 2.0}

    class Activation:
        calls = 0

        def action_breakdown(self, _left, _right):
            self.calls += 1
            return {"delta_J_AQ": 3.0}

    space = _strict_space()
    baseline = CandidateGenotype(
        pruning_width_genes={"cnn::a": 8},
        precision_genes={"precision::a": "FP32"},
    )
    candidate = CandidateGenotype(
        pruning_width_genes={"cnn::a": 8},
        precision_genes={"precision::a": "FP16"},
    )
    common = {
        "space": space,
        "baseline": baseline,
        "structure_proxy": Structure(),
        "weight_proxy": Weight(),
        "bops_evaluator": lambda _value: {"R_bops_vs_fp32": 0.1},
        "size_evaluator": lambda _value: {
            "R_parameter_retention": 1.0,
            "R_size_vs_fp32": 0.5,
        },
        "target": 0.1,
    }
    config_path = PROJECT_ROOT / "search/configs/unified/lidar_pyramid_ga.yaml"
    disabled_config = load_search_config(
        config_path,
        overrides={"proxy.include_activation_taylor": False},
        allow_unresolved=True,
    )
    disabled = build_strict_stage1(
        disabled_config,
        activation_cache=None,
        **common,
    )(candidate)
    assert disabled["J_AQ"] == 0.0
    assert disabled["activation_taylor_included"] is False

    cache = Activation()
    enabled_config = load_search_config(
        config_path,
        overrides={"proxy.include_activation_taylor": True},
        allow_unresolved=True,
    )
    enabled = build_strict_stage1(
        enabled_config,
        activation_cache=cache,
        **common,
    )(candidate)
    assert enabled["J_AQ"] == 3.0
    assert enabled["activation_taylor_included"] is True
    assert cache.calls == 1


def test_strict_ga_rejects_tolerances_above_point_005() -> None:
    with pytest.raises(ValueError, match="bops_tolerance"):
        StrictGAConfig(target_bops_retention=0.1, tolerance_abs=0.00501)
    with pytest.raises(ValueError, match="accuracy_tolerance"):
        StrictGAConfig(
            target_bops_retention=0.1,
            stage2_accuracy_tolerance=0.00501,
        )
