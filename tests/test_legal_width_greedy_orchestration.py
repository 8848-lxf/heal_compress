from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pruning.types import AtomicPruneUnit


def _context_and_proxy():
    from search.canonicalization import (
        SearchSpaceSpec,
        canonicalize_legal_width_candidate,
    )
    from search.decoding.fixed_taylor_width_decoder import FixedTaylorWidthDecoder
    from search.hashing import candidate_hash
    from search.space.legal_width_inventory import build_legal_width_inventory

    units = [
        AtomicPruneUnit(
            "scope",
            "conv",
            "out",
            [index],
            [f"c{index}"],
            float(index),
            _stable_id=f"u{index}",
        )
        for index in range(100)
    ]
    inventory = build_legal_width_inventory(
        units,
        minimum_retained_ratio=0.05,
        minimum_retained_channels=5,
        dense_alignment=1,
        per_domain_max_prune_rate=0.95,
    )
    domain_id = inventory.domain_ids[0]
    decoder = FixedTaylorWidthDecoder(
        inventory,
        [
            {
                "domain_id": domain_id,
                "physical_group_id": 0,
                "atomic_unit_id": f"u{index}",
                "first_order_score": float(index),
                "second_order_score": float(index),
            }
            for index in range(100)
        ],
    )
    space = SearchSpaceSpec(
        pruning_unit_ids=list(inventory.unit_ids),
        precision_layer_ids=["pg"],
        default_precision="FP32",
        structure_gene_type="legal_keep_width",
        legal_width_inventory=inventory,
        fixed_width_decoder=decoder,
        precision_action_space={"pg": ("FP32",)},
    )
    context = SimpleNamespace(
        search_space=space,
        code_commit="test-commit",
        checkpoint_hash="checkpoint",
    )

    class Proxy:
        def __init__(self) -> None:
            self.call_count = 0
            self.batch_call_count = 0

        def evaluate(self, genotype, *, generation=0, outer_round=0):
            del generation, outer_round
            self.call_count += 1
            keep = inventory.domains[0].legal_keep_widths[
                genotype.width_genes[domain_id]
            ]
            phenotype = canonicalize_legal_width_candidate(genotype, space)
            return {
                "R_BOPS": 0.4 * keep / 100.0,
                "R_bops_vs_fp32": 0.4 * keep / 100.0,
                "L_joint_raw": (100 - keep) * 0.01,
                "L_joint_first_order": (100 - keep) * 0.006,
                "L_joint_second_order": (100 - keep) * 0.004,
                "R_prune": (100 - keep) / 100.0,
                "F1": (100 - keep) * 0.01,
                "candidate_hash": candidate_hash(phenotype, space),
                "phenotype": phenotype.to_dict(),
            }

        def evaluate_batch(self, genotypes, *, generation=0, outer_round=0):
            self.batch_call_count += 1
            return SimpleNamespace(
                metrics=[
                    self.evaluate(
                        genotype,
                        generation=generation,
                        outer_round=outer_round,
                    )
                    for genotype in genotypes
                ]
            )

    return context, Proxy()


def test_six_budget_greedy_writes_endpoints_and_fixed_scale(tmp_path: Path) -> None:
    from search.orchestration.legal_width_greedy import run_six_budget_greedy

    context, proxy = _context_and_proxy()
    result = run_six_budget_greedy(
        context=context,
        proxy=proxy,
        run_dir=tmp_path,
        config={
            "targets": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
            "primary_tolerance": 0.005,
            "expanded_tolerance": 0.0075,
            "frontier_size": 8,
            "max_expansions": 4096,
            "anchor_scale_rows": [
                {"phenotype_hash": "anchor", "L_joint_raw": 2.0}
            ],
        },
    )

    assert result["target_count"] == 6
    assert result["feasible_budget_count"] == 6
    assert len(result["endpoint_records"]) == 6
    assert (tmp_path / "greedy" / "greedy_summary.json").is_file()
    assert (tmp_path / "greedy" / "budget_005_trace.json").is_file()
    assert (tmp_path / "greedy" / "budget_030_endpoint.json").is_file()
    trace = json.loads(
        (tmp_path / "greedy" / "budget_005_trace.json").read_text(
            encoding="utf-8"
        )
    )
    assert trace["trace_schema"] == "greedy-compact-v1"
    assert "evaluated_states" not in trace
    assert len(trace["evaluated_state_digest"]) == 64
    assert trace["evaluated_state_count"] >= 1
    scale_path = tmp_path / "joint_loss_scale.json"
    assert scale_path.stat().st_mode & 0o222 == 0
    assert result["joint_loss_scale"]["mapping"] == "linear_fixed_scale"
    assert result["joint_loss_scale"]["member_count"] > 6
    assert all(
        row["normal_candidate_repair_invoked"] is False
        for row in result["path_states"]
    )
    assert proxy.call_count <= 96
    assert proxy.batch_call_count > 0
    assert all(
        abs(row["actual_bops"] - row["target_bops"]) <= 0.005 + 1.0e-12
        for row in result["endpoints"]
    )


def test_greedy_config_contract_is_raw_weight_only_and_has_no_ap_gate() -> None:
    import yaml

    path = (
        Path(__file__).resolve().parents[1]
        / "search/configs/lidar_pyramid_4090_greedy_six_budget.yaml"
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert payload["structure_gene"]["type"] == "legal_keep_width"
    assert payload["structure_gene"]["ranking_depends_on_precision"] is False
    assert payload["candidate_proxy"]["mode"] == (
        "joint_pruning_quantization_second_order_fisher"
    )
    assert payload["proxy"]["task_score_mapping"] == "raw_joint_loss"
    assert payload["proxy"]["include_activation_taylor"] is False
    assert payload["proxy"]["sqnr_main_objective_weight"] == 0.0
    assert payload["proxy"]["bops_reference"] == "original_strict_fp32"
    assert payload["greedy"]["targets"] == [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    assert payload["greedy"]["frontier_size"] == 1
    assert payload["greedy"]["max_expansions"] == 16_384
    assert payload["greedy"]["primary_tolerance"] == pytest.approx(0.005)
    assert payload["greedy"]["expanded_tolerance"] == pytest.approx(0.0075)
    assert "min_map" not in payload.get("stage2", {})
    assert "min_ap07" not in payload.get("stage2", {})
    assert "max_map_drop" not in payload.get("stage2", {})


def test_joint_proxy_scale_loader_routes_explicit_mappings(tmp_path: Path) -> None:
    from search.orchestration.lidar_pyramid_search import _load_joint_proxy_scale
    from search.proxy.joint_loss_scale import (
        calibrate_joint_loss_scale,
        write_joint_loss_scale,
    )

    path = write_joint_loss_scale(
        tmp_path / "joint_loss_scale.json",
        calibrate_joint_loss_scale(
            [{"phenotype_hash": "p", "L_joint_raw": 1.0}],
            code_commit="abc",
        ),
    )

    assert _load_joint_proxy_scale(
        {
            "proxy_mode": "joint_taylor_second_order_fisher_diag",
            "task_score_mapping": "raw_joint_loss",
        }
    ) == {}
    loaded = _load_joint_proxy_scale(
        {
            "proxy_mode": "joint_taylor_second_order_fisher_diag",
            "task_score_mapping": "linear_fixed_scale",
            "joint_loss_scale_path": str(path),
        }
    )
    assert loaded["mapping"] == "linear_fixed_scale"
    assert loaded["value"] == pytest.approx(1.0)
    with pytest.raises(RuntimeError, match="task_score_mapping_required"):
        _load_joint_proxy_scale(
            {"proxy_mode": "joint_taylor_second_order_fisher_diag"}
        )


def test_cli_accepts_greedy_only_and_joint_loss_scale() -> None:
    from search.cli import parse_args

    args = parse_args(
        [
            "--config",
            "config.yaml",
            "--greedy-only",
            "--joint-loss-scale",
            "scale.json",
            "--greedy-run",
            "outputs/greedy",
        ]
    )

    assert args.greedy_only is True
    assert args.joint_loss_scale == "scale.json"
    assert args.greedy_run == "outputs/greedy"


def test_bind_greedy_run_records_endpoint_and_full_validation_lineage(
    tmp_path: Path,
) -> None:
    from search.cli import _bind_greedy_run

    run_dir = tmp_path / "greedy-run"
    endpoint_dir = run_dir / "greedy"
    endpoint_dir.mkdir(parents=True)
    (endpoint_dir / "greedy_summary.json").write_text(
        '{"endpoints": []}', encoding="utf-8"
    )
    full_dir = run_dir / "greedy_full_validation"
    full_dir.mkdir()
    full_manifest = full_dir / "greedy_full_validation.json"
    full_manifest.write_text(
        '{"successful_count": 0, "successful_candidates": []}',
        encoding="utf-8",
    )
    config: dict = {}
    search_config: dict = {}

    _bind_greedy_run(config, search_config, run_dir)

    assert search_config["greedy_endpoint_manifest"] == str(
        endpoint_dir.resolve()
    )
    assert search_config["greedy_full_validation_manifest"] == str(
        full_manifest.resolve()
    )
    lineage = config["runtime_provenance"]["greedy_run_override"]
    assert lineage["run_dir"] == str(run_dir.resolve())
    assert len(lineage["endpoint_manifest_sha256"]) == 64
    assert len(lineage["full_validation_manifest_sha256"]) == 64


def test_greedy_stage2_resume_uses_terminal_endpoints_and_closes_pool(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import search.orchestration.lidar_pyramid_search as module

    endpoints = [{"candidate_hash": "a"}, {"candidate_hash": "b"}]
    calls: dict[str, object] = {}

    def fake_load_endpoints(path):
        calls["endpoint_path"] = str(path)
        return endpoints

    monkeypatch.setattr(module, "load_greedy_endpoints", fake_load_endpoints)
    monkeypatch.setattr(
        module,
        "_select_runtime_stage2_gpus",
        lambda **kwargs: {"selected_gpu_ids": [6, 7]},
    )
    monkeypatch.setattr(
        module,
        "_build_shared_stage2_reference",
        lambda **kwargs: {"reference_hash": "reference", "forward_p50_ms": 10.0},
    )

    class FakePool:
        def __init__(self, **kwargs):
            calls["pool_kwargs"] = kwargs
            calls["pool"] = self
            self.closed = False

        def close(self):
            self.closed = True

    monkeypatch.setattr(module, "PersistentStage2ProcessPool", FakePool)

    def fake_full_validation(**kwargs):
        calls["full_validation_kwargs"] = kwargs
        return {"successful_count": 2}

    monkeypatch.setattr(
        module, "run_greedy_endpoint_full_validation", fake_full_validation
    )
    config = {
        "stage2": {"score_mode": "map_minus_latency_ratio"},
        "full_validation": {
            "num_frames": 1789,
            "required_evaluated_frames": 1789,
            "required_skipped_frames": 0,
            "num_workers": 8,
            "ap_iou_backend": "gpu",
        },
        "stage2_parallel": {
            "enabled": True,
            "gpu_ids": [],
            "max_memory_fraction": 0.5,
            "startup_timeout_seconds": 10,
            "task_timeout_seconds": 20,
            "poll_interval_seconds": 0.1,
        },
    }

    result = module._run_greedy_endpoint_stage2_resume(
        run_dir=tmp_path,
        config=config,
        checkpoint=tmp_path / "model.pth",
        code_commit="commit",
    )

    assert result["greedy_endpoint_stage2"] is True
    assert result["result"]["successful_count"] == 2
    assert calls["endpoint_path"].endswith("/greedy")
    assert calls["pool_kwargs"]["gpu_ids"] == [6, 7]
    worker_config = calls["pool_kwargs"]["worker_payload"]["config"]
    assert worker_config["stage2"]["num_frames"] == 1789
    assert worker_config["stage2"]["num_workers"] == 8
    assert calls["full_validation_kwargs"]["endpoints"] == endpoints
    assert calls["pool"].closed is True


def test_greedy_config_includes_full_validation_gpu_protocol() -> None:
    import yaml

    path = (
        Path(__file__).resolve().parents[1]
        / "search/configs/lidar_pyramid_4090_greedy_six_budget.yaml"
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert payload["stage2_parallel"]["enabled"] is True
    assert payload["stage2_parallel"]["gpu_ids"] == []
    assert payload["stage2_parallel"]["max_memory_fraction"] == pytest.approx(0.5)
    assert payload["full_validation"]["num_frames"] == 1789
    assert payload["full_validation"]["required_evaluated_frames"] == 1789
    assert payload["full_validation"]["required_skipped_frames"] == 0
    assert payload["full_validation"]["num_workers"] == 8
    assert payload["full_validation"]["ap_iou_backend"] == "gpu"


def test_runner_routes_greedy_stage2_resume_before_context_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import search.orchestration.lidar_pyramid_search as module

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    checkpoint = tmp_path / "model.pth"
    checkpoint.touch()
    calls: dict[str, object] = {}

    def fake_resume(**kwargs):
        calls.update(kwargs)
        return {"run_dir": str(run_dir), "greedy_endpoint_stage2": True}

    monkeypatch.setattr(module, "_run_greedy_endpoint_stage2_resume", fake_resume)
    monkeypatch.setattr(
        module,
        "build_lidar_pyramid_context",
        lambda **kwargs: pytest.fail("context build must be skipped"),
    )
    runner = module.LidarPyramidTwoStageSearch(
        config={"greedy": {"enabled": True}},
        checkpoint=checkpoint,
        output_root=tmp_path,
        resume=run_dir,
    )

    result = runner.run(stage2_only=True)

    assert result["greedy_endpoint_stage2"] is True
    assert calls["run_dir"] == run_dir
    assert calls["checkpoint"] == checkpoint.resolve()


def test_resume_stage2_uses_separate_resolved_config_snapshot(tmp_path: Path) -> None:
    from search.cli import _resolved_config_output_path

    assert _resolved_config_output_path(
        tmp_path, resume=True, stage2_only=True
    ).name == "resolved_stage2_config.yaml"
    assert _resolved_config_output_path(
        tmp_path, resume=False, stage2_only=True
    ).name == "resolved_config.yaml"
    assert _resolved_config_output_path(
        tmp_path, resume=True, stage2_only=False
    ).name == "resolved_config.yaml"
