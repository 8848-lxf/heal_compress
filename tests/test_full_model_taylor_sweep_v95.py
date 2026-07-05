from __future__ import annotations


def test_v95_build_pruner_command_uses_full_model_surface(tmp_path):
    from tools.latency_lut.run_full_model_taylor_sweep_v95 import build_pruner_command

    cmd = build_pruner_command(
        policy="A",
        ratio=0.05,
        output_dir=tmp_path / "model",
        checkpoint="ckpt.pth",
        model_config="cfg.yaml",
        heal_root="/heal",
        device="cuda:0",
        max_frames=200,
        num_calib_batches=1,
    )

    assert "--prunable-surface" in cmd
    assert "full_model_all_safe_coupled_units" in cmd
    assert "--importance-mode" in cmd
    assert "first_order_taylor" in cmd


def test_v95_target_ratio_semantics_prunable_surface():
    from tools.latency_lut.run_full_model_taylor_sweep_v95 import target_ratio_semantics

    report = target_ratio_semantics(
        target_ratio=0.5,
        total_model_params=1000,
        total_prunable_surface_params=400,
        actual_full_model_param_prune_ratio=0.2,
    )

    assert report["target_ratio_type"] == "prunable_surface_param_ratio"
    assert report["achievable_param_prune_upper_bound"] == 0.4
    assert report["actual_param_prune_ratio_of_prunable_surface"] == 0.5
