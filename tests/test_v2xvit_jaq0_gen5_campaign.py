from __future__ import annotations

import inspect

import pytest


def test_v2xvit_campaign_is_exactly_five_generations() -> None:
    import scripts.run_v2xvit_six_budget_formal_ga_gen10 as runner

    assert runner.GENERATIONS == 5
    source = inspect.getsource(runner.run)
    assert 'generation_contract="formal_gen5"' in source
    assert "range(1, GENERATIONS + 1)" in source
    assert "requires_exactly_five_generations" in source


def test_v2xvit_campaign_defaults_to_jaq0_and_one_gpu() -> None:
    import scripts.run_v2xvit_six_budget_formal_ga_gen10 as runner
    import scripts.run_v2xvit_six_budget_proxy as proxy

    assert "default=0.0" in inspect.getsource(runner.main)
    assert "default=0.0" in inspect.getsource(proxy.main)
    assert "v2xvit_campaign_forbids_cross_gpu_stage2" in inspect.getsource(
        runner.run
    )
    assert "activation_quantization_used_in_deployment" in inspect.getsource(
        runner.run
    )


def test_v2xvit_campaign_visible_gpu_binding_executes(monkeypatch) -> None:
    import scripts.run_v2xvit_six_budget_formal_ga_gen10 as runner

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    runner._validate_visible_gpu(3)
    with pytest.raises(RuntimeError, match="cuda_visible_devices_mismatch"):
        runner._validate_visible_gpu(4)
