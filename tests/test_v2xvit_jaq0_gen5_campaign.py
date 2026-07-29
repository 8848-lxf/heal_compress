from __future__ import annotations

import inspect


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
