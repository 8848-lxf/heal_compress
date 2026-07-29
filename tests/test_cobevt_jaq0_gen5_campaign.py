from __future__ import annotations

import inspect


def test_cobevt_campaign_defaults_to_gen5_jaq0_and_one_gpu() -> None:
    import scripts.run_cobevt_formal_ga_gen10 as runner
    from search.ga.cnn_stage12_v3 import PreparedCNNFormalSearch
    from search.ga.transformer_stage12_v3 import prepare_cobevt_search

    source = inspect.getsource(runner.run)
    assert "requires_exactly_five_generations" in source
    assert "generations=5" in source
    assert "default=0.0" in inspect.getsource(runner.main)
    assert "CUDA_VISIBLE_DEVICES" in source
    assert PreparedCNNFormalSearch.__dataclass_fields__[
        "activation_taylor_fitness_weight"
    ].default == 0.0
    assert prepare_cobevt_search.__kwdefaults__[
        "activation_taylor_fitness_weight"
    ] == 0.0


def test_cobevt_deployment_quantization_remains_enabled() -> None:
    import scripts.run_cobevt_formal_ga_gen10 as runner

    source = inspect.getsource(runner.run)
    assert "activation_quantization_used_in_deployment" in source
