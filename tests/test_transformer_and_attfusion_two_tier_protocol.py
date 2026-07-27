from __future__ import annotations

import inspect

from search.ga import cnn_stage12_v3 as protocol


def test_attfusion_and_cobevt_use_frozen_300_then_500_protocol() -> None:
    assert protocol.STAGE2_SCREENING_FRAMES == 300
    assert protocol.STAGE2_SCREENING_WARMUP_FRAMES == 100
    assert protocol.GENERATION_WINNER_FRAMES == 500
    assert protocol.GENERATION_WINNER_WARMUP_FRAMES == 200
    assert protocol.EVALUATION_MANIFEST_FRAMES == 500
    assert protocol.EVALUATION_MANIFEST_WARMUP_FRAMES == 200


def test_generation_winner_is_revalidated_without_rebuilding_engine() -> None:
    source = inspect.getsource(protocol.validate_generation_winner)
    assert "reevaluate_existing_candidate_engine" in source
    assert '"engine_rebuilt_for_validation": False' in source


def test_attfusion_is_treated_as_supported_cnn_family() -> None:
    spec = protocol.MODEL_SPECS["attfusion"]
    assert spec.family_id == "heal_lidar_attfusion"
