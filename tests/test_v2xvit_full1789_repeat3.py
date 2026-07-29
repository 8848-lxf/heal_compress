from __future__ import annotations

import pytest

from scripts.run_v2xvit_sixbudget_full1789_repeat3 import (
    _control_record,
    _parse_labels,
    summarize_repetitions,
)


def _row(index: int) -> dict[str, float | int]:
    return {
        "repeat": index,
        "AP@0.3": 0.7 + index * 0.001,
        "AP@0.5": 0.6 + index * 0.001,
        "AP@0.7": 0.4 + index * 0.001,
        "mAP": 0.5 + index * 0.001,
        "forward_p50_ms": 10.0 + index,
        "forward_p90_ms": 11.0 + index,
        "forward_p99_ms": 12.0 + index,
    }


def test_repeat3_summary_uses_all_three_repetitions() -> None:
    result = summarize_repetitions([_row(1), _row(2), _row(3)])
    assert result["mAP_mean"] == pytest.approx(0.502)
    assert result["mAP_std"] == pytest.approx(0.001)
    assert len(result["repetitions"]) == 3


def test_repeat3_summary_fails_closed_on_missing_repeat() -> None:
    with pytest.raises(RuntimeError, match="repeat_count_mismatch"):
        summarize_repetitions([_row(1), _row(2)])


def test_repeat5_summary_uses_all_five_repetitions() -> None:
    result = summarize_repetitions(
        [_row(index) for index in range(1, 6)], repeat_count=5
    )
    assert result["mAP_mean"] == pytest.approx(0.503)
    assert result["forward_p90_ms_mean"] == pytest.approx(14.0)
    assert result["forward_p99_ms_mean"] == pytest.approx(15.0)
    assert len(result["repetitions"]) == 5


def test_budget_subset_parser_is_ordered_and_fail_closed() -> None:
    assert _parse_labels("005,030,010") == ("005", "030", "010")
    with pytest.raises(ValueError, match="labels_unknown"):
        _parse_labels("005,007")
    with pytest.raises(ValueError, match="labels_invalid"):
        _parse_labels("005,005")


def test_control_record_uses_authoritative_physical_report_hash(
    tmp_path, monkeypatch
) -> None:
    import hashlib
    import json
    from scripts import run_v2xvit_sixbudget_full1789_repeat3 as runner

    candidate_hash = "candidate-a"
    candidate_dir = tmp_path / "ga/stage2_cache/budget_005" / candidate_hash
    engine = candidate_dir / "JMIX-FRESH/candidate.plan"
    engine.parent.mkdir(parents=True)
    engine.write_bytes(b"engine")
    physical = {
        "structure_hash": "physical-a",
        "physical_parameter_count": 5,
        "original_parameter_count": 10,
    }
    (candidate_dir / "physical_report.json").write_text(
        json.dumps(physical), encoding="utf-8"
    )
    monkeypatch.setattr(runner, "_candidate_dir", lambda *_args: candidate_dir)
    monkeypatch.setattr(
        runner,
        "_engine_acceptance",
        lambda _path: {
            "precision_realization_validation": {
                "realized_fp16_count": 1,
                "realized_int8_count": 0,
                "requested_int8_count": 0,
            }
        },
    )
    candidate = {
        "complete_phenotype_hash": candidate_hash,
        "genotype": {"precision_genes": {"unit": "FP16"}},
        "metadata": {
            "engine_path": str(engine),
            "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
        },
    }

    result = _control_record(
        label="005", method="GA-final", candidate=candidate, source_root=tmp_path
    )

    assert result["physical_structure_hash"] == "physical-a"


def test_control_record_rejects_stale_duplicated_physical_hash(
    tmp_path, monkeypatch
) -> None:
    import json
    from scripts import run_v2xvit_sixbudget_full1789_repeat3 as runner

    candidate_dir = tmp_path / "candidate"
    engine = candidate_dir / "JMIX-FRESH/candidate.plan"
    engine.parent.mkdir(parents=True)
    engine.write_bytes(b"engine")
    (candidate_dir / "physical_report.json").write_text(
        json.dumps(
            {
                "structure_hash": "physical-new",
                "physical_parameter_count": 5,
                "original_parameter_count": 10,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "_candidate_dir", lambda *_args: candidate_dir)
    monkeypatch.setattr(runner, "_engine_acceptance", lambda _path: {})
    candidate = {
        "complete_phenotype_hash": "candidate-a",
        "genotype": {"precision_genes": {}},
        "metadata": {
            "engine_path": str(engine),
            "physical_structure_hash": "physical-stale",
        },
    }

    with pytest.raises(RuntimeError, match="physical_structure_hash_mismatch"):
        _control_record(
            label="005",
            method="GA-final",
            candidate=candidate,
            source_root=tmp_path,
        )
