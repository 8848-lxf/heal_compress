from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_full_val_manifest_records_order_hash_and_skip_policy(tmp_path: Path) -> None:
    from search.baselines.full_validation import write_full_validation_manifest

    manifest = write_full_validation_manifest(
        tmp_path / "full_val_manifest.json",
        frame_ids=["0003", "0001", "0002"],
        dataset_split="validate",
        dataset_config_hash="cfg",
        postprocess_config={"nms": "heal"},
        skip_policy="record_and_common_subset",
    )

    assert manifest["dataset_split"] == "validate"
    assert manifest["frame_ids"] == ["0003", "0001", "0002"]
    assert manifest["total_manifest_frames"] == 3
    assert manifest["manifest_hash"]


def test_common_subset_uses_same_frame_ids_across_all_baselines() -> None:
    from search.baselines.full_validation import compute_common_evaluated_subset

    subset = compute_common_evaluated_subset(
        {
            "strict_fp32": {"evaluated_frame_ids": ["a", "b", "c"], "skipped_frame_ids": ["d"]},
            "strict_fp16": {"evaluated_frame_ids": ["b", "c", "d"], "skipped_frame_ids": ["a"]},
            "maximal_legal_int8": {"evaluated_frame_ids": ["c", "b"], "skipped_frame_ids": []},
        }
    )

    assert subset["common_subset_frames"] == 2
    assert subset["common_frame_ids"] == ["b", "c"]
    assert subset["skip_sets_comparable"] is True


def test_stage2_eval_manifest_uses_real_fixed_warmup_and_evaluation_ids(tmp_path: Path) -> None:
    import json

    from search.integration.data_provider import write_eval_manifest

    path = tmp_path / "eval_manifest.json"
    manifest = write_eval_manifest(
        path,
        num_frames=3,
        warmup_frames=2,
        available_frame_ids=["000211", "000212", "000214", "000215", "000216", "000217"],
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert manifest.frame_ids == ["000211", "000212", "000214", "000215", "000216"]
    assert payload["warmup_frame_ids"] == ["000211", "000212"]
    assert payload["evaluation_frame_ids"] == ["000214", "000215", "000216"]
    assert payload["warmup_frames"] == 2
    assert payload["num_frames"] == 3
    assert payload["manifest_hash"] == manifest.manifest_hash


def test_stage2_eval_manifest_can_reset_before_full_validation(tmp_path: Path) -> None:
    import json

    from search.integration.data_provider import write_eval_manifest

    path = tmp_path / "eval_manifest_reset.json"
    frame_ids = [f"{index:06d}" for index in range(5)]
    manifest = write_eval_manifest(
        path,
        num_frames=5,
        warmup_frames=2,
        available_frame_ids=frame_ids,
        reset_after_warmup=True,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["reset_after_warmup"] is True
    assert payload["warmup_frame_ids"] == frame_ids[:2]
    assert payload["evaluation_frame_ids"] == frame_ids
    assert payload["frame_ids"] == frame_ids
    assert manifest.frame_ids == frame_ids


def test_budget_final_manifest_offset_avoids_stage2_evaluation_frames(tmp_path: Path) -> None:
    import json

    from search.integration.data_provider import write_eval_manifest

    frame_ids = [f"{index:06d}" for index in range(12)]
    stage2 = write_eval_manifest(
        tmp_path / "stage2.json",
        num_frames=5,
        warmup_frames=2,
        available_frame_ids=frame_ids,
        reset_after_warmup=True,
        evaluation_offset=0,
    )
    budget_final = write_eval_manifest(
        tmp_path / "budget_final.json",
        num_frames=5,
        warmup_frames=2,
        available_frame_ids=frame_ids,
        reset_after_warmup=True,
        evaluation_offset=5,
    )
    stage2_payload = json.loads(stage2.path.read_text(encoding="utf-8"))
    final_payload = json.loads(budget_final.path.read_text(encoding="utf-8"))

    assert stage2_payload["evaluation_frame_ids"] == frame_ids[:5]
    assert final_payload["evaluation_frame_ids"] == frame_ids[5:10]
    assert set(stage2_payload["evaluation_frame_ids"]).isdisjoint(final_payload["evaluation_frame_ids"])
    assert final_payload["evaluation_offset"] == 5
