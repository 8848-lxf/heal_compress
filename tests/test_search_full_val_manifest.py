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
