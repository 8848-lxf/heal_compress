from __future__ import annotations

from pathlib import Path


def test_formal_attention_candidates_cover_qk_b1_and_one_b2():
    from search.orchestration.lidar_cobevt_attention_pruning import (
        formal_candidate_specs,
    )

    specs = formal_candidate_specs()
    assert [row.candidate_id for row in specs] == [
        "baseline_d32",
        "qk_only_d24",
        "qk_only_d16",
        "b1_uniform_d24",
        "b1_uniform_d16",
        "b2_global_d24",
    ]
    assert [(row.d_qk, row.d_v, row.embed_dim) for row in specs] == [
        (32, 32, 256),
        (24, 32, 256),
        (16, 32, 256),
        (24, 24, 256),
        (16, 16, 256),
        (24, 24, 192),
    ]
    assert all(row.heads == 8 for row in specs)


def test_fixed500_manifest_uses_gpu_evaluation_protocol(tmp_path: Path):
    from search.orchestration.lidar_cobevt_attention_pruning import (
        write_attention_evaluation_manifests,
    )

    ids = [f"frame_{index:04d}" for index in range(600)]
    result = write_attention_evaluation_manifests(tmp_path, ids)

    assert result["fixed500"].path.is_file()
    assert len(result["fixed500"].frame_ids) == 500
    assert result["fixed500"].frame_ids[0] == "frame_0020"
    assert result["smoke10"].frame_ids[0] == "frame_0020"
    assert result["protocol"] == {
        "ap_iou_backend": "gpu",
        "dataloader_workers": 8,
        "evaluation_frames": 500,
        "warmup_frames": 20,
    }


def test_attention_mask_json_round_trip_preserves_identity(tmp_path: Path):
    from search.model_families.lidar_cobevt.attention_dim_pruning import (
        AttentionDimMask,
        attention_masks_structure_hash,
    )
    from search.orchestration.lidar_cobevt_attention_pruning import (
        load_attention_masks,
        write_attention_masks,
    )

    masks = {
        "fusion_net.layers.0.window_attention.fn": AttentionDimMask(
            tuple(tuple(range(24)) for _ in range(8)),
            tuple(tuple(range(8, 32)) for _ in range(8)),
        )
    }
    path = tmp_path / "mask.json"
    write_attention_masks(path, masks)
    loaded = load_attention_masks(path)

    assert loaded == masks
    assert attention_masks_structure_hash(loaded) == attention_masks_structure_hash(
        masks
    )


def test_prepare_directory_allows_only_known_incomplete_cache(tmp_path: Path):
    from search.orchestration.lidar_cobevt_attention_pruning import (
        initialize_prepare_output,
    )

    (tmp_path / "attention_mean_gradients.pt").write_bytes(b"cache")
    (tmp_path / "attention_mean_gradients.manifest.json").write_text("{}")
    initialize_prepare_output(tmp_path)
    assert (tmp_path / "candidate_masks").is_dir()


def test_prepare_directory_rejects_unknown_existing_artifacts(tmp_path: Path):
    import pytest

    from search.orchestration.lidar_cobevt_attention_pruning import (
        initialize_prepare_output,
    )

    (tmp_path / "unrelated.bin").write_bytes(b"do not overwrite")
    with pytest.raises(RuntimeError, match="unknown_existing_artifacts"):
        initialize_prepare_output(tmp_path)


def test_production_gpu_ap_helper_handles_empty_detection_without_test_imports():
    import torch

    from search.integration.gpu_ap_iou import calculate_gpu_tp_fp_for_threshold

    stats = {0.5: {"tp": [], "fp": [], "gt": 0, "score": []}}
    calculate_gpu_tp_fp_for_threshold(
        None,
        None,
        torch.zeros(3, 8, 3),
        stats,
        0.5,
        torch.device("cpu"),
    )

    assert stats[0.5] == {"tp": [], "fp": [], "gt": 3, "score": []}


def test_cobevt_postprocess_mapping_has_one_ego_record():
    import torch

    from search.orchestration.lidar_cobevt_attention_pruning import (
        cobevt_postprocess_mapping,
    )

    outputs = {"cls_preds": torch.ones(1)}
    mapped = cobevt_postprocess_mapping(outputs)

    assert list(mapped) == ["ego"]
    assert mapped["ego"] is outputs


def test_candidate_result_upsert_replaces_failed_attempt():
    from search.orchestration.lidar_cobevt_attention_pruning import (
        upsert_candidate_result,
    )

    rows = [
        {"candidate_id": "a", "status": "evaluation_failed"},
        {"candidate_id": "b", "status": "ok"},
    ]
    updated = upsert_candidate_result(rows, {"candidate_id": "a", "status": "ok"})

    assert updated == [
        {"candidate_id": "a", "status": "ok"},
        {"candidate_id": "b", "status": "ok"},
    ]
