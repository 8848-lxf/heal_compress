from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def test_repaired_top5_manifest_writes_ranked_unique_configs_and_recovers_metrics(tmp_path: Path) -> None:
    from search.stage2.repaired_topk_manifest import write_repaired_topk_manifest

    run_dir = tmp_path / "run"
    round_dir = run_dir / "round_000"
    phenotype_a = {
        "pruned_unit_ids": [],
        "precision_profile": {"conv": {"requested_precision": "FP16", "realized_precision": "FP16"}},
        "metadata": {
            "requested_group_profile": {"pg0": "FP16"},
            "stage1_legalized_group_profile": {"pg0": "FP16"},
            "group_keep_map_by_scope": {},
            "group_prune_map_by_scope": {},
            "keep_value": 1,
            "prune_value": 0,
            "repair_mode": "mask_preserving_monotonic_floor",
        },
    }
    phenotype_b = {
        "pruned_unit_ids": ["u0", "u1"],
        "precision_profile": {"conv": {"requested_precision": "INT8", "realized_precision": "INT8"}},
        "metadata": {
            "requested_group_profile": {"pg0": "INT8"},
            "stage1_legalized_group_profile": {"pg0": "INT8"},
            "group_keep_map_by_scope": {"scope": {0: [0, 1, 2, 3], 1: [0, 2, 4, 6]}},
            "group_prune_map_by_scope": {"scope": {0: [4, 5, 6, 7], 1: [1, 3, 5, 7]}},
            "group_keep_map": {"conv::out::scope": {0: [0, 1, 2, 3], 1: [0, 2, 4, 6]}},
            "group_prune_map": {"conv::out::scope": {0: [4, 5, 6, 7], 1: [1, 3, 5, 7]}},
            "keep_value": 1,
            "prune_value": 0,
            "repair_mode": "mask_preserving_monotonic_floor",
        },
    }
    _write_json(
        round_dir / "stage1_topk.json",
        [
            {"role": "repaired", "candidate_hash": "aaa", "F1": 0.1, "phenotype": phenotype_a},
            {"role": "repaired", "candidate_hash": "bbb", "F1": 0.2, "phenotype": phenotype_b},
        ],
    )
    (round_dir / "generation_000.csv").parent.mkdir(parents=True, exist_ok=True)
    with (round_dir / "generation_000.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["candidate_hash", "F1", "L_fisher", "L_sqnr", "R_size", "R_bops", "P_bops"])
        writer.writeheader()
        writer.writerow({"candidate_hash": "aaa", "F1": "0.1", "L_fisher": "0.0", "L_sqnr": "0.01", "R_size": "0.5", "R_bops": "0.25", "P_bops": "0.0"})
    archive_dir = run_dir / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    with (archive_dir / "proxy_archive.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "candidate_hash": "proxy-key",
                    "phenotype": phenotype_b,
                    "F1": 0.2,
                    "L_fisher": 0.03,
                    "L_sqnr": 0.02,
                    "R_size": 0.48,
                    "R_bops": 0.24,
                    "P_bops": 0.0,
                },
                sort_keys=True,
            )
            + "\n"
        )

    manifest = write_repaired_topk_manifest(run_dir, round_index=0)

    assert manifest["status"] == "ok"
    assert [row["candidate_rank"] for row in manifest["candidates"]] == [0, 1]
    assert [row["repaired_phenotype_hash"] for row in manifest["candidates"]] == ["aaa", "bbb"]
    assert manifest["checks"]["repaired_phenotype_hash_unique"] is True
    assert manifest["checks"]["top5_sorted_by_repaired_F1"] is True
    assert manifest["candidates"][0]["L_SQNR"] == 0.01
    assert manifest["candidates"][1]["R_Fisher"] == 0.03
    assert manifest["candidates"][1]["group_keep_map"] == {"conv::out::scope": {"0": [0, 1, 2, 3], "1": [0, 2, 4, 6]}}
    assert (round_dir / "repaired_top5_manifest.json").is_file()
    assert (round_dir / "stage2_candidate_configs" / "candidate_00_aaa.json").is_file()
    assert json.loads((round_dir / "stage2_candidate_configs" / "candidate_01_bbb.json").read_text())["pruned_unit_ids"] == ["u0", "u1"]


def test_repaired_top5_manifest_scans_proxy_archive_once_for_multiple_archive_hits(tmp_path: Path, monkeypatch) -> None:
    from search.stage2.repaired_topk_manifest import build_repaired_topk_manifest

    run_dir = tmp_path / "run"
    round_dir = run_dir / "round_000"
    phenotype_a = {
        "pruned_unit_ids": ["u0"],
        "precision_profile": {"conv": {"requested_precision": "FP16", "realized_precision": "FP16"}},
        "metadata": {"stage1_legalized_group_profile": {"pg0": "FP16"}},
    }
    phenotype_b = {
        "pruned_unit_ids": ["u1"],
        "precision_profile": {"conv": {"requested_precision": "FP16", "realized_precision": "FP16"}},
        "metadata": {"stage1_legalized_group_profile": {"pg0": "FP16"}},
    }
    _write_json(
        round_dir / "stage1_topk.json",
        [
            {"role": "repaired", "candidate_hash": "aaa", "F1": 0.1, "phenotype": phenotype_a},
            {"role": "repaired", "candidate_hash": "bbb", "F1": 0.2, "phenotype": phenotype_b},
        ],
    )
    archive = run_dir / "archives" / "proxy_archive.jsonl"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with archive.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"candidate_hash": "p0", "phenotype": phenotype_a, "F1": 0.1, "L_fisher": 0.01}) + "\n")
        handle.write(json.dumps({"candidate_hash": "p1", "phenotype": phenotype_b, "F1": 0.2, "L_fisher": 0.02}) + "\n")

    original_open = Path.open
    archive_open_count = {"value": 0}

    def counting_open(self: Path, *args, **kwargs):
        if self == archive:
            archive_open_count["value"] += 1
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counting_open)

    manifest = build_repaired_topk_manifest(run_dir, round_index=0)

    assert [row["R_Fisher"] for row in manifest["candidates"]] == [0.01, 0.02]
    assert archive_open_count["value"] == 1
