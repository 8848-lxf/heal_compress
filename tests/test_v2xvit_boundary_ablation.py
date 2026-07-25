from __future__ import annotations

from search.reporting.v2xvit_boundary_ablation import (
    adjacent_successor,
    initial_genotype,
    make_restore_ablations,
    stable_mapping_hash,
)


def _contract():
    return {
        "domains": {
            "ffn_hidden::a": {"domain_type": "ffn_hidden", "original_width": 256, "legal_widths": (4, 252, 256)},
            "shrinker_m1.x": {"domain_type": "cnn_channel", "original_width": 256, "legal_widths": (4, 28, 256)},
            "attention_dh::a": {"domain_type": "attention_dh", "original_width": 32, "legal_widths": (4, 24, 32)},
            "backbone_m1.blocks.2.1::out": {"domain_type": "cnn_channel", "original_width": 256, "legal_widths": (4, 56, 256)},
            "backbone_m1.blocks.1.1::out": {"domain_type": "cnn_channel", "original_width": 128, "legal_widths": (4, 64, 128)},
        },
        "precision_groups": {
            "p": {"allowed_precisions": ("FP32", "FP16", "INT8"), "protected": False},
        },
    }


def test_adjacent_replay_changes_exactly_one_locus_and_hashes_deterministically():
    contract = _contract()
    current = initial_genotype(contract)
    after_width = adjacent_successor(
        current,
        action_type="domain_width",
        gene_id="attention_dh::a",
        contract=contract,
    )
    assert after_width["pruning_width_genes"]["attention_dh::a"] == 24
    assert after_width["precision_genes"] == current["precision_genes"]
    after_precision = adjacent_successor(
        current,
        action_type="precision",
        gene_id="p",
        contract=contract,
    )
    assert after_precision["precision_genes"]["p"] == "FP16"
    assert stable_mapping_hash(after_precision["pruning_width_genes"]) == stable_mapping_hash(
        current["pruning_width_genes"]
    )


def test_restore_ablations_change_only_requested_domains():
    contract = _contract()
    base = initial_genotype(contract)
    base["pruning_width_genes"].update(
        {
            "ffn_hidden::a": 252,
            "shrinker_m1.x": 28,
            "attention_dh::a": 4,
            "backbone_m1.blocks.2.1::out": 56,
            "backbone_m1.blocks.1.1::out": 64,
        }
    )
    winner005 = {"candidate_hash": "005", "genotype": base}
    width010 = dict(base["pruning_width_genes"])
    width010["attention_dh::a"] = 24
    winner010 = {
        "candidate_hash": "010",
        "genotype": {**base, "pruning_width_genes": width010},
    }
    rows = make_restore_ablations(winner005, winner010, contract)
    assert rows["A1_restore_ffn"]["changed_domains"] == ["ffn_hidden::a"]
    assert rows["A2_restore_shrinker"]["changed_domains"] == ["shrinker_m1.x"]
    assert rows["A3_restore_attention_to_010"]["changed_domains"] == ["attention_dh::a"]
    assert rows["A4_restore_stage2_backbone"]["changed_domains"] == [
        "backbone_m1.blocks.2.1::out"
    ]
    for row in rows.values():
        assert row["genotype"]["pruning_width_genes"]["backbone_m1.blocks.1.1::out"] == 64
