from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pruning.types import AtomicPruneUnit


def _inventory():
    from search.space.legal_width_inventory import build_legal_width_inventory

    units = [
        AtomicPruneUnit(
            "scope",
            "conv",
            "out",
            [index],
            [f"c{index}"],
            float(index),
            _stable_id=f"u{index}",
        )
        for index in range(16)
    ]
    return build_legal_width_inventory(units, dense_alignment=4)


def test_random_legal_width_genotypes_are_legal_by_construction() -> None:
    from search.encoding.legal_width_genotype import random_legal_width_genotype

    inventory = _inventory()
    rng = random.Random(20260716)
    for _ in range(10_000):
        genotype = random_legal_width_genotype(
            inventory,
            precision_actions={"pg0": ("FP16", "INT8")},
            rng=rng,
        )
        genotype.validate(inventory, {"pg0": ("FP16", "INT8")})


def test_width_index_out_of_range_fails_closed() -> None:
    from search.encoding.legal_width_genotype import LegalWidthGenotype

    inventory = _inventory()
    domain_id = inventory.domain_ids[0]
    genotype = LegalWidthGenotype(
        width_genes={domain_id: 999},
        precision_genes={"pg0": "FP16"},
    )
    with pytest.raises(ValueError, match="width_gene_index_out_of_range"):
        genotype.validate(inventory, {"pg0": ("FP16", "INT8")})


def test_hashes_separate_width_and_precision_identity() -> None:
    from search.encoding.legal_width_genotype import LegalWidthGenotype

    inventory = _inventory()
    domain_id = inventory.domain_ids[0]
    fp16 = LegalWidthGenotype({domain_id: 0}, {"pg0": "FP16"})
    int8 = LegalWidthGenotype({domain_id: 0}, {"pg0": "INT8"})

    assert fp16.width_vector_hash == int8.width_vector_hash
    assert fp16.precision_hash != int8.precision_hash
    assert fp16.genotype_hash != int8.genotype_hash


def test_formal_codec_and_canonicalization_use_width_genes_not_group_mask() -> None:
    from search.candidate_codec import decode_candidate, encode_candidate
    from search.canonicalization import (
        SearchSpaceSpec,
        canonicalize_legal_width_candidate,
    )
    from search.decoding.fixed_taylor_width_decoder import FixedTaylorWidthDecoder
    from search.encoding.legal_width_genotype import LegalWidthGenotype

    inventory = _inventory()
    domain_id = inventory.domain_ids[0]
    decoder = FixedTaylorWidthDecoder(
        inventory,
        [
            {
                "domain_id": domain_id,
                "physical_group_id": 0,
                "atomic_unit_id": f"u{index}",
                "first_order_score": float(index),
                "second_order_score": float(index),
            }
            for index in range(16)
        ],
    )
    candidate = LegalWidthGenotype({domain_id: 1}, {"layer": "FP16"})
    encoded = encode_candidate(candidate)
    decoded = decode_candidate(encoded)
    space = SearchSpaceSpec(
        pruning_unit_ids=list(inventory.unit_ids),
        precision_layer_ids=["layer"],
        structure_gene_type="legal_keep_width",
        legal_width_inventory=inventory,
        fixed_width_decoder=decoder,
        precision_action_space={"layer": ("FP16", "INT8")},
    )
    phenotype = canonicalize_legal_width_candidate(decoded, space)

    assert "group_mask" not in encoded
    assert decoded == candidate
    assert phenotype.pruned_unit_ids == [f"u{index}" for index in range(8)]
    assert phenotype.metadata["structure_hash"] == decoder.decode(
        candidate.width_genes
    ).structure_hash
