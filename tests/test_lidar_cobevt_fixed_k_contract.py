from __future__ import annotations

import pytest


def test_fixed_k_is_derived_and_has_zero_overflow():
    from search.model_families.lidar_cobevt.input_contract import derive_fixed_k

    contract = derive_fixed_k([101, 256, 513], alignment=256)

    assert contract.fixed_k == 768
    assert contract.source_max_k == 513
    assert contract.overflow_count == 0
    assert contract.record_count == 3


def test_cobevt_fixed_k_does_not_inherit_pyramid_constant():
    from search.model_families.lidar_cobevt.input_contract import derive_fixed_k

    contract = derive_fixed_k([24000], alignment=256)

    assert contract.fixed_k == 24064
    assert contract.fixed_k != 29696


def test_fixed_k_manifest_hash_is_order_sensitive_and_stable():
    from search.model_families.lidar_cobevt.input_contract import derive_fixed_k

    first = derive_fixed_k([100, 200], alignment=128)
    repeated = derive_fixed_k([100, 200], alignment=128)
    reordered = derive_fixed_k([200, 100], alignment=128)

    assert first.manifest_sha256 == repeated.manifest_sha256
    assert first.manifest_sha256 != reordered.manifest_sha256


@pytest.mark.parametrize("records", [[], [0], [10, -1]])
def test_fixed_k_rejects_empty_or_nonpositive_records(records):
    from search.model_families.lidar_cobevt.input_contract import derive_fixed_k

    with pytest.raises(ValueError):
        derive_fixed_k(records, alignment=256)


def test_fixed_k_rejects_nonpositive_alignment():
    from search.model_families.lidar_cobevt.input_contract import derive_fixed_k

    with pytest.raises(ValueError, match="alignment_must_be_positive"):
        derive_fixed_k([10], alignment=0)
