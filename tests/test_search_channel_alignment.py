from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_dense_alignment_requires_multiple_of_four_except_original_width() -> None:
    from search.pruning_space.local_domains import legal_dense_widths

    widths = legal_dense_widths(original_width=33, minimum_retained_ratio=0.10, alignment=4)

    assert 4 in widths
    assert 32 in widths
    assert 33 in widths
    assert all(width == 33 or width % 4 == 0 for width in widths)


def test_grouped_allowed_width_set_contains_256() -> None:
    from search.pruning_space.action_catalog import DEFAULT_GROUPED_CHANNELS_PER_GROUP

    assert 256 in DEFAULT_GROUPED_CHANNELS_PER_GROUP
    assert DEFAULT_GROUPED_CHANNELS_PER_GROUP == (4, 8, 16, 32, 64, 128, 256, 512)
