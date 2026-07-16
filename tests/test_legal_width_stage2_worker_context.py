from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_legal_width_worker_requires_global_pruning_context() -> None:
    from search.stage2.candidate_worker import _requires_global_pruning_context

    assert _requires_global_pruning_context(
        {"pruning": {"gene_type": "legal_keep_width"}}
    )
    assert _requires_global_pruning_context(
        {"joint_taylor_anchor_sweep": {"enabled": True}}
    )
    assert not _requires_global_pruning_context(
        {"pruning": {"gene_type": "coupled_channel_keep_mask"}}
    )
