from __future__ import annotations

from pathlib import Path

from tools.check_public_release import audit


def test_public_tree_has_no_private_or_generated_artifacts() -> None:
    root = Path(__file__).resolve().parents[1]
    assert audit(root, maximum_bytes=8 * 1024 * 1024) == []
