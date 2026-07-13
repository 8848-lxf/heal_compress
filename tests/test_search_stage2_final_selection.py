from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_collect_round_winners_uses_round_directory_index_for_cached_rows(tmp_path: Path) -> None:
    from search.stage2.final_selection import collect_round_winners

    run_dir = tmp_path / "run"
    round_dir = run_dir / "round_002"
    round_dir.mkdir(parents=True)
    (round_dir / "round_best_candidate.json").write_text(
        json.dumps(
            {
                "round_index": 1,
                "candidate_hash": "cached_from_old_round",
                "F2": 0.4,
            }
        ),
        encoding="utf-8",
    )

    winners = collect_round_winners(run_dir)

    assert winners == [
        {
            "round_index": 2,
            "round_dir": str(round_dir),
            "candidate_hash": "cached_from_old_round",
            "F2": 0.4,
        }
    ]
