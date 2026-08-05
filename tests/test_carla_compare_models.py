import json
from pathlib import Path

import pytest

from carla_integration.compare_models import _named_ints, _named_paths, _search_metrics


def test_named_paths_require_unique_name_equals_path(tmp_path: Path):
    parsed = _named_paths([f"Pyramid={tmp_path / 'result.json'}"])
    assert parsed["Pyramid"] == (tmp_path / "result.json").resolve()
    with pytest.raises(ValueError, match="duplicate"):
        _named_paths(["Pyramid=one.json", "Pyramid=two.json"])


def test_search_metrics_support_budget_array(tmp_path: Path):
    path = tmp_path / "search.json"
    path.write_text(
        json.dumps(
            [
                {
                    "mAP": 0.5,
                    "AP@0.3": 0.6,
                    "AP@0.5": 0.5,
                    "AP@0.7": 0.4,
                    "forward_p50_ms": 3.0,
                    "num_evaluated_frames": 10,
                }
            ]
        ),
        encoding="utf-8",
    )
    assert _search_metrics(path, 0)["map"] == 0.5
    assert _named_ints(["Pyramid=1"]) == {"Pyramid": 1}
