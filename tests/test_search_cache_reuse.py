from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from search.cache.artifact_cache import ArtifactCache
from search.cache.proxy_cache import ProxyCache
from search.cache.real_eval_cache import RealEvalCache


def test_proxy_cache_hit(tmp_path: Path) -> None:
    cache = ProxyCache(tmp_path / "proxy.jsonl")
    cache.put("hash-a", {"F1": 1.25, "R_size": 0.5})

    assert cache.get("hash-a") == {"F1": 1.25, "R_size": 0.5}
    assert cache.get("missing") is None


def test_real_eval_cache_hit_prevents_engine_build(tmp_path: Path) -> None:
    cache = RealEvalCache(tmp_path / "real.jsonl")
    cache.put("hash-a", {"F2": 0.1, "status": "ok", "engine_hash": "engine"})
    calls = {"build": 0}

    def build_engine() -> dict[str, object]:
        calls["build"] += 1
        return {"F2": 9.9}

    cached = cache.get_or_evaluate("hash-a", build_engine)

    assert cached["F2"] == 0.1
    assert calls["build"] == 0


def test_real_eval_cache_uses_strict_eval_cache_key(tmp_path: Path) -> None:
    cache_path = tmp_path / "real.jsonl"
    cache = RealEvalCache(cache_path)
    cache.put(
        "eval-key",
        {
            "candidate_hash": "candidate-a",
            "cache_key": "eval-key",
            "status": "ok",
            "engine_hash": "engine-a",
        },
    )

    reloaded = RealEvalCache(cache_path)

    assert reloaded.get("eval-key")["engine_hash"] == "engine-a"
    assert reloaded.get("candidate-a") is None


def test_physical_artifact_reuse_and_engine_separation(tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path / "artifacts.jsonl")
    cache.put_physical("physical-1", {"pruned_model": "a/pruned_model.pth"})
    cache.put_engine("candidate-a", "engine-env-a", {"engine": "a.engine"})

    assert cache.get_physical("physical-1") == {"pruned_model": "a/pruned_model.pth"}
    assert cache.get_physical("physical-1") is not None
    assert cache.get_engine("candidate-a", "engine-env-a") == {"engine": "a.engine"}
    assert cache.get_engine("candidate-a", "engine-env-b") is None


def test_same_pruning_different_precision_reuses_physical_not_engine(tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path / "artifacts.jsonl")
    cache.put_physical("physical-shape", {"snapshot_hash": "shape"})
    cache.put_qdq_onnx("candidate-fp16", {"qdq": "fp16.onnx"})

    assert cache.get_physical("physical-shape") is not None
    assert cache.get_qdq_onnx("candidate-int8") is None
