"""Tests for search module: encoding, objectives, and genetic search."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from heal_compress.search.search_space import CandidateEncoding, SearchSpaceEncoder
from heal_compress.search.proxy_objective import (
    ProxyObjectiveEvaluator, BOPSProxy, SizeProxy, DeployPenalty,
)
from heal_compress.search.genetic_search import GeneticSearchEngine, SearchConstraints
from heal_compress.tracer.coupled_channel_group import CoupledChannelGroup


def _make_dummy_groups(n=5):
    """Create dummy coupled channel groups for testing."""
    groups = []
    for i in range(n):
        groups.append(CoupledChannelGroup(
            group_id=f"group_{i}",
            source_modules=[f"layer_{i}"],
            channel_indices=list(range(16)),
            is_prunable=(i > 0),
            is_protected=(i == 0),
        ))
    return groups


def _make_dummy_model(num_layers=5):
    """Create a simple sequential model for testing."""
    layers = []
    ch = 16
    for i in range(num_layers):
        layers.extend([
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(),
        ])
    model = nn.Sequential(*layers)
    # Name layers to match group source_modules
    named = {}
    for i, (name, module) in enumerate(model.named_modules()):
        if isinstance(module, nn.Conv2d):
            named[f"layer_{i // 3}"] = name
    return model


class TestCandidateEncoding:
    """Tests for CandidateEncoding."""

    def test_full_candidate(self):
        """Test creating a full-precision candidate."""
        c = CandidateEncoding.full(
            ["g0", "g1", "g2"], ["l0", "l1"], "FP16"
        )
        assert all(v == 1 for v in c.prune_vars.values())
        assert all(v == "FP16" for v in c.bitwidth_vars.values())

    def test_random_candidate(self):
        """Test creating a random candidate."""
        c = CandidateEncoding.random(
            ["g0", "g1", "g2"], ["l0", "l1"],
            weight_bits=["FP16", "INT8", "INT4"],
        )
        assert len(c.prune_vars) == 3
        assert len(c.bitwidth_vars) == 2

    def test_clone(self):
        """Test that clone produces an independent copy."""
        c = CandidateEncoding.full(["g0"], ["l0"])
        c2 = c.clone()
        c2.prune_vars["g0"] = 0
        assert c.prune_vars["g0"] == 1

    def test_serialization(self):
        """Test to_dict / from_dict roundtrip."""
        c = CandidateEncoding.random(["g0", "g1"], ["l0", "l1"])
        d = c.to_dict()
        c2 = CandidateEncoding.from_dict(d)
        assert c2.prune_vars == c.prune_vars
        assert c2.bitwidth_vars == c.bitwidth_vars


class TestSearchConstraints:
    """Tests for SearchConstraints."""

    def test_repair_protected(self):
        """Test that repair forces protected groups to keep=1."""
        constraints = SearchConstraints(
            protected_group_ids={"g0"},
            weight_bits=["FP16", "INT8"],
        )
        c = CandidateEncoding({"g0": 0, "g1": 0}, {"l0": "FP16"})
        repaired = constraints.repair(c)
        assert repaired.prune_vars["g0"] == 1

    def test_repair_min_ratio(self):
        """Test that repair ensures minimum retention ratio."""
        constraints = SearchConstraints(
            protected_group_ids=set(),
            weight_bits=["FP16"],
            min_stage_ratio=0.5,
        )
        c = CandidateEncoding(
            {f"g{i}": 0 for i in range(10)},
            {"l0": "FP16"},
        )
        repaired = constraints.repair(c)
        kept = sum(v for v in repaired.prune_vars.values())
        assert kept >= 5

    def test_repair_bitwidth(self):
        """Test that repair snaps invalid bit-widths."""
        constraints = SearchConstraints(
            protected_group_ids=set(),
            weight_bits=["FP16", "INT8"],
        )
        c = CandidateEncoding({"g0": 1}, {"l0": "INT4"})
        repaired = constraints.repair(c)
        assert repaired.bitwidth_vars["l0"] in ["FP16", "INT8"]


class TestProxyObjective:
    """Tests for ProxyObjectiveEvaluator."""

    def test_full_candidate_score(self):
        """Test that a full candidate has a finite score."""
        c = CandidateEncoding.full(["g0", "g1"], ["l0", "l1"])
        evaluator = ProxyObjectiveEvaluator()
        result = evaluator.evaluate(c)
        assert "score" in result
        assert result["score"] >= 0

    def test_feasibility(self):
        """Test feasibility flag when no penalties."""
        c = CandidateEncoding.full(["g0"], ["l0"])
        evaluator = ProxyObjectiveEvaluator()
        result = evaluator.evaluate(c)
        assert result["feasible"] is True


class TestGeneticSearch:
    """Tests for GeneticSearchEngine."""

    def test_small_search(self):
        """Test running a small GA search."""
        group_ids = ["g0", "g1", "g2"]
        layer_names = ["l0", "l1"]
        evaluator = ProxyObjectiveEvaluator()
        constraints = SearchConstraints(
            protected_group_ids={"g0"},
            weight_bits=["FP16", "INT8"],
        )
        engine = GeneticSearchEngine(
            group_ids=group_ids,
            layer_names=layer_names,
            evaluator=evaluator,
            constraints=constraints,
            pop_size=10,
            max_generations=3,
            elite_size=2,
            output_dir="/tmp/test_ga_search",
        )
        best, score = engine.run(topk=3)
        assert isinstance(best, CandidateEncoding)
        assert score >= 0
        # Protected group should be kept
        assert best.prune_vars["g0"] == 1
