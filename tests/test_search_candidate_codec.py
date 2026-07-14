from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import replace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from search.candidate import CandidateGenotype, CandidatePhenotype, PrecisionDecision
from search.candidate_codec import decode_candidate, encode_candidate
from search.canonicalization import SearchSpaceSpec, canonicalize_candidate, repair_genotype
from search.hashing import candidate_hash


def _space(order: list[str] | None = None) -> SearchSpaceSpec:
    return SearchSpaceSpec(
        pruning_unit_ids=order or ["u1", "u2", "u3"],
        precision_layer_ids=["layer.a", "layer.b"],
        protected_pruning_unit_ids={"u1"},
        default_precision="FP16",
        trace_snapshot_hash="trace-hash",
        calibration_manifest_hash="calib-hash",
        onnx_export_config_hash="onnx-hash",
        tensorrt_version="10.9",
        gpu_compute_capability="8.9",
        builder_flags={"fp16": True, "int8": True},
        plugin_hashes={"scatter": "plugin-hash"},
    )


def test_candidate_genotype_codec_roundtrip() -> None:
    genotype = CandidateGenotype(
        pruning_genes={"u2": 0, "u1": 1},
        precision_genes={"layer.b": "INT8", "layer.a": "FP32"},
        meta={"origin": "unit-test"},
    )

    encoded = encode_candidate(genotype)
    decoded = decode_candidate(encoded)

    assert decoded == genotype
    assert list(encoded["pruning_genes"]) == ["u1", "u2"]
    assert list(encoded["precision_genes"]) == ["layer.a", "layer.b"]


def test_candidate_codec_exposes_group_mask_and_layer_bitwidth_contract() -> None:
    genotype = CandidateGenotype(
        pruning_genes={"prune::b": 0, "prune::a": 1},
        precision_genes={"quant::b": "INT8", "quant::a": "FP16"},
    )

    encoded = encode_candidate(genotype)
    decoded = decode_candidate(
        {
            "group_mask": encoded["group_mask"],
            "layer_bitwidth": encoded["layer_bitwidth"],
        }
    )

    assert encoded["group_mask"] == {"prune::a": 1, "prune::b": 0}
    assert encoded["layer_bitwidth"] == {"quant::a": "FP16", "quant::b": "INT8"}
    assert decoded == genotype


def test_repair_protects_units_and_snaps_precision() -> None:
    repaired = repair_genotype(
        CandidateGenotype(
            pruning_genes={"u1": 0, "u2": 0, "unknown": 0},
            precision_genes={"layer.a": "INT4", "unknown": "INT8"},
        ),
        _space(),
    )

    assert repaired.pruning_genes == {"u1": 1, "u2": 0, "u3": 1}
    assert repaired.precision_genes == {"layer.a": "FP16", "layer.b": "FP16"}


def test_same_phenotype_has_same_hash_despite_input_order() -> None:
    genotype = CandidateGenotype(
        pruning_genes={"u3": 0, "u1": 1, "u2": 1},
        precision_genes={"layer.b": "INT8", "layer.a": "FP16"},
    )
    phenotype_a = canonicalize_candidate(genotype, _space(["u1", "u2", "u3"]))
    phenotype_b = canonicalize_candidate(genotype, _space(["u3", "u2", "u1"]))

    assert phenotype_a.pruned_unit_ids == phenotype_b.pruned_unit_ids == ["u3"]
    assert candidate_hash(phenotype_a, _space()) == candidate_hash(phenotype_b, _space())


def test_precision_fallback_hash_uses_realized_profile() -> None:
    phenotype_requested_int8 = CandidatePhenotype(
        pruned_unit_ids=[],
        precision_profile={
            "layer.a": PrecisionDecision("INT8", "FP16", "unsupported"),
        },
    )
    phenotype_requested_fp16 = CandidatePhenotype(
        pruned_unit_ids=[],
        precision_profile={
            "layer.a": PrecisionDecision("FP16", "FP16", ""),
        },
    )

    assert candidate_hash(phenotype_requested_int8, _space()) == candidate_hash(phenotype_requested_fp16, _space())


def test_candidate_hash_changes_when_code_commit_changes() -> None:
    phenotype = canonicalize_candidate(
        CandidateGenotype(
            pruning_genes={"u1": 1, "u2": 1, "u3": 1},
            precision_genes={"layer.a": "FP16", "layer.b": "INT8"},
        ),
        _space(),
    )
    first = replace(_space(), code_commit="commit-a")
    second = replace(_space(), code_commit="commit-b")

    assert candidate_hash(phenotype, first) != candidate_hash(phenotype, second)


def test_phenotype_keeps_requested_and_realized_precision() -> None:
    phenotype = canonicalize_candidate(
        CandidateGenotype(
            pruning_genes={"u1": 1, "u2": 1, "u3": 1},
            precision_genes={"layer.a": "INT8", "layer.b": "FP16"},
        ),
        _space(),
        realized_precision={"layer.a": ("FP16", "grouped_channels_per_group_not_allowed")},
    )

    assert phenotype.precision_profile["layer.a"].requested_precision == "INT8"
    assert phenotype.precision_profile["layer.a"].realized_precision == "FP16"
    assert phenotype.precision_profile["layer.a"].fallback_reason == "grouped_channels_per_group_not_allowed"
