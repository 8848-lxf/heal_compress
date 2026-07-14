from __future__ import annotations

import sys
import json
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _mapping(*entries):
    from quantization.types import CanonicalPrecisionEntry, CanonicalPrecisionMappingResult

    return CanonicalPrecisionMappingResult(
        entries=[
            CanonicalPrecisionEntry(
                requested_precision="fp16",
                realized_request_precision="fp16",
                **entry,
            )
            for entry in entries
        ]
    )


def _group(group_id: str, module_paths: tuple[str, ...], *, source: str = "precision_coupling_tracer"):
    from search.quantization_space.types import QuantizationSearchGroup

    return QuantizationSearchGroup(
        group_id=group_id,
        module_paths=module_paths,
        canonical_node_ids=(),
        allowed_precisions=("FP16", "INT8"),
        protected=False,
        protection_reason="",
        ordering=0,
        parameter_count=1,
        baseline_macs=1.0,
        metadata={
            "group_namespace": "quantization",
            "precision_group_source": source,
        },
    )


def test_group_separation_audit_reports_distinct_namespaces_and_crosswalk() -> None:
    from search.group_separation_audit import audit_pruning_quantization_groups

    report = audit_pruning_quantization_groups(
        pruning_group_ids=("prune::conv_a::out0", "prune::conv_b::out0"),
        pruning_group_metadata={
            "prune::conv_a::out0": {"root_module_path": "conv_a"},
            "prune::conv_b::out0": {"root_module_path": "conv_b"},
        },
        quantization_groups=(
            _group("quant::conv_a", ("conv_a",)),
            _group("quant::conv_b", ("conv_b",)),
        ),
        canonical_mapping=_mapping(
            {
                "module_path": "conv_a",
                "canonical_node_name": "canonical_conv_a",
                "weight_initializer": "conv_a.weight",
                "precision_group": "quant::conv_a",
            },
            {
                "module_path": "conv_b",
                "canonical_node_name": "canonical_conv_b",
                "weight_initializer": "conv_b.weight",
                "precision_group": "quant::conv_b",
            },
            {
                "module_path": "pyramid_backbone.functional_affine_grid_matmul",
                "canonical_node_name": "canonical_affine_grid",
                "weight_initializer": "",
                "precision_group": "protected_functional_affine_grid",
            },
        ),
    )

    assert report["passed"] is True
    assert report["pruning_group_count"] == 2
    assert report["quantization_group_count"] == 2
    assert report["pruning_group_ids"] == ["prune::conv_a::out0", "prune::conv_b::out0"]
    assert report["quantization_group_ids"] == ["quant::conv_a", "quant::conv_b"]
    assert report["crosswalk_mapping_count"] == 2
    assert report["id_collision_count"] == 0
    assert report["unmapped_weighted_layer_count"] == 0
    assert report["duplicate_canonical_mapping_count"] == 0
    assert report["precision_inherited_from_pruning_scope_count"] == 0


def test_group_separation_audit_fails_closed_on_mixed_group_ownership() -> None:
    from search.group_separation_audit import audit_pruning_quantization_groups

    report = audit_pruning_quantization_groups(
        pruning_group_ids=("shared",),
        pruning_group_metadata={"shared": {"root_module_path": "conv_a"}},
        quantization_groups=(
            _group("shared", ("conv_a",), source="pruning_scope"),
            _group("quant::duplicate", ("conv_a",), source="pruning_scope"),
        ),
        canonical_mapping=_mapping(
            {
                "module_path": "conv_a",
                "canonical_node_name": "canonical_conv_a",
                "weight_initializer": "conv_a.weight",
                "precision_group": "shared",
            },
            {
                "module_path": "conv_b",
                "canonical_node_name": "canonical_conv_b",
                "weight_initializer": "conv_b.weight",
                "precision_group": "quant::missing",
            },
        ),
    )

    assert report["passed"] is False
    assert report["id_collision_count"] == 1
    assert report["unmapped_weighted_layer_count"] == 1
    assert report["duplicate_canonical_mapping_count"] == 1
    assert report["precision_inherited_from_pruning_scope_count"] == 2
    assert set(report["failure_reasons"]) == {
        "pruning_quant_group_collision",
        "unmapped_weighted_layer",
        "duplicate_canonical_mapping",
        "precision_inherited_from_pruning_scope",
    }


def test_group_separation_audit_writer_persists_report_and_fails_closed(tmp_path: Path) -> None:
    from search.group_separation_audit import write_group_separation_audit

    output = tmp_path / "group_separation_audit.json"
    with pytest.raises(RuntimeError, match="pruning_quant_group_audit_failed"):
        write_group_separation_audit(
            output,
            pruning_group_ids=("shared",),
            pruning_group_metadata={"shared": {"root_module_path": "conv"}},
            quantization_groups=(_group("shared", ("conv",), source="pruning_scope"),),
            canonical_mapping=_mapping(
                {
                    "module_path": "conv",
                    "canonical_node_name": "canonical_conv",
                    "weight_initializer": "conv.weight",
                    "precision_group": "shared",
                },
            ),
        )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert report["id_collision_count"] == 1
