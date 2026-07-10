from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _minimal_v98_fixture(root: Path) -> Path:
    _write_json(root / "trace_graph_coverage_report.json", {"dynamic_branch_enumeration_enabled": False})
    _write_json(root / "coupled_channel_units_full_model.json", [])
    _write_json(root / "coupled_channel_unit_completeness_report.json", {})
    _write_json(root / "tp_oracle_sampling_diff_report.json", {})
    _write_json(root / "mask0_physical_removal_dryrun_report.json", {})
    _write_json(root / "full_model_surface_after_unprotect_grouped_input.json", {})
    _write_csv(root / "operator_dependency_coverage_matrix.csv", [], ["op_type"])
    _write_csv(root / "grouped_conv_dependency_proof.csv", [], ["module_name"])
    _write_csv(root / "convtranspose_dependency_proof.csv", [], ["module_name"])
    return root


def _proof_fields() -> list[str]:
    return [
        "node_name",
        "node_type",
        "proof_pass",
        "num_input_branches",
        "input_branches",
        "matched_scope_id",
        "concat_offset_recorded",
        "downstream_conv_input_offset_recorded",
    ]


def _read_report(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open(newline="", encoding="utf-8")))


def test_single_trace_limitation_is_coverage_insufficient_not_unsupported(tmp_path: Path) -> None:
    from tools.latency_lut.audit_dependency_graph_v991 import build_v991_dependency_audit_reports

    v98 = _minimal_v98_fixture(tmp_path / "v98")
    _write_json(
        v98 / "op_graph_path_0.json",
        {
            "nodes": {
                "branch_a": {"op_type": "Conv", "output_shapes": [[1, 4, 8, 8]]},
                "branch_b": {"op_type": "Conv", "output_shapes": [[1, 4, 8, 8]]},
                "cat": {"op_type": "Cat", "cat_dim": 1, "input_shapes": [[1, 4, 8, 8], [1, 4, 8, 8]], "output_shapes": [[1, 8, 8, 8]]},
                "consumer": {"op_type": "Conv", "groups": 1, "input_shapes": [[1, 8, 8, 8]]},
            },
            "edges": [
                {"src": "branch_a", "dst": "cat", "input_index": 0},
                {"src": "branch_b", "dst": "cat", "input_index": 1},
                {"src": "cat", "dst": "consumer", "input_index": 0},
            ],
        },
    )
    _write_csv(
        v98 / "residual_concat_full_model_proof.csv",
        [
            {
                "node_name": "cat",
                "node_type": "concat",
                "proof_pass": "False",
                "num_input_branches": 2,
                "input_branches": '["branch_a", "branch_b"]',
                "matched_scope_id": "scope::cat",
                "concat_offset_recorded": "True",
                "downstream_conv_input_offset_recorded": "True",
            }
        ],
        _proof_fields(),
    )

    out = tmp_path / "v991"
    build_v991_dependency_audit_reports(v98, out)
    rows = _read_report(out / "residual_concat_fail_root_cause_report.csv")

    assert rows[0]["failure_category"] == "coverage_insufficient_single_trace"
    assert rows[0]["evidence_status"] == "coverage_insufficient"
    assert rows[0]["theoretical_prunability"] == "unknown_until_multitrace"
    assert "dynamic_path_not_covered" not in rows[0]["failure_category"]
    assert rows[0]["current_action"] == "do_not_mark_unsupported_due_to_single_trace"


def test_linear_pfn_and_scatter_are_classified_with_specific_resolvers(tmp_path: Path) -> None:
    from tools.latency_lut.audit_dependency_graph_v991 import build_v991_dependency_audit_reports

    v98 = _minimal_v98_fixture(tmp_path / "v98")
    _write_json(
        v98 / "op_graph_path_0.json",
        {
            "nodes": {
                "pfn_cat": {"op_type": "Cat", "cat_dim": 1, "input_shapes": [[16, 8], [16, 8]], "output_shapes": [[16, 16]]},
                "encoder_m1.pillar_vfe.pfn_layers.0.linear": {"op_type": "Linear", "raw_type": "Linear"},
                "scatter_add": {"op_type": "Add", "input_shapes": [[1, 64, 10, 10], [1, 64, 10, 10]], "output_shapes": [[1, 64, 10, 10]], "module_scope": ["encoder_m1.scatter"]},
                "op::scatter_consumer": {"op_type": "Other", "raw_type": "grid_sample"},
            },
            "edges": [
                {"src": "pfn_cat", "dst": "encoder_m1.pillar_vfe.pfn_layers.0.linear", "input_index": 0},
                {"src": "scatter_add", "dst": "op::scatter_consumer", "input_index": 0},
            ],
        },
    )
    _write_csv(
        v98 / "residual_concat_full_model_proof.csv",
        [
            {
                "node_name": "pfn_cat",
                "node_type": "concat",
                "proof_pass": "False",
                "num_input_branches": 2,
                "input_branches": "[]",
                "matched_scope_id": "",
                "concat_offset_recorded": "False",
                "downstream_conv_input_offset_recorded": "False",
            },
            {
                "node_name": "scatter_add",
                "node_type": "residual_add",
                "proof_pass": "False",
                "num_input_branches": 2,
                "input_branches": "[]",
                "matched_scope_id": "",
                "concat_offset_recorded": "False",
                "downstream_conv_input_offset_recorded": "False",
            },
        ],
        _proof_fields(),
    )

    out = tmp_path / "v991"
    build_v991_dependency_audit_reports(v98, out)
    by_node = {row["node_name"]: row for row in _read_report(out / "residual_concat_fail_root_cause_report.csv")}

    assert "linear_feature_dim_mapping_missing" in by_node["pfn_cat"]["failure_category"]
    assert "pfn_feature_dim_resolver_missing" in by_node["pfn_cat"]["failure_category"]
    assert by_node["pfn_cat"]["evidence_status"] == "resolver_missing"
    assert "scatter_geometry_index_op_protected" in by_node["scatter_add"]["failure_category"]
    assert by_node["scatter_add"]["theoretical_prunability"] == "geometry_or_index_op_protected"


def test_grouped_convtranspose_is_resolver_missing_not_permanently_unsupported(tmp_path: Path) -> None:
    from tools.latency_lut.audit_dependency_graph_v991 import build_v991_dependency_audit_reports

    v98 = _minimal_v98_fixture(tmp_path / "v98")
    _write_json(
        v98 / "op_graph_path_0.json",
        {
            "nodes": {
                "cat": {"op_type": "Cat", "cat_dim": 1, "input_shapes": [[1, 8, 8, 8], [1, 8, 8, 8]], "output_shapes": [[1, 16, 8, 8]]},
                "grouped_deconv": {"op_type": "ConvTranspose2d", "groups": 4, "input_shapes": [[1, 16, 8, 8]]},
            },
            "edges": [{"src": "cat", "dst": "grouped_deconv", "input_index": 0}],
        },
    )
    _write_csv(
        v98 / "residual_concat_full_model_proof.csv",
        [
            {
                "node_name": "cat",
                "node_type": "concat",
                "proof_pass": "False",
                "num_input_branches": 2,
                "input_branches": "[]",
                "matched_scope_id": "",
                "concat_offset_recorded": "True",
                "downstream_conv_input_offset_recorded": "False",
            }
        ],
        _proof_fields(),
    )

    out = tmp_path / "v991"
    build_v991_dependency_audit_reports(v98, out)
    row = _read_report(out / "residual_concat_fail_root_cause_report.csv")[0]

    assert "grouped_convtranspose_resolver_missing" in row["failure_category"]
    assert "currently_rejected_for_safety" in row["failure_category"]
    assert "theoretically_prunable_with_group_balanced_input_output_resolver" in row["failure_category"]
    assert row["evidence_status"] == "resolver_missing"
    assert row["theoretical_prunability"] == "prunable_with_new_resolver"
