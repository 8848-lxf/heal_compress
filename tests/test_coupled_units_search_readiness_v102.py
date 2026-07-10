from types import SimpleNamespace
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.latency_lut.audit_coupled_units_search_readiness_v102 import (
    build_search_space_encoding_schema,
    build_search_variable_rows,
    build_selector_stable_trace_rows,
    build_selector_universe_report,
    build_stability_report,
    stable_unit_id_for_unit,
)


def _unit(unit_id="domain::idx0", idx=0, module="conv"):
    return SimpleNamespace(
        unit_id=unit_id,
        scope_id="domain",
        root_idx=idx,
        root_channel_index=idx,
        members=[
            {
                "module_name": module,
                "module_type": "Conv2d",
                "axis": "out_channels",
                "local_index": idx,
                "index": idx,
                "concat_offset": 0,
                "residual_add_id": "",
                "grouped_conv_role": "",
                "transpose_conv_role": "",
            },
            {
                "module_name": f"{module}.bn",
                "module_type": "BatchNorm2d",
                "axis": "bn_channel",
                "local_index": idx,
                "index": idx,
                "concat_offset": 0,
                "residual_add_id": "",
                "grouped_conv_role": "",
                "transpose_conv_role": "",
            },
        ],
        dependency_types=["conv_out_to_bn"],
        constraints={"group_type": ""},
        metadata={"group_type": ""},
        importance=0.25,
        importance_mode="first_order_taylor",
        protected=False,
        protected_reason=None,
        unsupported_reason="",
        is_grouped_conv_related=False,
        grouped_conv_info=None,
    )


def test_stable_unit_id_ignores_member_order():
    unit_a = _unit()
    unit_b = _unit()
    unit_b.members = list(reversed(unit_b.members))

    assert stable_unit_id_for_unit(unit_a) == stable_unit_id_for_unit(unit_b)


def test_search_variable_rows_include_required_search_fields():
    rows = build_search_variable_rows([_unit()], candidates_by_source={"domain::idx0": []})

    row = rows[0]
    assert row["stable_unit_id"].startswith("cu_")
    assert row["is_searchable"] is True
    assert row["importance_score_available"] is True
    assert row["importance_score"] == 0.25
    assert row["contains_conv"] is True
    assert row["contains_bn"] is True
    assert row["bn_closure_complete"] is True
    assert row["supports_A"] is True
    assert row["supports_B"] is True


def test_stability_report_detects_changed_ids():
    stable_a = stable_unit_id_for_unit(_unit(idx=0))
    stable_b = stable_unit_id_for_unit(_unit(idx=1, unit_id="domain::idx1"))

    report = build_stability_report(
        [
            [{"unit_id": "domain::idx0", "stable_unit_id": stable_a, "domain_id": "domain", "members_json": "a"}],
            [{"unit_id": "domain::idx1", "stable_unit_id": stable_b, "domain_id": "domain", "members_json": "b"}],
        ]
    )

    assert report["stable_id_set_equal_across_runs"] is False
    assert report["verdict"] == "unstable_unit_ids"


def test_selector_universe_report_maps_source_units_to_search_table():
    rows = build_search_variable_rows([_unit()], candidates_by_source={"domain::idx0": []})
    report = build_selector_universe_report(
        selector_candidates=[
            {
                "candidate_id": "cand0",
                "source_coupled_units": ["domain::idx0", "domain::idx_missing"],
            }
        ],
        search_rows=rows,
    )

    assert report["selector_candidate_universe_mismatch"] is True
    assert report["selector_only_units"] == ["domain::idx_missing"]
    assert report["matched_source_units"] == ["domain::idx0"]


def test_search_space_schema_uses_stable_unit_ids_and_policy_masks():
    rows = build_search_variable_rows([_unit()], candidates_by_source={"domain::idx0": []})
    schema = build_search_space_encoding_schema(rows)

    assert schema["encoding_version"] == "v102"
    assert schema["variable_id_field"] == "stable_unit_id"
    assert schema["variables"][0]["actions"] == ["keep", "prune"]
    assert schema["variables"][0]["policy_compatibility"]["A"] is True


def test_selector_stable_trace_rows_include_stable_unit_id():
    rows = build_search_variable_rows([_unit()], candidates_by_source={"domain::idx0": []})
    trace = build_selector_stable_trace_rows(
        [
            {
                "candidate_id": "cand0",
                "domain_id": "domain",
                "strategy_policy": "A1",
                "source_coupled_units": ["domain::idx0"],
            }
        ],
        rows,
    )

    assert trace[0]["source_unit_id"] == "domain::idx0"
    assert trace[0]["stable_unit_id"] == rows[0]["stable_unit_id"]
    assert trace[0]["source_unit_found_in_search_table"] is True
