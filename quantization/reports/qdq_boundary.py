"""Auditable graph and TensorRT realization reports for explicit Q/DQ boundaries."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..tensorrt.layer_info import has_canonical_identity, layer_metadata, layer_name, load_layer_info, precision_name


def _plain(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _plain(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(item) for item in value]
    return value


def _normalized_boundary_tensor(value: str) -> str:
    return str(value).replace("__before_output_qdq", "")


def enrich_weighted_qdq_boundary_audit(
    rows: Sequence[Mapping[str, Any]],
    layer_info: str | Path | Sequence[Mapping[str, Any]] | Mapping[str, Any],
) -> dict[str, Any]:
    """Join graph-level Q/DQ placement with the realized fused TensorRT layer."""

    layers = load_layer_info(layer_info)
    enriched: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        canonical = str(row.get("weighted_node", ""))
        matched = [layer for layer in layers if has_canonical_identity(layer, canonical)]
        compute_matched = [
            layer
            for layer in matched
            if "reformatting copynode" not in layer_name(layer).lower()
            and "reformat" not in str(layer.get("LayerType", "")).lower()
        ]
        metadata = " | ".join(layer_metadata(layer) for layer in compute_matched)
        following = [dict(value) for value in row.get("following_ops", [])]
        semantic_following = [value for value in following if not bool(value.get("next_weighted_node", False))]
        fused_following = [
            value
            for value in semantic_following
            if str(value.get("name", "")) and str(value.get("name", "")) in metadata
        ]
        realized = precision_name(compute_matched[0]) if len(compute_matched) == 1 else ""
        placement = str(row.get("qdq_placement", ""))
        q_inputs = [str(value) for value in row.get("q_node_actual_input_tensor", [])]
        weighted_outputs = [str(value) for value in row.get("weighted_output_tensor", [])]
        scale_owner = str(row.get("activation_scale_owner", ""))
        normalized_q_inputs = {_normalized_boundary_tensor(value) for value in q_inputs}
        scale_owner_matches_q = not q_inputs or (
            bool(scale_owner) and normalized_q_inputs == {scale_owner}
        )
        issues: list[str] = []
        warnings: list[str] = []
        if len(compute_matched) != 1:
            issues.append(f"canonical_engine_compute_layer_match_count:{len(compute_matched)}")
        expected = str(row.get("legalized_precision", "")).lower()
        if realized and expected and realized != expected:
            issues.append(f"realized_precision_mismatch:{realized}!={expected}")
        if not scale_owner_matches_q:
            issues.append("activation_scale_owner_does_not_match_q_input")

        first_semantic = semantic_following[0] if semantic_following else None
        first_type = str(first_semantic.get("op_type", "")) if first_semantic else ""
        if placement == "weighted_output_before_following_ops" and first_type == "Relu":
            relu_fused = bool(fused_following and str(fused_following[0].get("op_type", "")) == "Relu")
            issues.append("explicit_qdq_precedes_relu_semantic_boundary")
            effective = (
                "invalid_pre_relu_qdq_even_when_engine_fused"
                if relu_fused
                else "raw_weighted_output_before_relu"
            )
        elif placement == "weighted_output_before_following_ops" and first_type in {"Add", "Concat"}:
            effective = f"quantized_branch_output_then_{row.get('merge_policy', '')}_{first_type}"
        elif placement == "fp16_weighted_output_no_output_qdq":
            effective = "explicit_fp16_weighted_output_before_functional_or_merge_boundary"
        elif fused_following:
            effective = "fused_through:" + ",".join(str(value.get("op_type", "")) for value in fused_following)
        else:
            effective = placement or "unclassified"

        row.update(
            {
                "realized_precision": realized or "unresolved",
                "engine_fused_layer": [layer_name(layer) for layer in compute_matched],
                "engine_reformat_layers": [
                    layer_name(layer) for layer in matched if layer not in compute_matched
                ],
                "engine_fused_following_ops": fused_following,
                "engine_effective_boundary": effective,
                "activation_scale_owner_matches_q_input": scale_owner_matches_q,
                "boundary_issues": issues,
                "boundary_warnings": warnings,
                "boundary_passed": not issues,
            }
        )
        enriched.append(row)
    issues = [
        {"canonical_layer": row.get("canonical_layer", ""), "issues": list(row.get("boundary_issues", []))}
        for row in enriched
        if row.get("boundary_issues")
    ]
    return {
        "status": "passed" if not issues else "failed",
        "passed": not issues,
        "weighted_int8_layer_count": len(enriched),
        "raw_weighted_output_qdq_count": sum(
            str(row.get("qdq_placement", "")) == "weighted_output_before_following_ops" for row in enriched
        ),
        "fp16_output_no_output_qdq_count": sum(
            str(row.get("qdq_placement", "")) == "fp16_weighted_output_no_output_qdq" for row in enriched
        ),
        "issues": issues,
        "warnings": [
            {
                "canonical_layer": row.get("canonical_layer", ""),
                "warnings": list(row.get("boundary_warnings", [])),
            }
            for row in enriched
            if row.get("boundary_warnings")
        ],
        "layers": enriched,
    }


def write_production_qdq_boundary_reports(
    output_dir: str | Path,
    qdq_result: Any,
    layer_info: str | Path | Sequence[Mapping[str, Any]] | Mapping[str, Any],
) -> dict[str, Any]:
    """Write JSON, CSV, and a concise verdict for the production artifact."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    payload = _plain(qdq_result)
    metadata = dict(payload.get("calibration_metadata", {}))
    report = enrich_weighted_qdq_boundary_audit(
        metadata.get("weighted_qdq_boundary_audit", []),
        layer_info,
    )
    report["qdq_topology_hash"] = str(metadata.get("qdq_topology_hash", ""))
    report["merge_policy"] = str(metadata.get("merge_policy", ""))
    (destination / "production_qdq_boundary_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    fields = [
        "canonical_layer",
        "weighted_node",
        "weighted_output_tensor",
        "following_ops",
        "q_node_actual_input_tensor",
        "qdq_placement",
        "activation_scale_owner",
        "merge_policy",
        "requested_precision",
        "legalized_precision",
        "requested_output_precision",
        "realized_precision",
        "engine_fused_layer",
        "engine_reformat_layers",
        "engine_effective_boundary",
        "boundary_passed",
        "boundary_issues",
        "boundary_warnings",
    ]
    with (destination / "production_qdq_boundary_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for raw in report["layers"]:
            row = {key: raw.get(key, "") for key in fields}
            for key in (
                "weighted_output_tensor",
                "following_ops",
                "q_node_actual_input_tensor",
                "engine_fused_layer",
                "engine_reformat_layers",
                "boundary_issues",
                "boundary_warnings",
            ):
                row[key] = json.dumps(row[key], sort_keys=True)
            writer.writerow(row)

    shrink = [
        row
        for row in report["layers"]
        if str(row.get("canonical_layer", ""))
        in {"shrink_conv.layers.0.double_conv.0", "shrink_conv.layers.0.double_conv.2"}
    ]
    lines = [
        "# Production explicit Q/DQ boundary verdict",
        "",
        f"- status: `{report['status']}`",
        f"- Q/DQ topology hash: `{report['qdq_topology_hash']}`",
        f"- weighted INT8 layers: {report['weighted_int8_layer_count']}",
        f"- graph-level raw weighted-output Q/DQ: {report['raw_weighted_output_qdq_count']}",
        f"- explicit FP16 output before merge: {report['fp16_output_no_output_qdq_count']}",
        "",
        "Graph placement and the TensorRT effective fused boundary are reported separately. TensorRT fusion does not legalize a Q/DQ node that semantically precedes ReLU; production output Q/DQ must consume the post-ReLU tensor.",
        "",
        "## Shrink layers",
        "",
        "| layer | graph placement | engine effective boundary | passed |",
        "|---|---|---|---|",
    ]
    lines.extend(
        f"| `{row['canonical_layer']}` | `{row['qdq_placement']}` | `{row['engine_effective_boundary']}` | {str(bool(row['boundary_passed'])).lower()} |"
        for row in shrink
    )
    if report["issues"]:
        lines.extend(["", "## Issues", ""])
        lines.extend(f"- `{row['canonical_layer']}`: {', '.join(row['issues'])}" for row in report["issues"])
    (destination / "production_qdq_boundary_verdict.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
