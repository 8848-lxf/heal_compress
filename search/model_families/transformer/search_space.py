"""Generate a verified-only Transformer quantization search space."""

from __future__ import annotations

from typing import Any, Iterable, Mapping


GENE_ROLES = (
    "qk_projection_precision",
    "v_projection_precision",
    "output_projection_precision",
    "qk_operand_accumulator_profile",
    "softmax_precision",
    "av_operand_accumulator_profile",
    "layernorm_precision",
    "ffn1_precision",
    "ffn2_precision",
    "residual_precision",
)


def _accepted(row: Mapping[str, Any]) -> bool:
    return bool(row.get("alpha_contract_match", True)) and all(
        bool(row.get(key, False))
        for key in (
            "physical_graph_legal",
            "onnx_success",
            "engine_success",
            "requested_realized_match",
            "zero_skip",
            "fixed500_safe",
            "latency_benefit",
            "no_precision_conflict",
        )
    )


def build_profile_library(
    results_by_model: Mapping[str, Iterable[Mapping[str, Any]]],
) -> dict[str, Any]:
    library: dict[str, Any] = {}
    allowed_sets: dict[str, set[str]] = {}
    for model, source_rows in results_by_model.items():
        rows = [dict(row) for row in source_rows]
        allowed = [row for row in rows if _accepted(row)]
        experimental = [
            row for row in rows
            if not _accepted(row)
            and row.get("status") not in {"rejected", "unsupported", "precision_fallback"}
        ]
        rejected = [row for row in rows if row.get("status") == "rejected"]
        unsupported = [
            row for row in rows
            if row.get("status") in {"unsupported", "precision_fallback"}
        ]
        library[str(model)] = {
            "allowed": allowed,
            "experimental": experimental,
            "rejected": rejected,
            "unsupported": unsupported,
        }
        allowed_sets[str(model)] = {str(row["profile_id"]) for row in allowed}
    portable = sorted(set.intersection(*allowed_sets.values())) if allowed_sets else []
    model_specific = {
        model: sorted(values - set(portable)) for model, values in allowed_sets.items()
    }
    library["cross_model_profiles"] = {
        "portable": portable,
        "model_specific": model_specific,
    }
    level_a = sorted(
        {
            str(row.get("profile_id"))
            for rows in results_by_model.values()
            for row in rows
            if row.get("accumulator_evidence_level") == "A" and row.get("accumulator_searchable")
        }
    )
    unknown = sorted(
        {
            str(row.get("profile_id"))
            for rows in results_by_model.values()
            for row in rows
            if row.get("requested_accumulator") and row.get("accumulator_evidence_level") != "A"
        }
    )
    library["accumulator_profiles"] = {
        "level_a_allowed": level_a,
        "unknown_not_searchable": unknown,
    }
    library["latency"] = {
        "source": "isolated_full_engine",
        "subgraph_additive": False,
    }
    library["schema_version"] = "transformer-precision-profile-library-v1"
    return library


def build_search_space(profile_library: Mapping[str, Any]) -> dict[str, Any]:
    models = [name for name in ("cobevt", "v2xvit") if name in profile_library]
    profiles = {
        model: {
            str(row["profile_id"]): dict(row)
            for row in profile_library[model].get("allowed", ())
        }
        for model in models
    }
    genes = []
    role_to_field = {
        "qk_projection_precision": "qk_projection_precision",
        "v_projection_precision": "v_projection_precision",
        "output_projection_precision": "output_projection_precision",
        "qk_operand_accumulator_profile": "qk_operand_accumulator_profile",
        "softmax_precision": "softmax_precision",
        "av_operand_accumulator_profile": "av_operand_accumulator_profile",
        "layernorm_precision": "layernorm_precision",
        "ffn1_precision": "ffn1_precision",
        "ffn2_precision": "ffn2_precision",
        "residual_precision": "residual_precision",
    }
    for gene in GENE_ROLES:
        support = {}
        for model, rows in profiles.items():
            support[model] = sorted(
                {
                    str(row.get(role_to_field[gene], ""))
                    for row in rows.values()
                    if row.get(role_to_field[gene])
                }
            )
        genes.append(
            {
                "gene": gene,
                "legal_values": sorted({value for values in support.values() for value in values}),
                "model_support": support,
                "evidence": "allowed profile fixed500 plus isolated full-engine latency",
            }
        )
    verified_joint = {
        model: sorted(rows) for model, rows in profiles.items()
    }
    return {
        "schema_version": "transformer-quantization-search-space-v1",
        "genes": genes,
        "verified_joint": verified_joint,
        "forbidden_combinations": [
            "native_int8_qk unless a Level-A INT32 accumulator contract and fixed500 safety exist",
            "unknown accumulator profiles",
            "any Cartesian-product combination absent from verified_joint",
            "FP8/BF16 fallback profiles",
        ],
        "cartesian_product_automatically_opened": False,
        "latency_source": "isolated_full_engine",
        "subgraph_lut_additive": False,
    }


__all__ = ["GENE_ROLES", "build_profile_library", "build_search_space"]
