#!/usr/bin/env python3
"""Freeze Greedy005 attribution inputs and audit its physical/calibration artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch


CANDIDATE_HASH = "44551dcb6358b38447662e376ad1731d61862343d56da4c054103c784029547b"
CHECKPOINT_SHA256 = "890f7f4db7b92142c29789b4ee4649494004021eb521f94c345fbe439ca6e3ab"
STRUCTURE_HASH = "c045f7c1421948f651f83e4dce81ac7c68f0fecb515d792479b3ffe15695a082"


def read(path: Path) -> Any:
    if not path.is_file():
        raise RuntimeError(f"required_input_missing:{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _shape(state: Mapping[str, torch.Tensor], key: str) -> list[int]:
    if key not in state:
        raise RuntimeError(f"physical_state_key_missing:{key}")
    return [int(value) for value in state[key].shape]


def run(source: Path, output: Path) -> None:
    winner_path = source / "greedy/v2xvit_greedy_winner_config.json"
    candidate_dir = source / "structures/v2xvit_greedy005_stage2_final" / CANDIDATE_HASH
    physical_path = candidate_dir / "physical_candidate.json"
    state_path = candidate_dir / "physical_state_dict.pth"
    requested_path = candidate_dir / "requested_vs_realized.json"
    calibration_path = candidate_dir / "calibration_manifest.json"
    scales_path = candidate_dir / "calibration_scales.json"
    precision_path = candidate_dir / "canonical_precision_mapping.json"
    engine_acceptance_path = candidate_dir / "engine_build_acceptance.json"
    engine_path = candidate_dir / "candidate.plan"
    eval_request_path = source / "evaluation/v2xvit_greedy005_final_fixed50/evaluation_request.json"
    eval_result_path = source / "evaluation/v2xvit_greedy005_final_fixed50/evaluation.json"
    inventory_path = source / "inventory/v2xvit_module_inventory.json"

    winner = read(winner_path)
    physical = read(physical_path)
    requested = read(requested_path)
    calibration = read(calibration_path)
    scales = read(scales_path)
    precision = read(precision_path)
    engine = read(engine_acceptance_path)
    eval_request = read(eval_request_path)
    eval_result = read(eval_result_path)
    inventory = read(inventory_path)
    checkpoint = Path(inventory["checkpoint_path"]).resolve()
    config = Path(inventory["config_path"]).resolve()
    eval_manifest = Path(eval_request["eval_manifest_path"]).resolve()
    plugin = Path(eval_request["plugin_path"]).resolve()
    for path in (state_path, engine_path, checkpoint, config, eval_manifest, plugin):
        if not path.is_file():
            raise RuntimeError(f"provenance_file_missing:{path}")

    physical_report = physical["physical_report"]
    precision_profile = winner["phenotype"]["realized_precision_profile"]
    precision_profile_hash = stable_hash(precision_profile)
    provenance_checks = {
        "winner_hash_exact": winner.get("candidate_hash") == CANDIDATE_HASH,
        "physical_candidate_hash_exact": physical.get("candidate_hash") == CANDIDATE_HASH,
        "candidate_hash_replay_exact": bool(physical.get("candidate_hash_replay_exact")),
        "physical_structure_hash_exact": physical_report.get("structure_hash") == STRUCTURE_HASH,
        "requested_realized_exact": bool(requested.get("exact")),
        "requested_widths_equal_realized": requested.get("requested_widths") == requested.get("realized_widths"),
        "checkpoint_hash_exact": sha256(checkpoint) == CHECKPOINT_SHA256,
        "checkpoint_inventory_hash_exact": inventory.get("checkpoint_sha256") == CHECKPOINT_SHA256,
        "calibration_checkpoint_hash_exact": calibration.get("checkpoint_sha256") == CHECKPOINT_SHA256,
        "calibration_structure_hash_exact": calibration.get("physical_structure_hash") == STRUCTURE_HASH,
        "scale_manifest_hash_exact": scales.get("calibration_manifest_hash") == calibration.get("manifest_hash"),
        "engine_build_ok": engine.get("status") == "ok" and engine.get("build", {}).get("success") is True,
        "engine_hash_exact": sha256(engine_path) == engine.get("build", {}).get("engine_hash"),
        "precision_realization_passed": engine.get("precision_realization_validation", {}).get("passed") is True,
        "fixed50_status_ok": eval_result.get("status") == "ok",
        "fixed50_manifest_path_exact": str(eval_manifest) == str(eval_result.get("eval_manifest_path")),
        "fixed50_manifest_hash_exact": read(eval_manifest).get("manifest_hash") == eval_result.get("eval_manifest_hash"),
        "fixed50_frames_exact": int(eval_result.get("num_evaluated_frames", -1)) == 50,
        "fixed50_no_skips": int(eval_result.get("num_skipped_frames", -1)) == 0,
        "qk_precision_contract_present": all(
            value == "FP32" for key, value in winner["phenotype"]["metadata"]["constant_precision_group_profile"].items()
            if key.endswith("::qk_matmul")
        ),
    }
    complete = all(provenance_checks.values())
    provenance = {
        "schema_version": "v2xvit-greedy005-accuracy-input-provenance-v1",
        "complete": complete,
        "candidate_hash": CANDIDATE_HASH,
        "bops_retention": winner["metrics"]["R_bops"],
        "joint_taylor": winner["metrics"]["L_joint_weight_activation_taylor"],
        "source_root": str(source),
        "artifacts": {
            str(path): {"sha256": sha256(path), "size_bytes": path.stat().st_size}
            for path in (
                winner_path, physical_path, state_path, requested_path, calibration_path,
                scales_path, precision_path, engine_acceptance_path, engine_path,
                eval_request_path, eval_result_path, checkpoint, config, eval_manifest, plugin,
            )
        },
        "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
        "config": {"path": str(config), "sha256": sha256(config)},
        "fixed50_manifest": {
            "path": str(eval_manifest),
            "file_sha256": sha256(eval_manifest),
            "manifest_hash": read(eval_manifest).get("manifest_hash"),
            "protocol": eval_request.get("evaluation_protocol_version"),
            "fixed_k": eval_request.get("fixed_k"),
            "warmup_frames": eval_request.get("warmup_frames"),
            "evaluation_frames": eval_request.get("num_frames"),
        },
        "physical": {
            "state_dict": str(state_path),
            "structure_hash": physical_report.get("structure_hash"),
            "state_dict_shape_hash": physical_report.get("state_dict_shape_hash"),
            "requested_widths": requested.get("requested_widths"),
            "precision_profile_hash": precision_profile_hash,
        },
        "calibration": calibration,
        "engine": {
            "path": str(engine_path),
            "sha256": sha256(engine_path),
            "tensorrt_version": engine.get("build_environment_manifest", {}).get("TensorRT_version"),
            "cuda": engine.get("build_environment_manifest", {}).get("nvcc_version"),
            "requested_realized": engine.get("precision_realization_validation"),
        },
        "qk_softmax_contract": {
            "policy_version": physical["phenotype"].get("precision_policy_version"),
            "constant_precision_group_profile": winner["phenotype"]["metadata"]["constant_precision_group_profile"],
            "onnx_audit_path": str(candidate_dir / "qdq_attention_fp32_audit.json"),
            "trt_audit_path": str(candidate_dir / "trt_attention_fp32_audit.json"),
        },
        "checks": provenance_checks,
        "provenance_incomplete": not complete,
    }
    write(output / "reports/input_provenance.json", provenance)
    if not complete:
        raise RuntimeError(f"provenance_incomplete:{[key for key, value in provenance_checks.items() if not value]}")

    calibration_key_fields = {
        "physical_structure_hash": calibration.get("physical_structure_hash"),
        "precision_map_hash": calibration.get("precision_map_hash"),
        "calibration_manifest_hash": calibration.get("manifest_hash"),
        "checkpoint_hash": calibration.get("checkpoint_sha256"),
    }
    cache_audit = {
        "schema_version": "v2xvit-greedy005-calibration-cache-audit-v1",
        "calibration_manifest_path": str(calibration_path),
        "scales_path": str(scales_path),
        "calibration_key_fields": calibration_key_fields,
        "physical_structure_hash_present": bool(calibration_key_fields["physical_structure_hash"]),
        "physical_structure_hash_matches": calibration_key_fields["physical_structure_hash"] == STRUCTURE_HASH,
        "precision_map_hash_present": bool(calibration_key_fields["precision_map_hash"]),
        "derived_precision_profile_hash": precision_profile_hash,
        "checkpoint_hash_present": bool(calibration_key_fields["checkpoint_hash"]),
        "checkpoint_hash_matches": calibration_key_fields["checkpoint_hash"] == CHECKPOINT_SHA256,
        "calibration_manifest_hash_present": bool(calibration_key_fields["calibration_manifest_hash"]),
        "scales_bind_manifest": scales.get("calibration_manifest_hash") == calibration.get("manifest_hash"),
        "stale_cache_risk": not bool(calibration_key_fields["physical_structure_hash"]),
        "cache_key_complete": all(bool(value) for value in calibration_key_fields.values()),
        "finding": "precision_map_hash_missing_from_old_calibration_key" if not calibration_key_fields["precision_map_hash"] else "complete",
        "reuse_allowed_for_jmix_fresh": False,
    }
    write(output / "reports/calibration_cache_audit.json", cache_audit)

    state_payload = torch.load(state_path, map_location="cpu")
    state = state_payload.get("model", state_payload)
    widths = requested["realized_widths"]
    attention_rows = []
    static_issues = []
    operations_by_domain = {
        row["domain_id"]: row
        for row in physical_report["transformer_report"].get("operations", [])
        if row.get("domain_type") == "attention_dh"
    }
    domain_manifest = read(source / "inventory/adapter_validation_20260723T1223/domain_manifests/v2xvit_transformer_domains_diagnostic.json")
    for instance in domain_manifest["attention_instances"]:
        module = str(instance["module_path"])
        domain_id = f"attention_dh::{module}"
        heads = int(instance["heads"])
        realized_dh = int(widths[domain_id])
        layout = str(instance["qkv_layout"])
        row = {
            "domain_id": domain_id,
            "module_path": module,
            "family": instance["family"],
            "heads": heads,
            "realized_d_h": realized_dh,
            "expected_scale": 1.0 / math.sqrt(realized_dh),
            "qkv_layout": layout,
        }
        if layout == "fused_qkv":
            qkv_shape = _shape(state, f"{module}.to_qkv.weight")
            out_shape = _shape(state, f"{module}.to_out.0.weight")
            row.update({"qkv_weight_shape": qkv_shape, "out_weight_shape": out_shape})
            row["qkv_rows_exact"] = qkv_shape[0] == 3 * heads * realized_dh
            row["out_input_columns_exact"] = out_shape[1] == heads * realized_dh
        else:
            q_shapes = [_shape(state, path + ".weight") for path in instance["q_projection_paths"]]
            k_shapes = [_shape(state, path + ".weight") for path in instance["k_projection_paths"]]
            v_shapes = [_shape(state, path + ".weight") for path in instance["v_projection_paths"]]
            o_shapes = [_shape(state, path + ".weight") for path in instance["output_projection_paths"]]
            row.update({"q_shapes": q_shapes, "k_shapes": k_shapes, "v_shapes": v_shapes, "o_shapes": o_shapes})
            row["qkv_rows_exact"] = all(shape[0] == heads * realized_dh for shape in q_shapes + k_shapes + v_shapes)
            row["out_input_columns_exact"] = all(shape[1] == heads * realized_dh for shape in o_shapes)
        operation = operations_by_domain.get(domain_id)
        if operation:
            row["reported_scale"] = operation.get("scale_after")
            row["scale_exact"] = math.isclose(float(operation["scale_after"]), row["expected_scale"], rel_tol=0.0, abs_tol=1e-12)
            row["qk_retained_count"] = len(operation.get("qk_flattened_keep", []))
            row["vo_retained_count"] = len(operation.get("vo_flattened_keep", []))
            row["qk_count_exact"] = row["qk_retained_count"] == heads * realized_dh
            row["vo_count_exact"] = row["vo_retained_count"] == heads * realized_dh
            row["reshape_contract"] = operation.get("reshape_contract")
        else:
            row.update({"reported_scale": row["expected_scale"], "scale_exact": True, "qk_count_exact": True, "vo_count_exact": True, "unchanged_domain": True})
        if not all(bool(row.get(key)) for key in ("qkv_rows_exact", "out_input_columns_exact", "scale_exact", "qk_count_exact", "vo_count_exact")):
            static_issues.append(f"attention_contract_failed:{domain_id}")
        attention_rows.append(row)

    ffn_rows = []
    for instance in domain_manifest["ffn_instances"]:
        module = str(instance["module_path"])
        domain_id = f"ffn_hidden::{module}"
        width = int(widths[domain_id])
        first = f"{module}.net.0"
        second = f"{module}.net.3"
        w1 = _shape(state, first + ".weight")
        b1 = _shape(state, first + ".bias")
        w2 = _shape(state, second + ".weight")
        exact = w1[0] == width and b1[0] == width and w2[1] == width and w2[0] == 256
        ffn_rows.append({"domain_id": domain_id, "module_path": module, "realized_d_ff": width, "w1": w1, "b1": b1, "w2": w2, "coupling_exact": exact, "residual_d_model": w2[0]})
        if not exact:
            static_issues.append(f"ffn_contract_failed:{domain_id}")

    static = {
        "schema_version": "v2xvit-greedy005-static-structure-audit-v1",
        "passed": not static_issues,
        "candidate_hash": CANDIDATE_HASH,
        "structure_hash": physical_report["structure_hash"],
        "state_dict_shape_hash": physical_report["state_dict_shape_hash"],
        "cnn_domain_count": physical_report["cnn_domain_count"],
        "attention_instance_count": len(attention_rows),
        "ffn_instance_count": len(ffn_rows),
        "requested_realized_exact": requested["exact"],
        "mask_only": physical_report["mask_only"],
        "hidden_padding": physical_report["hidden_padding"],
        "state_dict_tensor_count": len(state),
        "strict_load_required_for_phase1": True,
        "attention": attention_rows,
        "ffn": ffn_rows,
        "cnn_widths": {key: value for key, value in widths.items() if not key.startswith(("attention_dh::", "ffn_hidden::"))},
        "issues": static_issues,
    }
    write(output / "reports/static_structure_audit.json", static)
    (output / "reports/static_structure_audit.md").write_text(
        "# Static structure audit\n\n"
        f"Passed: `{static['passed']}`. Candidate and physical hashes are exact; "
        f"{len(attention_rows)} Attention and {len(ffn_rows)} FFN instances were shape-audited. "
        f"Mask-only={static['mask_only']}, hidden-padding={static['hidden_padding']}.\n\n"
        f"Calibration key is structure-bound but `precision_map_hash_present={cache_audit['precision_map_hash_present']}`; "
        "JMIX-FRESH therefore must regenerate scales and may not reuse the prior scale artifact.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "ok", "provenance_complete": complete, "static_passed": static["passed"], "cache_key_complete": cache_audit["cache_key_complete"]}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.source_root.resolve(), args.output_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
