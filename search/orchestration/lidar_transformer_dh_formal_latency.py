"""Selected Phase-A and accepted Phase-B isolated full-engine latency."""

from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
from typing import Any, Mapping

from search.model_families.transformer.dh_alignment_audit import latency_beneficial
from search.orchestration.lidar_transformer_dh_phase_b import (
    PROFILES,
    _family_evidence,
)
from search.orchestration.lidar_transformer_dh_recovery import (
    ExperimentState,
    require_phase_a_certificate,
    require_phase_b_fixed500,
)
from search.orchestration.lidar_transformer_h800_latency import (
    TRT_ROOT,
    _compute_processes,
    _engine_counts,
    _gpu_telemetry,
    _real_inputs,
    _time_engine,
    audit_isolation,
)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def _nearest(values: list[int], target: int, divisor: int) -> int:
    accepted = [value for value in values if value % divisor == 0]
    if not accepted:
        raise RuntimeError(f"latency_aligned_control_unavailable:{target}:{divisor}")
    return min(accepted, key=lambda value: (abs(value - target), -value))


def _micro_best(output_root: Path, model: str, family: str, widths: list[int], d0: int) -> int:
    rows = _read(
        output_root / "microbenchmark" / model / family / "primitive_microbenchmark.json"
    )
    if not isinstance(rows, list):
        raise RuntimeError(f"microbenchmark_missing:{model}:{family}")
    qk = {
        int(row["d_h"]): float(row["p50_ms"])
        for row in rows
        if row.get("build") and row.get("primitive") == "qk"
    }
    if d0 not in qk:
        raise RuntimeError(f"microbenchmark_baseline_missing:{model}:{family}")
    candidates = [value for value in widths if value % 4 != 0 and value in qk]
    if not candidates:
        raise RuntimeError(f"microbenchmark_safe_non4_missing:{model}:{family}")
    return max(candidates, key=lambda value: (qk[d0] / qk[value], value))


def select_phase_a_latency(output_root: Path) -> list[dict[str, Any]]:
    require_phase_a_certificate(output_root)
    rows: list[dict[str, Any]] = []
    reasons: dict[str, Any] = {}
    for model in ("lidar_cobevt", "lidar_v2xvit"):
        families = _read(
            output_root / "inventory" / f"{model.removeprefix('lidar_')}_attention_families.json"
        )
        for family in families:
            family_id = str(family["family_id"])
            evidence = _family_evidence(output_root, model, family)
            widths = [int(value) for value in evidence["continuous_safe_widths"]]
            d0 = int(evidence["original_d_h"])
            conservative = min(
                (value for value in widths if value % 4 != 0),
                key=lambda value: abs(value - (d0 - 1)),
            )
            boundary = int(evidence["continuous_safe_lower_bound"])
            micro = _micro_best(output_root, model, family_id, widths, d0)
            role_widths = {
                "baseline": d0,
                "conservative_nonaligned": conservative,
                "continuous_safe_boundary": boundary,
                "microbenchmark_best_nonaligned": micro,
            }
            nonaligned = {value for value in role_widths.values() if value % 4 != 0}
            controls: dict[int, tuple[int, int]] = {
                value: (_nearest(widths, value, 4), _nearest(widths, value, 8))
                for value in nonaligned
            }
            for value, (control4, control8) in controls.items():
                role_widths[f"control4_for_dh{value}"] = control4
                role_widths[f"control8_for_dh{value}"] = control8
            by_width: dict[int, list[str]] = {}
            for role, value in role_widths.items():
                by_width.setdefault(value, []).append(role)
            for profile in PROFILES:
                for width, roles in sorted(by_width.items(), reverse=True):
                    directory = (
                        output_root
                        / "engines"
                        / model
                        / family_id
                        / f"dh_{width:03d}"
                        / profile
                    )
                    build = _read(directory / "baseline_result.json") or {}
                    fixed = _read(
                        directory / "evaluation" / "fixed500" / "evaluation_acceptance.json"
                    ) or {}
                    if (
                        build.get("status") != "ok"
                        or int(build.get("requested_realized_conflict_count", -1)) != 0
                        or fixed.get("status") != "ok"
                        or int(fixed.get("evaluated", -1)) != 500
                        or int(fixed.get("skipped", -1)) != 0
                    ):
                        raise RuntimeError(
                            f"phase_a_latency_selection_not_accepted:{model}:{family_id}:{profile}:{width}"
                        )
                    control4 = controls.get(width, (None, None))[0]
                    control8 = controls.get(width, (None, None))[1]
                    rows.append(
                        {
                            "model": model,
                            "attention_family": family_id,
                            "profile": profile,
                            "d_h": width,
                            "original_d_h": d0,
                            "selection_roles": roles,
                            "alignment_class": (
                                "multiple_of_8"
                                if width % 8 == 0
                                else "multiple_of_4"
                                if width % 4 == 0
                                else "non4"
                            ),
                            "control_4_d_h": control4,
                            "control_8_d_h": control8,
                            "engine_sha256": build["engine_sha256"],
                            "fixed500_mAP": fixed["mAP"],
                            "selection_accuracy_source": "same-profile fixed500",
                            "formal_latency_source": None,
                        }
                    )
            reasons[f"{model}:{family_id}"] = {
                "original_d_h": d0,
                "continuous_safe_widths": widths,
                "conservative_nonaligned": conservative,
                "continuous_safe_boundary": boundary,
                "microbenchmark_best_nonaligned": micro,
                "aligned_controls": {
                    str(value): {"multiple_of_4": pair[0], "multiple_of_8": pair[1]}
                    for value, pair in controls.items()
                },
                "microbenchmark_used_only_for_candidate_selection": True,
                "full_engine_predictor": False,
                "unit_latency_additive": False,
            }
    _write_csv(output_root / "phase_a_latency_selection.csv", rows)
    _write(output_root / "phase_a_latency_selection_reason.json", reasons)
    return rows


def _layer_signature(directory: Path) -> dict[str, Any]:
    path = directory / "engine_build" / "engine_layer_info.json"
    payload = _read(path)
    layers = payload.get("Layers", payload) if payload else []
    tactics = sorted(
        {
            str(row.get("TacticName", row.get("Tactic", "")))
            for row in layers
            if str(row.get("TacticName", row.get("Tactic", "")))
        }
    )
    cast_count = sum("cast" in str(row).lower() for row in layers)
    reformat_count = sum("reformat" in str(row).lower() for row in layers)
    return {
        "tactic_signature": _stable_hash(tactics),
        "tactics": tactics,
        "cast_count": cast_count,
        "reformat_count": reformat_count,
        "fusion_signature": _stable_hash(
            [str(row.get("Name", "")) for row in layers if "fused" in str(row).lower()]
        ),
    }


def _isolation_gate(
    output_root: Path, phase: str, physical_gpu: int, isolation_seconds: int
) -> dict[str, Any]:
    audit = audit_isolation(physical_gpu, isolation_seconds)
    path = output_root / "formal_latency_process_audit.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "phase": phase,
                    "event": "isolation_gate",
                    "physical_gpu": physical_gpu,
                    **audit,
                },
                sort_keys=True,
            )
            + "\n"
        )
    if not audit["isolated"]:
        raise RuntimeError(f"formal_latency_isolation_failed:{phase}:{physical_gpu}")
    return audit


def _time_group(
    *,
    output_root: Path,
    phase: str,
    model: str,
    profile: str,
    candidate_rows: list[dict[str, Any]],
    directory_for: Any,
    candidate_key: str,
    baseline_key: str,
    physical_gpu: int,
    device: Any,
    inputs: Mapping[str, Any],
    warmup: int,
    iterations: int,
    repeats: int,
) -> list[dict[str, Any]]:
    ordered = sorted(
        candidate_rows,
        key=lambda row: (row[candidate_key] != baseline_key, str(row[candidate_key])),
    )
    baseline = next(row for row in ordered if row[candidate_key] == baseline_key)
    sequence = [baseline, *[row for row in ordered if row is not baseline], baseline]
    timed: list[dict[str, Any]] = []
    process_path = output_root / "formal_latency_process_audit.jsonl"
    for replay_index, candidate in enumerate(sequence):
        directory = directory_for(candidate)
        engine = directory / "engine.plan"
        external_before = [
            row for row in _compute_processes(physical_gpu) if int(row["pid"]) != os.getpid()
        ]
        if external_before:
            raise RuntimeError(f"external_process_before_formal_latency:{external_before}")
        before = _gpu_telemetry(physical_gpu)
        aggregate, repetition_rows = _time_engine(
            engine,
            inputs,
            device,
            warmup=warmup,
            iterations=iterations,
            repeats=repeats,
        )
        after = _gpu_telemetry(physical_gpu)
        external_after = [
            row for row in _compute_processes(physical_gpu) if int(row["pid"]) != os.getpid()
        ]
        with process_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "phase": phase,
                        "event": "candidate_process_audit",
                        "model": model,
                        "profile": profile,
                        "candidate": candidate[candidate_key],
                        "external_before": external_before,
                        "external_after": external_after,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        if external_after:
            raise RuntimeError(f"external_process_during_formal_latency:{external_after}")
        repeat_p50 = [float(row["p50_ms"]) for row in repetition_rows]
        cv = statistics.pstdev(repeat_p50) / statistics.mean(repeat_p50)
        timed.append(
            {
                **candidate,
                "phase": phase,
                "baseline_replay": candidate[candidate_key] == baseline_key,
                "replay_index": replay_index,
                "engine_sha256": _sha256(engine),
                "gpu_physical_index": physical_gpu,
                "gpu_uuid": before.get("uuid", ""),
                "warmup": warmup,
                "iterations": iterations,
                "repeats": repeats,
                "repeat_p50_cv": cv,
                "device_resident": True,
                "cuda_event": True,
                "formal": True,
                "telemetry_before": before,
                "telemetry_after": after,
                "repetitions": repetition_rows,
                **aggregate,
                **_engine_counts(directory),
                **_layer_signature(directory),
            }
        )
    replays = [row for row in timed if row["baseline_replay"]]
    drift = abs(float(replays[-1]["p50_ms"]) / float(replays[0]["p50_ms"]) - 1.0)
    if drift > 0.03:
        raise RuntimeError(f"formal_latency_baseline_replay_drift:{drift}")
    first = replays[0]
    for row in timed:
        row["baseline_replay_p50_drift_ratio"] = drift
        row["speedup_vs_same_profile_baseline"] = float(first["p50_ms"]) / float(row["p50_ms"])
        row.update(
            latency_beneficial(
                baseline_p50_ms=float(first["p50_ms"]),
                candidate_p50_ms=float(row["p50_ms"]),
                baseline_repeat_cv=float(first["repeat_p50_cv"]),
                baseline_replay_drift=drift,
            )
        )
    return timed


def run_phase_a_formal_latency(
    *,
    output_root: Path,
    physical_gpu: int,
    plugin: Path,
    isolation_seconds: int = 300,
    warmup: int = 200,
    iterations: int = 2000,
    repeats: int = 5,
) -> list[dict[str, Any]]:
    selection = select_phase_a_latency(output_root)
    _isolation_gate(output_root, "phase_a_selected", physical_gpu, isolation_seconds)
    import torch
    from search.integration.runtime_environment import load_tensorrt_runtime, runtime_cuda_index_for_physical

    load_tensorrt_runtime(TRT_ROOT)
    ctypes.CDLL(str(plugin), mode=ctypes.RTLD_GLOBAL)
    runtime = runtime_cuda_index_for_physical(physical_gpu)
    torch.cuda.set_device(runtime)
    device = torch.device(f"cuda:{runtime}")
    state = ExperimentState(output_root)
    all_rows: list[dict[str, Any]] = []
    for model in ("lidar_cobevt", "lidar_v2xvit"):
        inputs = _real_inputs(model, device)
        groups = sorted(
            {
                (str(row["attention_family"]), str(row["profile"]))
                for row in selection
                if row["model"] == model
            }
        )
        for family, profile in groups:
            candidates = [
                row
                for row in selection
                if row["model"] == model
                and row["attention_family"] == family
                and row["profile"] == profile
            ]
            d0 = int(candidates[0]["original_d_h"])
            with state.attempt(
                phase="phase_a_formal_latency",
                candidate_id=f"{model}-{family}-{profile}",
                artifact_output_path=output_root / "formal-latency" / "phase_a",
            ):
                rows = _time_group(
                    output_root=output_root,
                    phase="phase_a_selected",
                    model=model,
                    profile=profile,
                    candidate_rows=candidates,
                    directory_for=lambda row: output_root
                    / "engines"
                    / model
                    / family
                    / f"dh_{int(row['d_h']):03d}"
                    / profile,
                    candidate_key="d_h",
                    baseline_key=d0,
                    physical_gpu=physical_gpu,
                    device=device,
                    inputs=inputs,
                    warmup=warmup,
                    iterations=iterations,
                    repeats=repeats,
                )
            by_width = {
                int(row["d_h"]): row
                for row in rows
                if not (row["baseline_replay"] and int(row["replay_index"]) == len(rows) - 1)
            }
            for row in rows:
                if row.get("alignment_class") != "non4":
                    continue
                for divisor, key in ((4, "control_4_d_h"), (8, "control_8_d_h")):
                    control = by_width.get(int(row[key])) if row.get(key) is not None else None
                    if control:
                        comparison = latency_beneficial(
                            baseline_p50_ms=float(control["p50_ms"]),
                            candidate_p50_ms=float(row["p50_ms"]),
                            baseline_repeat_cv=float(control["repeat_p50_cv"]),
                            baseline_replay_drift=float(row["baseline_replay_p50_drift_ratio"]),
                        )
                        row[f"speedup_vs_control{divisor}"] = float(control["p50_ms"]) / float(row["p50_ms"])
                        row[f"beneficial_vs_control{divisor}"] = comparison["latency_beneficial"]
            all_rows.extend(rows)
        del inputs
        torch.cuda.empty_cache()
    for model in ("lidar_cobevt", "lidar_v2xvit"):
        _write_csv(
            output_root / f"formal_latency_phase_a_{model.removeprefix('lidar_')}.csv",
            [row for row in all_rows if row["model"] == model],
        )
    _write_csv(
        output_root / "formal_latency_baseline_replay.csv",
        [row for row in all_rows if row["baseline_replay"]],
    )
    _write(output_root / "formal-latency" / "phase_a" / "formal_latency.json", all_rows)
    state.update_phase(
        "phase_a_formal_latency",
        status="complete",
        unique_candidates=len(
            {(row["model"], row["attention_family"], row["profile"], row["d_h"]) for row in all_rows}
        ),
    )
    return all_rows


def run_phase_b_formal_latency(
    *,
    output_root: Path,
    physical_gpu: int,
    plugin: Path,
    isolation_seconds: int = 300,
    warmup: int = 200,
    iterations: int = 2000,
    repeats: int = 5,
) -> list[dict[str, Any]]:
    require_phase_b_fixed500(output_root)
    _isolation_gate(output_root, "phase_b_joint", physical_gpu, isolation_seconds)
    import torch
    from search.integration.runtime_environment import load_tensorrt_runtime, runtime_cuda_index_for_physical

    load_tensorrt_runtime(TRT_ROOT)
    ctypes.CDLL(str(plugin), mode=ctypes.RTLD_GLOBAL)
    runtime = runtime_cuda_index_for_physical(physical_gpu)
    torch.cuda.set_device(runtime)
    device = torch.device(f"cuda:{runtime}")
    state = ExperimentState(output_root)
    all_rows: list[dict[str, Any]] = []
    accuracy_rows: list[dict[str, Any]] = []
    with (output_root / "phase_b_joint_fixed500.csv").open(newline="", encoding="utf-8") as handle:
        accuracy_rows = list(csv.DictReader(handle))
    for model in ("lidar_cobevt", "lidar_v2xvit"):
        result = _read(output_root / "reports" / f"{model}_phase_b_result.json")
        candidates_by_id = {row["candidate_id"]: row for row in result["results"]}
        inputs = _real_inputs(model, device)
        for profile in PROFILES:
            accepted_ids = {
                row["candidate_id"]
                for row in accuracy_rows
                if row["model"] == model
                and row["profile"] == profile
                and row["accuracy_status"] == "SAFE"
                and str(row["diagnostic"]).lower() not in {"true", "1"}
            }
            accepted_ids.add("B0_BASELINE")
            candidates = [
                {
                    "model": model,
                    "profile": profile,
                    "candidate_id": candidate_id,
                    "joint_id": candidates_by_id[candidate_id]["joint_id"],
                    "targets": candidates_by_id[candidate_id]["targets"],
                    "diagnostic": candidates_by_id[candidate_id]["diagnostic"],
                    "alignment_control_id": (
                        "B1_CONSERVATIVE_ALIGNED"
                        if candidate_id in {"B2_CONSERVATIVE_NONALIGNED", "B3_MODERATE_NONALIGNED"}
                        else "B4_BOUNDARY_ALIGNED"
                        if candidate_id == "B5_BOUNDARY_NONALIGNED"
                        else None
                    ),
                }
                for candidate_id in candidates_by_id
                if candidate_id in accepted_ids
            ]
            if not any(row["candidate_id"] == "B0_BASELINE" for row in candidates):
                raise RuntimeError(f"phase_b_latency_baseline_missing:{model}:{profile}")
            with state.attempt(
                phase="phase_b_formal_latency",
                candidate_id=f"{model}-{profile}",
                artifact_output_path=output_root / "formal-latency" / "phase_b",
            ):
                rows = _time_group(
                    output_root=output_root,
                    phase="phase_b_joint",
                    model=model,
                    profile=profile,
                    candidate_rows=candidates,
                    directory_for=lambda row: output_root
                    / "engines"
                    / model
                    / "joint"
                    / str(row["joint_id"])
                    / profile,
                    candidate_key="candidate_id",
                    baseline_key="B0_BASELINE",
                    physical_gpu=physical_gpu,
                    device=device,
                    inputs=inputs,
                    warmup=warmup,
                    iterations=iterations,
                    repeats=repeats,
                )
            by_id = {
                str(row["candidate_id"]): row
                for row in rows
                if not (row["baseline_replay"] and int(row["replay_index"]) == len(rows) - 1)
            }
            for row in rows:
                control_id = row.get("alignment_control_id")
                control = by_id.get(str(control_id)) if control_id else None
                if control:
                    comparison = latency_beneficial(
                        baseline_p50_ms=float(control["p50_ms"]),
                        candidate_p50_ms=float(row["p50_ms"]),
                        baseline_repeat_cv=float(control["repeat_p50_cv"]),
                        baseline_replay_drift=float(row["baseline_replay_p50_drift_ratio"]),
                    )
                    row["speedup_vs_aligned_control"] = float(control["p50_ms"]) / float(row["p50_ms"])
                    row["beneficial_vs_aligned_control"] = comparison["latency_beneficial"]
            all_rows.extend(rows)
        del inputs
        torch.cuda.empty_cache()
    for model in ("lidar_cobevt", "lidar_v2xvit"):
        _write_csv(
            output_root / f"formal_latency_phase_b_{model.removeprefix('lidar_')}.csv",
            [row for row in all_rows if row["model"] == model],
        )
    with (output_root / "formal_latency_baseline_replay.csv").open(
        "a", newline="", encoding="utf-8"
    ) as handle:
        # Keep the canonical complete replay table deterministic below.
        pass
    phase_a = _read(output_root / "formal-latency" / "phase_a" / "formal_latency.json") or []
    _write_csv(
        output_root / "formal_latency_baseline_replay.csv",
        [row for row in (*phase_a, *all_rows) if row["baseline_replay"]],
    )
    _write(output_root / "formal-latency" / "phase_b" / "formal_latency.json", all_rows)
    state.update_phase(
        "phase_b_formal_latency",
        status="complete",
        unique_candidates=len(
            {(row["model"], row["profile"], row["candidate_id"]) for row in all_rows}
        ),
    )
    return all_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--phase", choices=("phase-a", "phase-b"), required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--isolation-seconds", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args(argv)
    root = Path(args.output_root).resolve()
    if args.phase == "phase-a" and args.select_only:
        rows = select_phase_a_latency(root)
    elif args.phase == "phase-a":
        rows = run_phase_a_formal_latency(
            output_root=root,
            physical_gpu=args.physical_gpu,
            plugin=Path(args.plugin).resolve(),
            isolation_seconds=args.isolation_seconds,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
    else:
        rows = run_phase_b_formal_latency(
            output_root=root,
            physical_gpu=args.physical_gpu,
            plugin=Path(args.plugin).resolve(),
            isolation_seconds=args.isolation_seconds,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
    print(json.dumps({"rows": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
