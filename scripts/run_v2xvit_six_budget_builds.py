#!/usr/bin/env python3
"""Serially build B0 and six-budget S32/JMIX deployment-closed engines."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/lixingfeng/anaconda3/envs/modelopt/bin/python")
TARGETS = ("030", "025", "020", "015", "010", "005")


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _environment(physical_gpu: int) -> dict[str, str]:
    prefix = "/home/lixingfeng/anaconda3/envs/modelopt"
    env = dict(os.environ)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(int(physical_gpu)),
            "CONDA_PREFIX": prefix,
            "CUDA_HOME": prefix,
            "CUDA_PATH": prefix,
            "CUDACXX": f"{prefix}/bin/nvcc",
            "CMAKE_CUDA_COMPILER": f"{prefix}/bin/nvcc",
            "TORCH_CUDA_ARCH_LIST": "8.9",
            "PATH": f"{prefix}/bin:{env.get('PATH', '')}",
            "PYTHONPATH": ":".join(
                (
                    "/home/lixingfeng/UniAD_examine/HEAL",
                    "/home/lixingfeng/UniAD_examine/HEAL/prune_model/Model-Optimizer-0.29.0",
                    str(REPO),
                    env.get("PYTHONPATH", ""),
                )
            ),
            "PYTHONUNBUFFERED": "1",
        }
    )
    return env


def _run_build(
    *,
    genotype: Path,
    profile_id: str,
    override: str,
    functional_contract: str,
    output_dir: Path,
    search_space: Path,
    physical_gpu: int,
) -> dict[str, Any]:
    if output_dir.exists():
        if any(output_dir.iterdir()):
            result_path = output_dir / "candidate_result.json"
            engine_path = output_dir / "candidate.plan"
            if result_path.is_file() and engine_path.is_file():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if result.get("status") == "ok" and result.get(
                    "requested_realized_exact"
                ):
                    return {
                        "profile_id": profile_id,
                        "command": "resumed_current_run_verified_artifact",
                        "returncode": 0,
                        "status": "ok",
                        "requested_realized_exact": True,
                        "engine_sha256": result.get("engine_sha256"),
                        "candidate_dir": str(output_dir),
                        "success": True,
                        "resumed_current_run": True,
                    }
            raise RuntimeError(f"build_output_incomplete_or_stale:{output_dir}")
        output_dir.rmdir()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(PYTHON),
        str(REPO / "scripts/run_v2xvit_deployment_closed_profile.py"),
        "--genotype-json",
        str(genotype),
        "--profile-id",
        profile_id,
        "--override-precision",
        override,
        "--functional-contract",
        functional_contract,
        "--search-space",
        str(search_space),
        "--output-dir",
        str(output_dir),
        "--physical-gpu",
        str(int(physical_gpu)),
        "--fixed-k",
        "27904",
    ]
    completed = subprocess.run(
        command,
        cwd=REPO,
        env=_environment(physical_gpu),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "orchestration.log").write_text(completed.stdout, encoding="utf-8")
    result_path = output_dir / "candidate_result.json"
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {}
    return {
        "profile_id": profile_id,
        "command": command,
        "returncode": completed.returncode,
        "status": result.get("status", "failed_no_result"),
        "requested_realized_exact": result.get("requested_realized_exact", False),
        "engine_sha256": result.get("engine_sha256"),
        "candidate_dir": str(output_dir),
        "success": completed.returncode == 0 and result.get("status") == "ok",
    }


def _compact_b0(root: Path) -> Path:
    winner = json.loads((root / "budgets/030/winner.json").read_text(encoding="utf-8"))
    genotype = dict(winner["genotype"])
    search_space = json.loads((root / "search_space.json").read_text(encoding="utf-8"))
    genotype["pruning_width_genes"] = {
        str(domain["domain_id"]): int(domain["original_width"])
        for domain in search_space["pruning_domains"]
    }
    genotype["precision_genes"] = {
        key: "FP32" for key in genotype["precision_genes"]
    }
    genotype["meta"] = {"created_by": "six_budget_b0", "repair_count": 0}
    path = root / "provenance/b0_genotype.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != genotype:
            raise RuntimeError("b0_genotype_provenance_conflict")
    else:
        _write(path, genotype)
    return path


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.output_root.resolve()
    search_space = root / "search_space.json"
    if not search_space.is_file():
        raise RuntimeError(f"search_space_missing:{search_space}")
    audits = []
    budget_selections = []
    b0_genotype = _compact_b0(root)
    audits.append(
        _run_build(
            genotype=b0_genotype,
            profile_id="B0_P32",
            override="all_fp32",
            functional_contract="P32",
            output_dir=root / "engines/B0",
            search_space=search_space,
            physical_gpu=args.physical_gpus[0],
        )
    )
    if not audits[-1]["success"]:
        raise RuntimeError("six_budget_b0_build_failed")
    for index, budget in enumerate(TARGETS):
        gpu = args.physical_gpus[index % len(args.physical_gpus)]
        winner = root / "budgets" / budget / "winner.json"
        s32 = _run_build(
            genotype=winner,
            profile_id=f"B{budget}_S32",
            override="all_fp32",
            functional_contract="P32",
            output_dir=root / "budgets" / budget / "S32",
            search_space=search_space,
            physical_gpu=gpu,
        )
        audits.append(s32)
        if not s32["success"]:
            _write(root / "budgets" / budget / "build_audit.json", {"attempts": [s32], "deployment_invalid": True})
            budget_selections.append(
                {
                    "budget": budget,
                    "selected_s32_dir": s32["candidate_dir"],
                    "selected_jmix_dir": "",
                    "selected_success": False,
                    "fallback_used": False,
                }
            )
            continue
        jmix = _run_build(
            genotype=winner,
            profile_id=f"B{budget}_JMIX_FRESH",
            override="candidate",
            functional_contract="F3",
            output_dir=root / "budgets" / budget / "JMIX-FRESH",
            search_space=search_space,
            physical_gpu=gpu,
        )
        audits.append(jmix)
        attempts = [s32, jmix]
        selected_s32 = s32
        selected_jmix = jmix
        fallback_used = False
        if not jmix["success"]:
            selection = json.loads(
                (root / "budgets" / budget / "stage2_candidates.json").read_text(
                    encoding="utf-8"
                )
            )
            primary_hash = str(
                json.loads(winner.read_text(encoding="utf-8"))["candidate_hash"]
            )
            fallbacks = [
                row
                for row in selection.get("candidates", [])
                if str(row.get("candidate_hash")) != primary_hash
            ][:4]
            for fallback_index, row in enumerate(fallbacks, start=1):
                fallback_root = root / "budgets" / budget / f"fallback_{fallback_index:02d}"
                genotype_path = fallback_root / "genotype.json"
                _write(genotype_path, row["genotype"])
                fallback_s32 = _run_build(
                    genotype=genotype_path,
                    profile_id=f"B{budget}_FALLBACK{fallback_index}_S32",
                    override="all_fp32",
                    functional_contract="P32",
                    output_dir=fallback_root / "S32",
                    search_space=search_space,
                    physical_gpu=gpu,
                )
                attempts.append(fallback_s32)
                audits.append(fallback_s32)
                if not fallback_s32["success"]:
                    continue
                fallback_jmix = _run_build(
                    genotype=genotype_path,
                    profile_id=f"B{budget}_FALLBACK{fallback_index}_JMIX_FRESH",
                    override="candidate",
                    functional_contract="F3",
                    output_dir=fallback_root / "JMIX-FRESH",
                    search_space=search_space,
                    physical_gpu=gpu,
                )
                attempts.append(fallback_jmix)
                audits.append(fallback_jmix)
                if fallback_jmix["success"]:
                    selected_s32 = fallback_s32
                    selected_jmix = fallback_jmix
                    fallback_used = True
                    break
        audit_path = root / "budgets" / budget / "build_audit.json"
        audit_payload = {
                "attempts": attempts,
                "deployment_invalid": not selected_jmix["success"],
                "fallback_used": fallback_used,
                "precision_fallback_allowed": False,
                "physical_gpu": gpu,
                "selected_s32_dir": selected_s32["candidate_dir"],
                "selected_jmix_dir": selected_jmix["candidate_dir"],
            }
        if audit_path.exists():
            existing = json.loads(audit_path.read_text(encoding="utf-8"))
            if existing.get("deployment_invalid") or (
                existing.get("selected_s32_dir") != selected_s32["candidate_dir"]
                or existing.get("selected_jmix_dir") != selected_jmix["candidate_dir"]
            ):
                raise RuntimeError(f"build_audit_resume_conflict:{budget}")
        else:
            _write(audit_path, audit_payload)
        budget_selections.append(
            {
                "budget": budget,
                "selected_s32_dir": selected_s32["candidate_dir"],
                "selected_jmix_dir": selected_jmix["candidate_dir"],
                "selected_success": bool(
                    selected_s32["success"] and selected_jmix["success"]
                ),
                "fallback_used": fallback_used,
            }
        )
    result = {
        "schema_version": "v2xvit-six-budget-serial-build-v1",
        "engine_builds_serial": True,
        "physical_gpus": args.physical_gpus,
        "attempts": audits,
        "budget_selections": budget_selections,
        "all_success": bool(audits[0]["success"])
        and all(row["selected_success"] for row in budget_selections),
        "system_nvcc_allowed": False,
        "modelopt_python_sha256": _sha256(PYTHON),
    }
    _write(root / "reports/six_budget_build_matrix.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--physical-gpus", type=int, nargs="+", default=(0, 1, 2, 3, 4, 5))
    result = run(parser.parse_args())
    return 0 if result["all_success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
