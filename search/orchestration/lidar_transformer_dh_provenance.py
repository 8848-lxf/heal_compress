"""Write reproducible branch and artifact lineage records for the d_h run."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
from typing import Any


FORMAL_BRANCH = "feature/heal-unified-search-h800"
FORMAL_START = "139d2c351889405c66fef05e420995663c91f08e"
PHASE_A_IMPLEMENTATION = "d0964c39cd350239e9e831b737bed551e17ad313"
MICRO_FIX = "6cfde0a126b940be1aa8558725b60fc7bd49cb4c"


def _git(workdir: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=workdir, text=True).strip()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _mtime(path: Path) -> str | None:
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat() if path.exists() else None


def write_provenance(run_root: Path) -> dict[str, Any]:
    workdir = Path(__file__).resolve().parents[2]
    provenance = run_root / "provenance"
    current_branch = _git(workdir, "branch", "--show-current")
    current_head = _git(workdir, "rev-parse", "HEAD")
    experiment_remote = _git(
        workdir, "rev-parse", "origin/feature/h800-transformer-dh-alignment-sweep"
    )
    formal_remote = _git(workdir, "rev-parse", f"origin/{FORMAL_BRANCH}")
    branch_manifest = {
        "experiment_branch": current_branch,
        "starting_commit": MICRO_FIX,
        "current_head": current_head,
        "experiment_remote_head": experiment_remote,
        "local_remote_equal": current_head == experiment_remote,
        "working_tree_tracked_status": _git(workdir, "status", "--short", "--untracked-files=no"),
        "base_quantization_audit_commit": "6db0e483d61e6ae845a9eeb83632efe83616627d",
        "phase_a_implementation_commit": PHASE_A_IMPLEMENTATION,
        "microbenchmark_fix_commit": MICRO_FIX,
    }
    _write(provenance / "git_branch_manifest.json", branch_manifest)

    structure_paths = sorted(run_root.glob("structures/lidar_*/*/dh_*/structure_result.json"))
    engine_paths = sorted(run_root.glob("engines/lidar_*/*/dh_*/*/baseline_result.json"))
    fixed_paths = sorted(
        run_root.glob("engines/lidar_*/*/dh_*/*/evaluation/fixed500/evaluation_acceptance.json")
    )
    timestamps = [path.stat().st_mtime for path in (*structure_paths, *engine_paths, *fixed_paths)]
    phase_a = {
        "run_manifest": str(run_root / "run_manifest.json"),
        "run_manifest_generated_at": _mtime(run_root / "run_manifest.json"),
        "run_manifest_base_head": json.loads(
            (run_root / "run_manifest.json").read_text(encoding="utf-8")
        ).get("base_head"),
        "phase_a_implementation_commit": PHASE_A_IMPLEMENTATION,
        "artifact_commit_per_attempt_recorded": False,
        "artifact_code_lineage": (
            "Phase-A implementation entered git at d0964c39; individual workers did not "
            "persist code commits, so no later report or microbenchmark commit is assigned "
            "to every structure/engine/fixed500 artifact."
        ),
        "structures": len(structure_paths),
        "engines": len(engine_paths),
        "fixed500": len(fixed_paths),
        "artifact_time_min": datetime.fromtimestamp(min(timestamps)).astimezone().isoformat(),
        "artifact_time_max": datetime.fromtimestamp(max(timestamps)).astimezone().isoformat(),
        "root_report_generated_at": _mtime(run_root / "root_conclusion.md"),
        "attempt_lineage_complete": False,
    }
    _write(provenance / "phase_a_artifact_lineage.json", phase_a)

    micro_paths = sorted(run_root.glob("microbenchmark/lidar_*/*/primitive_microbenchmark.json"))
    micro = {
        "fix_commit": MICRO_FIX,
        "final_family_results": [
            {"path": str(path), "mtime": _mtime(path)} for path in micro_paths
        ],
        "final_refresh_min": min((_mtime(path) for path in micro_paths), default=None),
        "final_refresh_max": max((_mtime(path) for path in micro_paths), default=None),
        "root_report_generated_at": _mtime(run_root / "root_conclusion.md"),
        "root_report_predates_final_microbenchmark_refresh": (
            bool(micro_paths)
            and (run_root / "root_conclusion.md").stat().st_mtime
            < max(path.stat().st_mtime for path in micro_paths)
        ),
        "acceptance": str(run_root / "microbenchmark" / "final_microbenchmark_acceptance.json"),
    }
    _write(provenance / "microbenchmark_lineage.json", micro)

    formal = {
        "branch": FORMAL_BRANCH,
        "starting_remote_head": FORMAL_START,
        "current_remote_head": formal_remote,
        "unchanged": formal_remote == FORMAL_START,
        "checked_out_for_modification": False,
        "merged": False,
        "cherry_picked": False,
        "pushed": False,
    }
    _write(provenance / "untouched_formal_branch.json", formal)
    return {
        "branch": branch_manifest,
        "phase_a": phase_a,
        "microbenchmark": micro,
        "formal_branch": formal,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args(argv)
    result = write_provenance(Path(args.run_root).resolve())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
