#!/usr/bin/env python3
"""Run immutable fixed50 PyTorch controls for the accuracy attribution."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]


def write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise RuntimeError(f"refusing_to_overwrite:{path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    root = args.source_root.resolve()
    out = args.output_root.resolve()
    controls = json.loads((out / "controls/control_manifest.json").read_text(encoding="utf-8"))["controls"]
    config = "/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_v2xvit/config.yaml"
    checkpoint = "/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_v2xvit/net_epoch_bestval_at27.pth"
    # Resolve the immutable manifest from the inherited evaluation request.
    inherited_request = json.loads((root / "evaluation/v2xvit_greedy005_final_fixed50/evaluation_request.json").read_text(encoding="utf-8"))
    manifest = inherited_request["eval_manifest_path"]
    heal_root = inherited_request["heal_root"]
    ranking = root / "rankings/v2xvit_fixed_rankings.json"
    search_manifest = root / "greedy/v2xvit_greedy_search_manifest.json"
    results: dict[str, Any] = {}
    names = args.controls or ["B0", "S32", "S16", "Full-winner", "CNN-only", "Attention-only", "FFN-only", "CNN+Attention", "CNN+FFN", "Attention+FFN"]
    for name in names:
        destination = out / "pytorch" / name.replace("+", "_").replace("-", "_")
        destination.mkdir(parents=True, exist_ok=False)
        request: dict[str, Any] = {
            "model_name": "lidar_v2xvit",
            "config_path": config,
            "checkpoint_path": checkpoint,
            "eval_manifest_path": manifest,
            "heal_root": heal_root,
            "repo_root": str(REPO),
            "device": "cuda:0",
            "physical_gpu": 5,
            "warmup_frames": 200,
            "num_frames": 50,
            "dataloader_num_workers": 8,
            "torch_num_threads": 4,
            "conda_env": "univ2x-opt",
            "output_path": str(destination / "evaluation.json"),
            "diagnostic_control": True,
            "control_name": name,
            "ranking_path": str(ranking),
            "search_manifest_path": str(search_manifest),
            "autocast_dtype": "float16" if name == "S16" else "none",
        }
        if name != "B0":
            control_dir = Path(controls[name]["control_dir"])
            control = json.loads((control_dir / "control.json").read_text(encoding="utf-8"))
            request["physical_control_json"] = str(control_dir / "control.json")
            request["physical_state_dict_path"] = str(control_dir / "physical_state_dict.pth")
            request["physical_structure_hash"] = control["physical_report"]["structure_hash"]
            request["precision_mode"] = control["precision_mode"]
        request_path = destination / "request.json"
        write(request_path, request)
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "5"
        command = [sys.executable, "-m", "search.model_family.pytorch_evaluation_worker", "--request", str(request_path)]
        completed = subprocess.run(command, cwd=REPO, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        (destination / "worker.log").write_text(completed.stdout or "", encoding="utf-8")
        result_path = destination / "evaluation.json"
        result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.is_file() else {"status": "missing_output", "worker_returncode": completed.returncode}
        result["worker_returncode"] = completed.returncode
        result["diagnostic_control"] = True
        result["control_name"] = name
        write(destination / "control_result.json", result)
        results[name] = result
        if result.get("status") != "ok":
            raise RuntimeError(f"pytorch_control_failed:{name}:{result.get('failure_reason','')}")
    summary_path = out / "reports/phase1_pytorch_controls.json"
    if summary_path.is_file():
        prior = json.loads(summary_path.read_text(encoding="utf-8"))
        prior.setdefault("controls", {}).update(results)
        summary_path.unlink()
        write(summary_path, prior)
    else:
        write(summary_path, {"schema_version": "v2xvit-greedy005-phase1-pytorch-controls-v1", "diagnostic_control": True, "controls": results})
    print(json.dumps({"status": "ok", "controls": list(results), "diagnostic_control": True}, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--controls", nargs="*")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
