"""Unified strict formal-search adapter for Pyramid, DiscoNet and F-Cooper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..unified.config import PROJECT_ROOT, ResolvedSearchConfig


_MODEL_IDS = {
    "lidar_pyramid": "pyramid",
    "heal_lidar_disco": "disco",
    "heal_lidar_fcooper": "fcooper",
}


def _resolve(
    value: str | Path,
    *,
    label: str,
    file: bool = True,
) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    valid = path.is_file() if file else path.is_dir()
    if not valid:
        kind = "file" if file else "directory"
        raise RuntimeError(f"cnn_formal_required_{kind}_missing:{label}:{path}")
    return path


class CNNFormalSearch:
    """Run the audited integrated Greedy gate and strict V3 GA pipeline."""

    def __init__(
        self,
        *,
        config: ResolvedSearchConfig,
        output_root: str | Path,
    ) -> None:
        if config.family.family_id not in _MODEL_IDS:
            raise ValueError(
                f"cnn_formal_unsupported_family:{config.family.family_id}"
            )
        self.config = config
        self.payload = config.payload
        self.output_root = Path(output_root).expanduser().resolve()

    def run(self) -> dict[str, Any]:
        from scripts.run_cnn_formal_ga_gen5 import run as run_formal

        model = dict(self.payload.get("model", {}) or {})
        runtime = dict(self.payload.get("runtime", {}) or {})
        proxy = dict(self.payload.get("proxy", {}) or {})
        search = dict(self.payload.get("search", {}) or {})
        stage2 = dict(self.payload.get("stage2", {}) or {})
        full = dict(self.payload.get("full_validation", {}) or {})
        output = dict(self.payload.get("output", {}) or {})
        baselines = dict(self.payload.get("baselines", {}) or {})

        family_id = self.config.family.family_id
        model_id = _MODEL_IDS[family_id]
        experiment = str(output.get("experiment_name", f"{model_id}_formal_ga"))
        run_dir = self.output_root / experiment
        baseline_value = baselines.get("strict_fp32_engine")
        baseline = (
            _resolve(baseline_value, label="strict_fp32_engine")
            if baseline_value
            else None
        )
        if model_id != "pyramid" and baseline is None:
            raise RuntimeError(f"cnn_formal_baseline_engine_required:{model_id}")

        targets = tuple(
            sorted(
                {float(value) for value in search.get("bops_targets", ())},
                reverse=True,
            )
        )
        args = argparse.Namespace(
            model=model_id,
            output_root=run_dir,
            checkpoint=_resolve(model["checkpoint"], label="checkpoint"),
            model_config=_resolve(model["config"], label="model_config"),
            calibration_manifest=_resolve(
                proxy["quant_calibration_npz_manifest"],
                label="calibration_manifest",
            ),
            heal_root=_resolve(runtime["heal_root"], label="heal_root", file=False),
            baseline_engine=baseline,
            physical_gpu=int(runtime.get("physical_gpu", 0)),
            generations=int(search.get("generations_per_round", 10)),
            seed=int(search.get("seed", 0)),
            taylor_samples=int(proxy.get("fisher_calibration_batches", 8)),
            targets=",".join(str(value) for value in targets),
            resume=False,
            greedy_only=str(search.get("method", "ga")) == "greedy",
            plugin=_resolve(runtime["plugin_path"], label="scatter_plugin"),
            tensorrt_root=_resolve(
                runtime["tensorrt_root"], label="tensorrt_root", file=False
            ),
            activation_taylor=bool(proxy.get("include_activation_taylor", False)),
            objective_calibration=str(
                proxy.get("objective_calibration", "raw")
            ),
            objective_fit_batches=int(proxy.get("objective_fit_batches", 4)),
            objective_validation_batches=int(
                proxy.get("objective_validation_batches", 4)
            ),
            objective_fit_candidates_per_mode=int(
                proxy.get("objective_fit_candidates_per_mode", 4)
            ),
            objective_validation_candidates_per_mode=int(
                proxy.get("objective_validation_candidates_per_mode", 2)
            ),
            stage2_latency_rounds=int(stage2.get("latency_rounds", 3)),
            full_validation_frames=int(full.get("num_frames", 1789)),
            full_validation_warmup_frames=int(full.get("warmup_frames", 200)),
            full_validation_latency_rounds=int(full.get("latency_rounds", 3)),
        )
        return_code = int(run_formal(args))
        if return_code != 0:
            raise RuntimeError(
                f"cnn_formal_search_failed:{model_id}:returncode={return_code}:"
                f"output={run_dir}"
            )

        if args.greedy_only:
            report_path = run_dir / "reports/greedy_anchor_deployment_validation.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            anchors = dict(report["anchors"])
            primary_label = min(
                anchors,
                key=lambda value: abs(float(value) / 100.0 - 0.10),
            )
            return {
                "status": "ok",
                "family_id": family_id,
                "method": "greedy",
                "run_dir": str(run_dir),
                "anchors": anchors,
                "best": anchors[primary_label],
            }

        report_path = run_dir / "reports/formal_ga_results.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("failures"):
            raise RuntimeError(f"cnn_formal_search_reported_failures:{report['failures']}")
        results = dict(report["results"])
        primary_label = min(
            results,
            key=lambda value: abs(float(value) / 100.0 - 0.10),
        )
        return {
            "status": "ok",
            "family_id": family_id,
            "formal_protocol": "strict_stage12_v3",
            "run_dir": str(run_dir),
            "targets": results,
            "best": results[primary_label]["final_winner"],
        }


__all__ = ["CNNFormalSearch"]
