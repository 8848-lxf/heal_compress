"""Bounded CoBEVT compatibility search over the shared legal-width engines."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

from ..canonicalization import canonicalize_legal_width_candidate
from ..encoding.legal_width_genotype import LegalWidthGenotype
from .legal_width_greedy import run_six_budget_greedy
from .legal_width_joint_ga import run_legal_width_stage1_seeds


def select_smoke_band(reachable_counts: Mapping[float, int]) -> float:
    eligible = [
        (float(target), int(count))
        for target, count in reachable_counts.items()
        if int(count) > 0
    ]
    if not eligible:
        raise RuntimeError("cobevt_no_reachable_smoke_budget")
    return min(
        eligible,
        key=lambda row: (-row[1], abs(row[0] - 0.20), row[0]),
    )[0]


def discover_reachable_budget_counts(
    *,
    context: Any,
    targets: list[float],
    tolerance: float,
    output_path: str | Path | None = None,
) -> dict[float, int]:
    space = context.search_space
    domain = space.legal_width_inventory.domains[0]
    group_ids = tuple(sorted(space.precision_action_space))
    proposals: dict[str, LegalWidthGenotype] = {}
    for width_index in range(len(domain.legal_keep_widths)):
        fp16 = {group_id: "FP16" for group_id in group_ids}
        base = LegalWidthGenotype(
            {domain.domain_id: width_index},
            fp16,
            {"seed_family": "reachable_all_fp16"},
        )
        proposals[base.genotype_hash] = base
        for group_id in group_ids:
            if "FP32" not in space.precision_action_space[group_id]:
                continue
            precision = dict(fp16)
            precision[group_id] = "FP32"
            candidate = LegalWidthGenotype(
                {domain.domain_id: width_index},
                precision,
                {
                    "seed_family": "reachable_single_fp32",
                    "seed_precision_group": group_id,
                },
            )
            proposals[candidate.genotype_hash] = candidate
    rows = []
    counts = {float(target): 0 for target in targets}
    for genotype in proposals.values():
        phenotype = canonicalize_legal_width_candidate(genotype, space)
        bops = float(context.bops_proxy.evaluate_breakdown(phenotype)["R_bops_vs_fp32"])
        matched = []
        for target in counts:
            if abs(bops - target) <= float(tolerance) + 1.0e-12:
                counts[target] += 1
                matched.append(target)
        rows.append(
            {
                "genotype_hash": genotype.genotype_hash,
                "width_vector_hash": genotype.width_vector_hash,
                "precision_hash": genotype.precision_hash,
                "R_BOPS": bops,
                "matched_targets": matched,
                "seed_family": genotype.meta.get("seed_family", ""),
            }
        )
    if output_path is not None:
        Path(output_path).write_text(
            json.dumps(
                {
                    "counts": dict(sorted(counts.items())),
                    "proposal_count": len(rows),
                    "rows": rows,
                    "tolerance": float(tolerance),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return counts


def run_bounded_cobevt_stage1(
    *,
    context: Any,
    proxy: Any,
    run_dir: str | Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    counts = {
        float(target): int(count)
        for target, count in dict(config.get("reachable_budget_counts", {})).items()
    }
    selected = select_smoke_band(counts)
    tolerance = float(dict(config.get("budget", {})).get("tolerance", 0.0075))
    destination = Path(run_dir)
    destination.mkdir(parents=True, exist_ok=True)
    greedy_config = {
        "targets": [selected],
        "primary_tolerance": min(tolerance, 0.005),
        "expanded_tolerance": tolerance,
        "frontier_size": int(dict(config.get("greedy", {})).get("frontier_size", 8)),
        "max_expansions": int(dict(config.get("greedy", {})).get("max_expansions", 4096)),
    }
    greedy = run_six_budget_greedy(
        context,
        proxy,
        destination / "greedy_stage1",
        greedy_config,
    )
    ga_config = dict(config.get("ga", {}))
    ga_config.update(
        {
            "target_bops_retention": selected,
            "bops_tolerance": tolerance,
        }
    )
    ga = run_legal_width_stage1_seeds(
        context=context,
        proxy=proxy,
        run_dir=destination / "ga_stage1",
        search_config=ga_config,
    )
    result = {
        "selected_budget": selected,
        "bops_tolerance": tolerance,
        "reachable_budget_counts": dict(sorted(counts.items())),
        "greedy": greedy,
        "ga": ga,
    }
    serializable = {
        "selected_budget": selected,
        "bops_tolerance": tolerance,
        "reachable_budget_counts": dict(sorted(counts.items())),
        "greedy_summary": {
            key: value
            for key, value in greedy.items()
            if key != "endpoint_records"
        },
        "ga_summary": {
            key: value
            for key, value in ga.items()
            if key
            not in {
                "archive",
                "records_by_phenotype_hash",
                "all_metric_rows",
                "generation_records",
            }
        },
    }
    (destination / "bounded_stage1_summary.json").write_text(
        json.dumps(serializable, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return result


class LidarCobevtSmokeSearch:
    def __init__(
        self,
        *,
        config: dict[str, Any],
        checkpoint: str | Path,
        output_root: str | Path,
        resume: str | Path | None = None,
    ) -> None:
        if resume is not None:
            raise ValueError("cobevt_smoke_resume_forbidden")
        family = str(dict(config.get("model", {})).get("family", "")).lower()
        if family != "lidar_cobevt":
            raise ValueError(f"cobevt_smoke_model_family_mismatch:{family}")
        self.config = dict(config)
        self.checkpoint = Path(checkpoint)
        self.output_root = Path(output_root)

    def run(self, **kwargs: Any) -> Mapping[str, Any]:
        context = kwargs.get("context")
        proxy = kwargs.get("proxy")
        run_dir = kwargs.get("run_dir")
        if bool(kwargs.get("stage2_only", False)):
            raise RuntimeError("cobevt_stage2_only_requires_completed_stage1_manifest")
        if bool(kwargs.get("baseline_only", False)):
            raise RuntimeError("cobevt_baseline_only_not_supported_by_smoke_runner")
        components = None
        if context is None or proxy is None:
            from ..model_families.lidar_cobevt.stage1_capability import (
                build_cobevt_stage1_components,
            )

            timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime(
                "%Y%m%d_%H%M%S"
            )
            run_dir = self.output_root / f"4090_lidar_cobevt_stage1_smoke_{timestamp}"
            runtime = dict(self.config.get("runtime", {}))
            model = dict(self.config.get("model", {}))
            completed = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[2],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            code_commit = completed.stdout.strip() if completed.returncode == 0 else ""
            components = build_cobevt_stage1_components(
                checkpoint=self.checkpoint,
                model_config=model["config"],
                heal_root=model.get(
                    "heal_root", "/home/lixingfeng/UniAD_examine/HEAL"
                ),
                device=f"cuda:{int(runtime.get('stage1_gpu', 7))}",
                run_dir=run_dir,
                fisher_calibration_batches=int(
                    dict(self.config.get("proxy", {})).get(
                        "fisher_calibration_batches", 1
                    )
                ),
                code_commit=code_commit,
            )
            context = components.context
            proxy = components.proxy
        if run_dir is None:
            raise RuntimeError("cobevt_stage1_run_dir_required")
        budget = dict(self.config.get("budget", {}))
        targets = [float(value) for value in budget.get("candidates", ())]
        tolerance = float(budget.get("tolerance", 0.0075))
        counts = discover_reachable_budget_counts(
            context=context,
            targets=targets,
            tolerance=tolerance,
            output_path=Path(run_dir) / "reachable_budget_probe.json",
        )
        effective_config = dict(self.config)
        effective_config["reachable_budget_counts"] = counts
        result = run_bounded_cobevt_stage1(
            context=context,
            proxy=proxy,
            run_dir=run_dir,
            config=effective_config,
        )
        return {
            "run_dir": str(Path(run_dir).resolve()),
            "model_family": "lidar_cobevt",
            "stage1_only": bool(kwargs.get("stage1_only", False)),
            "stage2_started": False,
            "selected_budget": result["selected_budget"],
            "reachable_budget_counts": counts,
            "joint_loss_scale": (
                components.joint_loss_scale if components is not None else None
            ),
            "greedy_feasible_budget_count": int(
                result["greedy"].get("feasible_budget_count", 0)
            ),
            "ga_archive_summary": dict(
                result["ga"].get("archive_summary", {})
            ),
        }


__all__ = [
    "LidarCobevtSmokeSearch",
    "discover_reachable_budget_counts",
    "run_bounded_cobevt_stage1",
    "select_smoke_band",
]
