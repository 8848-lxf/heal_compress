"""Two-stage structured pruning plus mixed-precision mock/search runner."""

from __future__ import annotations

import csv
import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from ..cache.proxy_cache import ProxyCache
from ..cache.real_eval_cache import RealEvalCache
from ..candidate import CandidateGenotype, CandidatePhenotype
from ..canonicalization import SearchSpaceSpec, canonicalize_candidate
from ..ga.engine import GAConfig, GeneticSearchEngine
from ..hashing import candidate_hash
from ..proxy.objective import ProxyObjective
from ..stage1.proxy_evaluator import Stage1ProxyEvaluator
from ..stage1.topk_selector import ProxyCandidateRecord, TopKConfig, select_stage1_topk
from ..stage2.objective import Stage2ObjectiveConfig, compute_stage2_score


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True), encoding="utf-8")


class _ConstantProxy:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def evaluate(self, _phenotype: CandidatePhenotype) -> float:
        return self.value


class TwoStageSearchRunner:
    """Small runner retained for dry-run and unit-test compatibility.

    The real lidar_pyramid CLI path uses
    :class:`search.orchestration.lidar_pyramid_search.LidarPyramidTwoStageSearch`.
    This runner intentionally uses lightweight proxy values so existing unit
    tests do not need a HEAL checkpoint or Fisher cache.
    """

    def __init__(
        self,
        *,
        output_root: str | Path,
        pruning_unit_ids: list[str],
        precision_layer_ids: list[str],
        protected_pruning_unit_ids: set[str] | None = None,
        stage2_evaluator: Callable[[CandidatePhenotype], dict[str, Any]] | None = None,
        random_seed: int = 42,
        trace_snapshot_hash: str = "dryrun-trace",
        calibration_manifest_hash: str = "dryrun-calibration",
        onnx_export_config_hash: str = "dryrun-onnx",
        tensorrt_version: str = "",
        gpu_compute_capability: str = "",
        builder_flags: dict[str, Any] | None = None,
    ) -> None:
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.space = SearchSpaceSpec(
            pruning_unit_ids=pruning_unit_ids,
            precision_layer_ids=precision_layer_ids,
            protected_pruning_unit_ids=protected_pruning_unit_ids or set(),
            trace_snapshot_hash=trace_snapshot_hash,
            calibration_manifest_hash=calibration_manifest_hash,
            onnx_export_config_hash=onnx_export_config_hash,
            tensorrt_version=tensorrt_version,
            gpu_compute_capability=gpu_compute_capability,
            builder_flags=builder_flags or {},
        )
        self.stage2_evaluator = stage2_evaluator
        self.random_seed = int(random_seed)
        self.archive_root = self.output_root / "archives"
        self.proxy_cache = ProxyCache(self.archive_root / "proxy_archive.jsonl")
        self.real_cache = RealEvalCache(self.archive_root / "real_eval_archive.jsonl")

    def _new_run_dir(self) -> Path:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        candidate = self.output_root / f"two_stage_joint_search_{stamp}"
        suffix = 0
        while candidate.exists():
            suffix += 1
            candidate = self.output_root / f"two_stage_joint_search_{stamp}_{suffix:02d}"
        candidate.mkdir(parents=True)
        return candidate

    def run(
        self,
        *,
        outer_rounds: int,
        population_size: int,
        generations: int,
        topk_real: int,
        dry_run: bool = False,
        stage1_only: bool = False,
        baseline: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        run_dir = self._new_run_dir()
        (run_dir / "baseline").mkdir(parents=True, exist_ok=True)
        (run_dir / "archives").mkdir(parents=True, exist_ok=True)
        _write_json(
            run_dir / "run_manifest.json",
            {
                "dry_run": dry_run,
                "outer_rounds": outer_rounds,
                "population_size": population_size,
                "generations": generations,
                "topk_real": topk_real,
                "search_space": {
                    "pruning_unit_count": len(self.space.pruning_unit_ids),
                    "precision_layer_count": len(self.space.precision_layer_ids),
                },
            },
        )
        if not dry_run and not stage1_only and self.stage2_evaluator is None:
            raise RuntimeError("real_stage2_evaluator_required")
        if not dry_run and not stage1_only and baseline is None:
            raise RuntimeError("real_baseline_required")
        baseline_metrics = baseline or {"mAP": 0.0, "forward_mean_ms": 0.0, "forward_p50_ms": 0.0, "forward_p90_ms": 0.0}
        objective = ProxyObjective(
            fisher=_ConstantProxy(0.0),
            sqnr=_ConstantProxy(0.0),
            size=_ConstantProxy(1.0),
            bops=_ConstantProxy(1.0),
        )
        proxy = Stage1ProxyEvaluator(self.space, objective=objective, cache=self.proxy_cache)
        global_rows: list[dict[str, Any]] = []
        previous_elite: list[CandidateGenotype] = []
        previous_best: CandidateGenotype | None = None
        for round_index in range(int(outer_rounds)):
            round_dir = run_dir / f"round_{round_index:03d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            ga = GeneticSearchEngine(
                self.space,
                GAConfig(
                    population_size=int(population_size),
                    num_generations=int(generations),
                    random_seed=self.random_seed + round_index,
                ),
            )

            def evaluate_genotype(genotype: CandidateGenotype, generation: int) -> dict[str, Any]:
                return proxy.evaluate(genotype, generation=generation, outer_round=round_index)

            scored = ga.run(evaluate_genotype, previous_elite=previous_elite, previous_best=previous_best)
            records = []
            for genotype, score, metrics in scored:
                phenotype = canonicalize_candidate(genotype, self.space)
                key = candidate_hash(phenotype, self.space)
                records.append(ProxyCandidateRecord(key, genotype, phenotype, float(score), metrics))
            unique_records: dict[str, ProxyCandidateRecord] = {}
            for record in records:
                unique_records.setdefault(record.candidate_hash, record)
            records = sorted(unique_records.values(), key=lambda row: row.F1)
            self._write_stage1_scores(round_dir / "stage1_scores.csv", records)
            selected = select_stage1_topk(
                records,
                real_eval_hashes=set(),
                archive_genotypes=previous_elite,
                config=TopKConfig(topk_real=topk_real),
            )
            _write_json(
                round_dir / "stage1_topk.json",
                [
                    {
                        "role": item.role,
                        "candidate_hash": item.record.candidate_hash,
                        "F1": item.record.F1,
                        "phenotype": item.record.phenotype.to_dict(),
                    }
                    for item in selected
                ],
            )
            if dry_run or stage1_only:
                continue
            for item in selected:
                record = item.record

                def evaluate_stage2() -> dict[str, Any]:
                    assert self.stage2_evaluator is not None
                    stage2_raw = self.stage2_evaluator(record.phenotype)
                    scored_stage2 = compute_stage2_score(stage2_raw, baseline=baseline_metrics, config=Stage2ObjectiveConfig())
                    return {**stage2_raw, **scored_stage2, "candidate_hash": record.candidate_hash}

                result = self.real_cache.get_or_evaluate(record.candidate_hash, evaluate_stage2)
                candidate_dir = round_dir / "stage2" / record.candidate_hash
                _write_json(candidate_dir / "candidate_genotype.json", record.genotype.to_dict())
                _write_json(candidate_dir / "candidate_phenotype.json", record.phenotype.to_dict())
                _write_json(candidate_dir / "stage2_score.json", result)
                global_rows.append({"candidate_hash": record.candidate_hash, "F1": record.F1, **result})
            previous_elite = [row.genotype for row in records[: max(1, min(5, len(records)))]]
            previous_best = previous_elite[0] if previous_elite else None
            _write_json(round_dir / "round_summary.json", {"selected": len(selected), "best_F1": records[0].F1 if records else None})
        self._write_global_summary(run_dir, global_rows)
        self._snapshot_archives(run_dir)
        return {"run_dir": str(run_dir), "dry_run": dry_run, "evaluated": len(global_rows)}

    @staticmethod
    def _write_stage1_scores(path: Path, records: list[ProxyCandidateRecord]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["candidate_hash", "F1", "pruned_units", "int8_layers"])
            writer.writeheader()
            for record in records:
                writer.writerow(
                    {
                        "candidate_hash": record.candidate_hash,
                        "F1": record.F1,
                        "pruned_units": len(record.phenotype.pruned_unit_ids),
                        "int8_layers": sum(v == "INT8" for v in record.phenotype.realized_precision_profile.values()),
                    }
                )

    @staticmethod
    def _write_global_summary(run_dir: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            _write_json(run_dir / "global_summary.json", {"evaluated": 0, "best": None})
            (run_dir / "global_summary.csv").write_text("candidate_hash,F1,F2,status\n", encoding="utf-8")
            return
        best = min(rows, key=lambda row: float(row.get("F2", float("inf"))))
        _write_json(run_dir / "global_summary.json", {"evaluated": len(rows), "best": best})
        _write_json(run_dir / "global_best_candidate.json", best)
        with (run_dir / "global_summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
            writer.writeheader()
            writer.writerows(rows)

    def _snapshot_archives(self, run_dir: Path) -> None:
        destination = run_dir / "archives"
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("proxy_archive.jsonl", "artifact_index.jsonl", "real_eval_archive.jsonl"):
            source = self.archive_root / name
            target = destination / name
            if source.exists():
                shutil.copy2(source, target)
            elif not target.exists():
                target.write_text("", encoding="utf-8")
