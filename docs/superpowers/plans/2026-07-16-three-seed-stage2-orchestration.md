# Three-Seed GA And Stage-2 Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run three independent legal-width GA populations per budget, merge one global per-generation candidate ranking, and execute auditable real Stage-2 with zero/one/two-to-five candidate semantics.

**Architecture:** Extend the existing legal-width GA and persistent worker pool. Split deployment smoke from 500-frame evaluation so a sole candidate can bypass the comparison; key all reusable work by deployment plus protocol identity; keep one shared strict-FP32 reference and queue every generation winner for full validation.

**Tech Stack:** Python 3.10, PyTorch, TensorRT 10.9, CUDA 11.8, existing strongly typed explicit-QDQ builder, pytest, JSONL caches, YAML.

## Global Constraints

- Complete `2026-07-16-linear-joint-greedy-search.md` first; its APIs and pushed commit are prerequisites.
- Work only on `feature/heal-compress-h800-sync-4090`; preserve H800 ancestor `b862b3d8ad061bd12580776226c75f564918298d`.
- Use `univ2x-opt` for controller/GA/tests and `modelopt` for TensorRT/plugin/engine worker execution.
- Keep legal-width decoding, fixed prune-only Taylor ranking, physical replay, QDQ semantics, calibration recipe, merge contracts, and strongly typed precision realization unchanged.
- GA uses three seeds, population 64, offspring 64, and 20 generations for each of six budgets.
- Merge the three seed rankings per generation and deploy at most one globally unique Top-5, not Top-5 per seed.
- BOPS primary tolerance is `0.005`; expanded `0.0075` is available only after primary supply and deployment backfill are exhausted.
- There is no AP/mAP hard gate. Complete frame counts, zero skips, finite outputs, BOPS, structure, QDQ, merge, and precision identity remain mandatory.
- One candidate skips the 500-frame comparison and goes directly to the full-validation winner queue.
- Use one shared strict-FP32 reference result for all Stage-2 workers.
- Use all admissible RTX 4090 devices with memory occupancy at or below 50%, ordered by lowest sampled load. Never terminate foreign processes.
- Use `apply_patch` for manual edits and `git diff --check` before every commit.

## File Ownership Map

- `search/stage2/objective.py`: real mAP/p50 score semantics and failure direction.
- `search/cache/deployment_registry.py`: deployment, evaluation-protocol, and lineage identity.
- `search/stage2/lidar_pyramid_real_evaluator.py`: shared reference, build-smoke, and evaluation-only engine operations.
- `search/stage2/candidate_worker.py`: explicit worker protocol dispatch.
- `search/orchestration/stage2_process_pool.py`: protocol-specific task cache and work-conserving dispatch.
- `search/orchestration/gpu_scheduler.py`: 50%-memory eligibility and low-load ordering.
- `search/orchestration/generation_stage2.py`: BOPS backfill and zero/one/two-to-five candidate decisions.
- `search/orchestration/legal_width_joint_ga.py`: independent per-seed GA records.
- `search/orchestration/legal_width_six_budget_ga.py`: per-generation seed merge and six-budget control flow.
- `search/orchestration/legal_width_stage2.py`: greedy endpoint and generation-winner full validation.
- `search/orchestration/formal_latency.py`: isolated one-GPU formal replay.
- `search/reporting/pareto_frontier.py`: official full-val/formal-latency fronts only.

---

### Task 1: Direct Real mAP/Latency Stage-2 Score

**Files:**
- Modify: `search/stage2/objective.py`
- Modify: `search/stage2/candidate_artifacts.py`
- Modify: `search/stage2/lidar_pyramid_real_evaluator.py`
- Modify: `search/stage2/candidate_worker.py`
- Modify: `tests/test_two_stage_joint_search.py`
- Modify: `tests/test_search_stage2_candidate_artifacts.py`
- Modify: `tests/test_search_final_contract.py`
- Create: `tests/test_stage2_map_latency_score.py`

**Interfaces:**
- Produces: formal `Stage2ObjectiveConfig(score_mode="map_minus_latency_ratio", latency_weight=0.10, latency_metric="forward_p50_ms")`.
- Formal `compute_stage2_score` returns larger-is-better `F2` and `selection_direction="maximize"`.

- [ ] **Step 1: Write failing exchange-rate and no-AP-gate tests**

```python
def test_map_minus_latency_ratio_encodes_ten_percent_for_point_zero_one() -> None:
    config = Stage2ObjectiveConfig(
        score_mode="map_minus_latency_ratio",
        latency_weight=0.10,
        latency_metric="forward_p50_ms",
    )
    baseline = {"mAP": 0.73, "forward_p50_ms": 10.0}
    accurate = compute_stage2_score(
        {"status": "ok", "mAP": 0.72, "forward_p50_ms": 10.0},
        baseline=baseline, config=config,
    )
    faster = compute_stage2_score(
        {"status": "ok", "mAP": 0.71, "forward_p50_ms": 9.0},
        baseline=baseline, config=config,
    )
    assert accurate["F2"] == pytest.approx(faster["F2"])
    assert accurate["selection_direction"] == "maximize"


def test_formal_stage2_has_no_map_or_ap07_hard_gate() -> None:
    result = compute_stage2_score(
        {"status": "ok", "mAP": 0.01, "AP07": 0.0, "forward_p50_ms": 1.0},
        baseline={"mAP": 0.73, "forward_p50_ms": 10.0},
        config=Stage2ObjectiveConfig(score_mode="map_minus_latency_ratio", latency_weight=0.10,
                                     latency_metric="forward_p50_ms"),
    )
    assert result["status"] == "ok"
    assert math.isfinite(result["F2"])
```

Update the baseline test to expect one call to `strict_fp32`, with both mAP and
p50 from that result. Add a second test that supplies a signed shared reference
to the evaluator and makes `evaluate_original_baseline` raise if called; the
shared result must be returned unchanged.

- [ ] **Step 2: Run RED**

```bash
conda run -n univ2x-opt pytest -q \
  tests/test_stage2_map_latency_score.py \
  tests/test_two_stage_joint_search.py \
  tests/test_search_stage2_candidate_artifacts.py
```

Expected: FAIL because the formal score mode does not exist and the evaluator still combines FP32 accuracy with FP16 latency.

- [ ] **Step 3: Implement the formal score branch**

Add fields without breaking explicitly requested legacy mode:

```python
@dataclass(frozen=True)
class Stage2ObjectiveConfig:
    score_mode: str = "legacy_normalized_loss"
    latency_weight: float = 0.10
    latency_metric: str = "forward_p50_ms"
    # retain legacy fields for historical artifact replay
```

Formal branch:

```python
if policy.score_mode == "map_minus_latency_ratio":
    if not all(math.isfinite(value) for value in (cand_map, cand_latency, base_latency)):
        return {"F2": -float("inf"), "status": "nonfinite_stage2_metric",
                "selection_direction": "maximize"}
    latency_ratio = cand_latency / max(base_latency, policy.epsilon)
    return {
        "F2": cand_map - float(policy.latency_weight) * latency_ratio,
        "status": status,
        "selection_direction": "maximize",
        "R_latency_real": latency_ratio,
        "mAP_real": cand_map,
        "accuracy_admission_passed": True,
        "failure_reasons": [],
    }
```

Failure statuses return `-inf` in this mode. Required evaluated/skipped frame
checks remain runtime protocol checks; remove `min_map`, `min_ap07`, and
`max_map_drop` from the formal config, not from legacy replay support.
Add `failure_stage2_score(config)` and replace evaluator/worker hard-coded
`float("inf")` assignments so formal maximize mode consistently uses `-inf`
while explicit legacy minimize mode remains compatible.

- [ ] **Step 4: Use one strict-FP32 reference**

Replace `_stage2_reference_baseline` with one call:

```python
reference = self.evaluate_original_baseline("strict_fp32", full_validation=False)
combined = {
    "status": "ok",
    "mAP": float(reference["mAP"]),
    self.objective_config.latency_metric: float(reference[self.objective_config.latency_metric]),
    "accuracy_reference": "original_strict_fp32",
    "latency_reference": "original_strict_fp32",
    "strict_fp32": reference,
}
```

Add `shared_stage2_reference: Mapping[str, Any] | None` to the evaluator. A
provided reference must declare `reference_precision=strict_fp32`, contain
finite mAP and p50, and carry engine/evaluation hashes. Return that signed
payload without building another reference. Formal candidate workers must
receive this shared payload; per-worker reference construction is forbidden.

Update `stage2_objective_report.json` to write the formula
`mAP - 0.10 * (p50/strict_fp32_p50)` and remove `tau_AP` for formal rows.

- [ ] **Step 5: Run GREEN and commit**

```bash
conda run -n univ2x-opt pytest -q \
  tests/test_stage2_map_latency_score.py \
  tests/test_two_stage_joint_search.py \
  tests/test_search_stage2_candidate_artifacts.py \
  tests/test_search_final_contract.py
git diff --check
git add search/stage2/objective.py search/stage2/candidate_artifacts.py \
  search/stage2/lidar_pyramid_real_evaluator.py search/stage2/candidate_worker.py \
  tests/test_stage2_map_latency_score.py tests/test_two_stage_joint_search.py \
  tests/test_search_stage2_candidate_artifacts.py tests/test_search_final_contract.py
git commit -m "feat: score stage2 with direct map latency tradeoff"
```

### Task 2: Protocol-Specific Deployment Cache And Worker Tasks

**Files:**
- Create: `search/cache/deployment_registry.py`
- Modify: `search/cache/__init__.py`
- Modify: `search/orchestration/stage2_process_pool.py`
- Modify: `search/orchestration/lidar_pyramid_search.py`
- Modify: `search/stage2/candidate_worker.py`
- Modify: `search/stage2/lidar_pyramid_real_evaluator.py`
- Create: `tests/test_deployment_registry.py`
- Modify: `tests/test_search_stage2_process_pool.py`
- Modify: `tests/test_search_stage2_candidate_worker.py`

**Interfaces:**
- Produces: `deployment_identity(payload)`, `evaluation_identity(deployment_id, protocol, manifest_hash, config_hash)`, and `DeploymentRegistry`.
- Worker task protocols: `reference_strict_fp32`, `build_smoke`, `evaluate_500`, `full_validation`, and `formal_latency`.

- [ ] **Step 1: Write failing identity and pool-cache tests**

```python
def test_same_deployment_reuses_engine_but_not_different_evaluation_protocol(tmp_path: Path) -> None:
    registry = DeploymentRegistry(tmp_path / "registry.jsonl")
    deployment = deployment_identity({"physical_hash": "p", "precision_hash": "q",
                                      "calibration_signature": "c", "build_signature": "b"})
    eval_500 = evaluation_identity(deployment, "evaluate_500", "m500", "cfg")
    full = evaluation_identity(deployment, "full_validation", "m1789", "cfg")
    assert eval_500 != full
```

Extend the process-pool fake worker test with two tasks sharing
`candidate_hash` but different `task_cache_key`; assert neither is incorrectly
reused. A repeated identical `task_cache_key` must be reused.

- [ ] **Step 2: Run RED**

```bash
conda run -n univ2x-opt pytest -q \
  tests/test_deployment_registry.py \
  tests/test_search_stage2_process_pool.py \
  tests/test_search_stage2_candidate_worker.py
```

Expected: FAIL because cache identity is currently only `candidate_hash`.

- [ ] **Step 3: Implement the registry and strict identities**

```python
def deployment_identity(payload: Mapping[str, Any]) -> str:
    required = ("physical_hash", "precision_hash", "calibration_signature", "build_signature")
    missing = [key for key in required if not str(payload.get(key, ""))]
    if missing:
        raise ValueError(f"deployment_identity_missing:{','.join(missing)}")
    return canonical_json_hash({key: payload[key] for key in required})


def evaluation_identity(deployment_id: str, protocol: str,
                        manifest_hash: str, config_hash: str) -> str:
    return canonical_json_hash({"deployment_identity": deployment_id,
                                "protocol": protocol,
                                "manifest_hash": manifest_hash,
                                "config_hash": config_hash})
```

`DeploymentRegistry` appends JSONL records keyed by identity and keeps every
budget/seed/generation lineage reference rather than overwriting it.
The `calibration_signature` must bind physical hash, precision profile,
calibration manifest, calibration recipe, and QDQ topology. The
`build_signature` must bind code commit, base/typed ONNX hashes, canonical
mapping, plugin hash, TensorRT/CUDA versions, GPU architecture, builder flags,
and strongly typed mode.

- [ ] **Step 4: Make the process pool cache protocol-specific**

Replace pool cache lookup/store keys with mandatory `task_cache_key`. Keep
`candidate_hash` only as display lineage. Persist `task_protocol` and
`task_cache_key` in every result and pool manifest. Reject tasks missing either
field in formal mode.

- [ ] **Step 5: Split evaluator and worker protocols**

Extract the build/smoke/resource-audit portion of
`evaluate_candidate_two_level` into:

```python
def build_and_smoke_candidate(
    self, phenotype: CandidatePhenotype, *, output_dir: str | Path,
    candidate_hash: str, smoke_frames: int = 10,
    smoke_warmup_frames: int = 10,
) -> dict[str, Any]:
    """Build once, run smoke, validate BOPS and deployment, but do not run 500/full-val."""
```

Expose an evaluation-only method:

```python
def evaluate_existing_engine(
    self, engine_path: str | Path, *, output_dir: str | Path,
    deployment_metadata: Mapping[str, Any]
) -> dict[str, Any]:
    return {**self._evaluate_engine(str(engine_path), Path(output_dir)),
            **dict(deployment_metadata)}
```

Route worker tasks by explicit protocol. `build_smoke` calls the first method;
`evaluate_500` and `full_validation` call the second with protocol-specific
context. Do not rebuild an engine for evaluation-only tasks.

Before the multi-GPU candidate pool starts, launch one single-GPU modelopt
worker task with protocol `reference_strict_fp32`. Close that temporary pool,
validate and sign its engine/mAP/p50 result, then put the payload in every
candidate worker's initialization request as `shared_stage2_reference`.
Candidate workers must not run a reference task. The reference GPU may later
also join the candidate pool if it still passes occupancy selection.

- [ ] **Step 6: Run GREEN and commit**

```bash
conda run -n univ2x-opt pytest -q \
  tests/test_deployment_registry.py \
  tests/test_search_stage2_process_pool.py \
  tests/test_search_stage2_candidate_worker.py \
  tests/test_search_stage2_candidate_artifacts.py
git diff --check
git add search/cache/deployment_registry.py search/cache/__init__.py \
  search/orchestration/stage2_process_pool.py \
  search/orchestration/lidar_pyramid_search.py search/stage2/candidate_worker.py \
  search/stage2/lidar_pyramid_real_evaluator.py \
  tests/test_deployment_registry.py tests/test_search_stage2_process_pool.py \
  tests/test_search_stage2_candidate_worker.py
git commit -m "feat: split stage2 deployment and evaluation protocols"
```

### Task 3: Low-Occupancy Eight-GPU Scheduler

**Files:**
- Create: `search/orchestration/gpu_scheduler.py`
- Create: `tests/test_stage2_gpu_scheduler.py`
- Modify: `search/orchestration/lidar_pyramid_search.py`

**Interfaces:**
- Produces: `select_stage2_gpu_ids(gpu_rows, process_rows, max_memory_fraction=0.50) -> dict[str, Any]`.

- [ ] **Step 1: Write failing occupancy tests**

```python
def test_scheduler_excludes_over_half_memory_and_orders_low_load() -> None:
    report = select_stage2_gpu_ids(
        [
            {"index": 0, "uuid": "a", "memory_used_mib": 13000, "memory_total_mib": 24000, "utilization_gpu_pct": 0},
            {"index": 1, "uuid": "b", "memory_used_mib": 4000, "memory_total_mib": 24000, "utilization_gpu_pct": 30},
            {"index": 2, "uuid": "c", "memory_used_mib": 2000, "memory_total_mib": 24000, "utilization_gpu_pct": 5},
        ],
        [], max_memory_fraction=0.50,
    )
    assert report["selected_gpu_ids"] == [2, 1]
    assert report["excluded"][0]["reason"] == "memory_fraction_above_limit"
```

Also test that foreign processes are reported but not terminated, and an empty
selection returns `dispatch_allowed=false` rather than choosing an overloaded
GPU.

- [ ] **Step 2: Run RED**

Run: `conda run -n univ2x-opt pytest -q tests/test_stage2_gpu_scheduler.py`

- [ ] **Step 3: Implement scheduler using existing runtime queries**

Use `query_gpus()` and `query_compute_processes()` from
`search/integration/runtime_environment.py`. Calculate memory fraction from
numeric MiB fields. Sort eligible GPUs by utilization, memory fraction,
foreign-process count, then index. Return the full snapshot and reasons. The
runner writes `stage2_gpu_selection.json` before starting a pool and pauses
with pending tasks if no GPU is eligible.

- [ ] **Step 4: Run GREEN and commit**

```bash
conda run -n univ2x-opt pytest -q \
  tests/test_stage2_gpu_scheduler.py \
  tests/test_search_trt_runtime_provenance.py
git diff --check
git add search/orchestration/gpu_scheduler.py \
  search/orchestration/lidar_pyramid_search.py tests/test_stage2_gpu_scheduler.py
git commit -m "feat: schedule stage2 on low occupancy GPUs"
```

### Task 4: Zero/One/Two-To-Five Generation Semantics

**Files:**
- Modify: `search/orchestration/generation_stage2.py`
- Modify: `tests/test_search_generation_stage2.py`
- Create: `tests/test_generation_stage2_candidate_counts.py`

**Interfaces:**
- Produces: the complete keyword-only `run_generation_stage2` signature shown
  in Step 3, with separate build-smoke and evaluate-500 callbacks.

- [ ] **Step 1: Write failing candidate-count tests**

```python
def test_one_candidate_skips_500_and_becomes_winner(tmp_path: Path) -> None:
    eval_calls = []
    report = run_generation_stage2(
        ranked_records=[record("only", bops=0.20)], generation_index=0,
        output_dir=tmp_path, policy=BopsBandPolicy(target=0.20), topk=5,
        build_smoke_batch_fn=lambda rows: [ok_build("only")],
        evaluate_500_batch_fn=lambda rows: eval_calls.extend(rows) or [],
    )
    assert report["status"] == "single_candidate_direct_winner"
    assert report["winner"]["candidate_hash"] == "only"
    assert report["winner"]["evaluation_500_skipped"] is True
    assert eval_calls == []


def test_two_candidates_both_run_500_and_max_f2_wins(tmp_path: Path) -> None:
    report = run_generation_stage2(
        ranked_records=[record("a", 0.20), record("b", 0.20)],
        generation_index=0, output_dir=tmp_path,
        policy=BopsBandPolicy(target=0.20), topk=5,
        build_smoke_batch_fn=lambda rows: [ok_build(row.candidate_hash) for row in rows],
        evaluate_500_batch_fn=lambda rows: [ok_eval(row, f2=0.5 if row["candidate_hash"] == "a" else 0.6) for row in rows],
    )
    assert report["winner"]["candidate_hash"] == "b"


def test_zero_candidates_records_bops_funnel_without_stage2(tmp_path: Path) -> None:
    build_calls = []

    def fail_eval(_rows):
        raise AssertionError("zero candidates must not evaluate")

    report = run_generation_stage2(
        ranked_records=[record("low", 0.18), record("high", 0.22)],
        generation_index=0,
        output_dir=tmp_path,
        policy=BopsBandPolicy(target=0.20),
        topk=5,
        build_smoke_batch_fn=lambda rows: build_calls.extend(rows) or [],
        evaluate_500_batch_fn=fail_eval,
    )
    assert report["status"] == "no_bops_admissible_candidates"
    assert report["selected_count"] == 0
    assert report["bops_admission"]["nearest_misses"]
    assert build_calls == []
```

- [ ] **Step 2: Run RED**

```bash
conda run -n univ2x-opt pytest -q \
  tests/test_generation_stage2_candidate_counts.py \
  tests/test_search_generation_stage2.py
```

- [ ] **Step 3: Implement the two-phase generation runner**

```python
def run_generation_stage2(
    ranked_records: Sequence[Any], *, generation_index: int,
    output_dir: str | Path, policy: BopsBandPolicy,
    build_smoke_batch_fn: Callable[[list[Any]], list[dict[str, Any]]],
    evaluate_500_batch_fn: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
    topk: int = 5,
) -> dict[str, Any]:
    """BOPS-select, build/backfill, then apply 0/1/2-5 evaluation semantics."""
```

Try primary candidates first. If primary build/backfill yields zero, call the
expanded selector and try expanded-only candidates. Enforce unique physical
plus deployment hashes. A build is admissible only when both physical and
realized BOPS pass the currently active primary or expanded interval; record
proxy, physical, and realized values separately. A primary-proxy candidate
whose realized BOPS moves outside primary is not silently relabeled expanded
while another primary candidate remains. Build until five successes or supply
exhaustion. For two-to-five, require 500/500, zero skips, and finite formal-mode
F2; maximize F2 with candidate hash tie-break. Write top5, failures, Stage-2
CSV, winner, admission funnel, and count decision artifacts even when zero.

- [ ] **Step 4: Preserve the compatibility wrapper and run GREEN**

Keep `deploy_generation_with_backfill` for old tests/configs, implemented via
the new primitives where possible. Update old assertions only where the
approved semantics intentionally changed.

```bash
conda run -n univ2x-opt pytest -q \
  tests/test_generation_stage2_candidate_counts.py \
  tests/test_search_generation_stage2.py \
  tests/test_search_stage2_round_results.py
git diff --check
git add search/orchestration/generation_stage2.py \
  tests/test_generation_stage2_candidate_counts.py \
  tests/test_search_generation_stage2.py
git commit -m "feat: support variable per-generation stage2 supply"
```

### Task 5: Three-Seed Generation Merge And Six-Budget GA

**Files:**
- Modify: `search/orchestration/legal_width_joint_ga.py`
- Create: `search/orchestration/legal_width_six_budget_ga.py`
- Create: `search/configs/lidar_pyramid_4090_joint_six_budget_ga.yaml`
- Modify: `search/orchestration/lidar_pyramid_search.py`
- Modify: `tests/test_legal_width_ga_orchestration.py`
- Create: `tests/test_three_seed_generation_merge.py`

**Interfaces:**
- Produces: `run_six_budget_joint_ga(context: Any, proxy: Any, stage2_pool: Any, run_dir: str | Path, config: Mapping[str, Any]) -> dict[str, Any]`.
- Consumes: immutable `joint_loss_scale.json`, greedy/anchor seeds, persistent Stage-2 pool, and generation runner.

- [ ] **Step 1: Write failing merge tests**

Use three fake seed outputs for the same generation containing duplicate and
unique phenotype/deployment identities. Assert:

```python
merged = merge_seed_generation_records(seed_outputs, generation=3)
assert merged[0].metrics["J1"] >= merged[1].metrics["J1"]
assert len({row.phenotype.metadata["phenotype_hash"] for row in merged}) == len(merged)
assert {row.metrics["seed_index"] for row in merged} == {0, 1, 2}
```

Add a 20-generation fake orchestration test proving exactly one generation
Stage-2 call per budget/generation, not three calls.
Add a config contract test proving the formal GA config selects
`task_score_mapping=linear_fixed_scale`, requires the CLI-provided scale,
uses no activation Taylor/SQNR objective weight, and contains no AP gate.

- [ ] **Step 2: Run RED**

```bash
conda run -n univ2x-opt pytest -q \
  tests/test_three_seed_generation_merge.py \
  tests/test_legal_width_ga_orchestration.py
```

- [ ] **Step 3: Expose per-seed, per-generation records**

Extend `run_legal_width_stage1_seeds` to return:

```python
"generation_records": {
    generation: {seed_index: list[ProxyCandidateRecord]}
    for generation in range(generations)
}
```

Do not change independent population evolution. Remove formal reliance on
`S_task`; archive/ranking uses finite `J1`, `R_param`, `R_BOPS`, and hashes.
Continue using infeasible distance only as a lexicographic evolution guide;
never admit such rows to Stage-2.

- [ ] **Step 4: Implement the six-budget orchestrator**

For each target in ascending configuration order:

1. build 3 unique initial populations from greedy endpoint/neighbors, anchors,
   and legal random diversity;
2. run each population for 20 generations with 64 population and offspring;
3. merge all three seed records for generation `g` by descending J1;
4. call `run_generation_stage2` once for that merged ranking;
5. append any winner to the budget's full-validation queue;
6. persist seed supply, BOPS funnel, backfill, worker, and cache lineage.

Zero-candidate generations are recorded and search continues to later
generations. Do not fabricate a winner.

- [ ] **Step 5: Add the formal config and runner route**

The config must explicitly contain:

```yaml
search:
  targets: [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
  primary_bops_tolerance: 0.005
  expanded_bops_tolerance: 0.0075
  independent_seeds: 3
  initial_population_size: 64
  population_size: 64
  offspring_size: 64
  generations: 20
  topk_stage2: 5

proxy:
  proxy_mode: joint_taylor_second_order_fisher_diag
  task_score_mapping: linear_fixed_scale
  joint_loss_scale_path: null
  include_activation_taylor: false
  sqnr_main_objective_weight: 0.0

stage2:
  score_mode: map_minus_latency_ratio
  latency_weight: 0.10
  latency_metric: forward_p50_ms
  ap_iou_backend: gpu
  strict_gpu_ap_iou: true
  num_frames: 500
  warmup_frames: 20
  required_evaluated_frames: 500
  required_skipped_frames: 0
  shared_reference_precision: strict_fp32
  num_workers: 8

full_validation:
  ap_iou_backend: gpu
  strict_gpu_ap_iou: true
  num_frames: 1789
  required_evaluated_frames: 1789
  required_skipped_frames: 0
  num_workers: 8

runtime:
  stage2_gpu_ids: auto
  max_stage2_gpu_memory_fraction: 0.50
```

`joint_loss_scale_path: null` is intentional in the committed config: the
formal run must supply the exact read-only greedy artifact through
`--joint-loss-scale PATH`, and the resolved config records its absolute path
and SHA256. Starting GA without that CLI value fails closed.

Reuse the accepted checkpoint, manifests, strongly typed builder, plugin,
calibration, and legal-width settings.

- [ ] **Step 6: Run GREEN and commit**

```bash
conda run -n univ2x-opt pytest -q \
  tests/test_three_seed_generation_merge.py \
  tests/test_legal_width_ga_orchestration.py \
  tests/test_generation_stage2_candidate_counts.py \
  tests/test_legal_width_genotype.py \
  tests/test_exception_only_repair.py
git diff --check
git add search/orchestration/legal_width_joint_ga.py \
  search/orchestration/legal_width_six_budget_ga.py \
  search/orchestration/lidar_pyramid_search.py \
  search/configs/lidar_pyramid_4090_joint_six_budget_ga.yaml \
  tests/test_three_seed_generation_merge.py \
  tests/test_legal_width_ga_orchestration.py
git commit -m "feat: orchestrate three-seed six-budget joint ga"
```

### Task 6: Generation-Winner Full Validation And Formal Latency

**Files:**
- Modify: `search/orchestration/legal_width_stage2.py`
- Modify: `search/orchestration/legal_width_greedy.py`
- Create: `search/orchestration/formal_latency.py`
- Modify: `search/orchestration/legal_width_six_budget_ga.py`
- Modify: `search/reporting/pareto_frontier.py`
- Create: `tests/test_generation_winner_full_validation.py`
- Create: `tests/test_formal_latency_replay.py`
- Modify: `tests/test_pareto_frontier.py`

**Interfaces:**
- Produces the public functions `run_greedy_endpoint_full_validation`,
  `run_generation_winner_full_validation`, and `run_formal_latency_replay`,
  with the keyword arguments exercised by the Step-1 tests, plus one final
  winner per budget.

- [ ] **Step 1: Write failing full-validation selection tests**

```python
def test_greedy_builds_only_one_unique_endpoint_per_budget(tmp_path: Path) -> None:
    endpoints = [endpoint(0.05, "a"), endpoint(0.10, "a"), endpoint(0.15, "b")]
    result = run_greedy_endpoint_full_validation(
        endpoints=endpoints, stage2_pool=fake_pool, run_dir=tmp_path,
        required_evaluated_frames=1789, required_skipped_frames=0,
    )
    assert result["unique_deployment_count"] == 2
    assert result["budget_lineage_count"] == 3


def test_all_unique_generation_winners_are_full_validated_once(tmp_path: Path) -> None:
    winners = [winner("a", generation=0), winner("a", generation=1), winner("b", generation=2)]
    result = run_generation_winner_full_validation(
        generation_winners=winners, stage2_pool=fake_pool, run_dir=tmp_path,
        required_evaluated_frames=1789, required_skipped_frames=0,
    )
    assert result["unique_deployment_count"] == 2
    assert result["lineage_reference_count"] == 3
```

Add formal latency tests that reject any active worker/build/calibration
process and ensure all rows share one GPU UUID and the strict-FP32 formal
reference.

- [ ] **Step 2: Run RED**

```bash
conda run -n univ2x-opt pytest -q \
  tests/test_generation_winner_full_validation.py \
  tests/test_formal_latency_replay.py
```

- [ ] **Step 3: Implement greedy endpoint deployment and validation**

Read the six proxy endpoint records written by Plan 1. Deduplicate by
phenotype/deployment identity, build each unique endpoint once through the
same `build_smoke` protocol, then run its engine on the complete validation
manifest. Do not construct or evaluate nonterminal greedy path states. Preserve
every budget-to-deployment lineage and write `greedy_full_validation.json/csv`.

- [ ] **Step 4: Implement explicit generation-winner validation**

Unlike the old screening nondominated selector, accept the explicit winner
list, deduplicate by deployment identity, reuse each engine, and evaluate every
unique winner on the same signed 1,789-frame manifest. Require 1789/1789,
zero skips, finite outputs, and precision identity. Preserve every duplicate
generation as lineage.

- [ ] **Step 5: Implement serial formal replay**

Extract/reuse `assert_formal_latency_isolation` and run one evaluation-only
worker on the selected quiet GPU. Replay strict FP32 first, then every unique
full-val engine with identical warmup and measured frames. Return p50/p95,
GPU UUID, clocks/load audit, engine hash, and reference hash. Any concurrent
worker or external compute process fails the formal replay rather than
publishing contaminated latency.

- [ ] **Step 6: Select budget winners**

For each full-val row compute:

```python
formal_f2 = full_validation_map - 0.10 * (
    formal_p50_ms / strict_fp32_formal_p50_ms
)
```

Choose maximum formal F2, then higher mAP, then lower p50, then candidate hash.
Write `budget_005_winner.json` through `budget_030_winner.json` and a combined
CSV.

Merge strict baselines, greedy full-validation endpoints, all unique GA
generation winners, and budget winners into the official result table. Call
`write_official_pareto_artifacts` only after formal latency fields are joined.
Extend plot source markers with `greedy` and keep the existing rule that
screening rows cannot enter an official front.

All three phases must be resume-idempotent. On `--stage2-only --resume`, query
the deployment registry and protocol-specific task keys, reuse complete valid
build/full-val rows, and dispatch only missing or invalid formal-latency work.
Never repeat a completed 1,789-frame evaluation merely to reach the latency
phase.

- [ ] **Step 7: Run GREEN and commit**

```bash
conda run -n univ2x-opt pytest -q \
  tests/test_generation_winner_full_validation.py \
  tests/test_formal_latency_replay.py \
  tests/test_pareto_frontier.py \
  tests/test_search_stage2_process_pool.py
git diff --check
git add search/orchestration/legal_width_stage2.py \
  search/orchestration/legal_width_greedy.py \
  search/orchestration/formal_latency.py \
  search/orchestration/legal_width_six_budget_ga.py \
  search/reporting/pareto_frontier.py \
  tests/test_generation_winner_full_validation.py \
  tests/test_formal_latency_replay.py tests/test_pareto_frontier.py
git commit -m "feat: validate generation winners and formal latency"
```

### Task 7: Plan-2 Regression Gate And Handoff

**Files:**
- Modify: `docs/codex_handoffs/4090_GA_EXPLICIT_QDQ_PROGRESS_20260714.md`

- [ ] **Step 1: Run focused and deployment regressions**

```bash
conda run -n univ2x-opt pytest -q \
  tests/test_stage2_map_latency_score.py \
  tests/test_deployment_registry.py \
  tests/test_stage2_gpu_scheduler.py \
  tests/test_generation_stage2_candidate_counts.py \
  tests/test_three_seed_generation_merge.py \
  tests/test_generation_winner_full_validation.py \
  tests/test_formal_latency_replay.py \
  tests/test_search_stage2_candidate_worker.py \
  tests/test_search_stage2_process_pool.py \
  tests/test_search_stage2_physical_validation.py \
  tests/test_search_stage2_realized_bops.py \
  tests/test_search_group_separation_audit.py \
  tests/test_search_quantization_groups.py \
  tests/test_strongly_typed_builder.py \
  tests/test_strongly_typed_qdq_graph.py \
  tests/test_search_merge_realization.py \
  tests/test_legal_width_ga_orchestration.py
```

Expected: all selected tests pass; record every command and result rather than
silently skipping a failure.

- [ ] **Step 2: Compile every changed Python file**

Compile the explicit Python files changed by this plan:

```bash
conda run -n univ2x-opt python -m py_compile \
  search/stage2/objective.py \
  search/stage2/candidate_artifacts.py \
  search/stage2/lidar_pyramid_real_evaluator.py \
  search/cache/deployment_registry.py \
  search/orchestration/stage2_process_pool.py \
  search/stage2/candidate_worker.py \
  search/orchestration/gpu_scheduler.py \
  search/orchestration/generation_stage2.py \
  search/orchestration/legal_width_joint_ga.py \
  search/orchestration/legal_width_six_budget_ga.py \
  search/orchestration/legal_width_stage2.py \
  search/orchestration/legal_width_greedy.py \
  search/orchestration/formal_latency.py \
  search/reporting/pareto_frontier.py
```

- [ ] **Step 3: Append handoff state**

Record implementation files, test count, cache identity, GPU selection, and:

```text
THREE_SEED_GA_ORCHESTRATION_IMPLEMENTED=true
VARIABLE_STAGE2_SUPPLY_IMPLEMENTED=true
SINGLE_CANDIDATE_500_SKIP_IMPLEMENTED=true
SHARED_FP32_REFERENCE_IMPLEMENTED=true
REAL_SEARCH_STARTED=false
```

- [ ] **Step 4: Commit and push only the 4090 branch**

```bash
git add docs/codex_handoffs/4090_GA_EXPLICIT_QDQ_PROGRESS_20260714.md
git commit -m "docs: record six-budget stage2 orchestration"
git diff --check
git fetch origin feature/heal-compress-h800-sync-4090
git rev-list --left-right --count HEAD...origin/feature/heal-compress-h800-sync-4090
git merge-base --is-ancestor b862b3d8ad061bd12580776226c75f564918298d HEAD
git push origin feature/heal-compress-h800-sync-4090
```
