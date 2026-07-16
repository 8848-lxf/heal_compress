# Six-Budget Real Experiment And Pareto Delivery Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the approved greedy-first and three-seed GA workflows on real RTX 4090 engines, complete 500-frame and full-validation evaluation, replay formal latency, and publish three official Pareto fronts.

**Architecture:** Run the proxy-only greedy phase first, deploy only its unique budget endpoints, freeze and verify the linear scale, then run six independent budget GA workflows. Use the persistent low-occupancy GPU pool for builds/AP evaluation and one quiet GPU for final serial latency; aggregate only full-validation plus formal-latency rows into official Pareto artifacts.

**Tech Stack:** `univ2x-opt`, `modelopt`, PyTorch, HEAL/OpenCOOD, TensorRT 10.9, CUDA 11.8, RTX 4090, GPU BEV-IoU postprocessing, pytest, Matplotlib.

## Global Constraints

- Complete and push both prior implementation plans before any experiment in this plan.
- Work and push only `feature/heal-compress-h800-sync-4090`; preserve ancestor `b862b3d8ad061bd12580776226c75f564918298d`; never force push.
- Do not modify tracer, dependency construction, physical pruning semantics, QDQ semantics, calibration recipe, or merge contracts to make an experiment pass.
- Do not reuse H800 plugins, ONNX, calibration caches, engines, or latency.
- Use fresh timestamped output directories; never delete or overwrite old outputs.
- Use GPU AP postprocessing and exactly eight DataLoader workers.
- Stage-2 uses all GPUs at or below 50% memory occupancy, sorted by lowest load. Do not kill foreign processes.
- BOPS uses primary `+/-0.005`; conditional `+/-0.0075` must be explicitly labeled and is allowed only after primary supply is exhausted.
- No AP/mAP hard gate. Do not hide low-accuracy results.
- One-candidate generations skip 500 frames and enter the full-validation winner queue; zero-candidate generations retain the complete failure funnel.
- Use one shared strict-FP32 reference. Official latency requires one quiet GPU after every parallel worker stops.
- Large checkpoints, ONNX, engines, caches, raw outputs, and tensor dumps remain ignored. Commit only code, tests, configs, reports, CSV summaries, and plots.

## Artifact Ownership Map

- Greedy run directory: proxy paths, six terminal candidates, immutable scale, unique endpoint engines, and endpoint full-validation rows.
- GA run directory: three-seed generation records, BOPS funnels, Stage-2 identities/results, generation winners, and budget winners.
- Deployment registry: cross-budget/seed/generation reuse and protocol-specific evaluation lineage.
- Formal latency replay: one-GPU strict-FP32 reference plus serial candidate p50/p95 evidence.
- Docs handoffs: lightweight result tables, final plots, reproduction commands, and timestamped progress state.

---

### Task 1: Entry Audit And Final Software Gate

**Files:**
- Read: `docs/superpowers/specs/2026-07-16-greedy-first-six-budget-joint-search-design.md`
- Read: the two implementation-plan handoff commits
- Output: `$ENTRY_AUDIT_DIR/entry_audit.json`

- [ ] **Step 1: Record branch, ancestry, remote, worktree, and stash**

Run:

```bash
export ENTRY_AUDIT_DIR="outputs/$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)_six_budget_entry_audit"
mkdir -p "$ENTRY_AUDIT_DIR"
git branch --show-current
git rev-parse HEAD
git status --short
git fetch origin feature/heal-compress-h800-sync-4090
git rev-list --left-right --count HEAD...origin/feature/heal-compress-h800-sync-4090
git merge-base --is-ancestor b862b3d8ad061bd12580776226c75f564918298d HEAD
git stash list
```

Expected: correct branch, zero divergence, ancestor exit 0, and no unexplained worktree changes. Preserve the named historical joint-Taylor stash; do not apply it wholesale.

After Steps 2 and 3, generate the structured audit with the same read-only
commands:

```bash
conda run -n univ2x-opt python - "$ENTRY_AUDIT_DIR/entry_audit.json" <<'PY'
import datetime
import json
import subprocess
import sys
from pathlib import Path

commands = {
    "branch": ["git", "branch", "--show-current"],
    "head": ["git", "rev-parse", "HEAD"],
    "status": ["git", "status", "--short"],
    "remote_divergence": ["git", "rev-list", "--left-right", "--count", "HEAD...origin/feature/heal-compress-h800-sync-4090"],
    "ancestor": ["git", "merge-base", "--is-ancestor", "b862b3d8ad061bd12580776226c75f564918298d", "HEAD"],
    "stash": ["git", "stash", "list"],
    "gpus": ["nvidia-smi", "--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
    "processes": ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"],
    "owned_or_related_processes": ["pgrep", "-af", "search|worker|stage2|trtexec|TensorRT|calibrat|evaluate"],
    "torch": ["conda", "run", "-n", "univ2x-opt", "python", "-c", "import torch; print(torch.__version__, torch.cuda.is_available())"],
    "tensorrt": ["conda", "run", "-n", "modelopt", "python", "-c", "import tensorrt as trt; print(trt.__version__)"],
    "plugin_sha256": ["sha256sum", "quantization/plugins/pointpillar_scatter_trt/build/libpointpillar_scatter_trt.so"],
}
results = {}
for name, command in commands.items():
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    results[name] = {"command": command, "exit_code": completed.returncode,
                     "stdout": completed.stdout, "stderr": completed.stderr}
payload = {"created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "checks": results}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
PY
```

- [ ] **Step 2: Record GPU and process occupancy**

```bash
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits
pgrep -af 'search|worker|stage2|trtexec|TensorRT|calibrat|evaluate' || true
```

Expected: every external process is identified. Do not dispatch to a GPU above 50% memory occupancy.

- [ ] **Step 3: Verify environments and production binaries**

```bash
conda run -n univ2x-opt python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
conda run -n modelopt python -c 'import tensorrt as trt; print(trt.__version__)'
sha256sum quantization/plugins/pointpillar_scatter_trt/build/libpointpillar_scatter_trt.so
conda run -n modelopt trtexec --version
```

Expected: TensorRT 10.9, CUDA-capable PyTorch, plugin file present, and hashes recorded.

- [ ] **Step 4: Run the complete pre-experiment test gate**

```bash
conda run -n univ2x-opt pytest -q \
  tests/test_joint_loss_scale.py \
  tests/test_joint_taylor_linear_score.py \
  tests/test_two_level_bops_admission.py \
  tests/test_greedy_legal_actions.py \
  tests/test_joint_budget_greedy_search.py \
  tests/test_legal_width_greedy_orchestration.py \
  tests/test_stage2_map_latency_score.py \
  tests/test_deployment_registry.py \
  tests/test_stage2_gpu_scheduler.py \
  tests/test_generation_stage2_candidate_counts.py \
  tests/test_three_seed_generation_merge.py \
  tests/test_generation_winner_full_validation.py \
  tests/test_formal_latency_replay.py \
  tests/test_legal_width_inventory.py \
  tests/test_deterministic_width_decoder.py \
  tests/test_precision_structure_orthogonality.py \
  tests/test_physical_replay_from_width_gene.py \
  tests/test_search_stage2_physical_validation.py \
  tests/test_strongly_typed_builder.py \
  tests/test_strongly_typed_qdq_graph.py \
  tests/test_search_merge_realization.py \
  tests/test_search_stage2_realized_bops.py \
  tests/test_search_stage2_process_pool.py \
  tests/test_pareto_frontier.py
```

Expected: all tests pass. Any failure blocks experiment start until diagnosed with `superpowers:systematic-debugging`.

### Task 2: Run Proxy-Only Six-Budget Greedy Search

**Files:**
- Config: `search/configs/lidar_pyramid_4090_greedy_six_budget.yaml`
- Output: `$GREEDY_RUN`

- [ ] **Step 1: Start a fresh foreground greedy run**

```bash
conda run -n univ2x-opt python -m search.cli \
  --config search/configs/lidar_pyramid_4090_greedy_six_budget.yaml \
  --output-root outputs \
  --greedy-only
```

Expected: one new run directory; no TensorRT engine build; six target reports plus a read-only `joint_loss_scale.json`.

Capture and validate the generated directory:

```bash
export GREEDY_RUN="$(find outputs -maxdepth 1 -type d -name '4090_greedy_six_budget_*' \
  -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
test -n "$GREEDY_RUN"
test -f "$GREEDY_RUN/joint_loss_scale.json"
```

- [ ] **Step 2: Audit every budget endpoint**

For each target, verify its endpoint JSON contains:

```text
target_bops
admission_mode
actual_proxy_bops
signed_bops_error
genotype_hash
width_vector_hash
structure_hash
precision_hash
phenotype_hash
L_joint_raw
R_prune
normal_candidate_repair_invoked=false
```

Expected: primary or explicitly expanded admission, or a documented infeasible result. No out-of-band endpoint is mislabeled feasible.

- [ ] **Step 3: Audit the loss scale**

```bash
stat -c '%a %n' "$GREEDY_RUN/joint_loss_scale.json"
sha256sum "$GREEDY_RUN/joint_loss_scale.json"
conda run -n univ2x-opt python -c \
  'import os, pprint; from search.proxy.joint_loss_scale import load_joint_loss_scale; pprint.pp(load_joint_loss_scale(os.path.join(os.environ["GREEDY_RUN"], "joint_loss_scale.json")))'
```

Expected: no write bits, `mapping=linear_fixed_scale`, positive nearest-rank P90, member hashes present, and no tau fields.

- [ ] **Step 4: Preserve the exact reproduction command**

Confirm `$GREEDY_RUN/commands.sh`, resolved config, entry commit, Fisher manifest, checkpoint hash, width-space hash, ranking hash, and scale hash exist before continuing.

### Task 3: Deploy And Full-Validate Greedy Endpoints

**Files:**
- Input/output: the greedy run directory
- Report: `docs/codex_handoffs/4090-greedy-six-budget-search-report.md`

- [ ] **Step 1: Resume the run in greedy endpoint deployment mode**

```bash
conda run -n univ2x-opt python -m search.cli \
  --config search/configs/lidar_pyramid_4090_greedy_six_budget.yaml \
  --output-root outputs \
  --resume "$GREEDY_RUN" \
  --stage2-only
```

Expected: only unique terminal phenotypes are built; duplicate endpoints across budgets reuse one deployment. Each unique endpoint passes build-smoke audits and runs 1789/1789 with zero skips.

- [ ] **Step 2: Verify deployment identity and no intermediate builds**

Compare endpoint identities against registry records. Assert:

```text
engine_build_count == unique_terminal_deployment_count
nonterminal_greedy_engine_build_count == 0
decoded_mask_hash == physical_replay_mask_hash
predicted_parameter_count == materialized_parameter_count
requested_precision_hash == realized_precision_hash
```

- [ ] **Step 3: Verify real metrics**

For every feasible target record AP03/AP05/AP07/mAP, 1789/1789, zero skips,
screening p50/p95, physical parameter retention, proxy/physical/realized BOPS,
engine hash, and GPU assignment. Low mAP is reported, not discarded.

- [ ] **Step 4: Write and commit the greedy report**

The report begins with:

```text
GREEDY_SIX_BUDGET_SEARCH_COMPLETE=true/false
GREEDY_FEASIBLE_BUDGET_COUNT=
GREEDY_UNIQUE_DEPLOYMENT_COUNT=
GREEDY_FULL_VAL_SUCCESS_COUNT=
LINEAR_LOSS_SCALE_HASH=
GA_STARTED=false
```

Include the action trace, strict/expanded admissions, endpoint table, failures,
full-val metrics, cache reuse, and exact reproduction commands.

```bash
git add docs/codex_handoffs/4090-greedy-six-budget-search-report.md \
  docs/codex_handoffs/4090_GA_EXPLICIT_QDQ_PROGRESS_20260714.md
git commit -m "exp: record six-budget greedy deployment results"
git push origin feature/heal-compress-h800-sync-4090
```

### Task 4: Launch The Fresh Six-Budget Three-Seed GA

**Files:**
- Config: `search/configs/lidar_pyramid_4090_joint_six_budget_ga.yaml`
- Output: `$GA_RUN`

- [ ] **Step 1: Bind the frozen greedy scale into a config snapshot**

Set the new run's `proxy.joint_loss_scale_path` to the absolute read-only scale
artifact from Task 2. Record its SHA256. Do not copy a mutable or regenerated
value into each generation.

- [ ] **Step 2: Start the real search in the foreground**

```bash
conda run -n univ2x-opt python -m search.cli \
  --config search/configs/lidar_pyramid_4090_joint_six_budget_ga.yaml \
  --output-root outputs \
  --joint-loss-scale "$GREEDY_RUN/joint_loss_scale.json"
```

Expected configured scale:

```text
budgets=6
independent_seeds_per_budget=3
population_size=64
offspring_size=64
generations=20
maximum_global_stage2_candidates_per_generation=5
stage2_frames=500
full_validation_frames=1789
```

Do not shorten these values to recover time.

Capture and validate the new run directory:

```bash
export GA_RUN="$(find outputs -maxdepth 1 -type d -name '4090_joint_six_budget_ga_*' \
  -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
test -n "$GA_RUN"
test -f "$GA_RUN/run_manifest.json"
```

- [ ] **Step 3: Monitor without changing search semantics**

At least every generation, inspect:

```text
candidate_count by seed
unique genotype/structure/phenotype counts
primary/expanded BOPS counts
selected Stage-2 count
backfill failures
repair invocation count
worker GPU assignment
cache hits
generation winner or zero-candidate reason
```

If GPU occupancy exceeds the memory threshold, allow the scheduler to pause
new dispatch; do not kill external jobs or silently move formal latency into a
contended interval.

- [ ] **Step 4: Validate variable candidate behavior during the run**

For a zero-candidate generation, require a persisted BOPS funnel and continue
to the next generation. For a one-candidate generation, require build/smoke,
no `evaluation_500` artifact, and a winner queued for full validation. For
two-to-five, require 500/500, zero skips, and maximum-F2 winner selection.

- [ ] **Step 5: Verify all six budgets finish or fail explicitly**

Each budget must have 20 generation records. A budget may have fewer than 20
winners only where zero-candidate generations are documented. No missing
generation may be silently treated as zero supply.

### Task 5: Full Validation Of Every Generation Winner

**Files:**
- Input/output: GA run directory

- [ ] **Step 1: Verify full-validation queue identity**

For each budget compare the generation winner list with the deduplicated
deployment registry. Preserve all source generations even where one engine is
reused.

- [ ] **Step 2: Complete full validation**

The orchestrator must evaluate every unique generation-winner deployment on
the same signed 1,789-frame manifest. Require:

```text
evaluation_protocol=full_validation
evaluated_frames=1789
skipped_frames=0
precision_identity_passed=true
full_validation_success=true
```

No AP threshold is applied.

- [ ] **Step 3: Reconcile counts**

Assert:

```text
winner_lineage_count == number of nonzero-supply generations
full_val_task_count == unique winner deployment count not already cached
full_val_result_count == unique winner deployment count
```

Every discrepancy must have a registry/cache reason.

### Task 6: Stop Parallel Work And Replay Formal Latency

**Files:**
- Output: `$GA_RUN/formal_latency_replay.json`

- [ ] **Step 1: Confirm all parallel workers are stopped**

```bash
pgrep -af 'candidate_worker|trtexec|TensorRT|calibrat|evaluate|legal_width_six_budget' || true
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits
```

Expected: no owned builder, calibration, GA controller, or evaluation worker remains. Do not proceed on a GPU with a competing compute process.

- [ ] **Step 2: Select one quiet RTX 4090 and record pre-audit**

Choose the least-loaded GPU that is actually free for formal timing. Record
UUID, clocks, temperature, power, memory, processes, driver, CUDA, TensorRT,
plugin hash, and engine hashes.

- [ ] **Step 3: Run strict-FP32 then all formal candidates serially**

Resume the formal-latency phase through the runner's configured phase command:

```bash
conda run -n univ2x-opt python -m search.cli \
  --config search/configs/lidar_pyramid_4090_joint_six_budget_ga.yaml \
  --output-root outputs \
  --resume "$GA_RUN" \
  --joint-loss-scale "$GREEDY_RUN/joint_loss_scale.json" \
  --stage2-only
```

Expected: the resume manifest recognizes completed builds/full-val and runs
only pending formal-latency work. Every formal row uses the same GPU UUID,
warmup, synchronization, measured inputs, and strict-FP32 reference.

- [ ] **Step 4: Verify formal timing integrity**

Require p50/p95 finite, no concurrent worker violations, reference engine hash
present, pre/during/post audits present, and no screening latency substituted
for formal fields.

### Task 7: Verify Budget Winners And Pareto Fronts

**Files:**
- Output: GA run Pareto CSV/PNG/PDF files
- Report: `docs/codex_handoffs/4090-six-budget-joint-ga-pareto-report.md`
- Plot copies: `docs/codex_handoffs/assets/4090_six_budget_joint_ga/`

- [ ] **Step 1: Verify budget-winner score calculation**

For every budget recompute:

```text
formal_F2 = full_val_mAP - 0.10 * formal_p50 / strict_FP32_formal_p50
```

Assert the recorded winner maximizes this value among that budget's valid
generation winners. Expanded-admission winners remain explicitly labeled.

- [ ] **Step 2: Verify official Pareto input protocol**

The official table may contain only rows with:

```text
full_validation_success=true
evaluation_protocol=full_validation
formal_latency_p50_ms finite
formal_latency_gpu_uuid == strict_fp32_reference_gpu_uuid
```

Reject any 500-frame mAP, screening p50, LUT latency, or proxy latency in the
official table.

- [ ] **Step 3: Verify all required artifacts**

```text
pareto_map_vs_bops_retention.csv/.png/.pdf
pareto_map_vs_param_retention.csv/.png/.pdf
pareto_map_vs_latency_p50.csv/.png/.pdf
pareto_combined_bops_map_latency.png
pareto_combined_param_map_latency.png
budget_winners.csv
```

Recompute dominance independently in the test helper and compare candidate
IDs with each CSV.

- [ ] **Step 4: Write the final report**

The report includes:

- branch, entry/final commits, configs, scale hash, checkpoint/manifests;
- greedy and GA search sizes and proxy evaluation counts;
- per-budget primary/expanded/zero-supply generation counts;
- every greedy endpoint and GA budget winner metric;
- all full-val generation winners or a linked committed CSV summary;
- build/evaluation/cache/failure matrices;
- requested/legalized/realized precision identity;
- physical parameter and BOPS reconciliation;
- formal GPU and latency isolation evidence;
- best mAP, lowest BOPS, lowest parameter retention, and lowest formal p50 candidates;
- three Pareto-front point counts and file paths;
- exact reproduction and resume commands;
- every unresolved limitation without proxy substitution.

Copy only lightweight final plots and CSV summaries into the docs asset
directory; leave raw output artifacts ignored.

### Task 8: Final Verification, Progress Anchor, Commit, And Push

**Files:**
- Modify: `docs/codex_handoffs/4090_GA_EXPLICIT_QDQ_PROGRESS_20260714.md`
- Add: final report, lightweight CSVs, and plots

- [ ] **Step 1: Run final tests and compilation**

Run the complete Task-1 test gate again, then:

```bash
mapfile -t CHANGED_PY < <(git diff --name-only \
  "$(git merge-base HEAD origin/feature/heal-compress-h800-sync-4090)"..HEAD \
  -- '*.py')
if ((${#CHANGED_PY[@]})); then
  conda run -n univ2x-opt python -m py_compile "${CHANGED_PY[@]}"
else
  printf '%s\n' 'No Python files changed since the remote merge base.'
fi
git diff --check
```

- [ ] **Step 2: Append final 4090 status fields**

```text
GREEDY_SIX_BUDGET_SEARCH_COMPLETE=
LINEAR_JOINT_SCORE_APPLIED=
GA_REAL_SEARCH_EXECUTED=
GA_POPULATION_SIZE=64
GA_GENERATIONS=20
GA_SEEDS=3
STAGE2_500_SUCCESSFUL_CANDIDATES=
FULL_VAL_SUCCESSFUL_CANDIDATES=
FORMAL_LATENCY_COMPLETE=
PARETO_BOPS_POINTS=
PARETO_PARAM_POINTS=
PARETO_LATENCY_POINTS=
STAGE_A_STARTED=false
STAGE_B_ALLOWED=false
```

End with a Shanghai timestamp and round delimiter.

- [ ] **Step 3: Commit lightweight evidence only**

```bash
git add docs/codex_handoffs/4090-six-budget-joint-ga-pareto-report.md \
  docs/codex_handoffs/4090-greedy-six-budget-search-report.md \
  docs/codex_handoffs/4090_GA_EXPLICIT_QDQ_PROGRESS_20260714.md \
  docs/codex_handoffs/assets/4090_six_budget_joint_ga
git diff --cached --check
git commit -m "report: record six-budget joint ga pareto results"
```

- [ ] **Step 4: Verify and push only the allowed branch**

```bash
git status --short
git log --oneline --decorate -10
git merge-base --is-ancestor b862b3d8ad061bd12580776226c75f564918298d HEAD
git fetch origin feature/heal-compress-h800-sync-4090
git rev-list --left-right --count HEAD...origin/feature/heal-compress-h800-sync-4090
git push origin feature/heal-compress-h800-sync-4090
git rev-parse HEAD
git rev-parse origin/feature/heal-compress-h800-sync-4090
```

Expected: local and remote commits match; H800 source branch is untouched; no force push.
