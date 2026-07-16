# Greedy-First Six-Budget Joint Search Design

## Status And Precedence

This specification defines the next formal search and deployment workflow for
the 4090 branch. It supersedes conflicting score, budget, Stage-2 admission,
and evaluation rules in the earlier joint-Taylor and legal-width designs. The
existing legal-width genotype, fixed precision-independent channel decoder,
physical replay, strongly typed explicit-QDQ deployment, and realization
audits remain authoritative where they do not conflict with this document.

The implementation and experiments remain confined to
`feature/heal-compress-h800-sync-4090`. They must preserve the H800 fix
ancestor and must not modify or push the H800 source branch.

## Goals

1. Replace the exponential Stage-1 task mapping with a transparent linear,
   fixed-scale joint weight Taylor objective.
2. Run a robust greedy iterative search first for BOPS retention targets
   `0.05`, `0.10`, `0.15`, `0.20`, `0.25`, and `0.30`.
3. Use the greedy paths and existing legal anchors to freeze the joint-loss
   scale before GA starts.
4. Run three independent GA populations for 20 generations per budget while
   deploying only one merged, globally unique Top-5 set per generation.
5. Build and evaluate real physically pruned, strongly typed mixed-precision
   engines with exact structure and precision identity audits.
6. Select generation and budget winners using real mAP and forward p50, then
   produce formal full-validation Pareto fronts.

Activation Taylor is outside this scope. SQNR remains diagnostic and is not a
main objective term. BOPS is an admission constraint and is not added to the
Stage-1 weighted objective. No AP or mAP hard gate is applied.

## Candidate Representation And Invariants

A candidate is `C=(s,b)`:

- `s` contains one legal retained-width index per existing local pruning
  domain;
- `b` contains one deployable action per canonical precision group.

The fixed prune-only Taylor decoder determines the structure mask:

`M = M(s)`.

Precision is not an input to the decoder. A precision-only operation must not
change pruned unit IDs, channel indices, the physical plan, parameter count,
or structure hash. A width-only operation must not change the precision
profile. Normal generated candidates are legal by construction and do not
invoke repair. Repair remains an exception-only compatibility path and may
neither add pruning nor change precision.

Physical export applies the decoded mask exactly. It must not rerank channels,
invoke conditional Taylor, run a new Top-K, or silently align widths. Export
must prove:

```text
decoded_mask_hash == physical_replay_mask_hash
predicted_parameter_count == sum(materialized_parameter.numel())
```

## Joint Weight Taylor Proxy

For effective weights

`W_eff(C) = M(s) * Q_b(W)`,

the perturbation is

`delta W(C) = M(s) * Q_b(W) - W`.

A pruned parameter contributes `delta w = -w` and does not also contribute a
quantization error. A retained parameter contributes the quantization error
for its actual canonical precision group. Overlapping dependency slices are
deduplicated by physical parameter element.

The default second-order empirical-Fisher score is

```text
L_joint(C) = sum_i |g_i * delta w_i|
           + 0.5 * sum_i h_i * delta w_i^2
```

where `h_i = mean((dL_task/dw_i)^2)`. It is an empirical Fisher diagonal, not
a full Hessian. The first-order form remains available for ablation. Both
forms use float64 accumulation and fail closed on nonfinite values or missing
precision mappings.

## Fixed Linear Stage-1 Objective

The exponential mapping `exp(-L_joint/tau)` is removed from the formal search.
The previously generated `tau=0.01027115735` is not reused because its anchor
boundary was underresolved and collapsed valid pruning candidates to nearly
zero task score.

After greedy search, collect positive finite `L_joint` values from:

- unique accepted states on the six greedy primary paths;
- unique legal anchor candidates used as search seeds.

Do not include every rejected one-step action, because their multiplicity
would bias the distribution. Deduplicate by phenotype hash, sort
deterministically, and compute the nearest-rank P90:

```text
index = ceil(0.90 * count) - 1
L_scale = sorted_positive_losses[index]
```

An empty or nonpositive pool fails closed. Persist the members, hashes,
quantile method, value, code commit, and creation time in a signed
`joint_loss_scale.json`. The artifact is read-only for all seeds, budgets, and
generations. It is never recomputed from generation statistics.

The formal Stage-1 score is maximized:

```text
R_prune = 1 - physical_parameter_count / original_parameter_count
J1 = -0.8 * (L_joint / L_scale) + 0.2 * R_prune
```

There is no clipping or hidden saturation. Lower joint Taylor loss and higher
structural parameter pruning improve `J1`. Precision bit-size changes do not
alter `R_prune`.

## BOPS Budgets And Two-Level Admission

The six target BOPS retention values are:

```text
0.05, 0.10, 0.15, 0.20, 0.25, 0.30
```

The primary interval for target `T` is `abs(R_BOPS-T) <= 0.005`. This interval
is always attempted first.

Only when no candidate survives the primary interval, including after all
primary candidates have exhausted deployment backfill, may the generation or
greedy endpoint selection use `abs(R_BOPS-T) <= 0.0075`. An expanded candidate
must be among the nearest legal candidates to the target and must not already
belong to an adjacent budget's primary interval. The expanded interval cannot
become the default configuration.

Every expanded admission records:

```text
admission_mode = expanded_bops_tolerance
target_bops
primary_tolerance = 0.005
effective_tolerance = 0.0075
actual_bops
signed_bops_error
nearest_adjacent_budget
reason = no_candidate_in_primary_interval
```

Proxy, physical, and realized BOPS are recorded separately. A profile or mask
must never be changed merely to make an already selected candidate appear
feasible.

If BOPS excludes an entire generation, report this funnel and the nearest
out-of-band candidates:

```text
raw_count
finite_count
structure_legal_count
precision_legal_count
bops_below_count
bops_primary_count
bops_expanded_only_count
bops_above_count
unique_count
```

## Robust Greedy Search

Greedy search runs before GA and independently for each target. Every run
starts from the all-keep strict-FP32 candidate. Legal actions are:

1. move one pruning domain to the next smaller legal retained width;
2. move one precision group down one actually deployable precision level.

For every successor, recompute the exact decoded mask, joint weight Taylor
loss, physical parameter proxy, and BOPS. An action must save positive BOPS and
remain structurally and precision legal. The primary action key is minimized:

```text
rho(action | C) = delta_L_joint / delta_R_BOPS_saved
```

Higher structural parameter reduction is the second key; stable action ID is
the final tie-break. The default bounded-backtracking frontier retains the
best eight alternative sibling states and permits at most 4,096 unique state
expansions per budget. A state is never expanded twice. These fixed limits
prevent a discrete action that jumps across a budget from prematurely making
the target unreachable while keeping this an interpretable greedy comparison,
not a second GA.

Search states and proxy computations may be memoized across budgets. Each
budget chooses one terminal phenotype, using the primary interval first and
the expanded interval only under the two-level rule. Among terminals in the
active interval, choose lower `L_joint`, then higher `R_prune`, then smaller
absolute BOPS error, then stable phenotype hash. If no terminal exists, that
budget is reported infeasible without relaxing legality.

Only the final unique terminal phenotype for each budget is physically built
and run on the complete validation manifest. If several budgets select the
same deployment identity, build and evaluate it once and retain all budget
lineage references.

## Three-Seed GA

For each budget, run three independent populations with different random
seeds, `population_size=64`, `offspring_per_generation=64`, and
`generations=20`. All populations use the same legal search space, fixed
ranking, `L_scale`, BOPS rules, and deployment cache. Greedy endpoints,
neighboring legal states, anchors, and diverse random legal candidates seed
the populations without cloning one phenotype to fill a population.

Structure crossover selects complete domain-width genes. Structure mutation
moves only among adjacent legal widths. Precision crossover and mutation
operate only on complete canonical precision groups. Each population evolves
independently, preserving stochastic diversity.

At each generation, merge ranked feasible candidates from all three
populations. Deduplicate by deployment identity and select at most five global
candidates, not five per seed. Ranked backfill continues after structure,
precision, BOPS, build, or realization failures until five successful
candidates exist or the merged ranking is exhausted.

## Stage-2 Candidate Count Semantics

Fewer than five candidates is not automatically a failed generation.

- **Zero candidates:** the generation fails and does not enter Stage-2. The
  complete admission funnel and nearest BOPS misses are mandatory.
- **One candidate:** build it, run physical/QDQ/strongly typed/merge/precision
  audits and a runtime smoke. Skip the generation's 500-frame comparison,
  register it directly as the generation winner, and queue it with all other
  generation winners for the common full validation. Its required AP
  evaluation is therefore full validation, not the skipped 500-frame screen.
- **Two to five candidates:** evaluate every candidate on the common fixed
  500-frame manifest and choose one generation winner using the real Stage-2
  score.

If deployment failures reduce a provisional set, backfill first. The above
count rule is applied only after the merged ranking and backfill candidates are
exhausted. No candidate is copied to reach five.

There is no mAP or AP hard gate. Runtime validity, complete evaluation,
`skipped=0`, BOPS admission, and deployment audits remain mandatory.

## Real Stage-2 And Winner Score

Use one signed strict-FP32 reference engine and one common reference result for
all RTX 4090 workers. Do not rebuild a reference per GPU. The reference and
candidate use the same manifest, evaluator, postprocessing, and latency
definition.

For candidates evaluated on 500 frames, maximize:

```text
R_latency = candidate_forward_p50 / strict_fp32_forward_p50
F2 = candidate_mAP - 0.10 * R_latency
```

This directly encodes the accepted exchange rate: a 10% forward-p50 reduction
can compensate an absolute mAP decrease of 0.01. The older `tau_AP` notation
is removed. AP03, AP05, AP07, p90, and p95 remain reported but are not separate
hard gates.

All generation winners, including single-candidate generations, run the same
complete 1,789-frame validation manifest. If the signed project full manifest
changes, every row in the comparison must use the same replacement manifest
and record its frame count and hash. After formal same-GPU latency replay,
choose the budget winner using full-validation mAP and formal p50 in the same
`F2` formula.

## Deployment Identity And Cache

The cache key includes at least:

```text
physical_hash
precision_profile_hash
calibration_manifest_hash
QDQ_topology_hash
plugin_binary_hash
TensorRT/CUDA/GPU-architecture signature
code/build signature
evaluation_manifest_hash
```

Physical model, ONNX, calibration cache, engine, 500-frame result, full-val
result, and formal latency result may be reused only when their relevant
identity fields match exactly. Reuse is recorded with source budget, seed,
generation, and candidate lineage. H800 engines, caches, and latency are never
reused on 4090.

Every deployment must pass:

- physical structure and decoded-mask identity;
- typed ONNX and semantic QDQ boundary audit;
- per-output-channel weight-scale and EntropyCalibration2 lineage audit;
- engine deserialize and EngineInspector audit;
- requested/legalized/realized precision identity;
- merge contracts, including `/Concat_9`;
- physical and realized BOPS admission.

## GPU Scheduling And Evaluation

Audit all eight RTX 4090 devices before and during work. Exclude devices whose
memory use exceeds 50%. Rank the remaining devices by sampled utilization,
memory use, and external process load, and schedule the least-loaded devices
first. Do not terminate or interfere with external processes. If all devices
exceed the admissible occupancy, pause dispatch and preserve pending work.

Build, calibration, 500-frame evaluation, and full validation use a persistent
work-conserving worker pool: whenever one worker finishes, it immediately
receives the next candidate. GPU AP postprocessing is mandatory and each
evaluation DataLoader uses eight workers.

Parallel latency is screening evidence only. After every required engine is
built and AP evaluation workers stop, select one then-idle RTX 4090, verify no
competing compute process, and replay all formal candidates serially with the
same warmup, inputs, synchronization, and measurement count. The strict-FP32
reference is replayed in the same formal session. Official latency Pareto
points use only this formal p50/p95.

## Pareto Outputs

The official candidate pool contains strict baselines, six greedy endpoints,
all full-validation GA generation winners, and final budget winners. Only rows
with successful full validation and formal same-GPU latency can enter official
fronts.

Produce at least:

```text
pareto_map_vs_bops_retention.csv/.png/.pdf
pareto_map_vs_param_retention.csv/.png/.pdf
pareto_map_vs_latency_p50.csv/.png/.pdf
pareto_combined_bops_map_latency.png
pareto_combined_param_map_latency.png
```

Screening mAP and proxy latency are kept in separate diagnostic tables and
must not be mixed into official fronts.

## Error Handling And Audit Records

Every failure is terminal for that candidate and retains its identity and
stage-specific reason. Required categories include missing mapping, illegal
structure, nonfinite proxy, BOPS below/above interval, physical replay
mismatch, parameter-count mismatch, ONNX/QDQ/calibration failure, engine build
or deserialize failure, precision fallback, merge failure, evaluation skip,
and cache signature mismatch.

Reports distinguish primary and expanded BOPS admissions, candidate supply by
seed, backfill attempts, final Stage-2 count, repair invocation count, cache
reuse, GPU assignment, and external occupancy. Normal-candidate repair rate
must be zero.

## Verification Strategy

Implementation proceeds with regression-first tests covering:

1. no exponential tau use in the formal objective;
2. deterministic P90 `L_scale`, signed persistence, and cross-generation
   immutability;
3. `J1` direction and the `0.8/0.2` weighting after normalization;
4. the `F2` 10%-latency/0.01-mAP exchange identity;
5. BOPS as the only performance admission gate;
6. primary then conditional expanded BOPS admission;
7. greedy action legality, exact successor rescoring, bounded backtracking,
   deterministic ties, and infeasible-budget closure;
8. three independent populations merged to one unique global Top-5;
9. zero/one/two-to-five Stage-2 count behavior;
10. exact mask replay, parameter count, and precision realization identity;
11. normal candidates never invoking repair;
12. deployment cache deduplication and cross-budget lineage;
13. GPU occupancy filtering and work-conserving dispatch;
14. 500-frame/full-validation protocol separation;
15. screening/formal latency separation and Pareto input filtering.

All changed Python files must compile. Focused and related regressions,
`git diff --check`, branch/ancestor checks, and output-ignore checks are
mandatory before each code or report commit.

## Delivery Order

1. Implement and test the linear objective, scale artifact, and two-level BOPS
   admission.
2. Implement and test robust six-budget greedy search.
3. Run greedy search; deploy and full-validate its unique endpoints.
4. Freeze `joint_loss_scale.json` from greedy paths and legal anchors.
5. Integrate and test the three-seed, 20-generation merged Top-5 GA workflow.
6. Run all six budget searches with real Stage-2 deployment and evaluation.
7. Full-validate generation winners, replay formal latency, compute budget
   winners and Pareto fronts.
8. Commit lightweight code, tests, configs, plots, summaries, and 4090 handoff
   reports; keep engines, ONNX, calibration caches, checkpoints, and raw
   outputs ignored.
