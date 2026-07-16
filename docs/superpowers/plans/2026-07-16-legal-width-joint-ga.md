# Legal-Width Joint GA Implementation Plan

**Goal:** Migrate the existing production search runner to deterministic legal
width structure genes, execute real multi-seed GA and deployment evaluation,
and publish three full-validation Pareto fronts.

**Architecture:** Keep the current phenotype, unified joint Taylor proxy,
physical replay, typed-QDQ, TensorRT, and Stage-2 contracts. Replace only the
formal structure chromosome and its generation/decode path, then add a shared
feasible archive and protocol-safe Pareto reporting.

## Task 1: Typed legal-width primitives

**Create:** `search/space/legal_width_inventory.py`,
`search/encoding/legal_width_genotype.py`,
`search/decoding/fixed_taylor_width_decoder.py`.

**Tests:** inventory, genotype, deterministic decoder, nested masks, grouped
decoder, and precision/structure orthogonality modules requested by the user.

- Write focused failing tests, including 10,000 random chromosomes.
- Enumerate dense/grouped legal widths from current physical constraints.
- Persist fixed first/second-order rankings and deterministic tie-breaks.
- Decode widths without any precision input and emit stable structure lineage.
- Run focused and existing grouped/physical regression tests.

## Task 2: Formal GA operators and compatibility boundary

**Create:** `search/operators/legal_width_mutation.py`,
`search/operators/legal_width_crossover.py`.

**Modify:** `search/candidate.py`, `search/canonicalization.py`, `search/ga/*`,
and candidate codec/hashing modules.

- Write failing mutation/crossover/repair/hash invariants.
- Generate all new structure chromosomes in legal index space.
- Keep an explicit legacy-mask import adapter; reject implicit binary offspring.
- Make normal candidate repair invocation zero and instrument exception repair.
- Preserve independent complete precision-group operations.

## Task 3: Joint proxy and production runner integration

**Modify:** current joint scalar/batched proxy, `SearchSpaceSpec`, and
`search/orchestration/lidar_pyramid_search.py`.

- Reuse only still-valid unified proxy pieces from `stash@{0}` by selective
  patching; do not apply its normal-shortlist-repair behavior.
- Decode width genes before proxy scoring and Stage-2 serialization.
- Verify scalar/batched parity and fixed-tau behavior.
- Verify physical replay mask and parameter count identity without reranking.
- Add the formal legal-width config and snapshot all inventory/ranking hashes.

## Task 4: Feasible archive, budget bands, and Pareto reporting

**Create:** `search/archive/feasible_pareto_archive.py`,
`search/reporting/pareto_frontier.py`.

- Write failing dominance, deduplication, protocol isolation, and latency-source
  tests.
- Maintain external non-dominated feasible candidates with structure/precision
  diversity.
- Derive reachable BOPS bands from anchor evidence when the main band is narrow.
- Generate CSV/PNG/PDF fronts only from full-val and formal-latency records.

## Task 5: Inventory, Fisher ranking, and anchor execution

- Create a new timestamped output root; never overwrite prior outputs.
- Reuse a Fisher cache only after complete checkpoint/config/manifest/code
  lineage validation, otherwise recollect in `univ2x-opt`.
- Emit legal-width inventory and fixed first/second-order rankings.
- Materialize unique anchor structures once and evaluate supported FP32/FP16/
  INT8 deployments in `modelopt` on GPUs 4-7.
- Calibrate/validate fixed tau and seed the legal-width population.

## Task 6: Real GA and Stage-2

- Run three seeds, each population >=64 and generations >=30, from generation 0.
- Maintain generation and external archive statistics; normal repair count must
  remain zero.
- Ensure >=30 unique feasible phenotypes and >=15 physical structures, using
  reachable budget bands if needed.
- Deploy ranked archive candidates with multi-GPU backfill until >=15 complete
  real 50-frame Stage-2 evaluations or the archive is exhausted.
- Preserve every build/evaluation/audit failure without relaxing constraints.

## Task 7: Full validation, formal latency, and reports

- Run one formal full-validation manifest for baselines, non-dominated points,
  band representatives, and at least five successful candidates.
- Stop all workers and measure every reportable engine serially on one idle GPU.
- Generate BOPS-mAP, physical-parameter-mAP, and real-p50-mAP fronts plus
  combined plots.
- Write `docs/codex_handoffs/4090-legal-width-joint-ga-search-report.md` and
  append a timestamped progress round.

## Task 8: Delivery

- Run all requested focused tests, relevant regressions, py_compile, and
  `git diff --check`.
- Commit implementation, tests, experiments, and reports in bounded stages.
- Fetch and verify no remote divergence before every push.
- Push only `feature/heal-compress-h800-sync-4090`, never force.
