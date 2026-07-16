# 4090 Six-Budget Greedy Search And Deployment Report

GREEDY_PROXY_SEARCH_COMPLETE = true
GREEDY_PRIMARY_ENDPOINT_COUNT = 6
GREEDY_EXPANDED_ENDPOINT_COUNT = 0
GREEDY_UNIQUE_PHENOTYPE_COUNT = 6
GREEDY_UNIQUE_PHYSICAL_STRUCTURE_COUNT = 4
GREEDY_ENGINE_BUILD_SUCCESS_COUNT = 6
GREEDY_FULL_VALIDATION_SUCCESS_COUNT = 6
GREEDY_FORMAL_LATENCY_COMPLETE = false
GA_SEARCH_STARTED = false
STAGE_A_STARTED = false
STAGE_B_ALLOWED = false

## Scope And Lineage

- Branch: `feature/heal-compress-h800-sync-4090`.
- Proxy run: `outputs/4090_legal_width_greedy_six_budget_20260716_131554`.
- Proxy entry commit: `49770d64c5119b3f4c26b9a5ad52bb2ec2b3a33f`.
- Deployment route commit: `b7a4e02`.
- Required H800 ancestor: `b862b3d8ad061bd12580776226c75f564918298d`.
- Checkpoint, legal-width inventory, Fisher manifest, canonical ranking, resolved
  config, and commands are preserved inside the proxy run directory.
- No GA, Stage A, or Stage B workflow ran in this phase.

The greedy comparator starts from the all-keep highest-deployable precision
profile and repeatedly selects the legal width or precision action with the
lowest incremental joint-Taylor loss per BOPS saved. It uses the fixed
precision-independent prune-only second-order Taylor channel ranking. Normal
candidate repair is disabled and was never invoked. The search is a bounded
single path with at most 16,384 unique proxy states.

## Fixed Linear Scale

The Stage-1 linear scale is frozen at:

```text
mapping = linear_fixed_scale
formula = J1=-0.8*(L_joint/L_scale)+0.2*R_prune
L_scale = 0.013404028918218002
quantile = 0.90
quantile_method = nearest_rank
member_count = 206
semantic_hash = fa9243cca1b09d42d92c8c23ce7e1e4e76ce0f09392e0f8ac9bb766a890915fd
file_sha256 = c0b273837408ff19650f707916140909007668a938493b72bc9c5260f3c1cefb
file_mode = 0444
```

There is no tau field. The scale is not recomputed during GA generations.

## Proxy Endpoints

All endpoints are in the primary `+/-0.005` BOPS interval. The conditional
`+/-0.0075` interval was not used.

| target | proxy BOPS | signed error | L_joint | R_prune | R_MAC | structure |
|---:|---:|---:|---:|---:|---:|---|
| 0.05 | 0.054956965 | +0.004956965 | 0.101383652 | 0.361826097 | 0.477385849 | `5a5714c3` |
| 0.10 | 0.104846135 | +0.004846135 | 0.007475357 | 0.213480076 | 0.561447501 | `adc79215` |
| 0.15 | 0.154047474 | +0.004047474 | 0.004206589 | 0.213480076 | 0.561447501 | `adc79215` |
| 0.20 | 0.204873204 | +0.004873204 | 0.002181608 | 0.213480076 | 0.561447501 | `adc79215` |
| 0.25 | 0.254855156 | +0.004855156 | 0.001262427 | 0.210132830 | 0.568324149 | `30311784` |
| 0.30 | 0.304221928 | +0.004221928 | 0.000636020 | 0.209643150 | 0.569328904 | `cb5f1986` |

The six genotype, precision, phenotype, and candidate hashes are all unique.
Targets 0.10, 0.15, and 0.20 intentionally share one physical structure while
using different precision profiles.

## Deployment Protocol

- One fresh strict-FP32 reference was built and evaluated on GPU 6.
- Endpoint build-smoke workers: GPU 6/7/1/5/0/4.
- Full-validation workers: GPU 2/3/6/7/1/5.
- Build command: typed ONNX, explicit Cast/QDQ, `--stronglyTyped --noTF32`.
- Forbidden production flags `--fp16`, `--int8`, precision constraints,
  layer precisions, and layer output types were absent.
- Each INT8 endpoint used an independent train200 EntropyCalibration2 cache.
- Smoke: 10/10, zero skip for every endpoint.
- Full validation: common manifest hash
  `f53bc4717da33ddde0e1d4d8078e1ed5cc71f0ed063d000f50cee8f016d264cb`.
- Evaluation: 1,789/1,789, zero skip, GPU IoU postprocessing, eight DataLoader
  workers, warmup 20 followed by latency reset.

## Full-Validation Results

The p50/p90/p95 values below are parallel screening measurements. They are
not formal latency and must not be used in the official latency Pareto front.

| target | realized BOPS | param retention | INT8/FP16/FP32 canonical | AP03 | AP05 | AP07 | mAP | delta mAP vs FP32 | p50/p90/p95 ms |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| FP32 | 1.000000000 | 1.000000000 | reference | 0.826146 | 0.781512 | 0.602062 | 0.736573 | 0.000000 | 7.421/8.273/9.805 |
| 0.05 | 0.054956966 | 0.638173903 | 32/38/0 | 0.794700 | 0.751606 | 0.565998 | 0.704101 | -0.032472 | 4.375/16.937/22.205 |
| 0.10 | 0.104846134 | 0.786519924 | 14/47/9 | 0.826207 | 0.781479 | 0.602128 | 0.736605 | +0.000032 | 4.392/16.751/22.773 |
| 0.15 | 0.154047469 | 0.786519924 | 6/35/29 | 0.826407 | 0.781816 | 0.602154 | 0.736792 | +0.000219 | 5.004/18.448/24.475 |
| 0.20 | 0.204873207 | 0.786519924 | 2/32/36 | 0.826339 | 0.781628 | 0.602299 | 0.736755 | +0.000182 | 5.340/18.748/24.851 |
| 0.25 | 0.254855157 | 0.789867170 | 0/32/38 | 0.826128 | 0.781677 | 0.602335 | 0.736713 | +0.000140 | 7.164/21.797/26.500 |
| 0.30 | 0.304221941 | 0.790356850 | 0/27/43 | 0.826214 | 0.781750 | 0.602363 | 0.736776 | +0.000203 | 7.455/21.264/26.481 |

No endpoint exhibited near-zero AP or shrink collapse. The 0.05 endpoint has
the expected accuracy tradeoff but remains a valid detector.

## Identity And Audit Evidence

| target | physical hash | physical params | engine hash | deployment hash | QDQ | reformat |
|---:|---|---:|---|---|---:|---:|
| 0.05 | `af186a9760b322386e250e1b66863349fbb14d4361935649e874cf7449b93fc8` | 3,487,487 | `09546004a44d134c2313fc8f9465031f2d4e236350b8c91d58888af769298639` | `286fec0981c93531ef9fa2c7bd17ab8b5bf28acec61360c8f49457d080ffa879` | 170 | 48 |
| 0.10 | `9877486b74abc78bd366fbdfa03c7f018d5e1f8aba3711882bdb078145125b89` | 4,298,167 | `db5bca818457a878d533876d1784c080d91a72939c9f81a0a2df07c95b1df2d7` | `eaec9e56e4ed3046c3cd7c33a480dcfb203f138cd0d76024382a600d256a2e1b` | 72 | 34 |
| 0.15 | `9877486b74abc78bd366fbdfa03c7f018d5e1f8aba3711882bdb078145125b89` | 4,298,167 | `5023b0a2eb89d27d44fcb60212fff26f71440466e574ea67072e932d7be8bce1` | `6e5f2618e9d1bf0e7a705662b6251e9133e13021595717f8be1df1d28666f122` | 28 | 43 |
| 0.20 | `9877486b74abc78bd366fbdfa03c7f018d5e1f8aba3711882bdb078145125b89` | 4,298,167 | `1265d1e05b786ce91dda05b51bebc2a0fd1c194d366f8f5e08f403f9c0092cc8` | `7c0f5c2f27f75d19b4fe7a21570a0521d8a4b8c89179908c7c38c62ff54ec2a1` | 10 | 49 |
| 0.25 | `9c36f769d09f0c761fed3ea44ba1493634b3500cb49b6c6197b6063876a1106b` | 4,316,459 | `e497114cdcfb6888e08b563e233c3ba29380bf09faf43cfebcc6c741be9e2426` | `14ee2fdafa2ad7febd9d9f36ef6d91739f236d56b522d79131810b66a58ee8d8` | 0 | 84 |
| 0.30 | `5caab916152f9b9e1411d0ecfcda80cc3dbc87f72a99b830e6c44356db6dfa82` | 4,319,135 | `4139eacbcd671631df97f01304327c55ae7801ded48c88ef66eea5951b608b9f` | `055a9fdad7b92dd778352fb19b53ec548f80b91ca8d6843ac9b0bbe8abb740cc` | 0 | 139 |

For every endpoint:

- `physical_plan_validation.passed=true`;
- frozen group keep/prune maps were verified;
- predicted and materialized parameter retention agree;
- physical, engine-structure, typed graph, semantic QDQ, merge, and
  pruning/quantization-group audits passed, including `/Concat_9`;
- unresolved precision count is zero;
- raw, repaired, requested, and realized precision hashes are identical;
- proxy versus realized BOPS differences are below numerical roundoff.

The canonical profile counts include the protected parameter-free affine-grid
functional MatMul. Engine realized weighted counts therefore contain one fewer
FP16 entry while preserving the same signed requested/realized profile hash.

## Failures And Remaining Work

Three earlier fresh proxy attempts are retained as failure evidence. They
identified and fixed a protected-FP16 legalizer error, shallow beam exhaustion,
and the insufficient 4,096-state bound. They are not reused by this result.
The successful run had no engine, calibration, smoke, evaluation, precision,
merge, or skip failure.

Formal single-GPU latency is intentionally pending until all later GA engines
have been built and every parallel worker has stopped. The official Pareto
fronts are also pending GA full-validation candidates and formal latency.

## Reproduction

```bash
conda run -n univ2x-opt python -m search.cli \
  --config search/configs/lidar_pyramid_4090_greedy_six_budget.yaml \
  --output-root outputs \
  --greedy-only

conda run -n univ2x-opt python -m search.cli \
  --config search/configs/lidar_pyramid_4090_greedy_six_budget.yaml \
  --output-root outputs \
  --resume outputs/4090_legal_width_greedy_six_budget_20260716_131554 \
  --stage2-only
```
