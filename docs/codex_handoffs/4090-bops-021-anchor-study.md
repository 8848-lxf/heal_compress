# 4090 BOPS 0.21 Anchor Study

Evidence timestamp: `2026-07-15 05:49:24 CST`

Branch: `feature/heal-compress-h800-sync-4090`

Experiment code commit: `4fb44fe73ce5906c84425af7b6c6acae4fd757c4`

Accepted H800 ancestor retained:
`b862b3d8ad061bd12580776226c75f564918298d`

Evidence directory:
`outputs/4090_bops_021_anchor_study_20260714_142336/`

## Verdict

The controlled study identifies two low-damage implementations of BOPS
retention near 0.21. Both anchor B and anchor C have observed mAP and AP@0.7
at least as high as the same-run strict FP32 reference while reducing forward
p50. This is a relative Pareto observation, not a formal AP admission gate.

Anchor C is the preferred primary path. It keeps every physical channel and
changes only `pg_0141` (`pyramid_backbone.deblocks.2.0`) from FP16 to INT8. It
reaches realized BOPS `0.2130358274`, mAP `0.725330`, and forward p50
`3.918936 ms`. Relative to all-FP16 anchor A, its mAP is `+0.000242` and p50 is
`-1.802677 ms`.

Anchor B proves that low-loss pure structural compression is also possible. It
prunes 72 of 256 output channels from the local
`shrink_conv.layers.0.double_conv.0` domain and reaches realized BOPS
`0.2110143492`, mAP `0.725272`, and p50 `5.663902 ms`. Its p90/p95 are noisier
and higher than A, while C is substantially faster across p50/p90/p95.

The recommended future Stage A region is therefore C-centered, with only
light late-domain pruning retained for hybrid exploration. Stage A was not
started because the proposed search-space restriction and formal AP hard gate
still require user approval, and the prior multi-GPU Top-5 gate remains false.

## Entry and production boundary audit

The entry gate was clean and satisfied:

- branch was `feature/heal-compress-h800-sync-4090`;
- local and remote entry HEAD were
  `36bf7a03e393db2750c75c9be52dd38e11c0832b`;
- the worktree was clean;
- the H800 fix commit was an ancestor, exit code 0.

All three anchors use the existing production deployment path:

- typed ONNX with explicit FP16/FP32 Cast and explicit Q/DQ for INT8;
- TensorRT `--stronglyTyped --noTF32`;
- no production `--fp16`, `--int8`, `precisionConstraints`,
  `layerPrecisions`, or `layerOutputTypes` precision control;
- no weakly typed production fallback;
- H800 semantic post-ReLU/post-merge QDQ boundaries;
- symmetric per-output-channel weights and TensorRT
  EntropyCalibration2 per-tensor activation calibration.

The EntropyCalibration2 scale-extraction engine is separate from the final
production engine. Its calibration-only TensorRT flags do not control the
strongly typed production engine; the calibrated scales are materialized as
typed graph Q/DQ tensors before the final build.

Per the follow-up instruction, plugin boundary precision was not re-optimized.
The accepted 4090 production setting, FP32, was fixed for all references and
anchors. EngineInspector reports three floating plugin inputs as `Float`, the
coordinate input as `Int32`, and the output as `Float`. PointPillarScatterTRT
matches zero quantization genes and is outside the 70 canonical compute-entry
profile and BOPS accounting.

The first generated summary displayed plugin boundary `UNKNOWN` despite
`passed=true`, because it expected exactly two floating inputs. The raw
EngineInspector evidence contained three valid Float inputs. A regression test
reproduced this report-only defect, and the summary now labels any all-Float
floating-input/Float-output passing boundary as FP32. No engine or evaluation
result was affected.

## Unified protocol

| item | fixed value |
|---|---|
| checkpoint | `net_epoch_bestval_at17.pth` |
| checkpoint SHA256 | `d20a01079cc09b1f313a37bd5ccd5174390a93f5c2b02dc9a942929e1227caca` |
| calibration | fixed train200, TensorRT EntropyCalibration2, fresh when INT8 is present |
| calibration tensor-manifest hash | `eb56308111e20ad7c789b18a8860289fe7357474722dadc43c810868554e0ec5` |
| validation manifest hash | `6f601374e573a5ed7da61eeac07259c0eea34fb5ed72d9bf52265d40a02c9f16` |
| plugin SHA256 | `91aec743dca383151b995a60d004cd254ce115ed16152873c1f46754bb15022d` |
| plugin boundary | fixed FP32, not a gene |
| GPU | physical GPU 4, RTX 4090 |
| GPU UUID | `GPU-166702d7-bf30-18e0-83ff-83b316d37c0e` |
| driver / CUDA / TensorRT | `580.105.08` / `11.8` / `10.9.0.34` |
| warmup | 20 frames, then latency collector reset |
| smoke | 10/10, zero skips, same engine later used for full measurement |
| measurement | 200/200, zero skips |
| BOPS | complete canonical weighted MAC x weight bits x activation bits |

GPU isolation passed before and after every smoke and 200-frame evaluation.
There were no foreign compute processes. Recorded telemetry at the individual
anchor gates was 0 percent instantaneous GPU utilization, 4,963 MiB used by
the study process, 40-42 C, and approximately 58-62 W. These are boundary
snapshots, not an average power measurement.

## Reference measurements

The strict FP32 reference was built and evaluated inside the same run on the
same manifest and GPU:

| reference | AP03 | AP05 | AP07 | mAP | p50 ms | p90 ms | p95 ms | frames/skips |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| strict FP32, same run | 0.810864 | 0.770644 | 0.593668 | 0.725059 | 7.308166 | 7.568362 | 7.764058 | 200/0 |
| current 4090 strongly typed E67 | 0.755723 | 0.706405 | 0.490781 | 0.650970 | 3.988057 | 4.106739 | 4.413895 | 200/0 |
| accepted H800 E67-ENT | 0.755606 | 0.703319 | 0.484407 | 0.647778 | 2.814116 | not retained here | not retained here | 200/0 |

The 4090 E67 row uses the same validation-manifest hash. The H800 row is a
historical cross-GPU reference; its AP delta is useful context, but its p50 is
not a controlled 4090 latency comparison.

Strict FP32 engine SHA256:
`9ad1c7bb6a7cc50f317f1189845eb168c8f9f75f1de2b818036cdb444e91afa2`.

## Anchor construction

### Anchor A: all-keep FP16

All pruning genes are keep and all 69 weighted quantization genes are FP16.
The protected parameter-free affine-grid MatMul remains FP16, producing a
canonical profile of 0 INT8 / 70 FP16 / 0 FP32. Scatter remains outside this
profile.

This anchor confirms the actual weighted BOPS ratio is exactly the theoretical
FP16 ratio:

`R_BOPS_theoretical = (16 x 16) / (32 x 32) = 0.25`

`R_BOPS_proxy = R_BOPS_physical = R_BOPS_realized = 0.25`

There is no difference from protected operations in the BOPS result because
the protected functional canonical entry is FP16. Scatter and non-weighted
auxiliary operations are deliberately outside the weighted BOPS denominator.
Their runtime cost is real, which is why BOPS is not treated as an engine
latency predictor.

### Anchor B: FP16 plus local structural pruning

No GA operator was invoked. The deterministic study-only inventory reused the
formal trace and exposed two late local roots:

- `shrink_conv.layers.0.double_conv.0`, 256 units;
- `shrink_conv.layers.0.double_conv.2`, 256 units.

Early backbone roots and detection heads were protected. Grouped-conv domains
were not modified. The scan retained the production alignment of 4, local
domain semantics, minimum retained ratio, and the physical pruning replay.
The configured deblock patterns did not occur as eligible physical roots in
the trace and were not synthesized.

Twelve legal physical width candidates were materialized in the BOPS interval.
The selected candidate had the lowest predicted pruning loss:

- root: `shrink_conv.layers.0.double_conv.0`;
- retained width: 184/256;
- pruned units: 72;
- proxy pruning loss: `9.42949277476685e-19`;
- physical parameters: 5,049,999, retention `0.9240973717`;
- global canonical MAC retention: `0.8440573967`.

Cached gradients and empirical Fisher diagonals were nonzero. No raw/global
ranking or undeclared L1 fallback was used. Channel repair did not alter a
single precision gene.

### Anchor C: all-keep FP16/INT8

All pruning genes remain keep. Sixty-seven legal INT8 groups were ranked by
FP16-to-per-channel-INT8 Taylor/Fisher perturbation, SQNR, MAC contribution,
and declared sensitivity priors. Selection is by MAC, not layer count.

The first legal prefix contains one group:

- group: `pg_0141`;
- module: `pyramid_backbone.deblocks.2.0`;
- canonical MACs: `17,179,869,184`;
- full canonical MAC share: `0.1971422541`;
- Taylor/Fisher perturbation: `0.0030363163`;
- SQNR loss proxy: `0.0623297152`;
- ranking score: `3.8048037938e-12`;
- weight-code saturation ratio: `0.0002460480`.

The ConvTranspose weight quantizer is symmetric, zero point 0, per-output
channel on axis 1, with a 128-element scale matching its physical output
width. Activation scales came from a fresh 200-sample IInt8EntropyCalibrator2
run; its cache was not reused.

## Complete anchor metrics

| anchor | pruning | MAC retention | INT8 MAC share | realized profile | BOPS | AP03 | AP05 | AP07 | mAP | p50 ms |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| A | none | 1.000000 | 0.000000 | 0 INT8 / 70 FP16 / 0 FP32 | 0.250000 | 0.810845 | 0.770418 | 0.594003 | 0.725088 | 5.721614 |
| B | 72/256 channels in one local shrink root | 0.844057 | 0.000000 | 0 INT8 / 70 FP16 / 0 FP32 | 0.211014 | 0.811078 | 0.770363 | 0.594374 | 0.725272 | 5.663902 |
| C | none | 1.000000 | 0.197142 | 1 INT8 / 69 FP16 / 0 FP32 | 0.213036 | 0.810842 | 0.769888 | 0.595260 | 0.725330 | 3.918936 |

Additional runtime and artifact metrics:

| anchor | p90 ms | p95 ms | FPS | params | param retention | engine bytes | Q/DQ nodes | reformats | saturation |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| A | 5.859551 | 6.677257 | 174.775866 | 5,464,791 | 1.000000 | 39,426,908 | 0 Q / 0 DQ | 98 | 0.0000000000 |
| B | 7.020254 | 8.713158 | 176.556712 | 5,049,999 | 0.924097 | 38,274,276 | 0 Q / 0 DQ | 142 | 0.0000000000 |
| C | 4.554928 | 5.580226 | 255.171271 | 5,464,791 | 1.000000 | 28,042,748 | 2 Q / 2 DQ | 27 | 0.0002460480 |

Saturation is defined here as the fraction of selected INT8 weight codes with
`abs(code) >= 127`, weighted by parameter count. It is zero when there are no
INT8 weights; it is not an activation tensor parity statistic.

All anchors completed smoke 10/10 and measurement 200/200 with zero skips.

## BOPS theory and realization

| anchor | composed theory | proxy | physical | realized | Delta repair | Delta physical | Delta realization |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 0.2500000000 | 0.2500000000 | 0.2500000000 | 0.2500000000 | 0 | 0 | 0 |
| B | 0.2110143492 | 0.2110143453 | 0.2110143492 | 0.2110143492 | 0 | +3.8835e-9 | 0 |
| C | 0.2130358274 | 0.2130358219 | 0.2130358219 | 0.2130358274 | 0 | 0 | +5.4482e-9 |

Anchor B composed theory is `0.25 x 0.8440573967`. Anchor C composed theory is
`0.25 x (1 - 0.1971422541) + 0.0625 x 0.1971422541`. The tiny proxy/physical
differences are floating-point/runtime-shape accounting differences, not a
precision or structure substitution.

The full precision-only A/B factor is 0.25. Anchor B reaches 0.21 by physical
MAC reduction; anchor C reaches it through the lower 8-bit x 8-bit factor on
19.714 percent of canonical MACs. No anchor uses metadata-only compression.

## Precision identity and deployment hashes

| anchor | raw = repaired = requested = realized | profile SHA256 | physical SHA256 | engine SHA256 |
|---|---|---|---|---|
| A | yes | `0d498418624ad79dfc071fd28bd6dc5375baaa8790d82ae87d49520659c48660` | `aed843d03dcd7b4619aad4ace865f1763ccbc136f73f0fef7e119bb060723b38` | `f2b1ae7c570ebc40174d20764f289f18332bc1869682e7358c7a3dcfd1d46776` |
| B | yes | `0d498418624ad79dfc071fd28bd6dc5375baaa8790d82ae87d49520659c48660` | `70e8f8d7fe4217195caa236bea98c8985d356dc06876986325b7a60c18b8bb73` | `4cdd2521b9e4d31a40789ea5c82405a3fa2a2f272d218a2718b5751045d7d860` |
| C | yes | `03da4b43731c91a5200336063780050f27fe667bd547f0845e54baf37c1fe1a1` | `aed843d03dcd7b4619aad4ace865f1763ccbc136f73f0fef7e119bb060723b38` | `5b996c245a1621bc22d8f0ba4a64f4c569d2bb32907cd92c04eb1090bd83bbdf` |

The raw, repaired, requested, and EngineInspector-realized hashes are equal
within every anchor. Therefore neither channel repair nor TensorRT changed the
precision profile. Anchor C shares the all-keep physical hash with A but has a
different deployment/profile/engine hash, as required.

Deployment SHA256 values are:

- A: `99327862b1ceabb6b769fcc01864582f3deb795b04682a9aa501d31aa8fd3737`;
- B: `71ceec7966dfb6f4f50b6e41a7dc53689de09e7d3a840fed242953874526b519`;
- C: `0f63b5534526c2f5180c0791282f9d97c951508b359f3e4824ac50096c1a04ab`.

QDQ topology SHA256 is
`4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`
for the no-INT8 A/B graphs and
`722a28a5b149add746f2a03afcec070c357ccb5289d80f113eb70a01f33c2a98`
for the one-group C graph. These intentionally differ from the H800 E67
67-group topology; the semantic boundary recipe is unchanged, while the
selected INT8 profile is different.

## Fail-closed audit results

All anchors report:

- canonical compute entries 70 and unresolved count 0;
- pruning/quantization ID collisions 0;
- unmapped weighted layers 0;
- duplicate canonical mappings 0;
- precision inherited from pruning scope 0;
- pre-ReLU QDQ 0;
- invalid raw Conv-output QDQ 0;
- orphan Q and DQ 0;
- duplicate QDQ 0;
- merge contract failures 0;
- unexpected FP32 fallbacks 0;
- strongly typed requested/realized profile audit passed;
- physical structure validation passed;
- realized BOPS gate passed for B and C.

`/Concat_9` is realized as FP16 for A, B, and C, with Half engine inputs and
Half engine output. In C, the INT8 deblock branch explicitly dequantizes and
casts to the FP16 merge contract. There is no mixed or undeclared FP32 merge.

The scatter plugin remains Float/Float, is absent from `layer_bitwidth`, and
does not alter the canonical precision counts. The production builder is
strongly typed for all anchors.

## Relative accuracy and latency

Controlled deltas against the same-run strict FP32 and all-FP16 A references:

| anchor | Delta mAP vs FP32 | Delta AP07 vs FP32 | Delta p50 ms vs FP32 | Delta mAP vs A | Delta AP07 vs A | Delta p50 ms vs A |
|---|---:|---:|---:|---:|---:|---:|
| A | +0.000030 | +0.000334 | -1.586552 | 0.000000 | 0.000000 | 0.000000 |
| B | +0.000213 | +0.000705 | -1.644263 | +0.000183 | +0.000371 | -0.057711 |
| C | +0.000272 | +0.001592 | -3.389229 | +0.000242 | +0.001257 | -1.802677 |

Historical context against E67:

| anchor | Delta mAP vs H800 E67 | Delta AP07 vs H800 E67 | Delta p50 ms vs H800 E67 | Delta mAP vs 4090 E67 | Delta AP07 vs 4090 E67 | Delta p50 ms vs 4090 E67 |
|---|---:|---:|---:|---:|---:|---:|
| A | +0.077310 | +0.109596 | +2.907498 | +0.074118 | +0.103221 | +1.733556 |
| B | +0.077494 | +0.109967 | +2.849786 | +0.074302 | +0.103592 | +1.675845 |
| C | +0.077552 | +0.110853 | +1.104820 | +0.074360 | +0.104479 | -0.069121 |

The H800 p50 deltas are listed only because the requested comparison requires
them. They are cross-GPU observations and must not be used to rank 4090
candidates. The controlled 4090 A/B/C comparison is the decision evidence.

## Low-damage path decision

Both B and C are low-damage observed points. C is preferred because it keeps
the physical model intact, matches B's accuracy within measurement noise, and
has a `1.744966 ms` lower p50 than B. It also has lower p90/p95 and fewer
realized reformats. Pure FP16 pruning remains useful as a secondary mechanism,
but the data do not support making it the primary budget lever.

The result favors a C-centered hybrid region rather than an unconstrained
random mix:

- use low-sensitivity INT8 MACs as the primary reduction;
- permit only light, late-domain Fisher-ranked pruning;
- exclude early backbone, heads, pillar VFE, functional affine-grid, scatter,
  and shrink-sensitive domains from unrestricted mutation;
- use exact B and C evidence to seed deterministic neighborhoods.

No hybrid B+C engine was measured in this study. The pruning x quantization
interaction therefore remains unknown and must be measured in the next
approved Top-5 smoke, not assumed to be zero.

## Proposed Stage A search changes, not applied

These values are recommendations awaiting explicit approval:

- recommended hard `R_MAC` floor: `0.95`;
- recommended full-canonical INT8 MAC share: `0.14-0.22`, always followed by
  exact realized BOPS admission in `[0.205, 0.215]`;
- pruning variables: only late, low-sensitivity local domains; keep early
  backbone, pillar VFE, detection heads, grouped-conv group count, scatter,
  affine-grid MatMul, and `shrink_conv.layers.0.double_conv.2` protected;
- precision variables: a ranked legal allowlist centered on `pg_0141`, late
  stage-2 groups, and adjacent low-sensitivity deblocks; keep early backbone,
  shrink, and heads protected unless new direct evidence supports them;
- FP32 gene: do not expose it as a free Stage A gene. The all-FP16 anchor is
  already accuracy-equivalent to FP32, and free FP32 would force additional
  pruning/INT8 damage to meet 0.21. Fixed FP32 plugin boundaries remain outside
  the gene space.

Proposed unique initial population composition:

- 35 percent no-prune C-neighborhood profiles, with exact C inserted once and
  unique adjacent MAC-ranked INT8 prefixes;
- 35 percent hybrids at `R_MAC=0.95-1.00`, combining C-like precision with
  small late-domain Fisher masks;
- 20 percent B-derived masks with channels monotonically restored until
  `R_MAC>=0.95`, followed by low-sensitivity INT8 budget projection;
- 10 percent fresh constrained candidates generated only inside the same
  protected region.

Every seed must still pass repair, uniqueness, physical materialization,
strongly typed realization, and exact BOPS admission. Exact anchor B is a
validated boundary/control point but is outside the proposed 0.95 MAC floor;
its Fisher channel order, not its exact 0.844 mask, seeds the legal hybrid
population.

Proposed Stage-1 loss model:

`L = L_prune + L_quant_incremental + L_prune_x_quant + L_MAC_weighted`

- `L_prune`: local Fisher/Taylor loss after repair, based on the physical mask;
- `L_quant_incremental`: FP16-to-INT8 Taylor/Fisher perturbation plus SQNR for
  each newly enabled group, rather than an absolute E67 score;
- `L_prune_x_quant`: conservative interaction on shared/downstream domains and
  merge branches; do not set it to zero before hybrid evidence exists;
- `L_MAC_weighted`: normalize sensitivity by the actual full-canonical MAC
  reduction, with exact BOPS still enforced separately.

Calibration plan for the proxy:

- A sets the FP16 intercept;
- B isolates observed pruning loss for one physical local-domain mask;
- C isolates observed incremental quantization loss for one all-keep profile;
- the first approved hybrid Top-5 smoke supplies the missing interaction term;
- subsequent real Stage-2 points update residual calibration without changing
  the formal target interval.

Candidate formal AP hard gate, pending user approval only:

- `mAP >= anchor_A_mAP - 0.020`, currently `0.705088`;
- `AP@0.7 >= anchor_A_AP07 - 0.030`, currently `0.564003`;
- required evaluated count and skips remain 300/0 for Stage A.

These are proposed values, not active configuration. The prior smoke
`max_map_drop=0.75` was not promoted. No Stage A code or population was
changed, and no generation was started.

## Implementation and tests

Implementation introduced by the anchor work:

- `search/anchors/bops_021.py`: deterministic A/B/C contracts, theoretical and
  realized BOPS audits, precision hash identity, MAC-ranked quantization
  sensitivity, manifest consistency, and relative Pareto observation;
- `search/anchors/runner.py`: no-GA planner, late local width scan, production
  physical/typed deployment, smoke10 plus measured200 reuse, complete evidence
  output, and corrected FP32 scatter summary;
- `search/configs/lidar_pyramid_4090_bops_021_anchors.yaml`: fixed protocol and
  fail-closed study configuration;
- `tests/test_bops_021_anchors.py`: 17 focused anchor and reporting contracts.

TDD evidence for the final report changes:

- scatter summary regression first failed `UNKNOWN != FP32`, then passed;
- relative Pareto helper first failed import, then passed with no absolute AP
  threshold;
- complete focused and related regression: `95 passed`;
- modified Python files are covered by `py_compile` in final verification;
- `git diff --check` is covered by final verification.

## Reproduction

Planning only, with no ONNX/engine/evaluation:

```bash
conda run -n univ2x-opt python -m search.anchors.runner \
  --config search/configs/lidar_pyramid_4090_bops_021_anchors.yaml \
  --output-root outputs \
  --gpu-id 4 \
  --planning-only
```

Fresh strict FP32 plus all A/B/C smoke10 and 200-frame measurements:

```bash
conda run -n univ2x-opt python -m search.anchors.runner \
  --config search/configs/lidar_pyramid_4090_bops_021_anchors.yaml \
  --output-root outputs \
  --gpu-id 4
```

Related regression:

```bash
conda run -n univ2x-opt pytest -q \
  tests/test_bops_021_anchors.py \
  tests/test_search_bops_budget.py \
  tests/test_search_bops_proxy.py \
  tests/test_search_final_contract.py \
  tests/test_search_quantization_groups.py \
  tests/test_search_baseline_precision_validation.py \
  tests/test_search_stage2_realized_bops.py \
  tests/test_search_generation_stage2.py \
  tests/test_strongly_typed_builder.py \
  tests/test_strongly_typed_qdq_graph.py \
  tests/test_search_merge_realization.py \
  tests/test_global_physical_prune_plan_v94.py \
  tests/test_search_full_val_manifest.py \
  tests/test_search_stage2_candidate_artifacts.py \
  tests/test_search_stage2_physical_validation.py
```

`ANCHOR_FP16_BASELINE_PASS = true`

`ANCHOR_FP16_PRUNING_PASS = true`

`ANCHOR_MIXED_NO_PRUNE_PASS = true`

`LOW_DAMAGE_BOPS_021_PATH_IDENTIFIED = true`

`MULTIGPU_TOP5_SMOKE_PASS = false`

`STAGE_A_ALLOWED = false`

`STAGE_A_STARTED = false`

`STAGE_B_ALLOWED = false`
