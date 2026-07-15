# 4090 constrained generation-0 Top-5 smoke report

## Decision

The approved C-centered search region is implemented and the fresh four-GPU
generation-0 smoke passed. Five unique candidates completed independent
physical materialization, typed ONNX export, EntropyCalibration2, strongly
typed TensorRT build, smoke10, and the fixed 200-frame evaluation. All five
passed the BOPS, R_MAC, INT8-MAC-share, AP, precision-identity, physical,
Q/DQ, and merge gates.

This result unlocks Stage A but does not start it. The user explicitly required
another approval boundary after this smoke.

## Repository and entry audit

- repository: `/home/lixingfeng/UniAD_examine/heal_compress`
- branch: `feature/heal-compress-h800-sync-4090`
- requested entry HEAD: `12d9b56a708937e00d0649404981ae11f346f37d`
- constrained-search implementation: `ecbf19f7456ac2184134d0b37a752f78ba667fbc`
- experiment HEAD: `8a938c42b7d53965a5ac1eba59b4f1cd53d12a10`
- required H800 ancestor: `b862b3d8ad061bd12580776226c75f564918298d`
- ancestor check: exit 0
- final report delivery commit: the commit containing this file; use
  `git log -1 --oneline` after synchronization

The experiment used the committed config
`search/configs/lidar_pyramid_4090_ga_stage2_constrained_smoke.yaml` and fresh
output directory:

`outputs/4090_ga_qdq_stage2_constrained_smoke_20260714_164648/`

The production build request records `strongly_typed=true`, `no_tf32=true`,
`enable_fp16=false`, `enable_int8=false`, and `precision_constraints=none`.
The actual `trtexec` command contains `--stronglyTyped --noTF32` and contains
none of `--fp16`, `--int8`, `--precisionConstraints`, `--layerPrecisions`, or
`--layerOutputTypes`. Precision is expressed only by typed ONNX Cast and
explicit Q/DQ. No weakly typed fallback was admitted.

PointPillarScatterTRT stayed at the accepted FP32 boundary. It is absent from
the precision genes and canonical BOPS rows. Plugin SHA256 is
`91aec743dca383151b995a60d004cd254ce115ed16152873c1f46754bb15022d`.
TensorRT is `10.9.0.34`, CUDA is `11.8`, GPU architecture is SM89, and the
driver is `580.105.08`.

## Implemented search space

The implementation adds a separate config and constrained package; it does
not modify the tracer or core pruning dependency graph.

- hard `R_MAC >= 0.95`, computed over full canonical weighted MAC;
- hard full-canonical INT8 MAC share in `[0.14, 0.22]`;
- hard proxy, repaired, physical, and realized BOPS interval
  `[0.205, 0.215]`;
- precision genes are FP16/INT8 only, default FP16; free FP32 mutation is
  rejected;
- only `shrink_conv.layers.0.double_conv.0` is exposed as the late local
  pruning root, with alignment 4 and at most 20 light-pruned units;
- pruning propagation still reaches the dependent input of
  `shrink_conv.layers.0.double_conv.2` through the existing formal closure;
- protected early backbone, VFE, scatter, heads, affine-grid functional
  MatMul, sensitive grouped-conv domains, and unapproved roots are unchanged;
- exact Anchor C is inserted once; Anchor B contributes its Fisher order only
  after channels are restored above the R_MAC floor;
- smoke10 is operational only; the formal mAP/AP07 gate is applied only to the
  fixed 200-frame result;
- Stage A auto-start is disabled even after a passing verdict.

The runtime INT8 audit considered all 69 parameterized canonical groups and
wrote one row per group with canonical MAC/share, Taylor/Fisher perturbation,
SQNR loss, sensitivity prior, ranking score, legality, selection, and rejection
reason. It selected 24 legal late groups from a total canonical MAC of
`87,144,530,560`. `pg_0141` remained first priority with canonical MAC
`17,179,869,184`, share `0.1971422541`, Taylor/Fisher perturbation
`0.0028043917`, SQNR loss `0.0623297152`, and legal INT8 status true.

The Stage-1 proxy is now explicitly decomposed as `L_prune +
L_quant_incremental + L_prune_x_quant_prior + L_MAC_weighted`, followed by
hard constraints. The interaction term is conservative and was not fitted or
changed online during this run.

## Source changes

- constrained configuration and contracts:
  `search/configs/lidar_pyramid_4090_ga_stage2_constrained_smoke.yaml`,
  `search/constrained/{context,policy,population}.py`;
- Stage-1 generation, repair, hard gates, and proxy decomposition:
  `search/ga/engine.py`, `search/proxy/{bops_proxy,gpu_batch_proxy,objective}.py`,
  `search/stage1/repair_selection.py`;
- per-generation parallel deployment and final unlock verdict:
  `search/orchestration/{generation_stage2,lidar_pyramid_search}.py`;
- persistent worker, two-level evaluation, physical/realized BOPS, precision
  identity, and AP admission:
  `search/stage2/{candidate_worker,lidar_pyramid_real_evaluator,objective,realized_bops}.py`;
- regression coverage:
  `tests/test_constrained_stage_a_search.py`,
  `tests/test_search_gpu_batch_integration.py`, and
  `tests/test_search_stage2_candidate_worker.py`.

The first live attempt exposed a real manifest-context bug: smoke10 was given
the fixed 200-frame manifest and failed closed with
`eval_manifest_count_mismatch:warmup=20!=10:eval=200!=10`. No result from that
attempt was accepted. Commit `8a938c4` adds a dedicated deterministic
smoke10 manifest, restores the full200 context in `finally`, and includes a
regression test. The final run starts fresh from that commit.

## Population supply and Stage-1

Requested and accepted initial population composition is exact:

| family | requested | accepted | fraction |
|---|---:|---:|---:|
| C-neighborhood | 358 | 358 | 34.96% |
| hybrid | 358 | 358 | 34.96% |
| B-derived restored-Fisher | 205 | 205 | 20.02% |
| constrained fresh | 103 | 103 | 10.06% |
| total | 1024 | 1024 | 100.00% |

Seed construction counters:

- generated unique genotype proposals: 5,117;
- repair legal: 5,117;
- R_MAC eligible: 4,118;
- INT8-MAC-share eligible: 1,979;
- legalized-BOPS eligible: 1,365;
- infeasible proposal rejection: 3,094;
- duplicate proposal rejection: 205;
- accepted unique genotypes: 1,024;
- exact Anchor C count: 1;
- precision illegal count: 0;
- precision repair identity failure count: 0.

Generation-0 processed all 1,024 candidates on GPU5 in eight CUDA proxy
batches. All 1,024 were repair legal and passed the post-repair R_MAC,
INT8-share, and legalized-BOPS gates. They represent 35 unique pruning plans;
989 duplicate physical-plan proposals were removed before deployment. This is
a physical-plan dedup count, not an AP or compression failure count: precision
profiles can differ while sharing a pruning plan.

B-derived seeds were present in the requested 20% population but did not rank
into the first eight deployments. That is a Stage-1 ranking outcome, not a
repair, BOPS, engine, or AP failure.

## Four-GPU deployment evidence

At startup GPUs 4/5/6/7 had no foreign processes and respectively 5/1/1/1 MiB
used. Their UUIDs were:

| GPU | UUID |
|---:|---|
| 4 | `GPU-166702d7-bf30-18e0-83ff-83b316d37c0e` |
| 5 | `GPU-d4b8342a-7038-f567-ce40-a4705b23854b` |
| 6 | `GPU-d98c7029-672d-47b6-acc2-8c5936495349` |
| 7 | `GPU-a4b5f0d9-c77c-2c99-b4c3-1b3811b835e3` |

One persistent worker ran on each physical GPU. Tasks 1-4 were assigned to
4/5/6/7 and tasks 5-8 were again assigned to 4/5/6/7. Every worker built its
same-GPU strict references, then each candidate independently built its
physical checkpoint, typed ONNX, Q/DQ graph, fresh train200 calibration cache,
and engine. Candidate calibration caches were not shared.

All eight deployed candidates passed smoke10 and full200. The dispatcher kept
five ranked unique candidates and recorded the other three as passing
speculative deployments caused by the four-wide batch. There were zero
Stage-2 failure records. All workers exited with return code 0, the pool
manifest is `stopped`, no worker/calibration/evaluation process remained, and
GPUs 4/5/6/7 returned to 5/1/1/1 MiB.

The common train200 frame-manifest hash is
`eb56308111e20ad7c789b18a8860289fe7357474722dadc43c810868554e0ec5`.
Candidate-specific calibration/deployment hashes differ as required by their
physical ONNX and precision profile. The common validation200 manifest hash is
`6f601374e573a5ed7da61eeac07259c0eea34fb5ed72d9bf52265d40a02c9f16`.

## Accepted Top-5

All pruning is confined to the local
`shrink_conv.layers.0.double_conv.0` root. Physical and realized BOPS are equal
because no precision substitution occurred. The proxy/physical differences
are only floating-point reporting differences below `6e-9`.

| rank | seed family | retained width | R_MAC | INT8 groups | INT8 MAC share | BOPS proxy | BOPS physical | BOPS realized | AP03 | AP05 | AP07 | mAP | p50 | p90 | p95 |
|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | hybrid | 244/256 | 0.974010 | pg_0141 | 0.197142 | 0.206538 | 0.206538 | 0.206538 | 0.810829 | 0.769633 | 0.595563 | 0.725342 | 4.184 | 10.948 | 15.683 |
| 2 | hybrid | 248/256 | 0.982673 | pg_0141 | 0.197142 | 0.208704 | 0.208704 | 0.208704 | 0.810820 | 0.769339 | 0.595289 | 0.725149 | 4.412 | 13.201 | 17.482 |
| 3 | hybrid | 252/256 | 0.991337 | pg_0141 | 0.197142 | 0.210870 | 0.210870 | 0.210870 | 0.810821 | 0.769993 | 0.595010 | 0.725275 | 5.247 | 12.326 | 16.520 |
| 4 | C-neighborhood | 256/256 | 1.000000 | pg_0141 | 0.197142 | 0.213036 | 0.213036 | 0.213036 | 0.811336 | 0.770351 | 0.594496 | 0.725394 | 4.670 | 12.043 | 19.173 |
| 5 | constrained fresh | 248/256 | 0.982673 | pg_0077, pg_0086, pg_0141 | 0.206383 | 0.206971 | 0.206971 | 0.206971 | 0.811268 | 0.770841 | 0.597449 | 0.726520 | 5.240 | 12.549 | 17.708 |

Every row is 200/200 with zero skips and FP32 scatter I/O. Ranks 1-4 realize
one INT8 / 69 FP16 canonical entries; rank 5 realizes three INT8 / 67 FP16.
The parameter-free protected affine-grid MatMul accounts for the one-entry
difference between the 70-entry engine profile and the 69 parameterized
weighted rows reported by the physical BOPS audit. Unresolved and unexpected
FP32 fallback counts are zero.

Q/DQ counts are 4 for ranks 1-4 and 16 for rank 5. Reformat counts are
69/41/40/47/52. Weight-code saturation ratios are `0.0002460480` for ranks
1-4 and `0.0005317802` for rank 5. Physical validation, semantic Q/DQ boundary,
per-channel scale lineage, strongly typed realization, every merge contract,
and `/Concat_9` all pass for all five.

The rank-1 row is the generation winner under the current F2 objective because
of its lower same-run latency. Rank 5 has the highest AP measurements, but this
smoke is an admission and parallelism gate, not a final model selection.

## Identity evidence

| rank | candidate hash | physical hash | deployment hash | engine SHA256 |
|---:|---|---|---|---|
| 1 | `41855cd36ddf9a4cc804c559f9fa1b22e7851254241c6fbb48ab326d895ef6f9` | `e3e9b791168f17c44b5520af64cbae7fa2a8bd76453454480fa7fddb600785f8` | `d91aef5e5db41e0c6efe9057811d341ae280ae10107d37b41e3c77abc45a6a95` | `a3f6edee10ff63e0180226b818ca76040e2b3a0eb6eadfd327b74c0c0c1116ae` |
| 2 | `5596bc0ed970c62d7706424412597265fa7cb1f134e32c910e0c6a62851a826e` | `3b51592f7c4a5d0525ed25aa210b7166feca464913cc3dc41e1c7aec573924a2` | `da44b52fc95628e237c56805989ab3c09607bb8f6a30c580bd021bc39a0d2ebe` | `90c580a289c0b3761916a29ac518756d0fcc1067ec47509e8a3c51ff4ea481f1` |
| 3 | `dabd48b53e08bc8508de84baab6ff15a4f790effdd96c2cac7d4533d09e4d242` | `8fe7cb38b7e9a293107b91668b3516760942a8127eda2449d7833a2104590906` | `ab46d78213d610d0744f5dd45f04834ec2d70f269ed17a9b7116370603370249` | `0b394b21feb37ee2cb77b78e1a64eb8a40eeadb3dcb62eca98f0cd5ac8524d01` |
| 4 | `08d11e11106601d0abad353d4d2efe1725102cab0805a68d79d3b91118c5b82e` | `aed843d03dcd7b4619aad4ace865f1763ccbc136f73f0fef7e119bb060723b38` | `0d08c51710066eecad7fde04af0279c442198419fb78677bedef33448f47dc20` | `36b0cc9dfa1bf6e53b6f39b6c829e21df96dba51b4ffddce0b7072feb9d9a955` |
| 5 | `e451c08897ec45399f737db9be2f36e018704ce890edc1ccdcc04b68070f1171` | `f51d34afe8b454a10e2a6fad42f92d595639bdc8f835cf211b115ba5dbbe7956` | `8c51831a1014085b0ff8d14a4396baa22e94ede85db87b22d74cbb82c65adca7` | `d798f4e45c7d708360f1f5b37834d84866fe10d3f5afbc7aba0455945c9ff44a` |

Ranks 1-4 share raw/repaired/requested/realized precision hash
`03da4b43731c91a5200336063780050f27fe667bd547f0845e54baf37c1fe1a1`.
Rank 5 shares hash
`1a4cf24a22749c1b3da398ff8dd8531daa830dd3626893461f03061e2bec1990`
across those same four stages. Thus pruning repair changed no precision gene,
and TensorRT changed no requested precision profile.

The five candidate, physical, deployment, and engine hash sets each have
cardinality five.

## Relative measurements

Each candidate is compared with the strict FP32 reference built and measured
on its own worker GPU. The tuple format is `delta mAP / delta AP07 / delta p50
ms`.

| rank | vs same-GPU strict FP32 | vs Anchor A | vs Anchor C | vs current 4090 E67 |
|---:|---|---|---|---|
| 1 | -0.000205 / +0.001361 / -5.924 | +0.000254 / +0.001560 / -1.537 | +0.000012 / +0.000303 / +0.265 | +0.074372 / +0.104782 / +0.196 |
| 2 | +0.000085 / +0.001547 / -4.318 | +0.000061 / +0.001286 / -1.310 | -0.000181 / +0.000029 / +0.493 | +0.074179 / +0.104508 / +0.424 |
| 3 | +0.000061 / +0.001128 / -3.096 | +0.000187 / +0.001007 / -0.474 | -0.000055 / -0.000250 / +1.328 | +0.074305 / +0.104229 / +1.259 |
| 4 | -0.000259 / -0.000156 / -3.689 | +0.000306 / +0.000493 / -1.051 | +0.000064 / -0.000764 / +0.752 | +0.074424 / +0.103715 / +0.682 |
| 5 | +0.000973 / +0.003247 / -4.869 | +0.001432 / +0.003446 / -0.482 | +0.001190 / +0.002189 / +1.321 | +0.075550 / +0.106668 / +1.252 |

All AP deltas are inside ordinary 200-frame run-to-run variation while every
candidate clears the approved absolute gates by a wide margin. Concurrent
p90/p95 values are noisier than the earlier single-GPU Anchor C measurement,
so future generation winner selection should keep using p50 and preserve the
same-GPU reference protocol.

## Seed-family outcomes and interaction observation

Stage-2 attempted five hybrid, one C-neighborhood, two constrained-fresh, and
zero B-derived candidates. All attempted candidates passed. The accepted
Top-5 contains three hybrid, one C-neighborhood, and one constrained-fresh
candidate. No family had a repair, BOPS, smoke10, AP, engine, Q/DQ, merge,
precision, or process-pool failure. B-derived had no Stage-2 denominator
because its candidates ranked below the first two four-wide batches.

Observed interaction is diagnostic only:

| rank | measured hybrid loss | predicted prune loss | predicted incremental quant loss | observed interaction loss |
|---:|---:|---:|---:|---:|
| 1 | -0.000254 | approximately 0 | 0.004542 | -0.004796 |
| 2 | -0.000061 | approximately 0 | 0.004542 | -0.004604 |
| 3 | -0.000187 | approximately 0 | 0.004542 | -0.004729 |
| 4 | -0.000306 | approximately 0 | 0.004542 | -0.004849 |
| 5 | -0.001432 | approximately 0 | 0.006849 | -0.008281 |

Negative observed interaction here means the measured 200-frame loss was
smaller than the conservative additive proxy prediction. It does not prove a
beneficial causal interaction because the proxy terms and AP loss are not yet
calibrated to a common statistical scale. The next proxy-calibration step
should retain these eight real measurements, fit only between generations,
and validate out of sample. No online scoring change was made in this smoke.

## Verification

- `pytest -q tests/test_search_*.py tests/test_constrained_stage_a_search.py`:
  184 passed, 6 environment warnings;
- every changed Python file passed `python -m py_compile`;
- `git diff --check` passed;
- the final experiment process exited 0;
- no residual Stage-2, calibration, evaluation, or controller process remains;
- generated ONNX, engines, caches, checkpoints, and output summaries remain
  ignored and are not part of the Git delivery.

## Reproduction

Fresh generation-0 smoke:

```bash
conda run -n univ2x-opt python -m search.cli \
  --config search/configs/lidar_pyramid_4090_ga_stage2_constrained_smoke.yaml \
  --output-root outputs
```

Focused verification:

```bash
conda run -n univ2x-opt pytest -q \
  tests/test_search_*.py tests/test_constrained_stage_a_search.py
git diff --check
git merge-base --is-ancestor \
  b862b3d8ad061bd12580776226c75f564918298d HEAD
```

The next action, only after explicit user confirmation, is to derive the
formal five-generation Stage A config from this committed constrained space
and start it fresh from generation 0. Stage B remains prohibited.

STRONGLY_TYPED_E67_READY_FOR_GA = true

ANCHOR_LOW_DAMAGE_PATH_IDENTIFIED = true

CONSTRAINED_SEARCH_SPACE_APPLIED = true

MULTIGPU_TOP5_SMOKE_PASS = true

STAGE_A_ALLOWED = true

STAGE_A_STARTED = false

STAGE_B_ALLOWED = false

--- Round 18 report completed: 2026-07-15 08:41:40 CST ---
