# 4090 strongly typed multi-GPU Stage-2 Top-5 smoke report

Date: 2026-07-15 CST  
Branch: `feature/heal-compress-h800-sync-4090`  
Code commit: `1f2c6a1`  
Required H800 ancestor: `b862b3d8ad061bd12580776226c75f564918298d`  
Run: `outputs/4090_ga_qdq_stage2_multigpu_smoke_20260714_125402/`

## Scope

This run is the required pre-Stage-A smoke after strongly typed E67 readiness.
It used a production-sized Stage-1 population (1024 initial, 512 population,
512 offspring), one generation, target BOPS 0.21 +/- 0.005 and real Top-5
backfill. Stage-1 ran on GPU5. Persistent Stage-2 workers ran on physical GPUs
4/5/6/7.

## GPU and reference evidence

| GPU | UUID | strict-FP32 mAP | fixed manifest hash |
| --- | --- | ---: | --- |
| 4 | `GPU-166702d7-bf30-18e0-83ff-83b316d37c0e` | 0.802498 | `c827031ab82bb1925f48ada20dd64d0fa395dbf8d919d04137e95a5d3f35aee4` |
| 5 | `GPU-d4b8342a-7038-f567-ce40-a4705b23854b` | 0.802498 | same |
| 6 | `GPU-d98c7029-672d-47b6-acc2-8c5936495349` | 0.802498 | same |
| 7 | `GPU-a4b5f0d9-c77c-2c99-b4c3-1b3811b835e3` | 0.802498 | same |

All workers passed isolation, built same-GPU strict FP32/FP16 references, and
wrote clean stop markers. No worker or GPU process remained after failure.

## Admission results

Stage-1 found 24 unique legal raw candidates in its proxy BOPS interval. The
post-repair BOPS gate retained 17 and rejected 7 before engine construction:

- legal repaired phenotypes: 24;
- post-repair BOPS eligible: 17;
- post-repair BOPS ineligible: 7;
- physical Stage-2 deployments attempted: 17;
- engine-realized BOPS out of budget: 4;
- accuracy hard-gate failures: 11;
- accepted unique physical/deployment pairs: 2;
- required: 5.

Every attempted candidate independently performed physical pruning, typed ONNX
and explicit-QDQ generation, fresh train200 EntropyCalibration2, strongly typed
engine build/deserialization and realization audits. Thirteen candidates ran
10/10 measured frames with zero skips; four failed the realized BOPS gate before
evaluation.

## Accepted candidates

| Stage-1 rank | physical hash | deployment hash | BOPS retention | mAP | realized precision |
| ---: | --- | --- | ---: | ---: | --- |
| 2 | `2e7166f254a5f7a4bdc69943c0eca28f9bb0133f474177abc79dd60f513877b7` | `086df989dbca5b28d2af0aac1b5d37ae128385aa22e8608884f0ce11afe4f1c8` | 0.211216 | 0.334322 | 21 INT8 / 29 FP16 / 19 FP32 |
| 10 | `3ac09e2047819f85bbd77e398a552d73c6322061ca2ace9d8003f08f3fe8b822` | `021de0b9eb45d0c7252f71cdfbd97bdb7dca9d056be38aa3b0074a993b83038a` | 0.208798 | 0.101695 | 22 INT8 / 29 FP16 / 18 FP32 |

The framework did not duplicate these candidates to fill the missing slots.
It terminated with:

`insufficient_unique_deployable_candidates:2<5`

## Loss attribution

Two same-physical-model strongly typed FP32 diagnostics used the same manifest:

- a mixed candidate with mAP 0.000000 recovered to FP32 mAP 0.114810;
- the accepted mAP 0.334322 candidate recovered to FP32 mAP 0.516949;
- both pairs retained identical physical hashes and parameter counts;
- all physical, QDQ boundary, precision, merge and engine-structure audits
  passed.

Physical pruning therefore causes material AP loss, and the candidate precision
profiles add further loss. The available evidence does not support weakening the
AP gate or treating the result as a typed-engine realization success.

## Decision

`STRONGLY_TYPED_E67_READY_FOR_GA = true`  
`MULTIGPU_TOP5_SMOKE_PASS = false`  
`STAGE_A_ALLOWED = false`  
`STAGE_A_STARTED = false`  
`STAGE_B_ALLOWED = false`

Stage A was not started. The next search change must increase the number of
repair-legal, realized-BOPS-valid, accuracy-valid candidates without weakening
the BOPS or AP admission rules. Any genotype, repair, Stage-1 objective or BOPS
change requires another fresh generation-0 smoke.

--- Report completed: 2026-07-15 04:42:23 CST ---
