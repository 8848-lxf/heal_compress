# 4090 Multi-GPU Stage-2 Integration

Status timestamp: 2026-07-14 19:20 CST.

## Scope and scheduling contract

The per-generation Stage-2 path now supports a persistent process pool with one
worker per physical GPU. The Stage-A configuration assigns GPUs 4, 5, 6 and 7:

- Stage-1 remains on physical GPU 5;
- after each generation is ranked, Stage-2 candidates are submitted in waves of
  four;
- each worker owns its CUDA model context, TensorRT subprocesses, strict-FP32 AP
  reference, strict-FP16 latency reference, artifact cache and real-evaluation
  cache;
- the coordinator consumes completed wave results in original Stage-1 rank
  order, then applies physical/deployment identity uniqueness and backfill;
- one worker process remains alive across all generations in the budget, so its
  context, references and exact-candidate cache are reused;
- workers close before the 500-frame budget-final evaluation so the final GPU
  has no resident Stage-2 competitor.

This is process isolation, not threads sharing one CUDA context. TensorRT build
and evaluation subprocesses continue to receive the assigned physical GPU via
their own `CUDA_VISIBLE_DEVICES` setting.

## Code changes

- `search/orchestration/stage2_process_pool.py` implements worker startup,
  readiness, atomic file queues, deterministic ordered result collection,
  timeout/crash propagation, exact candidate reuse and process cleanup.
- `search/stage2/candidate_worker.py` builds one formal lidar-pyramid context and
  `LidarPyramidRealEvaluator` per GPU, performs strict startup preflight, lazily
  creates same-GPU FP32/FP16 references, evaluates queued candidates and writes
  fail-closed result JSON.
- `search/orchestration/generation_stage2.py` accepts ranked parallel waves while
  preserving the existing uniqueness, finite-F2, failure and Top-5 gates.
- `search/orchestration/lidar_pyramid_search.py` creates the configured pool,
  sends repaired phenotype tasks, records worker GPU/PID metadata and closes the
  pool before budget final.
- `search/integration/lidar_pyramid_context.py` adds an explicit
  `allowed_gpu_pids` set. GPU 5 permits only the current GA controller PID while
  GPUs 4, 6 and 7 permit no foreign PID.
- `search/stage2/lidar_pyramid_real_evaluator.py` and
  `search/orchestration/budget_final.py` carry the same explicit PID allowlist
  into pre/post evaluation isolation gates.
- `search/configs/lidar_pyramid_4090_ga_stage_a.yaml` enables the four workers,
  declares timeouts and explicitly authorizes only the controller on Stage-1
  GPU 5.

## Determinism and cache behavior

`deploy_generation_with_backfill` evaluates a wave concurrently but handles its
results in Stage-1 rank order. Parallel completion order cannot change admission
or the generation winner. Failed candidates, missing identities, duplicate
physical/deployment identities and non-finite F2 remain fail-closed.

An exact candidate seen again in the same budget is returned from the process
pool cache with `pool_cache_hit`, `reused_pool_task_id` and the original worker
metadata. GPU-local evaluator caches remain separated because latency and the
strict-FP16 reference are GPU-specific. No H800 latency, engine, calibration or
plugin artifact is used.

## Verification

- focused multi-GPU/process/backfill/config suite: 18 passed;
- formal search/package suite: 313 passed, 7 known environment warnings;
- diagnostic broad repository suite: 801 passed and 4 pre-existing unrelated
  failures in grouped-conv v93 and legacy latency-LUT assertions;
- all modified Python files passed `python -m py_compile`;
- `git diff --check` passed before report generation.

Live startup smoke:
`outputs/4090_stage2_pool_startup_smoke_20260714_043413/`.

All four workers reached `ready` concurrently and exited cleanly:

| physical GPU | UUID | worker memory at ready | external PID allowlist |
| ---: | --- | ---: | --- |
| 4 | `GPU-166702d7-bf30-18e0-83ff-83b316d37c0e` | 850 MiB | empty |
| 5 | `GPU-d4b8342a-7038-f567-ce40-a4705b23854b` | 850 MiB | controller PID only |
| 6 | `GPU-d98c7029-672d-47b6-acc2-8c5936495349` | 850 MiB | empty |
| 7 | `GPU-a4b5f0d9-c77c-2c99-b4c3-1b3811b835e3` | 850 MiB | empty |

After shutdown the pool manifest was `stopped`, all four worker return codes
were zero, there was no `search.stage2.candidate_worker` process and GPUs 4-7
returned to 0 percent utilization. This smoke initialized formal contexts only.
It did not build a candidate engine and is not Stage-A evidence.

## Gate status

The multi-GPU infrastructure is implemented and locally validated, but the
fresh E67 readiness engine still realizes `/Concat_9` as FP32/mixed on this
4090 TensorRT 10.9 stack. Therefore:

`READY_FOR_GA = false`

`STAGE_A_STARTED = false`

--- Multi-GPU integration round completed: 2026-07-14 19:20:45 CST ---
