# 4090 CoBEVT Head-Dimension TensorRT Capability

## Scope

Branch: `feature/cobevt-head-dim-trt-capability`

Base: `a9c51511fa5b792b68b071a7ec740de6bb371ba5`

This branch builds an empirical TensorRT 10.9 capability matrix for CoBEVT
Attention. It is isolated from the Pyramid GA worktree and does not change the
Pyramid search or deployment path.

## Implemented Architecture

- `search/model_families/lidar_cobevt/head_dim_capability.py`
  defines immutable uniform, QK-only, and V-only candidate identities. Candidate
  hashes own graph kind, `d_qk`, `d_v`, precision profile, TensorRT version, and
  GPU architecture.
- `search/model_families/lidar_cobevt/head_dim_synthetic.py`
  exports canonical core/projection Attention graphs for P0 FP32, P1 native
  FP16, P2 F3, P3 INT8 projections, and P4 native INT8 requests. Explicit Q/DQ
  uses deterministic per-tensor activation and per-output-channel weight scales.
- `search/model_families/lidar_cobevt/head_dim_materializer.py`
  provides capability-only real-model resizing. Widths above 32 use zero-padded
  coordinates; Q/K surviving rows compensate the changed `1/sqrt(d_qk)` scale.
  This is a same-function control and is not a trained expansion recipe.
- `search/integration/lidar_cobevt_head_dim_runtime.py`
  separates production-engine timing from diagnostic-engine intermediate tensor
  parity. Diagnostic engines are never latency eligible.
- `search/reporting/cobevt_head_dim_capability.py`
  parses EngineInspector provenance, distinguishes primitive/partial/complete MHA
  fusion, keeps unknown accumulator precision unknown, calculates numerical
  parity, and derives a fail-closed search contract.
- `search/orchestration/lidar_cobevt_head_dim_capability.py`
  owns environment provenance, fresh export/build/runtime phases, deterministic
  eight-way sharding, same-shape P0 references, real-model structure inventory,
  and matrix assembly.
- `search/orchestration/lidar_cobevt_attention_pruning.py`
  reuses the established fixedK29696 CoBEVT exporter, strongly typed builder,
  F3 boundary contract, and GPU evaluator for capability-only structures.

## Candidate Matrix

Synthetic candidates: 282.

- Uniform widths: `4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 128`.
- QK-only widths: `8, 12, 16, 20, 24, 28, 32, 40, 48, 64` with `d_v=32`.
- V-only widths: `8, 12, 16, 20, 24, 28, 32, 40, 48, 64` with `d_qk=32`.
- Graphs: `core_attention` and `projection_attention`.
- Profiles: P0/P1/P2 on all applicable graphs; P3 on projection graphs; P4 on
  uniform core and projection graphs.

Real CoBEVT structures: 21 (`3 families x 7 representative widths`). All 21
strictly load the source checkpoint and physically replace six Attention modules.
The structure audit passed 21/21 without creating a CUDA context.

## Current Evidence

Development output:

`/data/lxf/heal_data/outputs/cobevt_head_dim_trt_capability_20260719_113405`

- 282/282 production ONNX exports passed.
- 282/282 diagnostic ONNX exports passed.
- 21/21 real CoBEVT physical structure audits passed.
- No TensorRT capability engine has been built in this development directory.
- The directory is not the final provenance directory because code changed after
  its initial `run_manifest.json`; final execution must use a fresh timestamp.

GPU execution is currently blocked by the active Pyramid six-budget GA. Every
RTX 4090 has a Pyramid persistent worker, and several cards also run Pyramid
calibration or `trtexec`. The capability CLI rejects any GPU containing a Pyramid
process even when instantaneous utilization is low.

## Fresh Reproduction

Set `OUT` to a new timestamped path after all code is committed and an RTX 4090
is genuinely idle.

```bash
OUT=/data/lxf/heal_data/outputs/cobevt_head_dim_trt_capability_$(date +%Y%m%d_%H%M%S)
PY=/home/lixingfeng/anaconda3/envs/univ2x-opt/bin/python

$PY -m search.orchestration.lidar_cobevt_head_dim_capability \
  --phase prepare --output-dir "$OUT" --physical-gpu 2
$PY -m search.orchestration.lidar_cobevt_head_dim_capability \
  --phase export --output-dir "$OUT" --trace-window-groups 1
$PY -m search.orchestration.lidar_cobevt_head_dim_capability \
  --phase prepare-real --output-dir "$OUT"
$PY -m search.orchestration.lidar_cobevt_head_dim_capability \
  --phase audit-real-structure --output-dir "$OUT"
```

When eight GPUs are idle, build production and diagnostic graphs with matching
shards. Each process must use the shard index as its physical GPU.

```bash
for GPU in 0 1 2 3 4 5 6 7; do
  /home/lixingfeng/anaconda3/envs/modelopt/bin/python \
    -m search.orchestration.lidar_cobevt_head_dim_capability \
    --phase build --output-dir "$OUT" --physical-gpu "$GPU" \
    --shard-index "$GPU" --shard-count 8 >"$OUT/logs/build_${GPU}.log" 2>&1 &
done
wait

for GPU in 0 1 2 3 4 5 6 7; do
  /home/lixingfeng/anaconda3/envs/modelopt/bin/python \
    -m search.orchestration.lidar_cobevt_head_dim_capability \
    --phase build-diagnostic --output-dir "$OUT" --physical-gpu "$GPU" \
    --shard-index "$GPU" --shard-count 8 >"$OUT/logs/build_diag_${GPU}.log" 2>&1 &
done
wait
```

Run all synthetic timings serially on one idle GPU so latency values are
comparable:

```bash
/home/lixingfeng/anaconda3/envs/modelopt/bin/python \
  -m search.orchestration.lidar_cobevt_head_dim_capability \
  --phase runtime --output-dir "$OUT" --physical-gpu 2
$PY -m search.orchestration.lidar_cobevt_head_dim_capability \
  --phase assemble --output-dir "$OUT"
```

Real CoBEVT builds/evaluations reuse
`search.orchestration.lidar_cobevt_attention_pruning` with
`$OUT/real_cobevt`, strict FP32, strict FP16, and
`F3_rest_fp16_qk_fp32_minimal_island`. Smoke10 precedes fixed50/fixed500;
fixed500 is only run for the specified boundary widths.

## Validation

Latest focused regression: 89 passed. Warnings are existing PyTorch tracing,
meshgrid, timm, and matplotlib deprecation warnings.

No Pyramid process was modified, paused, killed, or restarted.

---

Round 1 completed at 2026-07-19 12:19:06 Asia/Shanghai.

## Round 2: same-shape FP32 runtime reference gate

The first seven-way synthetic runtime attempt exposed a diagnostic-engine
comparison bug. A production engine and a diagnostic engine expose the same
Attention output, but registering QK/Softmax/AV tensors as additional outputs
changes TensorRT fusion and tactics. The two engines can therefore differ more
than a strict roundoff threshold in low precision even when both remain close to
the same-shape strict-FP32 reference.

The runtime evaluator now:

- retains `production_vs_diagnostic` parity as diagnostic provenance;
- evaluates production output against the same-shape P0 FP32 engine;
- separately evaluates all diagnostic tensors against that same P0 engine;
- applies numerical safety gates to both reference comparisons;
- never treats diagnostic-engine latency as deployment latency.

Changed files:

- `search/integration/lidar_cobevt_head_dim_runtime.py`: added
  `production_reference_parity` and removed the invalid production/diagnostic
  equality admission gate.
- `tests/test_lidar_cobevt_head_dim_capability.py`: added a RED-to-GREEN test in
  which production and diagnostic outputs differ from one another but are both
  within the accepted error bound of the same-shape FP32 reference.

Validation:

- focused regression: `35 passed`;
- modified Python files compile;
- `git diff --check` passes.

Operational evidence:

- 282/282 production engines and 282/282 diagnostic engines were already built
  successfully before this runtime-only correction;
- 44 reports emitted by the obsolete gate were retained under each candidate's
  `runtime_attempts/` directory;
- only the seven orphaned CoBEVT runtime children from that obsolete attempt
  were terminated; no Pyramid GA process was modified;
- user-approved shared GPUs are used for screening, so all current latency is
  labeled `screening_shared_gpu`, not formal isolated latency.

---

Round 2 completed at 2026-07-19 13:17:48 Asia/Shanghai.

---

## Round 3: full capability execution and real-model evidence

Completed at 2026-07-19 17:06:27 Asia/Shanghai.

This round resumed on `feature/cobevt-head-dim-trt-capability` after the user
explicitly allowed relatively idle shared 4090 cards. Pyramid workers were not
modified, paused, killed, or restarted. Because the cards were shared with
other workloads, every synthetic and real latency field is marked
`screening_shared_gpu`; no shared-card value is presented as isolated formal
latency.

### Synthetic matrix

- Candidate definitions: 282 (uniform, QK-only, and V-only families; core and
  projection graphs; P0-P4 profiles).
- Production ONNX export: 282/282.
- Diagnostic ONNX export: 282/282.
- Production TensorRT build: 282/282.
- Diagnostic TensorRT build: 282/282.
- Runtime execution: 282/282, with finite-output and same-shape FP32 reference
  gates applied independently to production and diagnostic graphs.
- Support classes: 140 `supported_primitive`, 12 `supported_fused_mha`, and
  130 `supported_with_fallback`.
- Complete fused-MHA detection occurred in 15 rows; only 12 passed the
  requested/realized identity and numerical gates. The three P4 fused-pattern
  detections remain fallback/unsafe evidence and are not eligible INT8 search
  actions.
- Non-8-aligned widths built and ran through primitive paths. In the projection
  uniform FP16 graph, complete fusion was observed at widths
  16/24/32/40/48/56/64/80/96/128. This demonstrates that the observed
  alignment rule is a fusion/tactic boundary rather than a universal ONNX or
  primitive-build restriction in this SM89/TensorRT 10.9 run.
- P2 F3 rows intentionally remain `supported_with_fallback` where the
  requested profile and realized graph differ. They are not silently promoted
  to exact F3 support.

### Real CoBEVT matrix

- Physical structures: 21 (uniform, QK-only, V-only).
- Fresh full-model engines: 21 FP32 + 21 FP16 + 21 F3 = 63/63 builds.
- Smoke10: 63/63 complete, 0 skipped.
- Fixed50: 14/14 selected rows complete, 0 skipped. The generated fixed50
  manifest overlaps the fixed500 source ordering; this is recorded in the
  local manifest provenance.
- Fixed500: six final uniform rows (d32/d48/d64, FP32 and F3), 6/6 complete,
  0 skipped. Exact mAP/p50/p90/p99 values are in
  `real_cobevt/real_cobevt_capability_matrix.csv` and
  `root_conclusion.md`.
- Same-shape FP32 references are used for precision deltas. This avoids
  attributing structural width loss to the precision profile.
- Real F3 fixed500 deltas versus same-shape FP32 were -0.000054 (d32),
  -0.000043 (d48), and -0.000601 (d64). These are screening results on a
  shared GPU, not a claim that all real widths are accuracy-safe.

### Provenance and code changes

- `search/orchestration/lidar_cobevt_head_dim_capability.py`: standardized
  real-matrix family/profile fields, recorded engine/precision completion
  provenance, and generated root/documentation reports during assemble.
- `search/reporting/cobevt_head_dim_capability.py`: added deterministic root
  conclusion and empirical-vs-documentation writers; deduplicated contract
  forbidden pairs and excluded fused widths from primitive-only lists.
- `tests/test_lidar_cobevt_head_dim_capability.py`: added report-writer and real
  matrix schema assertions.
- Output directory:
  `/data/lxf/heal_data/outputs/cobevt_head_dim_trt_capability_20260719_122046/`.

### Current limitations

- No full-model INT8 CoBEVT AP was run; synthetic INT8 rows are capability
  evidence only and several requested profiles fall back or fail numerical
  safety.
- Real fixed500 intentionally covers only the final d32/d48/d64 uniform FP32/F3
  subset; other rows remain smoke10/fixed50 evidence.
- A single isolated-GPU latency replay is still required before using latency
  as a production width-selection criterion.
