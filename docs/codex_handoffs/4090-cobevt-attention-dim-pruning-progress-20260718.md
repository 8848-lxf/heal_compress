# 4090 CoBEVT Attention Dimension Pruning Progress

## Round 1: Entry Audit and Experiment Design

- Branch: `feature/cobevt-attention-dim-pruning-audit`
- Entry commit: `252e46dd45446d4cc669556328abab7bf9dfaeca`
- CoBEVT worktree was clean before edits.
- Pyramid worktree remains on `feature/heal-compress-h800-sync-4090` at
  `d9e68ed54acb9dd878ea7949114dc3899348abf8`; its active GA process was not
  modified or interrupted.
- Old rejected Pyramid GA binary artifacts were removed from the two historical
  output roots. Lightweight genotype, pruning-plan, precision-profile,
  Inspector, evaluation and winner evidence was retained. About 79.9 GiB of
  actual disk space was recovered; cleanup manifests were written in both roots.
- Audited stock CoBEVT Attention: `E=256`, `H=8`, `d_qk=d_v=32`, six modules,
  fused `[768,256]` QKV projection, equal-width `chunk(3)`, relative-position
  bias, mask, Softmax, residual LayerNorm and FFN.
- Confirmed that unequal Q/K and V widths require explicit projection/split
  semantics; attribute-only edits would be invalid.
- Existing whole-head/global-embedding recipe remains untouched. The new work
  is an independent head-internal-dimension experiment path.
- GPU 2 selected for CoBEVT construction/evaluation. It had approximately
  0.9 GiB memory occupied and 0% utilization at audit time. Formal latency will
  only be accepted during an isolated measurement window.

Implementation files planned:

- `search/model_families/lidar_cobevt/attention_dim_pruning.py`
- `search/model_families/lidar_cobevt/attention_taylor.py`
- `search/model_families/lidar_cobevt/attention_microbenchmark.py`
- `search/orchestration/lidar_cobevt_attention_pruning.py`
- focused unit and integration tests under `tests/`

------------------------------------------------------------
Round completed: 2026-07-18  (Asia/Shanghai)
------------------------------------------------------------

## Round 2: Explicit Q/K/V Attention and B1 Physical Pruning

Implemented:

- `search/model_families/lidar_cobevt/attention_dim_pruning.py`
  - adds validated per-head `AttentionDimMask` identities;
  - adds explicit `q_proj`, `k_proj`, `v_proj`, `out_proj` Attention;
  - losslessly converts the stock fused QKV module at `d=32`;
  - permits `d_qk != d_v` with explicit projection shapes;
  - enforces shared Q/K positions and shared V/WO positions while allowing the
    two mask families to differ;
  - uses `1/sqrt(d_qk)` and retains original RPE/mask/Softmax semantics;
  - physically replaces all six real CoBEVT Attention modules for B1.
- `search/model_families/lidar_cobevt/attention_taylor.py`
  - computes unnormalized first-order `sum(abs(w * mean_gradient))` scores;
  - aggregates Q and K matching rows into QK units;
  - aggregates V rows and output-projection columns into VO units;
  - ranks independently per module and physical head with stable tie breaking.
- `tests/test_lidar_cobevt_attention_dim_pruning.py`
- `tests/test_lidar_cobevt_attention_taylor.py`

RED evidence: the initial test run failed with eight missing-module failures and
one fixture error before the implementation existed.

GREEN evidence:

- focused new tests: 9 passed;
- all CoBEVT, family-registry and cross-family-cache tests: 79 passed;
- modified Python files compiled successfully;
- `git diff --check` passed.

The existing whole-head CoBEVT pruning recipe and Pyramid production paths were
not changed.

------------------------------------------------------------
Round completed: 2026-07-18 08:15 CST
------------------------------------------------------------

## Round 3: B2 Global Embedding Closure and Model-Level Taylor Masks

- Added B2 materialization with `E=192`, `H=8`, `d_qk=d_v=24`.
- The global embedding keep set is an independent stratified axis; it is not
  inferred from an Attention-internal QK or VO mask.
- Synchronized the shrinker output, all fusion LayerNorms, Attention
  inputs/outputs, FFN residual boundaries, fusion MLP head and detection-head
  inputs.
- Preserved FFN hidden width at 256 and relative-position bias at eight heads.
- Added model-level decoding from full mean-gradient dictionaries to module-
  local, per-head QK and VO masks with deterministic ranking audit records.
- Verified that different Attention modules can select different local
  dimensions and that QK/VO families remain independent.

Validation:

- focused Attention tests: 13 passed;
- full CoBEVT/family isolation regression: 83 passed;
- `py_compile` and `git diff --check`: passed.

------------------------------------------------------------
Round completed: 2026-07-18 08:35 CST
------------------------------------------------------------

## Round 4: Resumable Runner and Preliminary Fixed-500 Validation

Implemented:

- resumable experiment phases for prepare, structure, PyTorch fixed-500,
  strongly typed FP16 export/build and TensorRT fixed-500 evaluation;
- streaming Attention-only mean-gradient collection on real HEAL task loss;
- fixed 20-frame warmup plus 500-frame validation manifest;
- GPU AP/IoU postprocessing with eight DataLoader workers;
- production GPU TP/FP helper independent of test-module import paths;
- per-candidate result upsert and preserved failed-protocol records.

Debug evidence:

- fixed GPU-weight/CPU-gradient Taylor scoring with an explicit cross-device
  regression test;
- fixed independent-worktree AP helper import failure;
- fixed malformed `OrderedDict` postprocess mapping before any frame was scored;
- all failed attempts remain in the output `failure_records/` directory.

Preliminary real run:

- output: `/data/lxf/heal_data/outputs/cobevt_attention_dim_pruning_20260718_082508`;
- Taylor calibration: 50 real train samples, micro-batch 1;
- all six candidates passed full-model physical forward;
- all six completed the same fixed validation set at 500/500, zero skip;
- preliminary FP32 PyTorch mAP: baseline `0.649178`, QK24 `0.647130`,
  QK16 `0.640181`, B1-24 `0.645439`, B1-16 `0.640215`, B2-24
  `0.485090`.

The B2 structure is legal but its zero-shot accuracy collapses by `0.164088`
absolute mAP. It is not accepted as a future legal search action from this
evidence. These measurements are marked preliminary because the runner changes
were not yet committed when they ran; they will be repeated from the clean
committed code before TensorRT construction.

Validation after implementation: 92 related tests passed; `py_compile` and
`git diff --check` passed.

------------------------------------------------------------
Round completed: 2026-07-18 09:00 CST
------------------------------------------------------------

## Round 5: Committed Fixed-500 Results and Build Resume Fix

Formal source run:

- code commit: `0a308544e8c77539d08219955cc149e99aa506d7`;
- output: `/data/lxf/heal_data/outputs/cobevt_attention_dim_pruning_20260718_085644`;
- Taylor samples: 50 real train frames, micro-batch 1;
- fixed-500 manifest hash:
  `5a2a81c05e635f71f277e9bd151dbb6d5dd0a8f9b08475d60511f67399824c19`;
- all structures passed full-model forward;
- all six candidates completed 500/500 with zero skip.

Formal FP32 PyTorch mAP:

- baseline d32: `0.648880`;
- QK-only d24: `0.646697` (`-0.002183`);
- QK-only d16: `0.640169` (`-0.008711`);
- B1 uniform d24: `0.644999` (`-0.003881`);
- B1 uniform d16: `0.639942` (`-0.008938`);
- B2 global d24: `0.485283` (`-0.163597`).

The first strongly typed FP16 baseline engine and layer-info files were built
successfully. Report creation then failed because the plain export dataclass
does not implement `to_dict`. Added RED/GREEN serialization and resume tests:
failed reports do not count as complete, and a completed engine plus layer-info
is reused only after reconstructing and validating the exact precision mapping.

------------------------------------------------------------
Round completed: 2026-07-18 09:20 CST
------------------------------------------------------------

## Round 6: Strongly Typed Full-Model FP16 Engines

Built all six formal candidates in the `modelopt` environment using TensorRT
10.9, `--stronglyTyped --noTF32`, the existing scatter plugin and no weak
precision flags.

Results:

- six of six engine builds passed;
- each graph realized 66 FP16 weighted groups, zero INT8 groups and zero
  unresolved precision layers;
- requested/realized precision audit passed for every engine;
- engine sizes range from 48,009,844 bytes (B2-24) to 50,425,628 bytes
  (baseline);
- no QK/V width candidate required nominal or hidden INT8 fallback.

The baseline and QK24 engines created before the report-serialization fix were
reused only after reconstructing the exact typed graph/mapping and passing the
same EngineInspector precision audit. All other engines were freshly built.

Added a required smoke10 admission phase before fixed-500 TensorRT evaluation;
an engine must complete exactly 10/10 with zero skip before the 500-frame phase.

------------------------------------------------------------
Round completed: 2026-07-18 09:30 CST
------------------------------------------------------------

## Round 7: Fixed-K Coverage Failure and Manifest-Bound Contract

The K=25,600 baseline engine passed smoke10 but failed the fixed-500 run at
216 evaluated frames because the next frame contained 26,931 real voxels.
This classified all K=25,600 engines as input-contract failures, not Attention
structure failures.

Added:

- an exact scan of all 20 warmup plus 500 evaluation frames;
- a 256-aligned fixed-K derivation bound to the evaluation manifest hash;
- `fixed_k_validated` fail-closed admission before export and evaluation;
- fixed-K-specific engine directories so incompatible input profiles cannot be
  reused.

The full 520-frame scan found:

- maximum real voxel count: `28,949`;
- aligned production fixed K: `29,184`;
- overflow count: `0`;
- fixed-500 manifest hash:
  `5a2a81c05e635f71f277e9bd151dbb6d5dd0a8f9b08475d60511f67399824c19`.

K=25,600 build reports, layer-info and failure logs remain as evidence; their
large ONNX/engine files are not eligible for reuse and will be removed before
the K=29,184 fresh build.

------------------------------------------------------------
Round completed: 2026-07-18 09:45 CST
------------------------------------------------------------

## Round 8: Pyramid-Compatible Full-Validation Fixed-K Contract

The fixed-500-only K=29,184 profile was stopped before evaluation and replaced
with the accepted Pyramid single-engine K=29,696 policy. Pyramid and CoBEVT use
the same DAIR-V2X point-cloud preprocessing contract:

- point-cloud range: `[-102.4, -51.2, -3.5, 102.4, 51.2, 1.5]`;
- voxel size: `[0.4, 0.4, 5]`;
- maximum points per voxel: `32`;
- configured test voxel ceiling: `70,000`.

The CoBEVT validation split was scanned in full without running model inference:

- validation records: `1,789`;
- full-validation maximum real voxel count: `29,164`;
- fixed-500 plus warmup records: `520`;
- fixed-500 maximum real voxel count: `28,949`;
- selected single-engine fixed K: `29,696`;
- full-validation overflows at K=29,696: `0`.

The fixed-K contract now records the subset-derived minimum separately from the
full-validation maximum and applies K=29,696 as the accepted Pyramid floor. If a
future CoBEVT split exceeds that floor, the scan raises the selected profile to
the next aligned value instead of clipping points. Export/build remains
fail-closed until a full-validation contract is present.

Per user direction, bucketed engines, multi-engine routing and over-limit
chunking are excluded. The production experiment remains one K=29,696 engine
per Attention candidate.

The interrupted K=29,184 attempt produced one baseline engine and partial QK24
ONNX files. Nine obsolete `.onnx`/`.plan` files (484,751,482 bytes) were removed;
the baseline build report, EngineInspector evidence and trtexec log were kept.

Validation: 28 CoBEVT Attention tests passed; modified Python files passed
`py_compile`; `git diff --check` passed.

------------------------------------------------------------
Round completed: 2026-07-19 00:42 CST
------------------------------------------------------------

## Round 9: Strict-FP32 Control Gate for TensorRT Accuracy Diagnosis

All six K=29,696 strongly typed FP16 engines built successfully, with exact
requested/realized FP16 precision audits. The baseline passed smoke10 and then
completed fixed500 at 500/500 with zero skip, but its TensorRT FP16 mAP was
`0.402059` versus `0.648880` for the same physical baseline in PyTorch FP32 on
the identical manifest. Remaining candidate evaluations were stopped rather
than treating this unexplained deployment delta as valid Attention-pruning
evidence.

The experiment runner now supports precision-specific FP32/FP16 control engine
directories and an optional candidate filter. This permits one exact baseline
FP32 control build/evaluation without rebuilding every candidate or mixing
engine identities. FP16 remains the default protocol.

Validation: 29 CoBEVT Attention tests passed; modified Python passed
`py_compile`; `git diff --check` passed.

------------------------------------------------------------
Round completed: 2026-07-19 01:02 CST
------------------------------------------------------------

## Round 10: FP32 Parity and FP16 Localization Harness

The strict FP32 K=29,696 baseline completed 500/500 with zero skip:

- AP30: `0.779501`;
- AP50: `0.683628`;
- AP70: `0.483247`;
- mAP: `0.648792`;
- screening forward p50: `7.901 ms`.

The same-manifest PyTorch FP32 mAP is `0.648880`, a delta of only `-0.000088`.
This proves the fixed-K wrapper, ONNX export semantics, scatter plugin, engine
output mapping and GPU postprocess are aligned. The strict FP16 mAP collapse is
therefore a precision-stability problem rather than a K/profile problem.

Added baseline-only mixed-precision diagnostic profiles and a smoke-only runner:

- `frontend_fp16`;
- `fusion_fp16`;
- `attention_fp16`;
- `ffn_heads_fp16`.

Each profile has an isolated engine identity and is explicitly marked
diagnostic. It cannot be confused with the formal strict-FP16 result. Smoke10 is
sufficient for localization because the strict FP32/FP16 smoke mAP values are
`0.521809` and `0.115378`, respectively.

Validation: all 16 orchestration tests passed; modified Python passed
`py_compile`; `git diff --check` passed.

------------------------------------------------------------
Round completed: 2026-07-19 01:18 CST
------------------------------------------------------------

## Round 11: CoBEVT FP16 Collapse Localized to Attention

Baseline-only smoke10 diagnostic results on the same manifest:

- strict FP32: mAP `0.521809`, p50 `8.181 ms`;
- strict FP16: mAP `0.115378`, p50 `5.120 ms`;
- frontend FP16 only: mAP `0.520389`, p50 `6.999 ms`;
- fusion FP16 only: mAP `0.117048`, p50 `6.406 ms`;
- Attention projection FP16 only: mAP `0.116860`, p50 `7.631 ms`;
- FFN/mlp/head FP16 only: mAP `0.519500`, p50 `7.464 ms`.

This isolates the collapse to the Attention FP16 path: Q/K/V/Out projections
cause the surrounding QK MatMul, mask/RPE, Softmax and AV MatMul chain to run in
FP16. Frontend and FFN/head FP16 are independently stable.

Added an explicit `attention_fp32_rest_fp16` profile for the next controlled
experiment. It protects only the 24 Attention projection groups in FP32 while
placing all remaining weighted groups in FP16. This profile is labeled mixed
precision and will not be reported as strict FP16.

------------------------------------------------------------
Round completed: 2026-07-19 01:35 CST
------------------------------------------------------------

## Round 12: Accuracy-Safe Mixed Engines and Attention INT8 Harness

The explicit `attention_fp32_rest_fp16` contract realizes 24 Attention
projection groups in FP32 and the remaining 42 weighted groups in FP16. All six
candidate engines built and passed the exact requested/realized audit. All six
also completed fixed500 at 500/500 with zero skip:

| candidate | AP30 | AP50 | AP70 | mAP | screening p50 ms |
|---|---:|---:|---:|---:|---:|
| baseline d32 | 0.779746 | 0.683858 | 0.483353 | 0.648986 | 6.388 |
| QK-only d24 | 0.779693 | 0.682241 | 0.477297 | 0.646411 | 6.409 |
| QK-only d16 | 0.776816 | 0.674873 | 0.466569 | 0.639419 | 6.151 |
| B1 uniform d24 | 0.778779 | 0.680786 | 0.477258 | 0.645608 | 5.948 |
| B1 uniform d16 | 0.774184 | 0.672532 | 0.470640 | 0.639119 | 5.506 |
| B2 global d24 | 0.752141 | 0.562050 | 0.141914 | 0.485368 | 6.056 |

These latency values remain screening measurements because Pyramid workers are
still resident on the server. B1 d24/d16 provide the best observed structural
latency tradeoff. B2 global d24 remains accuracy-invalid without recovery
training.

Added a real-activation Attention microbenchmark harness:

- uniform d_h scan: 8, 12, 16, 20, 24, 28 and 32;
- QK-only controls: (24,32), (16,32), and V-only (32,16);
- pure Attention and CoBEVT mask/RPE graph variants;
- typed FP16 and explicit-QDQ INT8 builds;
- symmetric per-output-channel linear weight Q/DQ;
- per-tensor activation Q/DQ;
- strongly typed TensorRT builds and EngineInspector realization records;
- actual captured CoBEVT activation/mask parity and 100-iteration screening
  latency.

The harness does not claim INT8 realization from build success alone; it records
projection, QK MatMul, Softmax, AV MatMul and out projection tensor formats.

------------------------------------------------------------
Round completed: 2026-07-19 02:08 CST
------------------------------------------------------------

## Round 13: Attention Microbenchmark Minimum Control

The d32 CoBEVT mask/RPE control produced successful strongly typed FP16 and
explicit-QDQ INT8 engines and executed the captured real activation for 20
warmup plus 100 measured iterations.

Preliminary control metrics:

- FP16 p50/p90/p99: `0.300/0.314/0.323 ms`;
- INT8 p50/p90/p99: `0.496/0.577/0.629 ms`;
- FP16 cosine vs PyTorch graph: `0.995172`;
- INT8 cosine vs PyTorch fake-quant graph: `0.995208`;
- explicit INT8 topology: 11 QuantizeLinear + 11 DequantizeLinear nodes;
- no non-finite output.

EngineInspector shows real INT8 realization:

- fused Q/K/V projection GEMM consumes and produces INT8;
- out projection consumes and produces INT8;
- QK MatMul, real mask/RPE, Softmax and AV MatMul are fused into
  `_gemm_mha_v2` with INT8 Q/K/V/AV boundaries;
- Softmax/RPE auxiliary tensors remain floating as expected.

The first two execution attempts failed before inference because the new runner
did not preload TensorRT shared libraries, then created the TensorRT context on
default GPU 0 while using a GPU 2 stream. Both were environment-boundary bugs,
not engine failures. The runner now explicitly loads `libnvinfer`,
`libnvinfer_plugin`, `libnvonnxparser`, and calls `torch.cuda.set_device` before
deserialization. Tests cover missing-library fail-closed behavior and fused MHA
metadata recognition.

At d32, INT8 is slower than FP16 in this real-shape subgraph, so INT8 build
success alone will not make a width eligible for future search.

------------------------------------------------------------
Round completed: 2026-07-19 02:28 CST
------------------------------------------------------------
