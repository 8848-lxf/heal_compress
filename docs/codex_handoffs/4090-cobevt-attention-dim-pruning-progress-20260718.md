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
