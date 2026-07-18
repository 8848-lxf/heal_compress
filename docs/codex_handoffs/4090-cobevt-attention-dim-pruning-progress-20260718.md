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
