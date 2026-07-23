# Transformer d_h Power Alignment 4090 Design

## Scope

Run the requested CoBEVT and V2XViT high-order `d_h` alignment experiment on
RTX 4090 while preserving the existing H800 implementation and evidence.
The worktree tracks `feature/h800-transformer-dh-alignment-sweep`; all new
code and artifacts identify the execution platform as RTX 4090/SM89.  No GA,
Greedy, full-1789 validation, formal-search migration, H800 engine reuse, or
cross-structure calibration reuse is allowed.

## Isolation

Existing `search/orchestration/lidar_transformer_dh_*` and
`search/model_families/transformer/dh_*` H800 modules remain unmodified.  A
4090-only adapter validates `/home/lixingfeng/anaconda3/envs/modelopt`, the
Conda CUDA 11.8 compiler, SM89, the installed TensorRT 10.9 tree, and the
PointPillar plugin before overriding imported runtime constants in process.
The adapter writes the H800 and formal-search branch heads into provenance and
fails if system CUDA or an H800 artifact is selected.

## Candidate Model

A pure candidate module generates the deduplicated power-of-two,
multiple-of-eight ladder, and nearby four-aligned controls for original widths
16, 32, and 64.  It records divisibility of both `d_h` and `H*d_h`, reduction
ratio, neighboring controls, and candidate class membership.  Joint CoBEVT and
V2XViT matrices are explicit data, not Cartesian products.

Each structure is materialized once with the frozen nested Taylor ranking.
P32, P16/F3, and P8/SQ1 use the existing physical rewrite and precision
contracts.  P8 calibration is fresh for each structure.  Every accepted engine
requires ONNX checking, exact requested/realized precision, zero fallback, and
hash-bound structure, ONNX, engine, scale, tactic, and fusion provenance.

## Execution

The scheduler creates fresh manifests, inventories, and Taylor rankings, then
assigns disjoint family or joint candidates to GPUs 4, 5, 6, and 7.  Each GPU
runs a serial build -> smoke10 -> fixed50 -> fixed500 chain so a candidate is
never evaluated before its build and precision gates pass.  Checkpoints are
append-only and resumable.  Failed candidates retain their reason and do not
block unrelated queues.

Formal latency is measured only after builds and evaluation finish.  One
isolated RTX 4090 runs same-profile baseline, candidates, and baseline replay
serially with 200 warmup, 2000 iterations, five repeats, device-resident input,
and CUDA events.  The result is explicitly a 4090 measurement; it is never
reported as H800 latency.  Candidates that initially pass the benefit gate and
their controls receive three fresh no-timing-cache builds.

## Evidence And Gates

The reporter separately computes same-profile structure speedup, precision-only
speedup from B0/P32, and total speedup from B0/P32.  Alignment advantage also
requires improvement over legal `target-4` and `target+4` controls.  Search
eligibility requires fixed500 accuracy, same-profile latency benefit, neighbor
advantage, independent-build stability, and a beneficial joint candidate.

Only compact CSV/JSON reports, manifests, hashes, inspector summaries, tests,
and the handoff document enter Git.  Checkpoints, materialized weights, ONNX,
engines, calibration caches, and raw output directories remain under `/data`.

