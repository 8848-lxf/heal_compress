# CoBEVT Head-Dimension TensorRT Capability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an auditable TensorRT 10.9 capability matrix for CoBEVT uniform, QK-only, and V-only Attention dimensions without interfering with Pyramid GA.

**Architecture:** Extend the existing explicit `d_qk`/`d_v` CoBEVT Attention implementation. A typed candidate schema owns shape/profile identities; canonical synthetic graph builders own FP32/FP16/F3/explicit-QDQ topology; a strict TensorRT provenance parser distinguishes primitive execution, local fusion, complete native MHA, and precision fallback; the orchestration entry writes append-only per-candidate evidence and aggregate matrices. Real CoBEVT evaluation reuses the current fixedK29696 exporter/evaluator and always binds a candidate to a same-shape FP32 reference.

**Tech Stack:** Python 3.9/3.10, PyTorch, ONNX opset 17, TensorRT 10.9 strongly typed networks, pytest, existing HEAL CoBEVT deployment adapters.

## Task 1: Candidate schema and matrix inventory

- [ ] Add failing tests for uniform/QK-only/V-only candidate counts, shape semantics, same-shape FP32 references, and identity hashes.
- [ ] Run the focused test and confirm RED due to the missing capability module.
- [ ] Implement immutable candidate/profile schemas and deterministic candidate generation.
- [ ] Run focused tests and confirm GREEN.
- [ ] Commit the schema and tests.

## Task 2: Canonical synthetic graphs and precision profiles

- [ ] Add failing tests for core/projection graph shapes and P0/P1/P2/P3/P4 boundary topology.
- [ ] Implement canonical Attention graphs using real CoBEVT head count, embedding width, token count, scale, mask, and RPE layout.
- [ ] Add explicit deterministic Q/DQ helpers for P3/P4 and explicit Cast boundaries for F3.
- [ ] Verify ONNX shape/type/QDQ inventories for every profile.
- [ ] Commit graph generation.

## Task 3: TensorRT provenance and support classification

- [ ] Add failing tests for primitive versus complete fused MHA, fallback detection, unknown accumulator precision, and runtime failure classification.
- [ ] Implement version-consistent TensorRT environment resolution from prior successful build evidence.
- [ ] Implement detailed trtexec build commands, EngineInspector parsing, fusion classification, and requested/realized comparison.
- [ ] Commit provenance parsing.

## Task 4: Runtime parity and matrix writer

- [ ] Add failing tests for deterministic, random, large-range, small-margin, and cancellation-heavy parity inputs.
- [ ] Implement runtime execution and QK/Softmax/AV/output parity metrics.
- [ ] Implement append-only CSV/JSON/Markdown matrix writers and final search-contract derivation.
- [ ] Commit parity and reporting.

## Task 5: Synthetic full matrix

- [ ] Prepare a fresh timestamped output directory and hardware manifest.
- [ ] Export all 282 candidate ONNX graphs without GPU interference.
- [ ] Wait for a GPU that is not occupied by Pyramid GA or external processes.
- [ ] Fresh-build and execute every eligible synthetic candidate under modelopt.
- [ ] Preserve failures, inspector reports, tactics, casts, reformats, hashes, and parity results.
- [ ] Commit only lightweight matrix summaries and reproduction metadata.

## Task 6: Real CoBEVT integration

- [ ] Add tests for deterministic shrink/expansion materialization at 8/12/16/24/32/48/64 while preserving residual embedding width.
- [ ] Build same-shape FP32, FP16, F3, and evidence-eligible INT8 engines.
- [ ] Run smoke10, gate fixed50, and run fixed500 only for specified boundary candidates.
- [ ] Mark latency screening-only unless a truly isolated non-Pyramid GPU is available.

## Task 7: Final evidence and delivery

- [ ] Generate capability matrices, search contract, empirical/documentation comparison, and root conclusion.
- [ ] Run focused and existing CoBEVT/Pyramid non-regression tests.
- [ ] Run py_compile and `git diff --check`.
- [ ] Update the 4090 handoff with timestamped round evidence.
- [ ] Commit, push only `feature/cobevt-head-dim-trt-capability`, and verify a clean worktree.

