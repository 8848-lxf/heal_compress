# FixedK Full Cover Train Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recompute LiDAROnly/lidar_pyramid TensorRT deployment results with a fixed_K that covers full validation and with all formal INT8 calibration generated from train split NPZ files.

**Architecture:** Add audit and orchestration scripts under `tests/quant_deploy/` without touching HEAL/OpenCOOD source. Reuse existing export/build/runner utilities where possible, and write all artifacts under `tests/quant_deploy/outputs/lidar_pyramid_agent_export_strategy_compare/`.

**Tech Stack:** Python, PyTorch/OpenCOOD runtime, TensorRT 10.9 in `modelopt`, existing PointPillarScatterTRT plugin, pytest.

---

### Task 1: K Distribution Evidence

**Files:**
- Create: `tests/quant_deploy/analyze_voxel_k_coverage.py`
- Modify: `tests/test_quant_deploy_utils.py`

- [ ] Add tests for percentile calculation, ceil-to-multiple, and K summary formatting.
- [ ] Implement split scanner for train/val using existing dataset/config helpers.
- [ ] Emit `debug/voxel_k_distribution_train.json`, `debug/voxel_k_distribution_val.json`, `summary/voxel_k_coverage_report.json`, and `.md`.
- [ ] Run the scanner in `modelopt` and derive `fixed_K_full_cover`.

### Task 2: Train Calibration NPZ Manifests

**Files:**
- Create: `tests/quant_deploy/dump_train_calibration_npz_for_all_strategies.py`
- Modify: `tests/test_quant_deploy_utils.py`

- [ ] Add tests for manifest hashing and split enforcement.
- [ ] Dump actual TensorRT input NPZ files from train split only.
- [ ] Save `manifest.json` with sample IDs, K/N distributions, input shapes/dtypes, and hashes.
- [ ] Produce calib50 and calib200 for dynamic bucket and single engine using the new fixed_K.

### Task 3: Rebuild Engines With Full-Cover fixed_K

**Files:**
- Create or modify orchestration under `tests/quant_deploy/` using existing builders.
- Output: `artifacts/onnx/fixedK<fixed_K>/...`, `artifacts/engines/fixedK<fixed_K>/...`

- [ ] Rebuild padded FP32/FP16 baselines.
- [ ] Rebuild dynamic bucket FP32/FP16 and INT8 train_calib50/train_calib200.
- [ ] Rebuild single_engine_maxK FP32/FP16 and INT8 train_calib50/train_calib200.
- [ ] Save build report JSON/Markdown with engine counts, paths, calibration cache paths, and TensorRT profile.

### Task 4: Idle-GPU Full-Val Retest

**Files:**
- Modify: `tests/quant_deploy/evaluate_all_deployment_engines_full_val_idle_gpu.py` or add fixedK-aware wrapper.

- [ ] Extend evaluator to accept fixedK output namespace and new calibration labels.
- [ ] Re-run full val on automatically selected idle GPU.
- [ ] Require evaluation split val, train calibration labels for INT8, and no K-based skips.
- [ ] Save per-mode evaluation, benchmark, GPU traces, and final summary reports.

### Task 5: INT8 Calibration Split Diagnosis

**Files:**
- Create: `tests/quant_deploy/diagnose_int8_calibration_split_and_accuracy_drop.py`

- [ ] Compare historical dynamic bucket INT8 with unknown calibration split against new train-calibrated results.
- [ ] Compare dynamic bucket train_calib200 vs single_engine train_calib200.
- [ ] Rank suspected causes and recommend mixed precision/QDQ next steps.

### Task 6: Verification

**Files:**
- Modify: `tests/test_quant_deploy_utils.py`

- [ ] Run `pytest tests/test_quant_deploy_utils.py -q`.
- [ ] Run `python -m py_compile tests/quant_deploy/*.py`.
- [ ] If plugin/build logic changed, run plugin build and plugin checks.
