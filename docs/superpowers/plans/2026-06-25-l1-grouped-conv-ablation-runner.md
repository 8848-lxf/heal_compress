# L1 Grouped Conv Ablation Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a one-command experiment runner for L1 grouped-conv pruning ablations across two grouped-conv selection modes and three pruning ratios, plus unified evaluation summaries.

**Architecture:** Keep pruning in `tests/test_general_pruner.py` and evaluation in the existing HEAL evaluation path. Add a thin runner that creates a unique experiment root, chooses a GPU, launches pruning/evaluation subprocesses, writes manifests and reproducibility commands, and aggregates stable CSV/JSON/Markdown summaries.

**Tech Stack:** Python stdlib subprocess/json/csv/pathlib, existing `utils.io_utils`, existing `tests/test_general_pruner.py`, existing `tests/test_prune_and_eval.py`, pytest.

---

### Task 1: GPU Selection

**Files:**
- Create: `utils/gpu_select.py`
- Test: `tests/test_l1_grouped_conv_ablation_runner.py`

- [ ] Write tests for default auto GPU exclusion, explicit GPU bypass, and no-available-GPU errors.
- [ ] Implement `parse_nvidia_smi_csv`, `select_gpu`, and `resolve_device_from_gpu_arg`.
- [ ] Run `pytest tests/test_l1_grouped_conv_ablation_runner.py -q`.

### Task 2: Runner Configuration and Commands

**Files:**
- Create: `tests/run_l1_grouped_conv_ablation.py`
- Test: `tests/test_l1_grouped_conv_ablation_runner.py`

- [ ] Write tests for six generated experiment configs and pruning command flags.
- [ ] Implement config generation, unique experiment root creation, and command rendering.
- [ ] Run the runner tests.

### Task 3: Evaluation Aggregation

**Files:**
- Create: `tests/evaluate_l1_grouped_conv_ablation.py`
- Test: `tests/test_l1_grouped_conv_ablation_runner.py`

- [ ] Write tests for reading baseline plus six manifests, failed model row preservation, and stable summary columns.
- [ ] Implement manifest loading, optional subprocess call to `tests/test_prune_and_eval.py`, JSON/CSV/Markdown aggregation.
- [ ] Run the runner tests.

### Task 4: Pruner Output Compatibility

**Files:**
- Modify: `tests/test_general_pruner.py`

- [ ] Add `forward_sanity_report.json`.
- [ ] Keep `pruned_model.pth` saving path intact for legal and forward-passing runs.
- [ ] Ensure pruning output includes the required stable files.

### Task 5: Shell Entrypoints

**Files:**
- Create: `scripts/run_l1_grouped_conv_ablation.sh`
- Create: `scripts/run_l1_grouped_conv_ablation_debug.sh`

- [ ] Add full and debug command wrappers.
- [ ] Mark shell scripts executable if filesystem permissions allow.

### Task 6: Verification

- [ ] Run `pytest -q tests/test_l1_grouped_conv_ablation_runner.py`.
- [ ] Run import/CLI help smoke for both new scripts.
- [ ] Run `git diff --check`.
