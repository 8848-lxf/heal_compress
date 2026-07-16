# Linear Joint Taylor And Greedy Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the formal exponential Taylor score with a fixed-scale linear objective and produce deterministic proxy-only greedy endpoints for six BOPS budgets.

**Architecture:** Keep the existing legal-width genotype, fixed Taylor decoder, scalar joint proxy, and CUDA batched proxy. Add one signed joint-loss scale artifact, one reusable two-level BOPS admission policy, and one bounded greedy search module; integrate them through the existing lidar-pyramid runner without creating a parallel model or pruning stack.

**Tech Stack:** Python 3.10, PyTorch, dataclasses, existing HEAL/OpenCOOD search modules, pytest, YAML.

## Global Constraints

- Work only on `feature/heal-compress-h800-sync-4090`; preserve ancestor `b862b3d8ad061bd12580776226c75f564918298d`.
- Use `univ2x-opt` for Python, PyTorch, proxy computation, and tests.
- Do not modify tracer semantics, pruning dependency construction, physical replay semantics, QDQ, TensorRT, or Stage-2 in this plan.
- Structure remains one legal retained-width index per local pruning domain; precision never changes the decoded mask.
- The formal proxy includes weight pruning and retained-weight quantization Taylor terms only; activation Taylor and SQNR are excluded from `J1`.
- Formal budgets are `0.05/0.10/0.15/0.20/0.25/0.30`; primary tolerance is `0.005`, conditional expanded tolerance is `0.0075`.
- No AP/mAP hard gate is introduced.
- Preserve historical exponential/tau helpers only for old artifact replay; the new formal config must not call them.
- The current uncommitted `test_tau_marks_underresolved_when_unsafe_exists_but_safe_boundary_is_coarse` hunk was created by Codex before the approved design changed. Remove only that hunk before Task 1; do not discard any unrelated worktree change.
- Use `apply_patch` for every manual file edit. Run `git diff --check` before every commit.

## File Ownership Map

- `search/proxy/joint_loss_scale.py`: deterministic calibration, signing, writing, and loading of the one formal linear scale.
- `search/proxy/task_score.py`: pure scalar score formulas; historical exponential and new linear modes remain explicit.
- `search/proxy/objective.py` and `search/proxy/gpu_batch_proxy.py`: scalar and batched routing with identical formal metrics.
- `search/admission/bops_band.py`: reusable primary/expanded BOPS classification and audit funnel.
- `search/greedy/legal_actions.py`: one-step legal width and precision transitions only.
- `search/greedy/joint_budget_search.py`: bounded deterministic greedy state exploration.
- `search/orchestration/legal_width_greedy.py`: six-budget context integration and artifacts.
- `search/cli.py`: explicit greedy-only and joint-loss-scale CLI inputs.

---

### Task 1: Fixed Joint-Loss Scale Artifact

**Files:**
- Create: `search/proxy/joint_loss_scale.py`
- Create: `tests/test_joint_loss_scale.py`
- Modify: `search/proxy/__init__.py`
- Modify: `tests/test_joint_taylor_tau_calibration.py` (remove only the obsolete uncommitted WIP test named in Global Constraints)

**Interfaces:**
- Consumes: rows containing `phenotype_hash` and `L_joint_raw`.
- Produces: `calibrate_joint_loss_scale(rows) -> dict[str, Any]`, `joint_loss_scale_hash(payload) -> str`, `write_joint_loss_scale(path, payload) -> Path`, and `load_joint_loss_scale(path) -> dict[str, Any]`.

- [ ] **Step 1: Remove the obsolete uncommitted tau test hunk**

Use `apply_patch` to remove the complete function named
`test_tau_marks_underresolved_when_unsafe_exists_but_safe_boundary_is_coarse`
from its `def` line through its final assertion. Do not remove any neighboring
historical tau test.

Run: `git diff -- tests/test_joint_taylor_tau_calibration.py`

Expected: no diff for that file unless an unrelated pre-existing user change remains.

- [ ] **Step 2: Write failing scale tests**

Create tests that require deterministic deduplication, nearest-rank P90, a signed read-only artifact, and fail-closed validation:

```python
def test_joint_loss_scale_uses_unique_positive_nearest_rank_p90(tmp_path: Path) -> None:
    rows = [
        {"phenotype_hash": f"p{i}", "L_joint_raw": float(i)}
        for i in range(1, 11)
    ] + [{"phenotype_hash": "p9", "L_joint_raw": 999.0}]
    payload = calibrate_joint_loss_scale(rows, code_commit="abc")
    assert payload["value"] == 9.0
    assert payload["member_count"] == 10
    path = write_joint_loss_scale(tmp_path / "joint_loss_scale.json", payload)
    assert path.stat().st_mode & 0o222 == 0
    assert load_joint_loss_scale(path)["joint_loss_scale_hash"] == payload["joint_loss_scale_hash"]


@pytest.mark.parametrize("loss", [0.0, -1.0, float("inf"), float("nan")])
def test_joint_loss_scale_rejects_empty_positive_pool(loss: float) -> None:
    with pytest.raises(ValueError, match="joint_loss_scale_positive_finite_pool_required"):
        calibrate_joint_loss_scale([{"phenotype_hash": "x", "L_joint_raw": loss}])


def test_joint_loss_scale_rejects_conflicting_duplicate_phenotype() -> None:
    with pytest.raises(ValueError, match="joint_loss_scale_duplicate_loss_conflict"):
        calibrate_joint_loss_scale([
            {"phenotype_hash": "same", "L_joint_raw": 1.0},
            {"phenotype_hash": "same", "L_joint_raw": 2.0},
        ])
```

- [ ] **Step 3: Run the tests and confirm RED**

Run: `conda run -n univ2x-opt pytest -q tests/test_joint_loss_scale.py`

Expected: FAIL because `search.proxy.joint_loss_scale` does not exist.

- [ ] **Step 4: Implement the scale artifact**

Implement the exact public behavior below. Hash a copy with the hash field removed, sort rows by phenotype hash before hashing, and use `ceil(0.9*n)-1`:

```python
def calibrate_joint_loss_scale(
    rows: Sequence[Mapping[str, Any]], *, code_commit: str = ""
) -> dict[str, Any]:
    unique: dict[str, float] = {}
    for row in rows:
        identity = str(row.get("phenotype_hash", ""))
        loss = float(row.get("L_joint_raw", float("nan")))
        if identity and math.isfinite(loss) and loss > 0.0:
            if identity in unique and not math.isclose(unique[identity], loss, rel_tol=0.0, abs_tol=1.0e-12):
                raise ValueError(f"joint_loss_scale_duplicate_loss_conflict:{identity}")
            unique.setdefault(identity, loss)
    if not unique:
        raise ValueError("joint_loss_scale_positive_finite_pool_required")
    ordered_losses = sorted(unique.values())
    value = ordered_losses[math.ceil(0.90 * len(ordered_losses)) - 1]
    payload = {
        "mapping": "linear_fixed_scale",
        "formula": "J1=-0.8*(L_joint/L_scale)+0.2*R_prune",
        "quantile": 0.90,
        "quantile_method": "nearest_rank",
        "value": float(value),
        "member_count": len(unique),
        "members": [
            {"phenotype_hash": key, "L_joint_raw": unique[key]}
            for key in sorted(unique)
        ],
        "code_commit": str(code_commit),
    }
    payload["joint_loss_scale_hash"] = joint_loss_scale_hash(payload)
    return payload
```

`write_joint_loss_scale` must write atomically, chmod `0444`, and reload through `load_joint_loss_scale`. The loader must reject writable files, wrong mapping/formula, nonpositive values, and hash mismatch.

- [ ] **Step 5: Run GREEN and regression tests**

Run:

```bash
conda run -n univ2x-opt pytest -q \
  tests/test_joint_loss_scale.py \
  tests/test_joint_taylor_tau_calibration.py
```

Expected: PASS. Historical tau tests remain green; the new formal scale tests pass.

- [ ] **Step 6: Commit the scale artifact**

```bash
git add search/proxy/joint_loss_scale.py search/proxy/__init__.py \
  tests/test_joint_loss_scale.py tests/test_joint_taylor_tau_calibration.py
git diff --cached --check
git commit -m "feat: add fixed linear joint loss scale"
```

### Task 2: Scalar And Batched Linear Joint Objective

**Files:**
- Modify: `search/proxy/task_score.py`
- Modify: `search/proxy/objective.py`
- Modify: `search/proxy/gpu_batch_proxy.py`
- Modify: `tests/test_joint_proxy_with_fixed_mask.py`
- Create: `tests/test_joint_taylor_linear_score.py`

**Interfaces:**
- Consumes: `L_joint`, `L_scale`, physical parameter counts, and optional first/second-order diagnostics.
- Produces: public `compute_linear_joint_j1` with the complete keyword-only
  signature shown in Step 3; also `ProxyObjectiveConfig.task_score_mapping`,
  `.joint_loss_scale`, `.task_weight`, and `.prune_weight`.

- [ ] **Step 1: Write failing scalar direction and scale tests**

```python
def test_linear_joint_j1_has_expected_direction_and_no_tau_fields() -> None:
    low = compute_linear_joint_j1(
        l_joint=2.0, l_scale=10.0, original_params=100, candidate_params=80
    )
    high = compute_linear_joint_j1(
        l_joint=4.0, l_scale=10.0, original_params=100, candidate_params=80
    )
    more_pruned = compute_linear_joint_j1(
        l_joint=2.0, l_scale=10.0, original_params=100, candidate_params=70
    )
    assert low["J1"] > high["J1"]
    assert more_pruned["J1"] > low["J1"]
    assert low["J1"] == pytest.approx(-0.8 * 0.2 + 0.2 * 0.2)
    assert low["F1"] == pytest.approx(-low["J1"])
    assert "tau" not in low
    assert "exponent_value" not in low
```

Add a scalar-vs-CUDA-batched parity test using the existing fake legal-width context in `tests/test_joint_proxy_with_fixed_mask.py`. Configure:

```python
ProxyObjectiveConfig(
    proxy_mode="joint_taylor_second_order_fisher_diag",
    task_score_mapping="linear_fixed_scale",
    joint_loss_scale=10.0,
    task_weight=0.8,
    prune_weight=0.2,
)
```

Assert parity for `L_joint_raw`, `normalized_joint_loss`, `R_prune`, `J1`, and `F1`.

- [ ] **Step 2: Run RED**

Run:

```bash
conda run -n univ2x-opt pytest -q \
  tests/test_joint_taylor_linear_score.py \
  tests/test_joint_proxy_with_fixed_mask.py
```

Expected: FAIL because `compute_linear_joint_j1` and the new config fields do not exist.

- [ ] **Step 3: Add the linear score helper**

Keep `compute_exponential_j1` for historical replay and add:

```python
def compute_linear_joint_j1(
    *,
    l_joint: float,
    l_scale: float,
    original_params: int,
    candidate_params: int,
    task_weight: float = 0.8,
    prune_weight: float = 0.2,
    proxy_mode: str = "joint_taylor_second_order_fisher_diag",
    l_joint_first_order: float | None = None,
    l_joint_second_order: float | None = None,
) -> dict[str, Any]:
    # Validate finite nonnegative loss, positive scale, nonnegative weights,
    # and 0 <= candidate_params <= original_params.
    normalized = float(l_joint) / float(l_scale)
    r_prune = 1.0 - float(candidate_params) / float(original_params)
    j1 = -float(task_weight) * normalized + float(prune_weight) * r_prune
    return {
        "task_score_mapping": "linear_fixed_scale",
        "L_joint_raw": float(l_joint),
        "L_joint_first_order": float(l_joint_first_order or 0.0),
        "L_joint_second_order": float(l_joint_second_order or 0.0),
        "L_scale": float(l_scale),
        "normalized_joint_loss": normalized,
        "original_params": int(original_params),
        "candidate_params": int(candidate_params),
        "R_prune": r_prune,
        "J1": j1,
        "F1": -j1,
        "proxy_mode": str(proxy_mode),
        "sqnr_main_objective_contribution": 0.0,
    }
```

- [ ] **Step 4: Route scalar formal joint mode through the linear helper**

Extend `ProxyObjectiveConfig`:

```python
task_score_mapping: str = "legacy"
joint_loss_scale: float | None = None
task_weight: float = 0.8
prune_weight: float = 0.2
```

In `ProxyObjective.evaluate`, select `compute_linear_joint_j1` only when
`task_score_mapping == "linear_fixed_scale"`. Require a positive
`joint_loss_scale`. Preserve the explicit exponential path only when the
mapping is `exponential`; do not silently fall back between mappings. Add a
calibration-only `raw_joint_loss` mapping that returns the joint first/second
terms, parameter counts, `R_prune`, and BOPS with `F1=L_joint`, but does not
produce `J1` or require `L_scale`. Only the greedy pre-calibration runner may
select this mapping.

- [ ] **Step 5: Route the CUDA batched scorer through the same formula**

In `TorchBatchedProxyScorer.evaluate_batch`, replace the formal exponential
branch with a mapping switch. For the linear mapping, compute in float64:

```python
normalized_joint_loss = l_joint / float(config.joint_loss_scale)
j1 = -float(config.task_weight) * normalized_joint_loss \
     + float(config.prune_weight) * r_prune
score = -j1
```

Add `normalized_joint_loss` and `L_scale` to output metrics. Do not emit
`S_task`, `tau`, exponent, or saturation fields for the linear formal mode.
Keep legacy tensors only inside the explicit exponential branch.

For `raw_joint_loss`, return the same raw joint/BOPS/parameter fields and use
`score=l_joint` solely to satisfy the generic batch evaluator interface. Do
not label that diagnostic score as formal J1.

- [ ] **Step 6: Run GREEN and scalar/batched regressions**

Run:

```bash
conda run -n univ2x-opt pytest -q \
  tests/test_joint_taylor_linear_score.py \
  tests/test_joint_proxy_with_fixed_mask.py \
  tests/test_joint_taylor_stage1.py \
  tests/test_joint_taylor_proxy.py
```

Expected: PASS with scalar/batched parity at the test tolerance.

- [ ] **Step 7: Commit the linear formal objective**

```bash
git add search/proxy/task_score.py search/proxy/objective.py \
  search/proxy/gpu_batch_proxy.py tests/test_joint_taylor_linear_score.py \
  tests/test_joint_proxy_with_fixed_mask.py
git diff --cached --check
git commit -m "feat: apply linear joint Taylor stage1 score"
```

### Task 3: Two-Level BOPS Admission Policy

**Files:**
- Create: `search/admission/__init__.py`
- Create: `search/admission/bops_band.py`
- Create: `tests/test_two_level_bops_admission.py`
- Modify: `search/orchestration/generation_stage2.py` (delegate the old single-row helper without changing Stage-2 count behavior yet)

**Interfaces:**
- Produces: `BopsBandPolicy`, `classify_bops_value`, and `select_bops_candidates`.
- Later plans consume the returned `admission_mode`, annotated rows, funnel, and nearest misses.

- [ ] **Step 1: Write failing policy tests**

```python
def test_primary_bops_candidates_suppress_expanded_candidates() -> None:
    policy = BopsBandPolicy(target=0.20, adjacent_targets=(0.15, 0.25))
    result = select_bops_candidates(
        [{"id": "primary", "R_BOPS": 0.204}, {"id": "expanded", "R_BOPS": 0.206}],
        policy=policy,
    )
    assert result["admission_mode"] == "primary_bops_tolerance"
    assert [row["id"] for row in result["admitted"]] == ["primary"]


def test_expanded_bops_is_used_only_when_primary_is_empty() -> None:
    policy = BopsBandPolicy(target=0.20, adjacent_targets=(0.15, 0.25))
    result = select_bops_candidates([{"id": "near", "R_BOPS": 0.206}], policy=policy)
    assert result["admission_mode"] == "expanded_bops_tolerance"
    assert result["admitted"][0]["effective_tolerance"] == 0.0075


def test_outside_expanded_bops_reports_nearest_misses() -> None:
    result = select_bops_candidates(
        [{"id": "low", "R_BOPS": 0.18}, {"id": "high", "R_BOPS": 0.22}],
        policy=BopsBandPolicy(target=0.20),
    )
    assert result["admitted"] == []
    assert result["funnel"]["bops_below_count"] == 1
    assert result["funnel"]["bops_above_count"] == 1
    assert {row["id"] for row in result["nearest_misses"]} == {"low", "high"}
```

- [ ] **Step 2: Run RED**

Run: `conda run -n univ2x-opt pytest -q tests/test_two_level_bops_admission.py`

Expected: FAIL because the admission package does not exist.

- [ ] **Step 3: Implement the immutable policy and selector**

```python
@dataclass(frozen=True)
class BopsBandPolicy:
    target: float
    primary_tolerance: float = 0.005
    expanded_tolerance: float = 0.0075
    adjacent_targets: tuple[float, ...] = ()


def select_bops_candidates(
    rows: Sequence[Mapping[str, Any]],
    *,
    policy: BopsBandPolicy,
    retention_key: str = "R_BOPS",
) -> dict[str, Any]:
    # Classify finite rows as below/primary/expanded_only/above.
    # Use primary rows whenever present. Otherwise use expanded_only rows that
    # are not in any adjacent target's primary interval. Sort admitted rows by
    # abs(error), then descending finite J1 when present, otherwise ascending
    # L_joint_raw, then phenotype/candidate hash.
    # Return the full funnel and the nearest row on each side when none pass.
```

Every admitted row must include target, primary/effective tolerances, signed
error, nearest adjacent budget, and reason. `fixed_bops_admission` becomes a
thin one-value compatibility wrapper over `classify_bops_value`.

- [ ] **Step 4: Run GREEN and existing BOPS tests**

Run:

```bash
conda run -n univ2x-opt pytest -q \
  tests/test_two_level_bops_admission.py \
  tests/test_search_generation_stage2.py \
  tests/test_search_bops_budget.py \
  tests/test_search_bops_proxy.py
```

Expected: PASS; existing fixed interval boundaries remain unchanged.

- [ ] **Step 5: Commit the admission policy**

```bash
git add search/admission search/orchestration/generation_stage2.py \
  tests/test_two_level_bops_admission.py
git diff --cached --check
git commit -m "feat: add auditable two-level BOPS admission"
```

### Task 4: Legal Greedy Action Enumeration

**Files:**
- Create: `search/greedy/__init__.py`
- Create: `search/greedy/legal_actions.py`
- Create: `tests/test_greedy_legal_actions.py`

**Interfaces:**
- Consumes: `LegalWidthGenotype`, `LegalWidthInventory`, and `precision_action_space`.
- Produces: `GreedyAction` and
  `enumerate_legal_actions(genotype: LegalWidthGenotype, *, inventory: LegalWidthInventory, precision_actions: Mapping[str, Sequence[str]]) -> list[tuple[GreedyAction, LegalWidthGenotype]]`.

- [ ] **Step 1: Write failing structure/precision orthogonality tests**

```python
def test_greedy_width_action_moves_one_adjacent_index_only(space) -> None:
    parent = full_fp32_genotype(space)
    actions = enumerate_legal_actions(parent, inventory=space.legal_width_inventory,
                                      precision_actions=space.precision_action_space)
    width_action, child = next(row for row in actions if row[0].kind == "width")
    changed = [key for key in parent.width_genes if parent.width_genes[key] != child.width_genes[key]]
    assert changed == [width_action.gene_id]
    assert child.width_genes[changed[0]] == parent.width_genes[changed[0]] - 1
    assert child.precision_genes == parent.precision_genes


def test_greedy_precision_action_preserves_width_vector(space) -> None:
    parent = full_fp32_genotype(space)
    action, child = next(
        row for row in enumerate_legal_actions(parent, inventory=space.legal_width_inventory,
                                               precision_actions=space.precision_action_space)
        if row[0].kind == "precision"
    )
    assert child.width_vector_hash == parent.width_vector_hash
    assert action.from_value == "FP32"
    assert action.to_value == "FP16"
```

- [ ] **Step 2: Run RED**

Run: `conda run -n univ2x-opt pytest -q tests/test_greedy_legal_actions.py`

Expected: FAIL because `search.greedy.legal_actions` does not exist.

- [ ] **Step 3: Implement deterministic adjacent actions**

```python
@dataclass(frozen=True)
class GreedyAction:
    kind: str
    gene_id: str
    from_value: int | str
    to_value: int | str

    @property
    def action_id(self) -> str:
        return f"{self.kind}:{self.gene_id}:{self.from_value}->{self.to_value}"
```

For each width gene, emit only `current_index-1` when valid. For each precision
group, sort its allowed actions by bits `FP32=32`, `FP16=16`, `INT8=8`, locate
the current action, and emit only the next lower deployable action. Validate
every child with `LegalWidthGenotype.validate`; sort by `action_id`.

- [ ] **Step 4: Run GREEN and legal-width invariants**

Run:

```bash
conda run -n univ2x-opt pytest -q \
  tests/test_greedy_legal_actions.py \
  tests/test_legal_width_genotype.py \
  tests/test_precision_structure_orthogonality.py \
  tests/test_nested_width_masks.py
```

Expected: PASS.

- [ ] **Step 5: Commit legal actions**

```bash
git add search/greedy tests/test_greedy_legal_actions.py
git diff --cached --check
git commit -m "feat: add legal greedy compression actions"
```

### Task 5: Bounded Targeted Greedy Search

**Files:**
- Create: `search/greedy/joint_budget_search.py`
- Create: `tests/test_joint_budget_greedy_search.py`
- Modify: `search/greedy/__init__.py`

**Interfaces:**
- Consumes: an initial `LegalWidthGenotype`, an evaluator callback, a successor callback, and `BopsBandPolicy`.
- Produces: `GreedySearchState`, `GreedyBudgetResult`, and the complete
  `run_targeted_greedy` callback interface exercised in Step 1.

- [ ] **Step 1: Write failing target, backtracking, and deterministic tie tests**

Use a fake state graph where the best immediate action jumps from `0.25` to
`0.18`, while the second action reaches `0.204`. Assert the eight-state
frontier finds the second path for target `0.20`. Also assert no state outside
`0.0075` is returned, no genotype is evaluated twice, and equal-rho actions
use higher `delta_R_prune` then stable action ID.

Core fixture:

```python
def evaluate(genotype: LegalWidthGenotype) -> Mapping[str, Any]:
    return metrics_by_hash[genotype.genotype_hash]

result = run_targeted_greedy(
    initial_genotype=initial,
    evaluate=evaluate,
    enumerate_successors=successors,
    policy=BopsBandPolicy(target=0.20, adjacent_targets=(0.15, 0.25)),
    frontier_size=8,
    max_expansions=4096,
)
assert result.terminal.metrics["R_BOPS"] == pytest.approx(0.204)
assert result.admission_mode == "primary_bops_tolerance"
```

- [ ] **Step 2: Run RED**

Run: `conda run -n univ2x-opt pytest -q tests/test_joint_budget_greedy_search.py`

Expected: FAIL because the bounded search module does not exist.

- [ ] **Step 3: Implement immutable states and results**

```python
@dataclass(frozen=True)
class GreedySearchState:
    genotype: LegalWidthGenotype
    metrics: Mapping[str, Any]
    parent_hash: str = ""
    action_id: str = ""


@dataclass(frozen=True)
class GreedyBudgetResult:
    target: float
    status: str
    terminal: GreedySearchState | None
    admission_mode: str
    accepted_path_states: tuple[GreedySearchState, ...]
    expansion_count: int
    failure_reason: str = ""
```

At each depth, expand the current frontier, reject nonfinite/nonpositive BOPS
savings, score transitions with `delta_L_joint/delta_R_BOPS_saved`, deduplicate
children by genotype hash, and retain the best eight. Record at most 4,096
unique evaluated states. Collect primary terminals first; only if none exist
after frontier exhaustion use expanded terminals. Select terminal by lower
`L_joint_raw`, higher `R_prune`, smaller BOPS error, then genotype hash.

- [ ] **Step 4: Run GREEN**

Run: `conda run -n univ2x-opt pytest -q tests/test_joint_budget_greedy_search.py`

Expected: PASS.

- [ ] **Step 5: Commit bounded greedy search**

```bash
git add search/greedy/joint_budget_search.py search/greedy/__init__.py \
  tests/test_joint_budget_greedy_search.py
git diff --cached --check
git commit -m "feat: add bounded joint BOPS greedy search"
```

### Task 6: Six-Budget Greedy Orchestration And Formal Config

**Files:**
- Create: `search/orchestration/legal_width_greedy.py`
- Create: `search/configs/lidar_pyramid_4090_greedy_six_budget.yaml`
- Create: `tests/test_legal_width_greedy_orchestration.py`
- Modify: `search/orchestration/lidar_pyramid_search.py`
- Modify: `search/cli.py`

**Interfaces:**
- Produces: `run_six_budget_greedy(context: Any, proxy: Any, run_dir: str | Path, config: Mapping[str, Any]) -> dict[str, Any]` and CLI flags `--greedy-only` and `--joint-loss-scale PATH`.
- Output includes one proxy endpoint per feasible budget, every accepted path state, the BOPS funnel, and `joint_loss_scale.json`.

- [ ] **Step 1: Write failing fake-context orchestration test**

```python
result = run_six_budget_greedy(
    context=fake_legal_width_context,
    proxy=fake_proxy,
    run_dir=tmp_path,
    config={
        "targets": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        "primary_tolerance": 0.005,
        "expanded_tolerance": 0.0075,
        "frontier_size": 8,
        "max_expansions": 4096,
    },
)
assert result["target_count"] == 6
assert (tmp_path / "greedy" / "greedy_summary.json").is_file()
assert (tmp_path / "joint_loss_scale.json").stat().st_mode & 0o222 == 0
assert all(row["normal_candidate_repair_invoked"] is False for row in result["path_states"])
```

Add a config contract test asserting weight-only Taylor,
`task_score_mapping=raw_joint_loss`, strict-FP32 BOPS reference, the six
targets, and no AP gate fields. The later Plan-2 config has a separate contract
test requiring `linear_fixed_scale` before GA starts.

- [ ] **Step 2: Run RED**

Run:

```bash
conda run -n univ2x-opt pytest -q \
  tests/test_legal_width_greedy_orchestration.py
```

Expected: FAIL because the orchestrator and config do not exist.

- [ ] **Step 3: Implement the orchestrator**

`run_six_budget_greedy` must:

1. create one all-keep strict-FP32 `LegalWidthGenotype`;
2. evaluate legal successors through the existing `Stage1ProxyEvaluator`;
3. run budgets independently while memoizing metrics by genotype hash;
4. write `budget_005`, `budget_010`, `budget_015`, `budget_020`, `budget_025`,
   and `budget_030` trace and endpoint JSON files;
5. merge unique accepted path states with configured anchor seed rows;
6. call `calibrate_joint_loss_scale` and write a read-only scale artifact;
7. return endpoint `ProxyCandidateRecord` objects for later deployment without
   building an engine in this plan.

The output summary must distinguish strict, expanded, and infeasible budgets.

- [ ] **Step 4: Add runner and CLI routing**

Add `--greedy-only` and `--joint-loss-scale PATH`. The latter overrides only
`proxy.joint_loss_scale_path` in the resolved runtime config and records the
absolute path and SHA256. In `LidarPyramidTwoStageSearch.run`, after legal-width
space/proxy construction and before GA, call the greedy orchestrator when
`greedy.enabled=true`. If `greedy_only`, write the run manifest and return
without Stage-2. Replace `_load_joint_proxy_scale` with a mapping-aware loader:
formal configs require `linear_fixed_scale`; legacy configs may explicitly
request `exponential`.

- [ ] **Step 5: Add the formal greedy config**

The YAML must include:

```yaml
greedy:
  enabled: true
  targets: [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
  primary_tolerance: 0.005
  expanded_tolerance: 0.0075
  frontier_size: 8
  max_expansions: 4096
  deploy_final_only: true

proxy:
  proxy_mode: joint_taylor_second_order_fisher_diag
  task_score_mapping: raw_joint_loss
  include_activation_taylor: false
  sqnr_main_objective_weight: 0.0
  bops_reference: original_strict_fp32
```

Copy checkpoint, Fisher, calibration manifest, legal-width, and strongly typed
paths from the accepted 4090 legal-width config. Do not duplicate model logic.

- [ ] **Step 6: Run GREEN and related Stage-1 tests**

Run:

```bash
conda run -n univ2x-opt pytest -q \
  tests/test_legal_width_greedy_orchestration.py \
  tests/test_joint_loss_scale.py \
  tests/test_joint_taylor_linear_score.py \
  tests/test_two_level_bops_admission.py \
  tests/test_greedy_legal_actions.py \
  tests/test_joint_budget_greedy_search.py \
  tests/test_legal_width_ga_orchestration.py \
  tests/test_joint_proxy_with_fixed_mask.py
```

Expected: PASS.

- [ ] **Step 7: Compile, inspect, and commit**

```bash
conda run -n univ2x-opt python -m py_compile \
  search/proxy/joint_loss_scale.py \
  search/proxy/task_score.py \
  search/proxy/objective.py \
  search/proxy/gpu_batch_proxy.py \
  search/admission/bops_band.py \
  search/greedy/legal_actions.py \
  search/greedy/joint_budget_search.py \
  search/orchestration/legal_width_greedy.py \
  search/orchestration/lidar_pyramid_search.py \
  search/cli.py
git diff --check
git status --short
git add search/orchestration/legal_width_greedy.py \
  search/orchestration/lidar_pyramid_search.py search/cli.py \
  search/configs/lidar_pyramid_4090_greedy_six_budget.yaml \
  tests/test_legal_width_greedy_orchestration.py
git commit -m "feat: orchestrate six-budget greedy joint search"
```

### Task 7: Plan-1 Regression Gate And Handoff

**Files:**
- Modify: `docs/codex_handoffs/4090_GA_EXPLICIT_QDQ_PROGRESS_20260714.md`

**Interfaces:**
- Produces: a committed, pushed implementation checkpoint ready for the Stage-2/GA orchestration plan.

- [ ] **Step 1: Run the complete focused regression gate**

```bash
conda run -n univ2x-opt pytest -q \
  tests/test_joint_loss_scale.py \
  tests/test_joint_taylor_linear_score.py \
  tests/test_two_level_bops_admission.py \
  tests/test_greedy_legal_actions.py \
  tests/test_joint_budget_greedy_search.py \
  tests/test_legal_width_greedy_orchestration.py \
  tests/test_joint_taylor_proxy.py \
  tests/test_joint_proxy_with_fixed_mask.py \
  tests/test_legal_width_genotype.py \
  tests/test_deterministic_width_decoder.py \
  tests/test_precision_structure_orthogonality.py \
  tests/test_nested_width_masks.py \
  tests/test_legal_width_mutation.py \
  tests/test_legal_width_crossover.py \
  tests/test_exception_only_repair.py \
  tests/test_search_bops_proxy.py \
  tests/test_search_bops_budget.py
```

Expected: all selected tests pass. Record exact count and duration.

- [ ] **Step 2: Append the 4090 progress anchor**

Record files, formulas, scale semantics, tests, and explicit states:

```text
LINEAR_JOINT_SCORE_IMPLEMENTED=true
TWO_LEVEL_BOPS_ADMISSION_IMPLEMENTED=true
GREEDY_PROXY_SEARCH_IMPLEMENTED=true
GREEDY_REAL_DEPLOYMENT_STARTED=false
GA_SEARCH_STARTED=false
```

End with a Shanghai timestamp and next round number.

- [ ] **Step 3: Commit, verify branch, and push**

```bash
git add docs/codex_handoffs/4090_GA_EXPLICIT_QDQ_PROGRESS_20260714.md
git commit -m "docs: record linear greedy search implementation"
git diff --check
git merge-base --is-ancestor b862b3d8ad061bd12580776226c75f564918298d HEAD
git fetch origin feature/heal-compress-h800-sync-4090
git rev-list --left-right --count HEAD...origin/feature/heal-compress-h800-sync-4090
git push origin feature/heal-compress-h800-sync-4090
```

Expected: push updates only the 4090 branch; no force push.
