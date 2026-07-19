# DiscoNet Family Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generalize the production legal-width Greedy/GA search to HEAL DiscoNet, reduce completed candidate artifact duplication, and run six real BOPS-budget searches with one 15-generation GA seed.

**Architecture:** Preserve the Pyramid path behind its current adapter and introduce a model-family protocol whose DiscoNet implementation owns model loading, fixedK export, canonical precision capability, and Stage-2 evaluation hooks. Completed candidate structure artifacts move to a run-level content-addressed gzip store with compact per-candidate manifests; active candidates retain temporary full plans until completion.

**Tech Stack:** Python 3.9/PyTorch in `univ2x-opt`; ONNX, TensorRT 10.9, explicit Q/DQ, and plugin runtime in `modelopt`; pytest; YAML; CUDA on RTX 4090.

## Global Constraints

- Work only on `feature/heal-compress-4090-cnn-softfusion-family` in `.worktrees/cnn-softfusion-family`.
- Preserve ancestor `b862b3d8ad061bd12580776226c75f564918298d`.
- Do not modify the running Pyramid worktree, process, outputs, or H800 branch.
- Budgets are exactly `0.05, 0.10, 0.15, 0.20, 0.25, 0.30`.
- GA uses one seed, population 64, offspring 64, 15 generations, and at most five Stage-2 engines per generation.
- Stage-2 remains physical pruning plus strongly typed TensorRT and real GPU evaluation.
- Do not commit PTH, ONNX, plan, calibration cache/engine, or tensor dumps.
- Use fail-closed precision, structure, BOPS, and evaluation gates.

---

### Task 1: Content-addressed completed-candidate audit artifacts

**Files:**
- Create: `search/artifacts/candidate_audit_store.py`
- Modify: `search/orchestration/stage2_artifact_compaction.py`
- Modify: `search/stage2/candidate_artifacts.py`
- Test: `tests/test_candidate_audit_store.py`
- Test: `tests/test_stage2_artifact_compaction.py`

**Interfaces:**
- Produces: `CandidateAuditStore.put_json(role, payload) -> ArtifactReference`.
- Produces: `finalize_completed_candidate(candidate_dir, store, completion_marker) -> dict[str, Any]`.
- Produces: `resolve_artifact_reference(run_root, reference) -> Any`.

- [x] **Step 1: Write failing tests for deterministic gzip blobs, cross-candidate deduplication, reference resolution, and active-generation refusal.**

```python
def test_completed_candidates_share_one_structure_blob(tmp_path):
    store = CandidateAuditStore(tmp_path / "audit_store")
    first = store.put_json("physical_plan", {"operations": [{"module": "x"}]})
    second = store.put_json("physical_plan", {"operations": [{"module": "x"}]})
    assert first.sha256 == second.sha256
    assert first.relative_path == second.relative_path
    assert len(list((tmp_path / "audit_store").rglob("*.json.gz"))) == 1
```

- [x] **Step 2: Run the focused test and verify RED.**

Run: `/home/lixingfeng/anaconda3/envs/univ2x-opt/bin/python -m pytest -q tests/test_candidate_audit_store.py`

Expected: import failure for `search.artifacts.candidate_audit_store`.

- [x] **Step 3: Implement deterministic canonical JSON, gzip with `mtime=0`, atomic replacement, SHA verification, and safe reference resolution.**

```python
@dataclass(frozen=True)
class ArtifactReference:
    role: str
    sha256: str
    relative_path: str
    uncompressed_size: int
    compressed_size: int

class CandidateAuditStore:
    def put_json(self, role: str, payload: Any) -> ArtifactReference: ...
```

- [x] **Step 4: Implement completion-gated finalization that writes `candidate_audit_manifest.json` and `structure_plan_summary.json`, verifies references, then removes only configured redundant aliases.**

```python
REDUNDANT_STRUCTURE_INPUTS = (
    "physical_plan.json",
    "physical_pruning_plan.json",
    "legalized_plan.json",
    "pruning_request.json",
    "sampling_pruning_request.json",
)
```

- [x] **Step 5: Extend legacy compaction with `--dry-run` and `--content-addressed`, requiring `stage2_artifact_retention.json` before mutation.**

- [x] **Step 6: Run tests and verify GREEN.**

Run: `/home/lixingfeng/anaconda3/envs/univ2x-opt/bin/python -m pytest -q tests/test_candidate_audit_store.py tests/test_stage2_artifact_compaction.py`

Expected: all pass.

- [x] **Step 7: Commit.**

```bash
git add search/artifacts/candidate_audit_store.py search/orchestration/stage2_artifact_compaction.py search/stage2/candidate_artifacts.py tests/test_candidate_audit_store.py tests/test_stage2_artifact_compaction.py
git commit -m "feat: compact completed candidate audit artifacts"
```

### Task 2: Model-family protocol and DiscoNet checkpoint compatibility

**Files:**
- Create: `search/integration/lidar_family.py`
- Create: `search/integration/lidar_family_registry.py`
- Create: `search/integration/disconet_compat.py`
- Modify: `search/integration/model_provider.py`
- Test: `tests/test_lidar_family_registry.py`
- Test: `tests/test_disconet_model_provider.py`

**Interfaces:**
- Produces: `HEALLidarFamilySpec` dataclass.
- Produces: `get_lidar_family_spec(name: str) -> HEALLidarFamilySpec`.
- Produces: `install_disconet_compat_module() -> dict[str, Any]`.
- Produces: `load_heal_lidar_model(..., family: HEALLidarFamilySpec) -> HEALLidarModelBundle`.

- [x] **Step 1: Write failing registry tests for `lidar_pyramid`, `lidar_disco`, and deferred `lidar_fcooper`.**

```python
def test_disco_family_has_weighted_soft_fusion():
    spec = get_lidar_family_spec("lidar_disco")
    assert spec.fusion_kind == "disconet"
    assert spec.export_recipe == "soft_fusion_fixed_k"
```

- [x] **Step 2: Write failing PixelWeightLayer topology and state-key tests.**

```python
def test_disconet_compat_pixel_weight_topology():
    layer = PixelWeightLayer(256)
    assert tuple(layer.conv1_1.weight.shape) == (128, 512, 1, 1)
    assert tuple(layer.conv1_4.weight.shape) == (1, 8, 1, 1)
```

- [x] **Step 3: Run tests and verify RED.**

Run: `/home/lixingfeng/anaconda3/envs/univ2x-opt/bin/python -m pytest -q tests/test_lidar_family_registry.py tests/test_disconet_model_provider.py`

- [x] **Step 4: Implement the family schema and registry without changing Pyramid defaults.**

```python
@dataclass(frozen=True)
class HEALLidarFamilySpec:
    name: str
    fusion_kind: str
    export_recipe: str
    output_names: tuple[str, ...]
    fixed_k: int
    plugin_boundary_dtype: str
    compatibility_installer: Callable[[], Mapping[str, Any]] | None = None
```

- [x] **Step 5: Implement and install the local DiscoNet compatibility module only when the family is `lidar_disco`.**

- [x] **Step 6: Generalize the model bundle and add strict weighted-key audit; preserve `load_lidar_pyramid_model` as a compatibility wrapper.**

- [x] **Step 7: Load the real checkpoint on CPU and assert zero missing/unexpected weighted keys, 171 source state entries, and checkpoint SHA identity.**

Run: `/home/lixingfeng/anaconda3/envs/univ2x-opt/bin/python -m pytest -q tests/test_disconet_model_provider.py -m integration`

- [x] **Step 8: Commit.**

```bash
git add search/integration/lidar_family.py search/integration/lidar_family_registry.py search/integration/disconet_compat.py search/integration/model_provider.py tests/test_lidar_family_registry.py tests/test_disconet_model_provider.py
git commit -m "feat: add HEAL lidar model family registry"
```

### Task 3: DiscoNet fixedK soft-fusion export wrapper

**Files:**
- Create: `search/integration/softfusion_trt_export.py`
- Modify: `search/integration/trt_compatible_export.py`
- Test: `tests/test_softfusion_trt_export.py`

**Interfaces:**
- Produces: `SearchTensorRTCompatibleSoftFusion`.
- Produces: `build_family_trt_export_module(model, family, output_names, fixed_k) -> nn.Module`.

- [x] **Step 1: Write failing shape and parity tests for DiscoFusion and MaxFusion wrappers.**

```python
def test_disco_wrapper_uses_per_modality_shrinker_before_fusion(model):
    wrapper = SearchTensorRTCompatibleSoftFusion(model, family="lidar_disco")
    assert wrapper.execution_contract == (
        "pillar_vfe", "scatter_plugin", "backbone", "shrinker", "disco_fusion", "heads"
    )
```

- [x] **Step 2: Run the test and verify RED.**

- [x] **Step 3: Implement fixedK PointPillar frontend reuse, BaseBEVBackbone execution, affine normalization, Max fusion, and Disco weighted fusion.**

- [x] **Step 4: Make family selection explicit and reject unsupported fusion recipes.**

- [x] **Step 5: Compare real checkpoint PyTorch model and export wrapper on a deterministic synthetic batch; finite bounded parity recorded on GPU 7.**

- [x] **Step 6: Run tests and commit.**

```bash
git add search/integration/softfusion_trt_export.py search/integration/trt_compatible_export.py tests/test_softfusion_trt_export.py
git commit -m "feat: export fixedK HEAL soft fusion models"
```

### Task 4: Family-aware context, canonical precision space, and Stage-2 hooks

**Files:**
- Create: `search/integration/lidar_family_context.py`
- Create: `search/stage2/lidar_family_real_evaluator.py`
- Modify: `search/integration/lidar_pyramid_context.py`
- Modify: `search/stage2/lidar_pyramid_real_evaluator.py`
- Modify: `search/stage2/candidate_worker.py`
- Test: `tests/test_lidar_family_context.py`
- Test: `tests/test_lidar_family_stage2.py`

**Interfaces:**
- Produces: `LidarFamilySearchContext` extending the shared context fields with `family_spec` and `export_module_factory`.
- Produces: `LidarFamilyRealEvaluator` using family hooks while retaining the production physical/QDQ/TRT/evaluation core.

- [ ] **Step 1: Write failing tests that Pyramid and Disco contexts select different exporters but identical shared BOPS and process-pool contracts.**

- [ ] **Step 2: Write failing tests that Disco weighted fusion layers enter canonical precision mapping and functional Softmax remains an explicit floating contract.**

- [ ] **Step 3: Run tests and verify RED.**

- [ ] **Step 4: Extract only family-varying hooks from the Pyramid context/evaluator; keep existing Pyramid public classes and defaults unchanged.**

- [ ] **Step 5: Add Disco merge/Softmax weighted-sum contracts and fail on unresolved weighted ONNX layers or precision fallback.**

- [ ] **Step 6: Route persistent workers by `model_family` in the request context.**

- [ ] **Step 7: Run context, QDQ, strongly typed, merge, realized-BOPS, worker, and Pyramid regression tests.**

- [ ] **Step 8: Commit.**

```bash
git add search/integration/lidar_family_context.py search/stage2/lidar_family_real_evaluator.py search/integration/lidar_pyramid_context.py search/stage2/lidar_pyramid_real_evaluator.py search/stage2/candidate_worker.py tests/test_lidar_family_context.py tests/test_lidar_family_stage2.py
git commit -m "feat: add family aware strongly typed stage2"
```

### Task 5: Single-seed six-budget orchestration and configs

**Files:**
- Modify: `search/orchestration/legal_width_six_budget_ga.py`
- Modify: `search/orchestration/lidar_pyramid_search.py`
- Modify: `search/cli.py`
- Create: `search/configs/lidar_disco_4090_greedy_six_budget.yaml`
- Create: `search/configs/lidar_disco_4090_joint_six_budget_ga.yaml`
- Test: `tests/test_single_seed_six_budget_ga.py`
- Test: `tests/test_lidar_family_cli.py`

**Interfaces:**
- Produces: family-neutral six-budget orchestration accepting `independent_seeds=1`.
- Produces: Disco configs with 15 generations and Stage-2 Top-5 cap.

- [ ] **Step 1: Write failing config-contract tests for the exact budgets, one seed, 15 generations, population/offspring 64, and Top-5 cap.**

```python
def test_disco_ga_contract(config):
    assert config["search"]["targets"] == [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    assert config["search"]["independent_seeds"] == 1
    assert config["search"]["generations"] == 15
    assert config["search"]["topk_stage2"] == 5
```

- [ ] **Step 2: Write failing orchestration tests proving build count never exceeds five and no backfill build occurs after the selected Top-5.**

- [ ] **Step 3: Implement family-neutral naming while preserving three-seed Pyramid behavior for its existing config.**

- [ ] **Step 4: Add the two Disco configs with content-addressed artifact retention enabled.**

- [ ] **Step 5: Run tests and commit.**

```bash
git add search/orchestration/legal_width_six_budget_ga.py search/orchestration/lidar_pyramid_search.py search/cli.py search/configs/lidar_disco_4090_greedy_six_budget.yaml search/configs/lidar_disco_4090_joint_six_budget_ga.yaml tests/test_single_seed_six_budget_ga.py tests/test_lidar_family_cli.py
git commit -m "feat: configure DiscoNet six budget search"
```

### Task 6: DiscoNet deployment readiness experiment

**Files:**
- Create: `docs/codex_handoffs/4090-DISCONET-STRONGLY-TYPED-READINESS.md`
- Update: `docs/codex_handoffs/4090-CNN-SOFTFUSION-GENERALIZATION-20260719.md`

**Interfaces:**
- Produces: timestamped readiness output and `DISCONET_READY_FOR_SEARCH=true|false`.

- [ ] **Step 1: Select GPUs that do not overlap active Pyramid processes and record UUID/utilization/memory/processes.**

- [ ] **Step 2: Generate fixed smoke10, calibration train200, fixed500, and full-validation manifests with hashes.**

- [ ] **Step 3: Run physical all-keep export parity and ONNX checker.**

- [ ] **Step 4: Build fresh strongly typed strict-FP32, strict-FP16, and supported explicit-QDQ baselines in `modelopt`.**

- [ ] **Step 5: Run smoke10 and fixed500 with GPU AP IoU, evaluated count exact, and zero skipped frames.**

- [ ] **Step 6: Audit EngineInspector, requested/realized precision, QDQ, fusion Softmax/weighted-sum contract, BOPS, and plugin boundary.**

- [ ] **Step 7: Stop if readiness fails; record direct evidence without weakening gates.**

- [ ] **Step 8: Commit lightweight evidence and report.**

```bash
git add docs/codex_handoffs/4090-DISCONET-STRONGLY-TYPED-READINESS.md docs/codex_handoffs/4090-CNN-SOFTFUSION-GENERALIZATION-20260719.md
git commit -m "docs: record DiscoNet deployment readiness"
```

### Task 7: Six-budget Greedy and real endpoint validation

**Files:**
- Update: `docs/codex_handoffs/4090-CNN-SOFTFUSION-GENERALIZATION-20260719.md`

**Interfaces:**
- Produces: six unique or explicitly infeasible Greedy endpoints with real full-validation and formal latency evidence.

- [ ] **Step 1: Start a fresh timestamped Greedy run using `lidar_disco_4090_greedy_six_budget.yaml`.**

- [ ] **Step 2: Verify all six budget endpoints have structure/precision/BOPS lineage and no normal repair invocation.**

- [ ] **Step 3: Build each unique endpoint once, run smoke10, then full validation.**

- [ ] **Step 4: Stop all builders and measure formal latency serially on one idle GPU.**

- [ ] **Step 5: Generate compact results and update the handoff report.**

### Task 8: One-seed 15-generation six-budget real GA

**Files:**
- Update: `docs/codex_handoffs/4090-CNN-SOFTFUSION-GENERALIZATION-20260719.md`

**Interfaces:**
- Produces: 90 generation records, at most 450 engine attempts, generation winners, full-validation budget winners, and Pareto summaries.

- [ ] **Step 1: Start a fresh GA run using `lidar_disco_4090_joint_six_budget_ga.yaml` and the completed Greedy seed manifest.**

- [ ] **Step 2: Monitor per-budget generation counts, proxy evaluations, BOPS admissions, Stage-2 build counts, and artifact-store growth without per-generation policy changes.**

- [ ] **Step 3: Verify each generation builds zero to five candidates and records the reason when fewer are admitted.**

- [ ] **Step 4: Run fixed500 on admitted candidates; a sole candidate becomes the generation winner without redundant comparison evaluation.**

- [ ] **Step 5: Full-validate all generation winners, select budget winners, and measure formal latency serially.**

- [ ] **Step 6: Generate mAP/BOPS, mAP/parameter-retention, and mAP/formal-latency Pareto tables and plots.**

### Task 9: Final verification, report, commit, and push

**Files:**
- Update: `docs/codex_handoffs/4090-CNN-SOFTFUSION-GENERALIZATION-20260719.md`
- Create: `docs/codex_handoffs/4090-DISCONET-GREEDY-GA-SEARCH-REPORT.md`

**Interfaces:**
- Produces: final branch history and remote evidence.

- [ ] **Step 1: Run all new tests and relevant Pyramid regression tests.**

Run: `/home/lixingfeng/anaconda3/envs/univ2x-opt/bin/python -m pytest -q tests/test_candidate_audit_store.py tests/test_stage2_artifact_compaction.py tests/test_lidar_family_registry.py tests/test_disconet_model_provider.py tests/test_softfusion_trt_export.py tests/test_lidar_family_context.py tests/test_lidar_family_stage2.py tests/test_single_seed_six_budget_ga.py tests/test_lidar_family_cli.py`

- [ ] **Step 2: Compile every modified Python file.**

Run: `/home/lixingfeng/anaconda3/envs/univ2x-opt/bin/python -m py_compile <modified Python files>`

- [ ] **Step 3: Run `git diff --check`, verify no large artifacts are staged, and verify the H800 ancestor.**

- [ ] **Step 4: Write the final report with Greedy/GA metrics, engine hashes, precision audits, failures, artifact savings, and reproduction commands.**

- [ ] **Step 5: Commit reports and push only `feature/heal-compress-4090-cnn-softfusion-family`.**

```bash
git fetch origin feature/heal-compress-4090-cnn-softfusion-family
git rev-list --left-right --count HEAD...origin/feature/heal-compress-4090-cnn-softfusion-family
git push origin feature/heal-compress-4090-cnn-softfusion-family
```
