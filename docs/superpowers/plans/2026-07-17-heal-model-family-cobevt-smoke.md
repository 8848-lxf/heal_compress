# HEAL Model-Family CoBEVT Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an isolated `lidar_cobevt` model-family recipe that reuses the formal legal-width greedy/GA framework and builds audited strongly typed TensorRT engines for one bounded real-GPU compatibility smoke without changing the accepted `lidar_pyramid` deployment path.

**Architecture:** A lazy model-family registry preserves the current pyramid runner as the default and dispatches CoBEVT to a separate recipe. CoBEVT owns checkpoint/input, pruning, typed export, canonical precision, QDQ/merge, plugin capability, and evaluation contracts while reusing existing search operators, cache signatures, strongly typed builder worker, and process orchestration. Operator capability is proven in stages; a new plugin is not written unless a minimized TensorRT 10.9 parser or parity failure demonstrates that ONNX-native lowering is insufficient.

**Tech Stack:** Python 3.9, PyTorch/HEAL/OpenCOOD, ONNX, TensorRT 10.9, CUDA 11.8, pytest, existing legal-width/joint-Taylor search modules, `univ2x-opt` for model/search and `modelopt` for TensorRT deployment.

## Global Constraints

- Develop and push only `feature/heal-compress-4090-cobevt-family`, derived from `feature/heal-compress-h800-sync-4090` at `6d184ac`.
- Preserve ancestor `b862b3d8ad061bd12580776226c75f564918298d`; never modify or push `feature/heal-compress-h800`; never force push.
- Do not stop, restart, resume, or alter the active six-budget pyramid search.
- Do not change pyramid canonical mapping, typed export, QDQ, plugin boundary, evaluator, or cache semantics.
- CoBEVT checkpoint is `/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_cobevt/net_epoch_bestval_at19.pth` with SHA256 `67b0f2f00d74fea4912b4fdb902c1150146c733201e5dc36d3358f5ba605cfd4`.
- Resolve TensorRT to `/home/lixingfeng/UniAD_examine/HEAL/prune_model/TensorRT-10.9_x86_cu118`; fail if the resolved tree/version differs.
- Production engines use `--stronglyTyped --noTF32` and reject `--fp16`, `--int8`, precision constraints, layer precision overrides, layer output-type overrides, and weakly typed fallback.
- Reuse `PointPillarScatterTRT` only after model-family capability validation; keep it floating point and outside precision genes.
- Use DataLoader workers 8 and the GPU post-processing path for fixed50 evaluation.
- Store large artifacts under `/data/lxf/heal_data/outputs/`; never commit PTH, ONNX, PLAN/engine, cache, NPZ/NPY, tensor dumps, or plugin binaries.
- Execute inline; do not delegate to subagents.

---

### Task 1: Add Lazy Model-Family Dispatch With Pyramid Freeze Guards

**Files:**
- Create: `search/model_families/__init__.py`
- Create: `search/model_families/contracts.py`
- Create: `search/model_families/registry.py`
- Modify: `search/cli.py`
- Create: `tests/test_model_family_registry.py`
- Create: `tests/test_pyramid_family_freeze.py`

**Interfaces:**
- Produces: `model_family_name(config: Mapping[str, Any]) -> str`.
- Produces: `create_family_runner(*, config, checkpoint, output_root, resume=None) -> SearchRunner`.
- Produces: `ModelFamilyCapabilityManifest` with family, recipe version, checkpoint/config hashes, export/plugin/precision capabilities, and stable SHA256.
- Preserves: configs without `model.family` instantiate the current `LidarPyramidTwoStageSearch` with unchanged arguments.

- [ ] **Step 1: Write failing dispatch and freeze tests**

```python
def test_missing_family_defaults_to_existing_pyramid_runner(monkeypatch, tmp_path):
    seen = {}
    class ExistingRunner:
        def __init__(self, **kwargs):
            seen.update(kwargs)
    monkeypatch.setattr(registry, "_load_pyramid_runner", lambda: ExistingRunner)
    runner = registry.create_family_runner(
        config={"model": {}}, checkpoint="model.pth", output_root=tmp_path
    )
    assert isinstance(runner, ExistingRunner)
    assert registry.model_family_name({"model": {}}) == "lidar_pyramid"

def test_cobevt_dispatch_is_lazy_and_does_not_import_pyramid_quant_modules(monkeypatch, tmp_path):
    class CobevtRunner: ...
    monkeypatch.setattr(registry, "_load_cobevt_runner", lambda: CobevtRunner)
    assert isinstance(registry.create_family_runner(
        config={"model": {"family": "lidar_cobevt"}},
        checkpoint="model.pth", output_root=tmp_path,
    ), CobevtRunner)
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `conda run -n univ2x-opt pytest -q tests/test_model_family_registry.py tests/test_pyramid_family_freeze.py`

Expected: collection fails because `search.model_families` does not exist.

- [ ] **Step 3: Implement contracts, lazy registry, and CLI dispatch**

Implement the registry with imports inside `_load_pyramid_runner()` and
`_load_cobevt_runner()`. Replace only the direct constructor in `search/cli.py`;
leave the pyramid runner and its deployment imports untouched.

- [ ] **Step 4: Run GREEN and existing CLI regression**

Run: `conda run -n univ2x-opt pytest -q tests/test_model_family_registry.py tests/test_pyramid_family_freeze.py tests/test_search_tool_adapters.py`

Expected: all pass; the old CLI tests still observe the same pyramid arguments.

- [ ] **Step 5: Commit and push**

```bash
git add search/model_families search/cli.py tests/test_model_family_registry.py tests/test_pyramid_family_freeze.py
git commit -m "feat: add isolated HEAL model-family dispatch"
git push -u origin feature/heal-compress-4090-cobevt-family
```

### Task 2: Add CoBEVT Model, Checkpoint, Manifest, And Fixed-K Capabilities

**Files:**
- Create: `search/model_families/lidar_cobevt/__init__.py`
- Create: `search/model_families/lidar_cobevt/model_capability.py`
- Create: `search/model_families/lidar_cobevt/input_contract.py`
- Create: `tests/test_lidar_cobevt_model_capability.py`
- Create: `tests/test_lidar_cobevt_fixed_k_contract.py`

**Interfaces:**
- Produces: `CobevtModelCapability.load() -> CobevtModelBundle` containing model, `HEALLiDARAdapter`, config, checkpoint-load report, and hashes.
- Produces: `derive_fixed_k(records, *, alignment) -> FixedKContract` with maximum observed voxels, rounded fixed K, overflow count, and manifest hash.
- Produces: `validate_scatter_capability(model) -> ScatterCapability` proving plugin input semantics and floating boundary.

- [ ] **Step 1: Write failing checkpoint and fixed-K tests**

```python
def test_checkpoint_hash_and_model_family_are_bound():
    capability = CobevtModelCapability(CHECKPOINT, CONFIG, HEAL_ROOT)
    report = capability.preflight()
    assert report.checkpoint_sha256 == EXPECTED_SHA
    assert report.model_family == "lidar_cobevt"

def test_fixed_k_is_derived_and_has_zero_overflow():
    contract = derive_fixed_k([101, 256, 513], alignment=256)
    assert contract.fixed_k == 768
    assert contract.overflow_count == 0
    assert contract.source_max_k == 513

def test_cobevt_fixed_k_does_not_inherit_pyramid_constant():
    contract = derive_fixed_k([24000], alignment=256)
    assert contract.fixed_k == 24064
    assert contract.fixed_k != 29696
```

- [ ] **Step 2: Verify RED**

Run: `conda run -n univ2x-opt pytest -q tests/test_lidar_cobevt_model_capability.py tests/test_lidar_cobevt_fixed_k_contract.py`

Expected: import failure for the new CoBEVT modules.

- [ ] **Step 3: Implement model and input contracts**

Load HEAL lazily through `HEALLiDARAdapter`, inspect `load_state_dict` missing and
unexpected keys, bind the exact hashes, and fail on a wrong model core method.
Implement fixed-K derivation from actual calibration/evaluation records without
embedding `29696` in CoBEVT code.

- [ ] **Step 4: Run GREEN and a real CPU model-load preflight**

Run:

```bash
conda run --no-capture-output -n univ2x-opt pytest -q \
  tests/test_lidar_cobevt_model_capability.py \
  tests/test_lidar_cobevt_fixed_k_contract.py
```

Expected: all tests pass and the real checkpoint report records no unexplained
missing/unexpected weighted parameters.

- [ ] **Step 5: Commit and push**

```bash
git add search/model_families/lidar_cobevt tests/test_lidar_cobevt_model_capability.py tests/test_lidar_cobevt_fixed_k_contract.py
git commit -m "feat: add CoBEVT model and input capabilities"
git push origin feature/heal-compress-4090-cobevt-family
```

### Task 3: Add CoBEVT Legal-Width Physical Pruning Recipe

**Files:**
- Create: `search/model_families/lidar_cobevt/pruning_recipe.py`
- Create: `tests/test_lidar_cobevt_pruning_recipe.py`
- Create: `tests/test_lidar_cobevt_attention_widths.py`
- Modify only if required by a failing generic test: `pruning/transformer_checker.py`
- Modify only if required by a failing generic test: `pruning/transformer_pruning_fns.py`

**Interfaces:**
- Produces: `CobevtPruningRecipe.build_inventory(model, trace) -> LegalWidthInventory`.
- Produces: `CobevtPruningRecipe.materialize(model, decoded_plan) -> PhysicalPruneReport`.
- Produces: `CobevtPruningRecipe.validate(model, decoded_plan) -> StructureAudit`.
- Consumes: existing deterministic legal-width decoder and tracer dependency groups without redefining tracer semantics.

- [ ] **Step 1: Write failing structural invariants**

```python
def test_attention_widths_are_head_aligned():
    widths = recipe.legal_fusion_widths(original_width=256, dim_head=32)
    assert widths
    assert all(width % 32 == 0 for width in widths)

def test_materialized_qkv_ffn_norm_and_heads_agree(cobevt_model, fusion_plan):
    report = recipe.materialize(cobevt_model, fusion_plan)
    assert report.passed
    assert report.qkv_out == 3 * report.embed_dim
    assert report.layernorm_dim == report.embed_dim
    assert report.prediction_head_in == report.embed_dim
    assert report.predicted_params == report.physical_params
```

- [ ] **Step 2: Verify RED**

Run: `conda run -n univ2x-opt pytest -q tests/test_lidar_cobevt_pruning_recipe.py tests/test_lidar_cobevt_attention_widths.py`

Expected: missing CoBEVT pruning recipe.

- [ ] **Step 3: Implement the minimal family pruning recipe**

Use existing Conv/Linear/LayerNorm slicing primitives. Preserve complete
`dim_head=32` heads and update QKV, output projection, feed-forward,
LayerNorm, relative-position embedding head columns, shrink output, and output
head inputs atomically. If a traced fusion closure is incomplete, mark that
domain protected with a reason instead of applying partial surgery.

- [ ] **Step 4: Verify deterministic decode and physical count**

Run:

```bash
conda run -n univ2x-opt pytest -q \
  tests/test_lidar_cobevt_pruning_recipe.py \
  tests/test_lidar_cobevt_attention_widths.py \
  tests/test_physical_replay_from_width_gene.py \
  tests/test_legal_width_inventory.py
```

Expected: all pass; decoded mask hash equals replay mask hash and predicted
parameter count equals `sum(p.numel())`.

- [ ] **Step 5: Commit and push**

```bash
git add search/model_families/lidar_cobevt/pruning_recipe.py tests/test_lidar_cobevt_pruning_recipe.py tests/test_lidar_cobevt_attention_widths.py pruning/transformer_checker.py pruning/transformer_pruning_fns.py
git commit -m "feat: add CoBEVT legal-width physical pruning recipe"
git push origin feature/heal-compress-4090-cobevt-family
```

### Task 4: Add Typed ONNX Export And Operator Capability Probe

**Files:**
- Create: `quantization/export/heal_lidar_cobevt.py`
- Create: `search/model_families/lidar_cobevt/export_recipe.py`
- Create: `search/model_families/lidar_cobevt/operator_probe.py`
- Create: `tests/test_lidar_cobevt_typed_export.py`
- Create: `tests/test_lidar_cobevt_operator_probe.py`
- Create: `tests/test_lidar_cobevt_scatter_contract.py`

**Interfaces:**
- Produces: `CobevtExportRecipe.export(model, batch, path, contract) -> TypedExportReport`.
- Produces: `probe_onnx_capabilities(onnx_path, trt_context) -> OperatorCapabilityReport`.
- Produces: a minimal reproducer path and exact failing node when checker, parser, or parity fails.

- [ ] **Step 1: Write failing export/probe tests**

```python
def test_scatter_is_float_and_not_a_precision_gene(export_report):
    assert export_report.scatter.input_dtype in {"FLOAT", "FLOAT16"}
    assert export_report.scatter.output_dtype == export_report.scatter.input_dtype
    assert "PointPillarScatterTRT" not in export_report.precision_gene_names

def test_native_attention_lowering_contains_no_unregistered_custom_ops(probe):
    assert probe.unregistered_custom_ops == []
    assert probe.weak_precision_fallback_requested is False
```

- [ ] **Step 2: Verify RED**

Run: `conda run -n univ2x-opt pytest -q tests/test_lidar_cobevt_typed_export.py tests/test_lidar_cobevt_operator_probe.py tests/test_lidar_cobevt_scatter_contract.py`

Expected: imports fail for the new export/probe modules.

- [ ] **Step 3: Implement typed wrapper and staged capability audit**

Lower einops to reshape/transpose/reduce, represent mask fill with typed ONNX
operations, keep grid coordinates floating point, and reuse the scatter custom
symbol without importing pyramid wrappers. Persist checker, shape/type,
operator inventory, and parser diagnostics separately.

- [ ] **Step 4: Run CPU tests and the real strict-FP32 export probe**

Run the tests, then export the exact checkpoint to a new timestamped directory
and invoke TensorRT parser with `--stronglyTyped --noTF32 --skipInference` in
`modelopt`. Expected outcomes are either a passing native graph or one
minimized unsupported-operation report. A parser failure is evidence, not a
reason to fall back to weak typing.

- [ ] **Step 5: Capability decision checkpoint**

If the report identifies a genuinely unsupported semantic op, stop this plan,
append the exact reproducer to the 4090 progress log, and write a focused plugin
spec before changing C++ code. If native export passes, record
`new_plugin_required=false` and continue.

- [ ] **Step 6: Commit and push**

```bash
git add quantization/export/heal_lidar_cobevt.py search/model_families/lidar_cobevt/export_recipe.py search/model_families/lidar_cobevt/operator_probe.py tests/test_lidar_cobevt_typed_export.py tests/test_lidar_cobevt_operator_probe.py tests/test_lidar_cobevt_scatter_contract.py
git commit -m "feat: add CoBEVT typed export capability probe"
git push origin feature/heal-compress-4090-cobevt-family
```

### Task 5: Add CoBEVT Canonical Precision And Strongly Typed Stage-2 Recipe

**Files:**
- Create: `search/model_families/lidar_cobevt/quantization_recipe.py`
- Create: `search/model_families/lidar_cobevt/deployment_recipe.py`
- Create: `search/stage2/lidar_cobevt_real_evaluator.py`
- Create: `tests/test_lidar_cobevt_precision_groups.py`
- Create: `tests/test_lidar_cobevt_qdq_contract.py`
- Create: `tests/test_lidar_cobevt_strongly_typed_builder.py`
- Create: `tests/test_cross_family_cache_isolation.py`

**Interfaces:**
- Produces: `CobevtQuantizationRecipe.build_capability(typed_onnx) -> PrecisionCapabilityManifest`.
- Produces: `CobevtDeploymentRecipe.build(candidate, output_dir) -> DeploymentReport`.
- Produces: `LidarCobevtRealEvaluator.evaluate(candidate, frames) -> CandidateMetrics`.
- Consumes: existing entropy calibration, strongly typed worker, cache/deployment signatures, and engine inspector parsers.

- [ ] **Step 1: Write failing precision, builder, and cache tests**

```python
def test_unsupported_int8_is_removed_not_fallback(capability):
    assert "INT8" not in capability.actions_for("fusion_net.layers.0.window_attention.norm")

def test_builder_command_is_strongly_typed(command):
    assert "--stronglyTyped" in command and "--noTF32" in command
    assert "--fp16" not in command and "--int8" not in command
    assert not any("precisionConstraints" in token for token in command)

def test_pyramid_artifact_cannot_satisfy_cobevt_cache_key(pyramid_key, cobevt_key):
    assert pyramid_key.model_family != cobevt_key.model_family
    assert pyramid_key.digest != cobevt_key.digest
```

- [ ] **Step 2: Verify RED**

Run: `conda run -n univ2x-opt pytest -q tests/test_lidar_cobevt_precision_groups.py tests/test_lidar_cobevt_qdq_contract.py tests/test_lidar_cobevt_strongly_typed_builder.py tests/test_cross_family_cache_isolation.py`

Expected: new recipe imports fail.

- [ ] **Step 3: Implement canonical mapping and explicit dtype contracts**

Map every Conv/Linear weight exactly once, protect scatter/Softmax/LayerNorm/
grid coordinates/residual merges/raw outputs until proven otherwise, and create
INT8 actions only for groups whose QDQ and inspector realization pass. Bind
model family, recipe version, fixed-K, plugin hashes, calibration manifest, and
profile hashes into cache identity.

- [ ] **Step 4: Implement the evaluator using GPU post-processing**

Reuse shared engine execution and HEAL AP utilities, select the GPU postprocess
path, configure DataLoader workers to 8, and enforce smoke10 before fixed50.

- [ ] **Step 5: Run GREEN plus existing strongly typed regressions**

Run:

```bash
conda run -n univ2x-opt pytest -q \
  tests/test_lidar_cobevt_precision_groups.py \
  tests/test_lidar_cobevt_qdq_contract.py \
  tests/test_lidar_cobevt_strongly_typed_builder.py \
  tests/test_cross_family_cache_isolation.py \
  tests/test_strongly_typed_qdq_graph.py \
  tests/test_strongly_typed_builder_policy.py
```

Expected: all pass with no changed pyramid expected counts.

- [ ] **Step 6: Commit and push**

```bash
git add search/model_families/lidar_cobevt/quantization_recipe.py search/model_families/lidar_cobevt/deployment_recipe.py search/stage2/lidar_cobevt_real_evaluator.py tests/test_lidar_cobevt_precision_groups.py tests/test_lidar_cobevt_qdq_contract.py tests/test_lidar_cobevt_strongly_typed_builder.py tests/test_cross_family_cache_isolation.py
git commit -m "feat: add CoBEVT strongly typed deployment recipe"
git push origin feature/heal-compress-4090-cobevt-family
```

### Task 6: Add Bounded Greedy And GA Smoke Orchestration

**Files:**
- Create: `search/orchestration/lidar_cobevt_smoke.py`
- Create: `search/configs/lidar_cobevt_4090_model_family_smoke.yaml`
- Modify: `search/model_families/registry.py`
- Create: `tests/test_lidar_cobevt_smoke_config.py`
- Create: `tests/test_lidar_cobevt_smoke_orchestration.py`

**Interfaces:**
- Produces: `LidarCobevtSmokeSearch.run() -> CobevtSmokeResult`.
- Reuses: `JointBudgetGreedySearch`, legal-width GA operators, feasible archive, and Stage-2 candidate cache.
- Enforces: one selected reachable band, one greedy endpoint, GA population 16, generations 3, one seed, Top-2 per generation.

- [ ] **Step 1: Write failing bounded-smoke tests**

```python
def test_smoke_scale_is_exact(config):
    assert config["ga"] == {"population_size": 16, "generations": 3, "seeds": [4090]}
    assert config["stage2"]["topk_per_generation"] == 2
    assert config["evaluation"]["smoke_frames"] == 10
    assert config["evaluation"]["screening_frames"] == 50
    assert config["evaluation"]["num_workers"] == 8

def test_band_selection_is_deterministic():
    counts = {0.10: 2, 0.15: 8, 0.20: 8, 0.25: 4, 0.30: 1}
    assert select_smoke_band(counts) == 0.20
```

- [ ] **Step 2: Verify RED**

Run: `conda run -n univ2x-opt pytest -q tests/test_lidar_cobevt_smoke_config.py tests/test_lidar_cobevt_smoke_orchestration.py`

Expected: missing CoBEVT smoke runner/config.

- [ ] **Step 3: Implement bounded orchestration**

Discover reachable counts for `{0.10,0.15,0.20,0.25,0.30}`, select the most
populated band with tie-break toward 0.20, freeze tolerance 0.0075, run one
greedy endpoint and the exact GA scale, and deduplicate physical/deployment
hashes. A generation with one candidate proceeds; zero records exhaustion.

- [ ] **Step 4: Run GREEN and a Stage-1-only real model smoke**

Run tests, then invoke the new config with Stage-1 only to prove real checkpoint
inventory, proxy, greedy, and GA execution without building engines. Confirm
at least one proxy-feasible phenotype exists before scheduling Stage-2.

- [ ] **Step 5: Commit and push**

```bash
git add search/orchestration/lidar_cobevt_smoke.py search/configs/lidar_cobevt_4090_model_family_smoke.yaml search/model_families/registry.py tests/test_lidar_cobevt_smoke_config.py tests/test_lidar_cobevt_smoke_orchestration.py
git commit -m "feat: add bounded CoBEVT greedy and GA smoke"
git push origin feature/heal-compress-4090-cobevt-family
```

### Task 7: Run The Real Strongly Typed CoBEVT Compatibility Experiment

**Files:**
- Output only: `/data/lxf/heal_data/outputs/4090_lidar_cobevt_model_family_smoke_<timestamp>/`
- Update: `docs/4090_HEAL_MODEL_FAMILY_COBEVT_PROGRESS_20260717.md`

**Interfaces:**
- Consumes: the frozen capability manifest and smoke config from Tasks 1-6.
- Produces: strict reference, greedy endpoint, per-generation GA Top-2 build/eval reports, hashes, AP, latency, GPU, and failure evidence.

- [ ] **Step 1: Record preflight and choose low-contention GPUs**

Record UUID, utilization, memory percentage, temperature, power, and processes.
Do not use a GPU above 50% memory unless no candidate is available; do not stop
external or pyramid processes. Create a fresh `/data` output directory.

- [ ] **Step 2: Freeze train200 and fixed50 manifests**

Scan voxel counts, derive fixed K, prove zero overflow, hash both manifests, and
write the capability manifest before engine construction.

- [ ] **Step 3: Build and evaluate the strict reference**

Use `modelopt` for typed export/parser/build and real GPU smoke10/fixed50.
Require engine deserialization, nonfinite count zero, evaluated 50, skipped 0,
and inspector/ONNX dtype identity.

- [ ] **Step 4: Run one greedy endpoint**

Use the selected frozen BOPS band, materialize once, build one independently
calibrated strongly typed engine, and complete smoke10/fixed50.

- [ ] **Step 5: Run GA 16 x 3 x one seed with Top-2**

Execute all three generations, build unique deployments once, backfill within
each generation when a candidate fails, and preserve every failure reason.

- [ ] **Step 6: Validate success conditions**

Require at least one greedy and one GA candidate with real engine inference and
fixed50. Report requested/legalized/realized precision identity, physical
parameter/BOPS identity, plugin boundary, AP, p50/p90/p95, and GPU provenance.

### Task 8: Final Verification, Evidence Report, And Branch Push

**Files:**
- Create: `docs/codex_handoffs/4090-lidar-cobevt-search-smoke-report.md`
- Update: `docs/4090_HEAL_MODEL_FAMILY_COBEVT_PROGRESS_20260717.md`
- Update: `docs/codex_handoffs/LATEST.md` only if it currently serves as a pointer and can reference both active tracks without hiding pyramid progress.

**Interfaces:**
- Produces: a lightweight committed report and append-only final round.
- Excludes: all large deployment artifacts.

- [ ] **Step 1: Run focused and pyramid regression suites**

Run all new tests plus legal-width, joint proxy, physical replay, QDQ, strongly
typed, cache, worker/process-pool, and CLI tests. Record exact pass/fail counts.

- [ ] **Step 2: Compile every changed Python file**

Run `conda run -n univ2x-opt python -m py_compile <changed Python files>` and
record success.

- [ ] **Step 3: Check repository hygiene**

Run `git diff --check`, `git status --short`, and an explicit tracked-file scan
for `.pth`, `.onnx`, `.plan`, `.engine`, `.cache`, `.npz`, `.npy`, `.so`, and
large files. Expected: no deployment artifacts staged.

- [ ] **Step 4: Write the report and append the timestamped progress round**

Include model/checkpoint/config hashes, fixed K, operator inventory, plugin
decision, legal widths, greedy/GA counts, all engine outcomes, fixed50 AP and
latency, requested/realized precision, GPU distribution, cache identity,
pyramid freeze evidence, failures, and reproduction commands.

- [ ] **Step 5: Commit and push**

```bash
git add docs/4090_HEAL_MODEL_FAMILY_COBEVT_PROGRESS_20260717.md docs/codex_handoffs/4090-lidar-cobevt-search-smoke-report.md docs/codex_handoffs/LATEST.md
git commit -m "docs: report CoBEVT model-family deployment smoke"
git fetch origin feature/heal-compress-4090-cobevt-family
git rev-list --left-right --count HEAD...origin/feature/heal-compress-4090-cobevt-family
git push origin feature/heal-compress-4090-cobevt-family
```

- [ ] **Step 6: Verify final lineage**

Run branch, HEAD/remote HEAD, clean status, last ten commits, diff check, and
H800 ancestor checks. Confirm the original pyramid worktree remains on
`feature/heal-compress-h800-sync-4090` and its process is still running or has
completed normally.

