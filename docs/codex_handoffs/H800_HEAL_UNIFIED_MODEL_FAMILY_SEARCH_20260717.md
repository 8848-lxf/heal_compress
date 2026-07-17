# H800 HEAL unified model-family search handoff

## Round 1 — freeze LiDAR-pyramid and establish the V2X-ViT admission layer

### Branch isolation and frozen source of truth

- Frozen LiDAR-pyramid H800 branch: `feature/heal-compress-h800`.
- Frozen commit: `1da7e3b37afa30549a81df4259b80312f449f9a2` (`docs(handoff): separate greedy and GA latency summaries`).
- New branch: `feature/heal-unified-search-h800`, created directly from the frozen commit.
- This round did not modify any existing `lidar_pyramid` search, pruning, Q/DQ, TensorRT, plugin, or evaluation source file.
- All V2X-ViT work is additive and isolated under `search/model_family/**`, plus two new scripts and one new test file.
- No GA, greedy search, physical pruning, calibration, Q/DQ rewrite, TensorRT engine build, or AP evaluation was started.

### Design decisions

The unified layer separates three states that the old model-specific path could conflate:

1. Static capability: an operator/domain could theoretically be pruned or quantized.
2. Production permission: the model-family adapter allows it after model-specific constraints.
3. Runtime evidence: ONNX mapping, semantic Q/DQ boundaries, plugin compatibility, strongly typed parsing, realized precision, physical reload, and real forward have actually passed.

GA/greedy orchestration must only consume state 3. A discovered Transformer layer or pruning domain is not automatically a legal gene.

The V2X-ViT pruning plan follows Torch-Pruning 1.6 concepts but does not directly reuse its `nn.MultiheadAttention` pruner, because HEAL uses custom Linear/Einsum attention:

- initial safe candidate: FFN hidden-width domains;
- SplitAttn hidden width: coupled `fc1 -> LayerNorm -> fc2` materializer required;
- window attention: prune whole Q/K/V head bundles and the matching output-projection inputs;
- heterogeneous HGT attention: couple all agent-type Q/K/V/A projections, both head axes of `relation_att`/`relation_msg`, and update `heads`;
- global embedding width remains 256 until residual and LayerNorm contracts are proven.

The deployment policy is fixed-K and fixed maximum agents for a single engine. `fixed_k` is deliberately mandatory and must later be derived from a frozen calibration/evaluation manifest; it is not inherited from LiDAR-pyramid `29696` or blindly copied from YAML `max_voxel_test`.

### Added production/source files

- `search/model_family/contracts.py`
  - Typed model-family contracts for weighted ops, legal pruning domains, merge boundaries, special deployment ops, plugins, and deterministic audit hashes.
- `search/model_family/registry.py`
  - Provider registration, explicit lookup, and YAML-based family detection.
- `search/model_family/heal_v2xvit.py`
  - HEAL LiDAR V2X-ViT provider.
  - Enumerates module and functional Einsum weights, candidate structured domains, FP16 merge contracts, scatter/warp/Einsum/window special ops, and explicit blockers.
- `search/model_family/model_provider.py`
  - Strict HEAL config/checkpoint loader and weighted-module runtime coverage hooks.
- `search/model_family/readiness.py`
  - Evidence-based admission gate. It exposes no formal search genes while canonical mapping, semantic Q/DQ, plugin, strongly typed parsing, precision realization, or physical pruning evidence is missing.
- `search/model_family/export/heal_v2xvit.py`
  - Isolated six-input fixed-K V2X-ViT ONNX wrapper.
  - Reuses the existing scatter symbolic without modifying the LiDAR-pyramid exporter.
  - Keeps fusion prewarp plus Transformer STTF/ROI GridSample semantics.
  - Replaces only the inverse of the model's hard-coded identity correction with an explicit identity sampling grid, removing unsupported `linalg_inv` without skipping the sampling operations.
- `search/model_family/export/__init__.py` and `search/model_family/__init__.py`
  - Public model-family interfaces.
- `scripts/audit_heal_model_family.py`
  - Strict model load, capability inventory, runtime weighted-branch coverage, and compact JSON audit.
- `scripts/smoke_export_heal_v2xvit.py`
  - Strict load, PyTorch wrapper parity, ONNX export/checker, graph inventory, hashes, and search-readiness evidence.
- `tests/test_search_model_family_v2xvit.py`
  - Registry, functional weights, ConvTranspose axis, pruning/merge/plugin gates, mandatory fixed-K, identity STTF/ROI, and readiness tests.

### Real V2X-ViT model evidence

Inputs:

- Config: `/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_v2xvit/config.yaml`
- Config SHA256: `801729a29db46b634646ab0db450671c13ab3cc65c1e7224ec372274a8448d5a`
- Checkpoint: `/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_v2xvit/net_epoch_bestval_at27.pth`
- Checkpoint SHA256: `890f7f4db7b92142c29789b4ee4649494004021eb521f94c345fbe439ca6e3ab`

Strict CUDA model smoke:

- `state_dict`: 282/282 tensors, strict load passed, no missing/unexpected/shape mismatch.
- Parameters: 13,453,197.
- Parameterized modules: 83 = 56 Linear + 24 Conv2d + 3 ConvTranspose2d.
- Functional weighted tensors: six `relation_att`/`relation_msg` parameters.
- Canonical static weighted entries: 89.
- Synthetic CUDA outputs: cls `[1,2,64,128]`, reg `[1,14,64,128]`, dir `[1,4,64,128]`.
- Runtime weighted-module coverage: 70/83 (84.34%).
- Uncalled paths: all twelve HGT type-1 Q/K/V/A projections and `fusion_net.fusion_net.encoder.prior_feed`.
- The current DAIR LiDAR wrapper creates zero agent-type encodings, so export specializes the active HGT type-0 path. This is not proof of general heterogeneous-type coverage.
- Audit artifact: `outputs/h800_heal_model_family_v2xvit_smoke_20260717_122228/model_family_audit.json`.
- Audit artifact SHA256: `42a320fc341ab42bb27c74a9c3b81d016951e2b761a66d8f153aeebba98ab68c`.
- Audit hash inside artifact: `bdde9519622352118fa888cf968e02770514daebd182415419846debe8e03a5e`.

Tracer smoke:

- Backend: `runtime_tensor_flow` after FX fallback.
- Inventory: 270 modules, 873 ops, 67 dependency edges, 50 scopes, 14,806 atomic units.
- Protected modules: 7; unresolved: 0; unsupported: 0.
- Weighted runtime coverage remains 70/83, so the trace is not accepted as full-branch coverage.
- Trace artifact: `outputs/h800_heal_model_family_v2xvit_trace_smoke_20260717_122117/trace_summary.json`.
- Trace artifact SHA256: `d7940aacb14de5559082e3786d2a0c6c52bd5a7668516ff35602fa173ed181a6`.

### Minimal ONNX smoke result

Authoritative artifact directory (ignored by Git):

`outputs/h800_heal_model_family_v2xvit_onnx_smoke_20260717_123358/`

- fixed-K: 64 for synthetic smoke only; maximum agents: 2.
- Six inputs: voxel features/coords/counts, pairwise transform, valid voxel mask, and agent mask.
- Wrapper versus original PyTorch passed `atol=5e-4, rtol=1e-4`:
  - cls max absolute error `0.0001716614`;
  - reg max absolute error `0.0001194477`;
  - dir max absolute error `0.0002928078`.
- ONNX opset 17 export passed.
- ONNX checker passed.
- Graph: 4,603 nodes, 182 initializers, 52,031,821 bytes.
- Contains one `trt::PointPillarScatterTRT` node.
- Contains three GridSample nodes, preserving the two-stage warp/ROI semantics.
- Contains no Inverse/LinalgInv node.
- Important parser risks: 1 If, 27 Einsum, 58 MatMul, 12 LayerNormalization, and type-0 specialization.
- ONNX SHA256: `e0f39d3e570204a12337b33fa92b158b45a1d87d9cb475c1a1748f9f42239a48`.
- Report SHA256: `cdd106442c09d1b056be6fce6de04f41fe679887e5a03887565aac3b246faf9a`.
- Active graph has 70 directly initialized weighted Conv/ConvTranspose/MatMul nodes, consistent with the 70/83 runtime module coverage; canonical mapping is not yet accepted because functional relation tensors and tied/reused MatMul paths need topology-aware mapping.

### Current admission status

```text
lidar_pyramid_frozen_at_1da7e3b: true
existing_lidar_pyramid_source_modified: false
v2xvit_config_available: true
v2xvit_checkpoint_available: true
v2xvit_strict_state_load: true
v2xvit_cuda_forward_smoke: true
v2xvit_capability_audit_complete: true
v2xvit_full_branch_trace_coverage: false
v2xvit_pytorch_export_wrapper_parity: true
v2xvit_onnx_export_smoke: true
v2xvit_onnx_checker: true
v2xvit_canonical_weight_mapping: false
v2xvit_semantic_qdq_boundary_audit: false
v2xvit_scatter_plugin_contract_verified: false
v2xvit_strongly_typed_tensorrt_parser: false
v2xvit_precision_realization_verified: false
v2xvit_physical_pruning_materializer: false
v2xvit_quantization_search_ready: false
v2xvit_pruning_search_ready: false
v2xvit_joint_search_ready: false
v2xvit_ga_or_greedy_started: false
```

### Reproduction commands

```bash
cd /home/lixingfeng/UniAD_examine/heal_compress
source /home/lixingfeng/miniconda3/etc/profile.d/conda.sh
conda activate univ2x-opt

python -m pytest -q tests/test_search_model_family_v2xvit.py

CUDA_VISIBLE_DEVICES=6 python scripts/audit_heal_model_family.py \
  --config /home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_v2xvit/config.yaml \
  --checkpoint /home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_v2xvit/net_epoch_bestval_at27.pth \
  --heal-root /home/lixingfeng/UniAD_examine/HEAL \
  --device cuda:0 --forward-smoke \
  --output outputs/<new-audit-dir>/model_family_audit.json

CUDA_VISIBLE_DEVICES=6 python scripts/smoke_export_heal_v2xvit.py \
  --config /home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_v2xvit/config.yaml \
  --checkpoint /home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_v2xvit/net_epoch_bestval_at27.pth \
  --heal-root /home/lixingfeng/UniAD_examine/HEAL \
  --device cuda:0 --fixed-k 64 --max-agents 2 \
  --output-dir outputs/<new-onnx-smoke-dir>
```

### Next round

1. Freeze a real calibration manifest and derive the V2X-ViT fixed-K contract from it.
2. Build topology-aware canonical mapping for active Conv/ConvTranspose/MatMul and functional HGT relation tensors; distinguish inactive type branches from unresolved mapping.
3. Remove or legalize the exported ONNX If and probe TensorRT parsing in `modelopt` using the required isolated toolchain.
4. Revalidate the PointPillar scatter plugin shape/precision contract for the V2X-ViT input profile.
5. Only after FP16 engine parity: add semantic post-activation/post-merge Q/DQ boundaries and strongly typed precision realization.
6. Only after quantization deployment smoke: implement and validate one conservative FFN hidden-width physical pruning domain, strict reload, and real forward.
7. Keep readiness gates closed until each piece of evidence is recorded; do not start broad greedy/GA before the minimal physical + deployment chain passes.

---

Round 1 timestamp: 2026-07-17 12:35 server-local / artifact sequence `20260717_123358`

---

## Round 2 — strongly typed TensorRT FP32 execution smoke

### Source change

`scripts/smoke_export_heal_v2xvit.py` now accepts `--save-engine-io`. It writes:

- raw binary inputs with exact binding dtype/shape/hash;
- NumPy reference outputs from the PyTorch export wrapper;
- their provenance in `export_report.json`.

This keeps engine runtime smoke reproducible without committing tensor dumps. The generated files remain below ignored `outputs/` directories.

### Isolated modelopt toolchain

The default interactive PATH placed `/usr/local/cuda/bin` before the Conda environment. Deployment commands must therefore prepend the environment explicitly:

```bash
source /home/lixingfeng/miniconda3/etc/profile.d/conda.sh
conda activate modelopt
export PATH="$CONDA_PREFIX/bin:$PATH"
export CUDA_HOME="$CONDA_PREFIX"
export CC="$CONDA_PREFIX/bin/gcc"
export CXX="$CONDA_PREFIX/bin/g++"
export TRT_ROOT=/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118
export LD_LIBRARY_PATH="$TRT_ROOT/lib:$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
```

Verified paths/versions after this correction:

- Python: `/home/lixingfeng/miniconda3/envs/modelopt/bin/python`.
- nvcc: `/home/lixingfeng/miniconda3/envs/modelopt/bin/nvcc`, CUDA 11.8 V11.8.89.
- gcc/g++: `/home/lixingfeng/miniconda3/envs/modelopt/bin/gcc` and `g++`, Anaconda 11.2.0.
- TensorRT root: `/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118`.
- TensorRT: 10.9.0.34.
- GPU: physical GPU 6, H800, compute capability 9.0.
- Existing plugin: `quantization/plugins/pointpillar_scatter_trt/build/libpointpillar_scatter_trt.so`.
- Plugin SHA256: `61d9adf44855ab2a595220718270d361c993f9ff281e986cdf8a62d5ca317ecd`.
- `ldd` resolves `libnvinfer.so.10` and `libnvinfer_plugin.so.10` from the required TensorRT root and `libcudart.so.11.0` from `modelopt`.

No plugin was rebuilt or overwritten in this round.

### Strongly typed build, inspector, and runtime

Input ONNX remained the Round 1 fixedK=64 synthetic-smoke graph. The first strongly typed build used TensorRT's default TF32 permission and proved parser/plugin compatibility. It built and deserialized a 74,110,876-byte engine, then ran one legal input successfully. Inspector showed Float compute with TF32 tactics, so it was not used for strict FP32 parity.

The authoritative no-TF32 smoke is:

`outputs/h800_heal_model_family_v2xvit_trt_no_tf32_smoke_20260717_124110/`

Build flags:

```text
--stronglyTyped
--noTF32
--builderOptimizationLevel=0
--profilingVerbosity=detailed
--staticPlugins=<pointpillar scatter .so>
```

Verified results:

- ONNX parse passed, including 27 Einsum, LayerNormalization, GridSample, and the exported If.
- `PointPillarScatterTRT` plugin was found and instantiated.
- Detected six inputs and three outputs.
- Engine build passed; size 75,416,172 bytes; SHA256 `f5064d954a9ee08c3bc70cb98ad01dc90d7aa15679158824847e9524d6452275`.
- Engine deserialize and execution-context creation passed.
- Binding order: voxel features, coords, point counts, pairwise transform, valid-voxel mask, agent mask, cls, reg, dir.
- One inference with legal saved inputs passed; evaluated as execution smoke only, not a latency benchmark.
- Inspector: 207 layers = 106 kgen + 70 gemm + 24 convolution + 3 deconvolution + 3 no-op + 1 plugin.
- Float is present in 205 layer rows; Int32 appears only in shape/index/plugin-support paths.
- No TF32 tactic and no FP16 tactic were present in the no-TF32 engine.
- Engine layer info SHA256: `9b254fc6b1c7e5b9855268a832d52affe1537a22cdcce577c48340de717791f2`.

TensorRT versus PyTorch wrapper on the synthetic input:

| Output | max abs | mean abs | cosine |
|---|---:|---:|---:|
| cls | 0.0045905 | 0.0004169 | 0.99999988 |
| reg | 0.0010176 | 0.0000759 | 1.00000000 |
| dir | 0.0039229 | 0.0007068 | 0.99999952 |

The remaining small difference is not treated as full numerical equivalence until repeated on real frames. Runtime parity artifact SHA256: `14395e21fceb5971679d3b805866ff38dddc63a7e7e19af2934fdd6297addc45`.

### Interpretation and gates

- `stronglyTyped` does not choose a faster precision. It preserves the types encoded in ONNX.
- Because this ONNX has no Cast/Q/DQ precision graph, the engine is FP32. This is expected and proves the future explicit-Q/DQ chain cannot rely on builder heuristics.
- Strongly typed parser, plugin loading, engine deserialization, context creation, binding layout, and one synthetic forward are now proven for the active type-0 FP32 graph.
- Production plugin compatibility remains gated because fixed-K is not yet derived from a real frozen manifest.
- Quantization readiness remains false: canonical mapping, semantic Q/DQ boundaries, entropy calibration, and requested-versus-realized INT8 precision are still absent.
- Physical pruning readiness remains false.

```text
v2xvit_strongly_typed_fp32_parse: true
v2xvit_strongly_typed_fp32_engine_build: true
v2xvit_engine_deserialize: true
v2xvit_execution_context: true
v2xvit_scatter_plugin_synthetic_runtime: true
v2xvit_binding_contract_synthetic: true
v2xvit_real_frame_tensor_parity: false
v2xvit_real_manifest_fixed_k: false
v2xvit_explicit_qdq: false
v2xvit_int8_precision_realization: false
v2xvit_quantization_search_ready: false
v2xvit_joint_search_ready: false
```

### Runtime reproduction

First generate legal input binaries in `univ2x-opt`:

```bash
CUDA_VISIBLE_DEVICES=6 python scripts/smoke_export_heal_v2xvit.py \
  --config /home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_v2xvit/config.yaml \
  --checkpoint /home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_v2xvit/net_epoch_bestval_at27.pth \
  --device cuda:0 --fixed-k 64 --max-agents 2 \
  --skip-onnx --save-engine-io \
  --output-dir outputs/<new-engine-io-dir>
```

Then switch to the isolated `modelopt` environment shown above and build with the required TensorRT root:

```bash
CUDA_VISIBLE_DEVICES=6 "$TRT_ROOT/bin/trtexec" \
  --onnx=outputs/<onnx-smoke-dir>/heal_lidar_v2xvit_fixedk.onnx \
  --stronglyTyped --noTF32 \
  --staticPlugins=quantization/plugins/pointpillar_scatter_trt/build/libpointpillar_scatter_trt.so \
  --saveEngine=outputs/<new-trt-dir>/v2xvit_strongly_typed_fp32.plan \
  --skipInference --builderOptimizationLevel=0 --profilingVerbosity=detailed
```

---

Round 2 timestamp: 2026-07-17 12:42 server-local / artifact sequence `20260717_124110`

---

## Round 3 — topology-aware V2X-ViT canonical weight mapping

### New implementation

- Added `search/model_family/onnx_mapping.py`.
- Export smoke now captures all weighted module calls during the real ONNX trace, invokes the existing formal module origin mapper, and layers V2X-ViT-only mapping on top.
- The accepted LiDAR-pyramid origin mapper under `quantization/**` remains unchanged.
- `scripts/smoke_export_heal_v2xvit.py` now emits `canonical_weight_mapping.json` and feeds realized-graph mapping evidence into the readiness report.

The enhanced mapper distinguishes:

1. active module weights, including repeated calls that share one HGT type-0 projection;
2. active functional `relation_att`/`relation_msg` initializers traced through Gather/Concat/Transpose into the exact Einsum node;
3. parameter-free grid-generation MatMul nodes, which are protected compute islands rather than precision genes;
4. activation-only attention Einsum nodes;
5. model capabilities inactive because of the frozen type-0 export contract;
6. truly unresolved active weighted paths, which fail the readiness gate.

### Real-model result

Authoritative ignored artifact directory:

`outputs/h800_heal_model_family_v2xvit_mapping_smoke_20260717_124907/`

Results:

- Static weighted capabilities: 89.
- Active weighted capabilities: 76.
  - active parameterized modules: 70;
  - active functional HGT relation parameters: 6.
- Active weighted compute nodes/calls: 88.
  - the extra 12 calls are the same HGT type-0 Q/K/V/A projection reused for both agents across three encoder layers;
  - every repeated call retains a distinct graph index and canonical call identity.
- Inactive weighted capabilities: 13.
  - twelve HGT type-1 Q/K/V/A projections, inactive because the current LiDAR-only wrapper supplies zero type encoding;
  - `fusion_net.fusion_net.encoder.prior_feed`, defined by HEAL but not called by `V2XTEncoder.forward`.
- Functional relation mapping: six initializer-to-Einsum paths, all resolved.
- Parameter-free affine/grid MatMul: three nodes grouped under a non-gene deployment identity.
- Activation-only attention/window Einsum: 21.
- Unresolved active weighted path: 0.
- `realized_graph_mapping_complete=true`.
- `full_static_branch_coverage=false`, intentionally, until a real manifest proves whether any nonzero agent type can occur.
- Mapping hash: `8c80fc412ba6dec5fdf01deb19acb411063a7f7a671b5cbf4c5febb8421ddf7d`.
- Mapping artifact SHA256: `93e226716c08af90c805a3b86b6515a4e00f3f8fa0762e02d6bdde7f27ce769a`.

### Tests and readiness

The new test builds a minimal ONNX graph with a normal Linear MatMul, a functional relation initializer consumed through Gather -> Einsum, and an inactive type-1 branch. It proves the three mapping states are disjoint and unresolved remains empty.

```text
v2xvit_realized_graph_canonical_weight_mapping: true
v2xvit_functional_relation_weight_mapping: true
v2xvit_parameter_free_grid_matmul_classified: true
v2xvit_activation_only_einsum_classified: true
v2xvit_unresolved_active_weighted_count: 0
v2xvit_full_static_branch_coverage: false
v2xvit_real_manifest_agent_type_policy_verified: false
v2xvit_semantic_qdq_boundaries: false
v2xvit_quantization_search_ready: false
v2xvit_joint_search_ready: false
```

The next quantization step must consume the 76 active weighted capability entries, not assume the 89 static entries are all present in the specialized graph and not create genes for the three grid MatMul or 21 activation-only Einsum nodes.

---

Round 3 timestamp: 2026-07-17 12:50 server-local / artifact sequence `20260717_124907`

---

## Round 4 — freeze the V2X-ViT train200 manifest and fixed-K contract

### Scope and selection contract

This round established a V2X-ViT-only calibration manifest. It did not reuse
the LiDAR-pyramid train200 list or `fixedK29696`, and it did not generate NPZ,
ONNX, Q/DQ, engine, evaluation, GA, or greedy-search artifacts.

- Dataset: real HEAL DAIR-V2X train split in train mode, `visualize=false`.
- Raw/valid train count: 4,811/4,811 for the synchronized dataset.
- Selection policy: `evenly_spaced_valid_train_indices_v1`.
- Selection order: 200 ascending indices spanning `[0, 4810]`; no shuffle.
- RNG contract: Python, NumPy, and PyTorch reset before each sample with
  `uint32(20260717 + dataset_index)`.
- Stable identity: DAIR vehicle frame ID, infrastructure frame ID, dataset
  index, and source-pair ID are recorded for every sample.
- Realized agent distribution: 19 one-agent frames and 181 two-agent frames.
- All realized modalities are `m1`, consistent with the current LiDAR-only
  mapping and active type-0 V2X-ViT export specialization.
- Direct indexed loading with zero workers is deliberate for manifest
  construction determinism; it does not change the previously requested
  eight-worker evaluation-loader policy.

### K statistics and frozen value

`K` is the total unpadded `inputs_m1.voxel_features.shape[0]` after HEAL's real
per-agent preprocessing and single-frame collate.

| Statistic | K |
|---|---:|
| min | 5,793 |
| p50 | 20,525.50 |
| p90 | 25,243.10 |
| p95 | 25,984.35 |
| p99 | 27,242.87 |
| max | 27,666 |
| mean | 19,793.995 |

The maximum is dataset index 4,496, vehicle frame `015547`, infrastructure
frame `001261`, with per-agent K `[14711, 12955]`.

The formal derivation is:

```text
fixed_K = ceil(max_observed_train200_K / 256) * 256
        = ceil(27666 / 256) * 256
        = 27904
```

- Alignment margin over the maximum sample: 238 voxels.
- Frozen-manifest truncation count: 0/200.
- Mean padding ratio: 0.2906395.
- The contract fails closed when a future input exceeds 27,904; it never
  silently truncates.
- This is only a zero-truncation guarantee for the exact frozen train200
  manifest. It is not claimed to upper-bound the remaining 4,611 train
  samples, the validation split, or a different preprocessing contract.

### Source and frozen artifact changes

- `search/model_family/calibration_manifest.py`
  - stable 200-frame selection;
  - per-sample RNG ownership;
  - K distribution and 256-aligned fixed-K derivation;
  - canonical manifest identity hashing;
  - fail-closed validation and frozen-manifest loading.
- `scripts/build_heal_v2xvit_train200_manifest.py`
  - builds the real HEAL train dataset without loading large calibration NPZ;
  - records frame provenance, per-agent K, dtypes, preprocessing and source
    hashes;
  - replays samples 0/100/199 and refuses to overwrite existing evidence;
  - only accepts exactly 200 samples for this schema.
- `search/model_family/manifests/heal_lidar_v2xvit_train200_fixed_k.json`
  - committed 160-KiB frozen manifest containing the ordered 200-frame list
    and all observed K evidence.
- `search/model_family/export/heal_v2xvit.py`
  - `HealV2XViTExportPolicy.from_frozen_train_manifest(...)` validates the
    manifest and imports fixed-K/max-agents/modality plus manifest hash.
- `scripts/smoke_export_heal_v2xvit.py`
  - supports mutually exclusive `--fixed-k` (synthetic diagnostic only) and
    `--fixed-k-manifest` (formal real-manifest path);
  - records manifest path/hash in export provenance.
- `tests/test_search_model_family_v2xvit.py`
  - deterministic selection, fixed-K formula, tamper detection, and export
    policy manifest-consumption tests.

Frozen identities:

- Manifest identity hash:
  `03d038c0b8a4d900d247e7e06d245ce15a7614910b4254b6179d76adb1cbd000`.
- Frozen JSON file SHA256:
  `5d5ce47fc333f027b09a23225ac1068a6b5ac15242a9db091e10db88e3453b55`.
- Config SHA256:
  `801729a29db46b634646ab0db450671c13ab3cc65c1e7224ec372274a8448d5a`.
- Checkpoint SHA256:
  `890f7f4db7b92142c29789b4ee4649494004021eb521f94c345fbe439ca6e3ab`.
- Train split SHA256:
  `865e0ff2c788a67bede72a93984cc6b5f3e506fcb99c95571803322356f79051`.
- Cooperative data-info SHA256:
  `30aa21051f56082cfb714bf8ad93bab8c018ae2ba1211b7873852c125f3e2659`.

Ignored evidence directories:

- `outputs/h800_heal_model_family_v2xvit_train200_manifest_20260717_130221/`
- `outputs/h800_heal_model_family_v2xvit_train200_manifest_replay_20260717_130450/`

The second independent build produced the same manifest identity and reported
`frozen_status=already_identical`; it did not rewrite the frozen file.

### Validation and reproduction

```text
tests/test_search_model_family_v2xvit.py: 10 passed
model-family + legacy domain-width + greedy regression: 25 passed
git diff --check: passed
```

```bash
cd /home/lixingfeng/UniAD_examine/heal_compress
source /home/lixingfeng/miniconda3/etc/profile.d/conda.sh
conda activate univ2x-opt

python scripts/build_heal_v2xvit_train200_manifest.py \
  --output-dir outputs/<new-v2xvit-train200-audit-dir> \
  --frozen-manifest \
    search/model_family/manifests/heal_lidar_v2xvit_train200_fixed_k.json

# Formal exporter use after the next real-frame export round:
CUDA_VISIBLE_DEVICES=6 python scripts/smoke_export_heal_v2xvit.py \
  --config /home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_v2xvit/config.yaml \
  --checkpoint /home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_v2xvit/net_epoch_bestval_at27.pth \
  --device cuda:0 \
  --fixed-k-manifest search/model_family/manifests/heal_lidar_v2xvit_train200_fixed_k.json \
  --output-dir outputs/<new-real-fixedk-export-dir>
```

### Readiness after this round

```text
v2xvit_train200_manifest_frozen: true
v2xvit_train200_manifest_replay_deterministic: true
v2xvit_fixed_k_27904_frozen: true
v2xvit_fixed_k_zero_truncation_on_train200: true
v2xvit_fixed_k_full_train_upper_bound_verified: false
v2xvit_real_manifest_agent_type_policy_verified: true
v2xvit_real_fixedk_onnx_export: false
v2xvit_real_fixedk_scatter_plugin_runtime: false
v2xvit_semantic_qdq_boundaries: false
v2xvit_quantization_search_ready: false
v2xvit_pruning_search_ready: false
v2xvit_joint_search_ready: false
v2xvit_ga_or_greedy_started: false
```

The next bounded task is one real train200 frame through the fixedK=27904
PyTorch wrapper/ONNX/plugin path, followed by real-frame FP16 parity. Q/DQ and
physical pruning remain gated.

---

Round 4 timestamp: 2026-07-18 04:05 CST (host shell: 2026-07-17 13:05 PDT) / artifact sequence `20260717_130450`

---
