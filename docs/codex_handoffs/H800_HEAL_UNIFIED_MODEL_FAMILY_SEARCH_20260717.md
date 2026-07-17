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
