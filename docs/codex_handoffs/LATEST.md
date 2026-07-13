# HEAL/OpenCOOD Joint Compression Search Handoff — 2026-07-12 Current Server

This handoff is based only on current-server code, files, hashes, and executions. The absent historical run `lidar_pyramid_joint_search_final_20260712_060922` is not used as completion evidence.

## Repository and environments

- Repository: `/home/lixingfeng/UniAD_examine/heal_compress`
- Branch: `feature/pruning-quant-toolkit-cleanupv1`
- Starting HEAD: `94c4ab6`
- Current run: `tests/outputs/lidar_pyramid_joint_search_final_20260712_115848`
- Model environment: `univ2x-opt`, torch `2.0.1+cu118`, CUDA available.
- Deployment environment: `modelopt`.
- TensorRT root: `/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118`, TensorRT `10.9.0.34`.
- `modelopt` tools: env Python, NVCC `11.8.89`, GCC/G++ `11.2.0`; subprocess activation pins `PATH`, `CUDA_HOME`, `CC`, `CXX`, and `CUDACXX` to the Conda environment.
- PointPillarScatterTRT plugin exists and loads from `quantization/plugins/pointpillar_scatter_trt/build/libpointpillar_scatter_trt.so`.
- Original checkpoint, config, and 1789-ID DAIR validation split exist and were hash/path verified.

## Implemented and verified fixes

- Stage-1 uses the direct four-term objective without hidden median normalization; Size and BOPS reference original strict FP32.
- BOPS feasibility ordering uses `R_bops_vs_fp32`.
- Formal CLI used CUDA-batched proxy on GPU 7 with `scalar_evaluate_call_count=0`.
- Real evaluation manifest contains fixed real frame IDs and fails closed on missing/skipped IDs.
- ModelOpt invocation uses explicit Conda activation because `conda run -n modelopt` is broken on this server.
- Physical width validation only checks axes actually pruned, avoiding false failures on protected heads.
- ONNX cache reuses the typed origin map and validated pruned ONNX.
- Q/DQ node counting correctly distinguishes QuantizeLinear from DequantizeLinear.
- Calibration is ONNX BatchNorm-fold aware: activation input comes from the Conv input, folded output scale comes from the paired BN output, and weight scale comes from the actual ONNX initializer.
- Calibration cache identity includes semantics version and ONNX hash, invalidating pre-fix scales.
- Stage-2 evaluation cache identity includes deployment/calibration pipeline version.
- GA diversity and block crossover were changed from quadratic Python work to exact O(N×genes) computations. At 512×1504, diversity was about 0.173 s and 512 crossovers about 0.453 s.
- Resume merges the existing run manifest and appends command history instead of overwriting evidence.
- Round winner aggregation accepts the formal evaluator filename `evaluation.json`.

## Nonzero physical pruning proof

Candidate `496f0bf8d8c5417efc44f119315dbea4fd48d7c6130fdd745525b072f440c1fe`:

- Pruned units: 660.
- Parameters: 5,464,791 → 5,012,691, an 8.27296% reduction.
- repaired→request: verified.
- request→legal physical plan indices: verified.
- grouped keep/prune maps: frozen and verified.
- Physical hash: `ec0b11e3f89345626e5aaab43bb9f584f720336592e90cddc338bd693f3b4552`.
- Strict checkpoint reload: no missing or unexpected keys.
- Real HEAL/OpenCOOD PyTorch forward: passed; every returned tensor value was finite.
- Pruned ONNX checker, shape inference, canonical mapping, and physical initializer shapes: passed.
- Corrected explicit Q/DQ: Q15/DQ15, 4 coupled INT8 groups, 5 INT8 layers, finite nonzero scales, no fallback.
- TensorRT engine: deserialize/context/bindings/plugin/precision/forward validation passed.
- Fixed 300-frame result: mAP `0.5188452457`, p50 `2.8151518588 ms`, F2 `8.6784237977`; technically valid but rejected for AP loss.

## Four-round results on the shared fixed 300-frame manifest

Strict references on the same manifest:

- strict FP32: mAP `0.7301969468`, 300/300, zero skips.
- strict FP16: p50 `2.5095428185 ms`, 300/300, zero skips.

| Round | Winner | Pruned units | mAP | p50 ms | F2 |
|---:|---|---:|---:|---:|---:|
| 0 | `ddd6a0f2…d0132` | 0 | 0.730513 | 2.503623 | 0.199528 |
| 1 | `6d2c9a06…003a6` | 656 | 0.604684 | 2.768489 | 5.241138 |
| 2 | `fbcabf46…db9bb` | 660 | 0.420867 | 2.649126 | 12.584319 |
| 3 | `57af0e04…d6401` | 792 | 0.459218 | 3.421780 | 11.111854 |

All 20 repaired Top-5 candidates completed real Stage-2 evaluation with 300/300 frames and zero skipped frames.

## Unified full validation and final winner

Manifest hash: `6e5e9dc739a67005d5d8caded2dd6a51011ede7fd78e8a7259a7e7d103761acf`; 1789 validation frames, zero warmup frames, latency rounds 3.

- strict FP32: mAP `0.7368540734`, 1789/1789, zero skips.
- strict FP16: p50 `2.5024222753 ms`, 1789/1789, zero skips.
- Round-0 winner: mAP `0.7362839016`, p50 `2.5152699867 ms`, F2 `0.2238336957`.
- Round-1 winner: mAP `0.6198493654`, p50 `2.7795983090 ms`, F2 `4.9023409402`.
- Round-2 winner: mAP `0.4475838657`, p50 `2.7114949965 ms`, F2 `11.7875179371`.
- Round-3 winner: mAP `0.4395245342`, p50 `3.4581916795 ms`, F2 `12.1695691078`.

Final winner: Round-0 candidate `ddd6a0f2ccdc3f37c8361c1cb090ee193100d63ff626d0f93018cdda267d0132`. It is an all-keep, all-FP16 candidate. The search is valid even though the AP-prioritized objective selected no physical pruning as the optimum; multiple nonzero candidates completed the full deployment chain and were rejected by real AP.

## Cache and resume status

- Physical cache: verified.
- ONNX cache: verified, including explicit cache-hit artifact.
- Evaluation cache: verified, including a real candidate cache hit.
- Completed-run resume tracked 174 artifacts; none changed hash, mtime, size, or presence.
- Dedicated deployment cache: not verified. `archives/artifact_index.jsonl` has physical and ONNX entries but no dedicated `qdq_onnx` or `engine` entry/hit. Do not mark `deployment_cache_verified` true until this is implemented and exercised independently of completed-round skipping.

## Tests and evidence

- Final search test command: `CUDA_VISIBLE_DEVICES=7 pytest -q tests/test_search*.py tests/test_two_stage_joint_search.py`
- Result: `117 passed`.
- `git diff --check`: passed.
- Migration audit: `migration_artifact_audit.json` and `.md`.
- Acceptance booleans: `final_acceptance_status.json` and `.md`.
- Full validation: `tests/outputs/lidar_pyramid_joint_search_final_20260712_115848/final_full_validation/final_full_validation.json`.
- Resume proof: `tests/outputs/lidar_pyramid_joint_search_final_20260712_115848/resume_without_rebuild_validation.json`.

## Remaining honest gap

The only false final acceptance field is `deployment_cache_verified`. Implement and test dedicated Q/DQ and engine cache entries/hits before changing it to true. No GA, physical model, ONNX, calibration, Q/DQ, engine, 300-frame evaluation, or full validation needs to be rerun merely to establish the current completion state.
