# H800 Transformer quantization audit handoff

## Scope and provenance

- Branch: `feature/h800-transformer-quantization-audit`
- Base: `1da7e3b37afa30549a81df4259b80312f449f9a2` (`feature/heal-compress-h800`)
- Production code commit: `edd0a09c` (`feat: audit H800 transformer quantization profiles`)
- Output root: `/data/lxf/heal_data/outputs/h800_transformer_quantization_20260721_100649`
- Models: `lidar_cobevt`, `lidar_v2xvit`
- Structure remained frozen. No Transformer pruning, GA, Greedy, or Pyramid search was run.
- The requested `/home/lixingfeng/anaconda3/envs/modelopt` does not exist on this host. The verified environment is `/home/lixingfeng/miniconda3/envs/modelopt`.
- TensorRT root: `/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118`, version `10.9.0.34`; GPU capability `(9, 0)`; the independent SM90 CUDA extension passed with `cpu_fallback=false`.

## Implementation delivered

- Added model-independent Transformer role, requested/realized precision, accumulator, SmoothQuant, FP8, Q/DQ adjacency, inventory, profile-library, and search-space contracts under `search/model_families/transformer/`.
- Added separate CoBEVT and V2XViT H800 profiles under `search/model_families/lidar_cobevt/` and `search/model_families/lidar_v2xvit/`. V2XViT does not reuse CoBEVT node-name rules.
- Added fresh strongly-typed ONNX/TRT builders and orchestration for manifests, inventory, baselines, role sensitivity, SmoothQuant, BF16/FP8, accumulator evidence, evaluation, formal latency, and reports under `search/orchestration/lidar_transformer_h800_*`.
- Production strongly-typed builds reject implicit precision flags and layer constraints in production mode. Non-production legacy callers retain compatibility while weak flags are omitted from the actual strongly-typed command.
- ModelOpt Q/DQ adjacency repair was extracted from the 4090 CoBEVT-specific toolchain into `search/model_families/transformer/qdq_adjacency.py`; INT8 and FP8 production paths share it.
- Requested precision, output precision, Q/DQ, fusion/tactic, Cast/Reformat, and accumulator evidence are audited independently. Profile names and output dtypes are never treated as realized-compute or accumulator proof.
- Evaluation uses fixed manifests, eight workers, warmup-reset, zero-skip enforcement, fixed-K, and CUDA AP/IoU. Formal latency is device-resident CUDA-event timing with 200 warmup, 2000 iterations, five repeats, and baseline replay.
- The selected SmoothQuant alpha is part of admission: CoBEVT `0.8`, V2XViT `0.75`. Earlier alpha `0.7` rows remain observational and cannot enter `allowed`.
- Out-of-scope incomplete 4090 head-dimension/Stage-1 search files were not ported. The existing Pyramid domain-width search path was not changed.

## Data and inventory

| Model | fixed-K | weighted entries | PyTorch parameters | unresolved accepted-engine mapping |
|---|---:|---:|---:|---:|
| CoBEVT | 29184 | 65 plus one protected functional affine MatMul | 10,500,260 | 0 |
| V2XViT | 27904 | 100 | 13,453,197 | 0 |

V2XViT has 13 parameterized branch modules absent from the accepted fixed `max_agents=2` ONNX trace (`prior_feed` and inactive `k/q/v/a_linears.1` branches). They are explicitly recorded as `missing_role_mapping` and excluded fail-closed; they are not unmapped weighted engine layers.

## Baselines and fixed500

| Model | Profile | fixed50 mAP | fixed500 mAP | Formal p50 ms | Accuracy conclusion |
|---|---|---:|---:|---:|---|
| CoBEVT | B0 PyTorch FP32 | — | 0.645145 | — | accepted checkpoint baseline |
| CoBEVT | B1 Attention FP32 TRT | 0.579157 | 0.644723 | 5.279552 | accepted TRT reference |
| CoBEVT | B2 strict FP16 | 0.228844 | not run | 3.112224 | catastrophic, rejected |
| CoBEVT | B3 F3 | 0.577108 | 0.644759 | 3.757568 | safe |
| V2XViT | B0 PyTorch FP32 | — | 0.658688 | — | accepted checkpoint baseline |
| V2XViT | B1 Attention FP32 TRT | 0.575245 | 0.658220 | 14.422208 | accepted TRT reference |
| V2XViT | B2 strict FP16 | 0.574546 | 0.657954 | 9.022000 | safe |
| V2XViT | B3 F3 | 0.572937 | 0.658173 | 11.593760 | safe |

CoBEVT isolated P1-P11 role changes were non-catastrophic, but P12 full-attention FP16 reproduced the collapse (`mAP=0.227923`). V2XViT P12 remained safe on fixed50 (`mAP=0.574960`). This shows a joint CoBEVT attention phenotype failure, not a claim that every individual role must be FP32.

## INT8, BF16, and FP8 conclusions

| Model | Profile | Realized low-precision contract rows | Q/DQ rows | fixed500 mAP | p50 ms | B1 speedup | Status |
|---|---|---:|---:|---:|---:|---:|---|
| CoBEVT | SQ1 Q/K INT8, DQ FP32 QK, alpha 0.8 | 12 INT8 | 48 | 0.644550 | 2.946176 | 1.792x | allowed |
| CoBEVT | FFN2 INT8 diagnostic, alpha 0.8 | 6 INT8 | 24 | 0.644744 | 3.040832 | 1.736x | allowed single-role phenotype |
| CoBEVT | H4 protected-QK BF16 | 60 BF16 | 0 | 0.645429 | 3.852896 | 1.370x | allowed |
| CoBEVT | H7 FFN FP8 E4M3 | 12 FP8 | 48 | 0.644451 | 3.052800 | 1.729x | allowed |
| V2XViT | SQ1 Q/K INT8, DQ FP32 QK, alpha 0.75 | 30 INT8 | 120 | 0.658316 | 11.058848 | 1.304x | allowed |
| V2XViT | FFN1 INT8 diagnostic, alpha 0.75 | 3 INT8 | 12 | 0.658047 | 10.855808 | 1.329x | allowed single-role phenotype |
| V2XViT | H4 protected-QK BF16 | 102 BF16 | 0 | 0.657924 | 13.175648 | 1.095x | allowed |
| V2XViT | H9 all legal Linear FP8 E4M3 | 66 FP8 | 264 | 0.658486 | 11.183776 | 1.290x | allowed |

CoBEVT SQ4/SQ5 joint FFN-related INT8 graphs and V2XViT SQ4 failed in TensorRT compilation. V2XViT SQ5 built but was `BORDERLINE` at fixed500 (`mAP=0.648566`, delta `-0.009654`). CoBEVT SQ6 maximal Linear INT8 was `UNSAFE` at fixed50 (`mAP=0.525043`). These remain negative controls, not allowed profiles.

The alpha `0.7` CoBEVT SQ3 row was safe and fast but does not match the selected CoBEVT alpha `0.8`; it is retained as observational evidence only. A selected-alpha rebuild is required before SQ3 can be admitted.

## Accumulator and execution phenotype

- Level-A independently searchable profiles are only `Q0_F32A32` and `A0_F32A32` for both models.
- FP16-default QK/AV executes but its accumulator is opaque (Level B), so it is only usable inside an already verified joint phenotype.
- Exact F16A32/BF16A32 is unavailable through the current TensorRT 10.9 network contract; H2 BF16A32 is a precision conflict because the accumulator remains unknown.
- Native INT8 QK/AV and native FP8 QK/AV were not implemented. Projection INT8/FP8 followed by DQ to FP32 QK is not mislabeled as a native low-precision attention core.
- No complete fused MHA was observed. Full-engine role latency cannot be separated after fusion, and subgraph values are not added to predict full-engine latency.
- H800 inspector rows contain a mixture of `sm80`- and `sm90`-named tactics. Exact H800-vs-4090 tactic comparison remains unverified because matching 4090 EngineInspector artifacts are absent locally; no 4090 engine or timing cache was reused.

## Formal latency protocol

- Physical GPU: `GPU-fc1ca600-f06e-f791-125e-102e4916449b` (H800, GPU 7).
- Five-minute external-process isolation passed separately for each model.
- B1 replay p50 drift: CoBEVT `0.676%`; V2XViT `0.047%`, both below the `3%` rejection threshold.
- Full p50/p90/p95/p99/mean/std, clocks, temperature, power, kernel count, and Cast/Reformat count are in `formal_latency_cobevt.csv` and `formal_latency_v2xvit.csv`.

## Final libraries and reports

- `cobevt_quantization_matrix.csv`
- `v2xvit_quantization_matrix.csv`
- `transformer_quantization_cross_model.csv`
- `precision/requested_realized_cobevt.csv`
- `precision/requested_realized_v2xvit.csv`
- `precision/accumulator_evidence.csv`
- `precision/qdq_inventory.csv`
- `transformer_precision_profile_library.json`
- `transformer_quantization_search_space.json`
- `root_conclusion.md`

Exact portable allowed profile IDs are `B3_F3` and `H4_FULL_TRANSFORMER_BF16_PROTECTED_QK`. SQ1 topology is safe for both models, but alpha/scales are model-specific, so the exact deployed profile is not portable byte-for-byte. Only profiles listed under `allowed` and `verified_joint` may seed a later small Greedy study; no Cartesian product is automatically opened.

## Verification commands

```bash
CUDA_VISIBLE_DEVICES='' /home/lixingfeng/miniconda3/envs/modelopt/bin/python -m pytest -q \
  tests/test_transformer_h800_quantization.py \
  tests/test_formal_packages_cpu.py \
  tests/test_search_full_val_manifest.py \
  tests/test_lidar_cobevt_*.py \
  tests/test_search_model_family_v2xvit.py \
  tests/test_two_stage_joint_search.py

/home/lixingfeng/miniconda3/envs/modelopt/bin/python -m compileall -q \
  quantization search tests/test_transformer_h800_quantization.py \
  tests/test_lidar_cobevt_*.py tests/test_search_model_family_v2xvit.py

git diff --check
```

Result: `244 passed`; `py_compile/compileall` passed; `git diff --check` passed. JUnit is retained at `<output>/tests/pytest.xml`.

## Remaining work

- Full1789 was intentionally not run in this audit; it remains required for any later final search winner.
- Rebuild CoBEVT SQ2/SQ3 at selected alpha `0.8` and V2XViT SQ2/SQ5 at selected alpha `0.75` only if those profiles are candidates for admission.
- Obtain matching 4090 EngineInspector artifacts before claiming exact tactic differences.
- Do not expose unknown accumulator, inactive V2XViT branches, failed joint graphs, or unverified Cartesian combinations as search genes.

---

Round timestamp: `2026-07-22 04:59:53 CST` (run directory dated `20260721`); production code commit `edd0a09c`.

---
