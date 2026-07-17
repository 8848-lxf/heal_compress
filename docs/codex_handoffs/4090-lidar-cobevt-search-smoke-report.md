# 4090 LiDAR CoBEVT Model-Family Search And Deployment Smoke

Date: 2026-07-18 (Asia/Shanghai)

## Direct Status

```text
BRANCH=feature/heal-compress-4090-cobevt-family
IMPLEMENTATION_COMMIT=55a1886d200f5cb5352d1c9bc5264879abb8851f
PYRAMID_PRODUCTION_PATH_MODIFIED=false
COBEVT_LEGAL_WIDTH_STAGE1_PASS=true
COBEVT_GREEDY_STAGE1_PASS=true
COBEVT_GA_16X3X1_STAGE1_PASS=true
COBEVT_STRONGLY_TYPED_BUILD_PASS=true
COBEVT_REAL_GPU_INFERENCE_PASS=true
COBEVT_SMOKE10_FIXED50_PASS=true
COBEVT_MIXED_INT8_DEPLOYMENT_AVAILABLE=false
COBEVT_FULL_VALIDATION_EXECUTED=false
STAGE_A_STARTED=false
STAGE_B_ALLOWED=false
```

This is a bounded model-family compatibility smoke. It proves that the shared
legal-width greedy/GA framework can load, prune, type, build, and execute the
LiDAR CoBEVT model with real TensorRT engines. It is not a full-validation
model selection result and does not modify the accepted lidar_pyramid path.

## Isolation And Identity

- Isolated worktree:
  `/home/lixingfeng/UniAD_examine/heal_compress/.worktrees/cobevt-family`
- Development branch: `feature/heal-compress-4090-cobevt-family`
- Source branch remains: `feature/heal-compress-h800-sync-4090`
- Required H800 ancestor `b862b3d8ad061bd12580776226c75f564918298d`:
  present.
- Checkpoint SHA256:
  `67b0f2f00d74fea4912b4fdb902c1150146c733201e5dc36d3358f5ba605cfd4`
- Model config SHA256:
  `a0ee9d64fd1b01af95b1c997937ab07e0e810440236accfb598d5462ba086c33`
- Plugin SHA256:
  `91aec743dca383151b995a60d004cd254ce115ed16152873c1f46754bb15022d`
- TensorRT: 10.9.0, CUDA 11.8, RTX 4090 (SM89).

## Model-Family Architecture

The new family path is dispatched lazily. Configurations without an explicit
family still instantiate the unchanged pyramid runner. CoBEVT owns these
contracts:

1. Checkpoint/config/input identity and a six-input fixed-K graph.
2. One legal-width attention domain with head-aligned widths
   `(64, 96, 128, 160, 192, 224, 256)`.
3. A deterministic pure-pruning Taylor ranking over eight attention heads.
4. Atomic physical replay across shrink Conv, attention QKV/out, FFN,
   LayerNorm, relative-position embeddings, and cls/reg/dir head inputs.
5. Fifty-three canonical weighted precision groups plus one protected
   parameter-free affine-grid MatMul.
6. CoBEVT-only typed closure for LayerNorm, elementwise attention branches,
   Where, and mixed floating Concat.
7. A separate GPU-only evaluation provider/worker using CoBEVT input packing.

The scatter plugin remains outside precision genes and BOPS accounting. All
five engines expose a Float-to-Float scatter boundary in EngineInspector.

## Fixed-K And Manifests

Artifact root (not tracked by Git):

`/data/lxf/heal_data/outputs/4090_lidar_cobevt_model_family_smoke_20260718_042721`

- Scanned records: train200 plus validation70, total 270.
- Maximum observed voxel count: `25412`.
- Derived fixed K (alignment 256): `25600`.
- Overflow count: `0`.
- Voxel-count manifest hash:
  `ebf6d260479bfc77c1b910272a68bb490b287f45c52b1aeea7816820e3ddc800`.
- smoke10 manifest hash:
  `21cb7f0a41e97acbb9bba8abf1e9c2bd5106116e30ec2d931e0f672cd029f973`.
- fixed50 manifest hash:
  `14e70f4974c7aec17fa9579b58a4282ebaeb188e59f4dde44a018482eaa63127`.
- Warmup: 20 frames; fixed50 evaluation uses the following 50 non-overlapping
  frames; latency collection resets after warmup.
- AP/IoU backend: GPU only; DataLoader workers: exactly 8.

## Bounded Stage-1 Search

- Fisher samples/micro-batch: `1/1` using empirical mean of squared gradient.
- Weighted runtime calls and precision groups: `53/53`.
- Width-space hash:
  `37b4fd2b89d6fbfba83e612dc28ade9782800bbb97aeed26e56a77ad2a6084f8`.
- Fixed second-order ranking hash:
  `55865b333d42326880d324355c7c64753b2ca42f9b507d33a3075fd71952a1d6`.
- Greedy: 990 unique proxy evaluations; endpoint BOPS `0.2543881352`.
- GA: population 16, 3 generations, 1 seed (`4090`), 48 unique
  genotypes/phenotypes, 6 observed structures, repair rate 0.
- BOPS-feasible archive: 9 phenotypes, 2 structures, 8 precision profiles.
- Feasible supply by generation: `9, 0, 0`.
- Generation 1/2 were not backfilled with out-of-band candidates; legal
  mutations moved above the `[0.2425, 0.2575]` smoke band.

## Strongly Typed Build Audit

Every production command contains `--stronglyTyped --noTF32` and omits
`--fp16`, `--int8`, precision constraints, layer precision overrides, and
layer output-type overrides. Requested versus realized canonical precision
passed for every engine, with zero unresolved layers and zero INT8 fallback.

The first greedy build failed at `/backbone_m1/Concat`: deblock0 was explicit
FP32 while deblock1/2 were explicit FP16. The CoBEVT auxiliary closure did not
previously type Concat. A RED test reproduced the exact mixed input contract.
The fix inserts one explicit FP32-to-FP16 Cast for this merge; the unchanged
graph then parsed and built. No weakly typed fallback was used.

## Real Fixed50 Results

All rows completed smoke10 `10/10, 0 skip` before fixed50 `50/50, 0 skip`.
Latency is parallel screening latency on the listed GPU, not isolated formal
latency and not full-validation latency.

| candidate | source | GPU | R_BOPS | R_param | weighted FP32/FP16/INT8 | mAP | AP03 | AP05 | AP07 | p50/p90/p95 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| strict_fp32_reference | baseline | 4 | 1.000000 | 1.000000 | 53/0/0 | 0.591499 | 0.743520 | 0.620491 | 0.410487 | 8.324/9.573/10.020 |
| strict_fp16_reference | baseline | 7 | 0.250000 | 1.000000 | 0/53/0 | 0.192366 | 0.253440 | 0.210204 | 0.113453 | 4.622/5.134/5.567 |
| greedy_endpoint | greedy | 4 | 0.254388 | 1.000000 | 30/23/0 | 0.590689 | 0.742410 | 0.619529 | 0.410128 | 6.630/7.860/8.170 |
| ga_gen0_rank1 | GA rank 1 | 6 | 0.250001 | 1.000000 | 1/52/0 | 0.193831 | 0.254929 | 0.211557 | 0.115006 | 4.592/5.244/5.630 |
| ga_gen0_pruned_diverse | GA structure-diverse | 7 | 0.247149 | 0.938587 | 0/53/0 | 0.175288 | 0.221205 | 0.190278 | 0.114382 | 4.443/5.380/5.886 |

The greedy mixed engine is the useful compatibility result. Relative to the
same-GPU strict FP32 reference, it loses only `0.000810` absolute mAP while
screening p50 changes from `8.324` to `6.630 ms` (`1.255x`). The all/near-all
FP16 engines lose about 0.40 mAP. Therefore CoBEVT attention and/or other
protected paths require a more conservative precision prior; BOPS feasibility
and the current weight-only Stage-1 proxy alone do not predict this numerical
sensitivity.

## Engine Identity

| candidate | parameters | engine bytes | engine SHA256 |
|---|---:|---:|---|
| strict_fp32_reference | 10,500,260 | 55,957,636 | `6bab10962a558a48fb41271cb700da4084f6c5ba8668d2983d40552373d30fec` |
| strict_fp16_reference | 10,500,260 | 50,330,740 | `a316ed947c0310b7db0b602017d59964124c9bb1d4936d429f306a658962cf43` |
| greedy_endpoint | 10,500,260 | 56,379,164 | `7568aea6e5d3e8cb1c3da0e7a801c30674254f993e60f600ad53b0569ae70496` |
| ga_gen0_rank1 | 10,500,260 | 50,759,460 | `312ff7c882672a8b90117145e961795c84c3139d63ed0a310aaa98bc2388b896` |
| ga_gen0_pruned_diverse | 9,855,410 | 49,156,996 | `168f73ebf1b389ddecb71072674df41f28711a17860ac6fcff94b3aa9101a629` |

The pruned candidate deterministically removes attention head 7, changes
fusion width `256 -> 224`, and matches predicted/physical parameter count
exactly (`9,855,410`). Export does not rerank channels.

## Tests And Hygiene

- Focused model-family, typed graph, builder, GPU protocol, and pyramid freeze
  regression: `89 passed, 26 warnings`.
- One initial linked-worktree import-path failure was reproduced and passed
  with explicit `PYTHONPATH=$PWD:/home/lixingfeng/UniAD_examine`; no production
  code change was needed.
- All changed Python files pass `py_compile`.
- `git diff --check` passes.
- No PTH, ONNX, PLAN, cache, tensor dump, or plugin binary is tracked.

## Limitations And Next Step

- Mixed INT8 remains deployment-unavailable and is removed from CoBEVT genes.
- No calibration cache was generated because no legal INT8 action was used.
- These are fixed50 screening results. No 1789-frame full validation or formal
  isolated single-GPU latency replay was run.
- A subsequent CoBEVT search should seed the demonstrated greedy profile,
  protect empirically sensitive attention paths, collect real Stage-2 labels,
  then rerun a wider bounded GA before selecting full-validation candidates.

Exact build and saved-request evaluation commands are in:

`/data/lxf/heal_data/outputs/4090_lidar_cobevt_model_family_smoke_20260718_042721/reproduction_commands.sh`
