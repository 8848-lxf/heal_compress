# 4090 DiscoNet Strongly Typed Deployment Readiness

```text
DISCONET_ONNX_EXPORT_PASS=true
DISCONET_STRICT_FP32_PASS=true
DISCONET_STRICT_FP16_PASS=true
DISCONET_MAXIMAL_LEGAL_INT8_PASS=true
DISCONET_READY_FOR_SEARCH=true
FORMAL_ISOLATED_LATENCY_PASS=false
FORMAL_ISOLATED_LATENCY_REASON=shared_gpu_authorized_screening_only
```

## Identity

```text
branch = feature/heal-compress-4090-cnn-softfusion-family
code commit = ba81fd38cd9d49fe5049c19dc4c84229b972f506
output = /var/tmp/lxf/heal_data/outputs/disco_strongly_typed_readiness_20260719_191700/4090_disco_joint_six_budget_ga_20260719_041947
checkpoint sha256 = cc69e872fab245d687260ec4c277fab531b66a0b1e4d89fedfd4605b8f13e96c
validation manifest sha256 = dea61ba10cf70f259801e049a4d7c38336dc635ebdadc60ba1d997c2473ea496
physical model hash = 48f0ce646b745064449b99655fcc0c658a353b725c7081b0c83e58e1bd0eb2d2
fixedK = 29696
plugin boundary = FP32
plugin sha256 = 91aec743dca383151b995a60d004cd254ce115ed16152873c1f46754bb15022d
trtexec sha256 = 4bd148a8f709f7c7c5f91b1a662a2e17d55710741fad67dc61fa078342a9523f
GPU = NVIDIA GeForce RTX 4090
GPU UUID = GPU-a4b5f0d9-c77c-2c99-b4c3-1b3811b835e3
compute capability = 8.9
driver = 580.105.08
CUDA = 11.8
TensorRT = 10.9.0.34
```

The run used GPU 7 as explicitly authorized by the user. A Pyramid persistent
worker and an external inference service remained resident. Accuracy and
precision-realization evidence is valid, but latency is screening-only until a
serial replay on an isolated GPU.

## Results

All rows use the same 500-frame manifest, GPU AP IoU, eight DataLoader workers,
20 warmup frames, and zero skipped frames.

| precision | AP30 | AP50 | AP70 | mAP | BOPS retention | p50 ms | p90 ms | p95 ms | engine SHA256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| strict FP32 | 0.703705 | 0.620369 | 0.485990 | 0.603354 | 1.000000 | 8.129 | 14.301 | 19.048 | `475c46243bbeebb10d576cb8de6d004af1aef414610f03586020136213761c31` |
| strict FP16 | 0.704203 | 0.620924 | 0.486718 | 0.603948 | 0.250000 | 4.845 | 12.837 | 16.371 | `b46b670036df0c4d65669c22913adb0d4441c6f2ea2f7f881fc05885311036ff` |
| maximal legal INT8 | 0.628350 | 0.545198 | 0.363737 | 0.512429 | 0.062500 | 3.876 | 8.974 | 12.954 | `c749c99330340dff0f259908d74d2511ae7d747fdb48740b7dad30fe0522bb4c` |

Screening deltas relative to strict FP32:

```text
strict FP16 delta mAP = +0.000594
strict FP16 p50 speedup = 1.678x
maximal legal INT8 delta mAP = -0.090926
maximal legal INT8 p50 speedup = 2.097x
```

The speedups are not formal latency claims because the GPU was shared.

## Deployment Audits

```text
weighted canonical layers = 32
precision groups = 32
protected precision groups = 1
pruning-scope precision coupling count = 0
strongly typed = true
unresolved typed tensors = 0
PointPillarScatterTRT QDQ count = 0
strict FP32 realized = 32 FP32
strict FP16 realized = 32 FP16
maximal legal INT8 realized = 31 INT8 + 1 protected FP16
requested INT8 groups = 31
realized INT8 groups = 31
precision fallback mismatches = 0
QuantizeLinear count = 88
DequantizeLinear count = 88
QDQ boundary audit = passed
merge realization audit = passed
physical structure audit = passed
engine deserialize = passed
calibration cache reused = false
calibration cache sha256 = 234472dbfe42f627f990e7c1bbc171ed4315f301b8f646f8a047692bd6abe3c0
```

Both fusion merges are proven FP16 semantic boundaries. `/Concat_7` is fused
with its downstream typed Cast; EngineInspector Metadata, the explicit FP16
branch Casts, and the strongly typed graph jointly establish its realization.

## Readiness Decision

`DISCONET_READY_FOR_SEARCH=true` because physical all-keep export, ONNX checker,
strongly typed FP32/FP16/INT8 builds, real runtime, exact 500-frame evaluation,
requested/realized identity, QDQ, plugin, merge, and BOPS audits all pass. The
INT8 AP loss is real search evidence, not a deployment-capability failure.

Formal latency must be replayed after candidate engines are built and all
builders/workers are stopped. No formal speed ranking is made from this run.

## Reproduction

```bash
cd /home/lixingfeng/UniAD_examine/heal_compress/.worktrees/cnn-softfusion-family
/home/lixingfeng/anaconda3/envs/univ2x-opt/bin/python -m search.cli \
  --config search/configs/lidar_disco_4090_joint_six_budget_ga.yaml \
  --baseline-only \
  --output-root /var/tmp/lxf/heal_data/outputs/disco_strongly_typed_readiness_<timestamp>
```

--- ROUND 1 | 2026-07-19 19:35:51 +0800 ---
