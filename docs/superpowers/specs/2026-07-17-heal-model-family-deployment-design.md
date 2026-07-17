# HEAL Model-Family Search And Deployment Design

## Decision Summary

Adopt a frozen-legacy, model-family-recipe architecture. The existing
`lidar_pyramid` search, typed-ONNX export, explicit-QDQ insertion, calibration,
TensorRT build, and evaluation path remains the production implementation and
is not rewritten. Shared services are exposed behind narrow contracts, while
`lidar_cobevt` receives a separate model-family recipe for physical pruning,
typed export, canonical precision mapping, plugin capability, and evaluation.

The first compatibility experiment is intentionally bounded:

- checkpoint:
  `/home/lixingfeng/UniAD_examine/Auto_Search/original_models/dairv2s/LiDAROnly/lidar_cobevt/net_epoch_bestval_at19.pth`;
- checkpoint SHA256:
  `67b0f2f00d74fea4912b4fdb902c1150146c733201e5dc36d3358f5ba605cfd4`;
- one deterministically selected feasible BOPS band;
- one greedy endpoint;
- one GA seed with population 16, three generations, and per-generation Top-2;
- real physical pruning, typed ONNX, explicit Cast/QDQ, strongly typed
  TensorRT, engine inspection, GPU smoke10, and fixed 50-frame evaluation.

This experiment validates compatibility. It does not replace the running
six-budget `lidar_pyramid` experiment and does not claim CoBEVT production
readiness or a final Pareto frontier.

## Considered Architectures

### Selected: frozen pyramid plus model-family recipes

Introduce common capability and deployment contracts, dispatch by model
family, and implement CoBEVT beside the existing pyramid path. The pyramid
factory delegates to the exact current runner and evaluator. This minimizes
regression risk, gives CoBEVT explicit ownership of its transformer contracts,
and creates a reusable seam for later HEAL families.

### Rejected for this phase: immediate generic rewrite

Moving both pyramid and CoBEVT onto a new generic FX/ONNX deployment engine
would produce a cleaner end state, but it would invalidate the accepted
pyramid lineage while the long search is active. It also combines interface
extraction with a production migration, making failures hard to attribute.

### Rejected: clone the entire pyramid path

A full CoBEVT copy would be quick initially, but would duplicate calibration,
cache, builder, process-pool, and audit logic. It would not establish a
maintainable HEAL-series architecture and would cause the two implementations
to drift.

## Non-Negotiable Isolation

1. Existing pyramid quantization and deployment modules retain their behavior,
   defaults, canonical mapping, plugin boundary, cache lineage, and public
   entry points.
2. The default CLI family remains `lidar_pyramid`; old configs without a model
   family field resolve to the existing `LidarPyramidTwoStageSearch`.
3. No CoBEVT code is imported by the pyramid deployment modules. Model-family
   selection occurs before the runner is constructed.
4. The active pyramid process and its output directory are not stopped,
   resumed, rewritten, or reused by the CoBEVT experiment.
5. CoBEVT artifacts use a new timestamped directory under
   `/data/lxf/heal_data/outputs/` and are ignored by Git.
6. ONNX, PTH, PLAN/engine, calibration caches, tensor dumps, and plugin
   binaries are never committed.

## Model-Family Contracts

The model-family layer exposes five focused capabilities.

### Model capability

Loads the exact config and checkpoint, creates train/calibration/validation
loaders, builds trace inputs, computes the HEAL task loss, and provides the
post-processing evaluator. It records checkpoint, config, dataset manifest,
and code hashes. `HEALLiDARAdapter` remains the common HEAL loader; a family
recipe supplies family-specific behavior.

### Pruning capability

Builds the current tracer-derived local domains and legal-width inventory,
maps a legal width genotype to a deterministic physical plan, materializes the
model, and validates parameter counts and tensor shapes. CoBEVT fusion widths
are legal only in multiples of `dim_head=32`; QKV, output projection, feed
forward, LayerNorm, positional embedding head dimension, shrink output, and
prediction-head inputs must remain consistent. If that closure cannot be
proven by the current tracer plus the family materializer, the fusion domain is
protected rather than partially pruned.

The older Auto_Search CoBEVT pruner is evidence for required slice semantics,
not a runtime dependency. Required logic is reimplemented behind the CoBEVT
recipe with tests; the external package is not imported by production search.

### Export capability

Defines fixed input names, dynamic axes/profile bounds, output names, custom
ONNX symbols, shape/type audits, and PyTorch-to-ONNX parity. CoBEVT uses padded
agents (`max_cav=2`) and a fixed-K point-pillar input contract. Fixed K is
derived from the exact train200 calibration and 50-frame validation manifests,
rounded to the declared packing alignment, and must have zero overflow for
those manifests. It is not copied blindly from the pyramid value `29696`.

### Quantization capability

Owns canonical weighted operations, precision groups, protected functional
operations, explicit FP16/FP32 Cast boundaries, INT8 QDQ boundaries, merge and
residual contracts, weight scales, calibration lineage, and realized-profile
audit. Precision actions are generated from demonstrated deployment
capability; unsupported INT8 actions are absent rather than silently legalized
to FP16 or FP32.

### Evaluation capability

Runs one baseline and candidates on the same deterministic 50-frame manifest,
with GPU post-processing, DataLoader workers set to 8, smoke10 before formal
screening, warmup/reset semantics, evaluated/skipped accounting, and latency
provenance. It records AP03/AP05/AP07/mAP and p50/p90/p95/FPS from real engine
execution. Screening latency may share a lightly occupied GPU and is labelled
as such; it is not presented as isolated formal latency.

## Plugin And Special-Operator Policy

Point-pillar encoding is common to both models, so CoBEVT first attempts to
reuse the accepted `PointPillarScatterTRT` plugin. Its current binary hash is
`91aec743dca383151b995a60d004cd254ce115ed16152873c1f46754bb15022d`.
The plugin stays floating point and outside precision genes.

`maxK` is initially an input packing/profile policy, not a TensorRT plugin.
Einops rearrange/reduce operations should lower to ONNX reshape, transpose, and
reduce nodes. CoBEVT attention should first use ONNX-native MatMul, Add, Where,
Softmax, LayerNorm decomposition, and projection operators. Grid warping should
first use TensorRT's supported ONNX representation.

A new plugin is allowed only after a reproducible capability probe proves one
of the following:

1. ONNX checker or type/shape inference cannot express the required semantic
   operation;
2. TensorRT 10.9 strongly typed parsing rejects the minimal reproducer;
3. the parsed engine runs but fails numerical parity at the isolated operator
   boundary;
4. the native graph requires a prohibited weakly typed fallback.

Any new plugin lives under its own CoBEVT-specific directory, has an isolated
CMake target and creator/version namespace, declares supported FP32/FP16
dtypes, is serialized/deserialized in a minimal test, and contributes its SHA
to deployment identity. It must not modify the scatter plugin source or the
pyramid plugin load list.

## Strongly Typed Deployment Sequence

The CoBEVT capability study and smoke follow this fail-closed sequence:

1. load the exact model and verify checkpoint coverage;
2. run a PyTorch forward and fixed-manifest baseline;
3. enumerate weighted and functional operations and build a canonical mapping;
4. export a strict FP32 typed ONNX graph using the family wrapper;
5. run ONNX checker, shape/type audit, parser audit, and PyTorch/ONNX parity;
6. minimize any parser/parity failure to the responsible subgraph;
7. use native ONNX decomposition when semantically equivalent;
8. add a model-specific plugin only when the plugin policy is satisfied;
9. build strict FP32 and FP16 strongly typed reference engines;
10. probe legal INT8 groups with explicit per-output-channel weight QDQ and
    train200 entropy-calibrated per-tensor activation QDQ;
11. remove any action that fails QDQ, merge, inspector, or realized-profile
    identity;
12. freeze the resulting deployment capability manifest before search.

Production commands use `--stronglyTyped --noTF32`. They must not use
`--fp16`, `--int8`, precision constraints, layer precision overrides, layer
output type overrides, or weakly typed fallback. Tensor dtypes originate only
from ONNX types, explicit Cast, Q/DQ, and plugin declarations.

The configured TensorRT path supplied in an earlier request,
`/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118`, is absent on this
host. The installed and currently exercised TensorRT 10.9 tree is
`/home/lixingfeng/UniAD_examine/HEAL/prune_model/TensorRT-10.9_x86_cu118`;
the experiment records this resolved path rather than silently using a missing
location.

## CoBEVT Precision Contracts

Canonical precision groups are generated from the exported CoBEVT graph and
cross-checked against PyTorch modules. Conv and Linear projections may expose
FP32, FP16, and INT8 only after actual realization tests. The following remain
floating point until separately proven safe:

- scatter input/output;
- Softmax and its mask/fill path;
- LayerNorm reductions and normalization;
- relative-position index and embedding lookup;
- residual/Add merge tensors;
- grid-warp coordinates and sampling;
- final cls/reg/dir raw outputs.

QKV and feed-forward Linear layers are not globally declared INT8 merely
because they have weights. Each action must preserve the surrounding floating
attention contract using explicit QDQ/Cast and must match the EngineInspector
realization. Requested, legalized, and realized profile hashes must be equal;
unresolved mappings or unexpected fallback reject the candidate.

## Search Smoke

The compatibility run uses the existing legal-width genotype, deterministic
prune-only Taylor decoder, joint pruning/weight-quantization Taylor candidate
score, BOPS gate, greedy implementation, GA operators, cache identity, and
Stage-2 process infrastructure. Model-specific behavior enters only through
the CoBEVT family recipe.

Before search, the capability probe computes reachable BOPS bands. From the
approved budget series `{0.10, 0.15, 0.20, 0.25, 0.30}`, the smoke selects the
band with the most unique legal proxy-feasible phenotypes; ties select the band
closest to `0.20`. Tolerance is `0.0075`. The selected band and candidate
counts are written before Stage-2 and are not changed after seeing AP results.
If no band contains a deployable candidate, the smoke fails closed.

The greedy run searches to one endpoint and deploys that unique endpoint. The
GA uses population 16, three generations, one seed, and per-generation Top-2.
Top-2 means up to six Stage-2 admissions; duplicate deployment identities are
built once and referenced by cache lineage. A generation with one feasible
candidate evaluates that candidate. A generation with none records the BOPS or
legality exhaustion reason.

Every admitted candidate executes:

`legal width -> deterministic mask -> physical model -> structure audit ->`
`typed ONNX -> explicit Cast/QDQ -> train200 calibration -> strongly typed`
`engine -> deserialize/inspector -> smoke10 -> fixed50 GPU evaluation`.

At least one greedy and one GA candidate must complete real engine inference
and fixed50 for the framework compatibility smoke to pass. Results are
diagnostic and do not unlock or modify pyramid Stage A/B.

## Deployment Identity And Cache Safety

The cache key contains model family and recipe version in addition to code
commit, checkpoint/config hashes, physical model and structure hashes, base
ONNX hash, canonical mapping and QDQ topology hashes, requested/legalized/
realized precision hashes, fixed-K/profile hash, calibration and validation
manifest hashes, calibration recipe, TensorRT/CUDA/GPU architecture, and every
plugin binary hash. Pyramid artifacts can never satisfy a CoBEVT cache key.

## Failure Handling

The run fails closed for checkpoint mismatch, physical replay failure,
parameter-count mismatch, fixed-K overflow, ONNX export/check/type failure,
parser failure, operator parity failure, missing canonical mapping, illegal
INT8, QDQ or scale audit failure, merge dtype failure, unexpected fallback,
strongly typed build/deserialization failure, inspector/profile mismatch,
BOPS violation, nonfinite output, evaluation skip, or cache signature
mismatch. Weakly typed engines and PyTorch-only candidates cannot be reported
as successful deployment results.

Failures preserve a minimal reproducer, command, stderr digest, graph/operator
inventory, and hashes. Large graphs and engines remain outside Git.

## Verification Strategy

Implementation follows RED-GREEN tests. Coverage includes:

- family registry defaults old configs to the unchanged pyramid runner;
- CoBEVT selection cannot mutate pyramid deployment configuration;
- capability manifests are model-family and recipe-version specific;
- fixed-K derivation has zero manifest overflow and does not assume 29696;
- scatter is reused, floating point, and excluded from precision genes;
- unsupported attention precision actions are absent;
- CoBEVT legal widths keep attention head and QKV/FFN/head inputs consistent;
- physical parameter predictions match the materialized model;
- ONNX dtype, QDQ, merge, and inspector identities agree;
- strongly typed commands reject weak precision flags;
- cross-family cache reuse is rejected;
- greedy and GA smoke sizes are exactly the approved bounds;
- smoke10 and fixed50 use real GPU engine execution and one manifest;
- DataLoader workers are 8 and post-processing uses the GPU path;
- pyramid regression tests retain accepted canonical and deployment behavior.

## Deliverables

Code and tests are committed in focused stages: model-family contracts and
dispatch, CoBEVT model/pruning capability, CoBEVT export capability probe,
CoBEVT strongly typed quantization/deployment, and bounded greedy/GA smoke.
Lightweight evidence is written to:

- `docs/codex_handoffs/4090-lidar-cobevt-search-smoke-report.md`;
- the existing 4090 progress handoff with a timestamped round separator;
- JSON/CSV summaries containing structure, precision, plugin, engine, AP,
  latency, GPU, and cache lineage.

The running pyramid search remains independent. No CoBEVT full validation,
multi-budget production search, or new plugin is implied by completion of this
bounded compatibility smoke.
