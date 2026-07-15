# Joint Taylor GA Design

## Scope

Add a parameter-level joint pruning and mixed-precision Taylor proxy without
changing the GA genotype, dependency graph, physical pruning replay, typed-QDQ
deployment, or Stage-2 objective. Calibrate a fixed exponential task score
from a global second-order Taylor pruning sweep, then run one fresh
generation-0 Top-5 validation. Stage A and Stage B remain stopped.

## Architecture

The implementation has four bounded units:

1. `search/proxy/joint_taylor.py` owns effective parameter perturbations,
   first/second-order loss estimates, conditional group marginal costs,
   parameter-slice deduplication, and the exponential `J1` mapping.
2. `search/proxy/fisher_statistics.py` collects and persists `mean(g)` and
   empirical Fisher `mean(g^2)` with a signed manifest. It never constructs a
   full Hessian.
3. `search/anchors/joint_taylor_sweep.py` converts a fixed FP16 conditional
   group ranking into legal structures for requested parameter prune rates,
   orchestrates the FP32/FP16/maximal-legal-INT8 engine matrix, validates full
   manifests, isolates formal latency replay, and calibrates immutable tau.
4. Existing `LidarPyramidSearchRunner` and Stage-1 repair receive narrow
   adapters. Raw GA masks stay untouched until shortlist repair; repair uses
   conditional group marginal cost only to choose a legal monotonic subset and
   never changes precision genes. Stage-2 remains unchanged.

## Data Flow

Fisher collection records gradients per micro-batch and accumulates both
`sum(g)` and `sum(g^2)`. The parameter-slice resolver maps each atomic pruning
unit to exact physical slices. Joint scoring first pseudo-quantizes each whole
parameter using its canonical precision group, then overwrites pruned slices
with zero. A boolean union mask ensures overlapping dependency closures count
each parameter element once.

For candidate `C`, `delta = M(C) * Q_b(C)(W) - W`. The first-order score is
`sum(abs(g_mean * delta))`; the second-order score adds
`0.5 * sum(fisher_diag * delta^2)`. No parameter-count or local-domain
normalization is applied. SQNR remains diagnostic and is not added to J1.

After tau calibration, Stage-1 maximizes
`J1 = 0.8 * exp(-L_joint/tau) + 0.2 * R_prune`. The exponent is computed in
float64 and capped at 80 with an explicit saturation flag. BOPS remains a hard
post-repair admission gate.

## Anchor Sweep

The anchor builder globally ranks legal unprotected coupled groups using the
second-order conditional FP16 marginal cost. It selects the legal structure
nearest each requested parameter prune rate in `{0.0,...,0.7}`, enforcing the
existing dense/grouped alignment rules and a per-domain prune cap of 0.8. It
records requested and realized rates separately and never uses this ranking to
replace GA candidate generation.

Every unique physical structure is deployed as strict FP32, strict FP16, and
maximal legal INT8 on GPUs 4-7 using the existing strongly typed production
path. All AP evaluations use one full validation manifest. After builds and AP
evaluations finish and workers stop, one idle 4090 replays every engine
serially for formal latency.

Tau uses the safe valid anchor with absolute mAP drop closest to, but not above,
0.1. Up to three legal bisection anchors refine a crossing. The fixed value is
`L_safe / ln(2)` and is written to a read-only `proxy_scale.json`. Missing or
invalid safe boundaries fail closed before generation-0.

## Error Handling

Nonfinite Fisher values, duplicate parameter slices, missing precision mapping,
illegal grouped-conv structures, deployment audit failures, manifest mismatch,
BOPS failures, concurrent formal-latency workers, or invalid tau all fail
closed. Results are never replaced with defaults and failed anchors are kept in
the failure matrix.

## Verification

Tests cover the 35 user-specified contracts, including exact perturbation
semantics, `mean(g^2)`, no normalization, raw-mask preservation, conditional
repair, tau invariance, absolute mAP drop, formal-latency isolation, lineage,
and SQNR exclusion. Existing search, physical pruning, typed-QDQ, merge,
realized-BOPS, worker, process-pool, and orchestration suites remain regression
gates. Large experiment artifacts stay under timestamped ignored `outputs/`.

