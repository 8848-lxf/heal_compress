# TensorRT deployment API

`build_trt_command` is pure command generation. It emits FP16/INT8 flags,
`--precisionConstraints=obey`, canonical `--layerPrecisions` and
`--layerOutputTypes`, workspace, min/opt/max shapes, plugin and layer-info
output. TensorRT/trtexec/plugin paths are caller supplied; no server path is a
default.

`build_trt_engine` is an explicit opt-in subprocess call with timeout, log,
return code, engine hash and failure report. It is never called by import,
profile generation, Q/DQ insertion or validation.

`validate_engine_structure` requires snapshot v2 by default, canonical layer
coverage and compatible physical/engine weight metadata. It never falls back
to sampling estimates. `validate_precision_realization` records requested
INT8, realized INT8/FP16, fallback, hidden casts, reformats, boundaries,
coverage and mismatch reasons. `validate_engine_provenance` checks physical,
profile, mapping, base/QDQ ONNX, engine/plugin hashes, TensorRT version and
build-policy version.

`load_trt_engine` and `run_engine_smoke` import TensorRT/CUDA only on explicit
use. No engine was built or run during formalization.

