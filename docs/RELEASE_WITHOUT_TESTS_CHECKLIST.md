# Release without tests checklist

Run from the repository root. No command below uses CUDA, a checkpoint,
TensorRT, a plugin or a dataset.

```bash
grep -RniE 'from[[:space:]]+tests|import[[:space:]]+tests|tests\.' tracer pruning quantization
python -m compileall -q tracer pruning quantization

mv tests tests.__disabled_for_release_check
python - <<'PY'
import tracer
import pruning
import quantization

from tracer.api import trace_model
from pruning.api import (
    score_pruning_units,
    select_pruning_request,
    build_physical_pruning_plan,
    materialize_pruning,
)
from quantization.api import (
    export_pruned_signal_maxk_onnx,
    generate_precision_profile,
    insert_explicit_qdq,
    build_trt_command,
)

print("formal package import smoke passed")
PY
mv tests.__disabled_for_release_check tests
```

Also scan Python sources for server-specific paths and path injection:

```bash
rg -n '<server-home>|sys\.path.*tests|runpy.*tests|subprocess.*tests' \
  tracer pruning quantization \
  -g '!**/__pycache__/**' -g '!**/build*/**'
```

Generated compiler caches and CMake/Ninja build metadata are not release
source and are excluded from the source-path gate.

The release gate additionally requires CPU unit tests for trace serialization,
Taylor normalization, global selection, grouped replay, direction-specific
protection, snapshot/hash v2, canonical naming, Q/DQ tracing, TRT command
generation and requested-versus-realized precision.
