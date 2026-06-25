#!/usr/bin/env bash
set -e

export TRT_ROOT=/home/lixingfeng/UniAD_examine/TensorRT-10.9_x86_cu118

export PATH=${TRT_ROOT}/bin:${TRT_ROOT}/targets/x86_64-linux-gnu/bin:${PATH}

export NVIDIA_PIP_LIBS=$(python - <<'PY'
import site, glob, os
paths = []
for sp in site.getsitepackages():
    paths += glob.glob(os.path.join(sp, "nvidia", "*", "lib"))
print(":".join(paths))
PY
)

export TORCH_LIB=$(python - <<'PY'
import torch, os
print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)

export LD_LIBRARY_PATH=${NVIDIA_PIP_LIBS}:${TRT_ROOT}/lib:${TRT_ROOT}/targets/x86_64-linux-gnu/lib:${TORCH_LIB}:${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib64:${LD_LIBRARY_PATH:-}

echo "[quant_deploy env] TRT_ROOT=${TRT_ROOT}"
echo "[quant_deploy env] PATH=${PATH}"
echo "[quant_deploy env] LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

which trtexec || true

python - <<'PY'
import sys
print("python:", sys.executable)
try:
    import torch
    print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
except Exception as e:
    print("torch import failed:", repr(e))

try:
    import tensorrt as trt
    print("tensorrt:", trt.__version__)
except Exception as e:
    print("tensorrt import failed:", repr(e))

try:
    import modelopt
    print("modelopt:", getattr(modelopt, "__version__", "unknown"))
except Exception as e:
    print("modelopt import failed:", repr(e))
PY
