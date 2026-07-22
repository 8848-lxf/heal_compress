"""Compile and execute a minimal SM90 CUDA extension in the active Conda toolchain."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import torch
from torch.utils.cpp_extension import load_inline


def main() -> int:
    cpp = """
    #include <torch/extension.h>
    torch::Tensor add_one_cuda(torch::Tensor input);
    """
    cuda = """
    #include <torch/extension.h>
    __global__ void add_one_kernel(float* value) { value[threadIdx.x] += 1.0f; }
    torch::Tensor add_one_cuda(torch::Tensor input) {
      add_one_kernel<<<1, 1>>>(input.data_ptr<float>());
      return input;
    }
    """
    module = load_inline(
        name="h800_sm90_extension_gate",
        cpp_sources=cpp,
        cuda_sources=cuda,
        functions=["add_one_cuda"],
        extra_cuda_cflags=["-lineinfo"],
        with_cuda=True,
        verbose=True,
    )
    value = torch.zeros(1, device="cuda", dtype=torch.float32)
    actual = module.add_one_cuda(value)
    torch.cuda.synchronize()
    cache = Path(os.environ["TORCH_EXTENSIONS_DIR"]) / "h800_sm90_extension_gate"
    objects = []
    for path in sorted(cache.glob("*.o")):
        try:
            dump = subprocess.check_output(
                [str(Path(os.environ["CONDA_PREFIX"]) / "bin/cuobjdump"), "--list-elf", str(path)],
                stderr=subprocess.STDOUT,
                text=True,
            )
            objects.append({"object": str(path), "cuobjdump": dump})
        except subprocess.CalledProcessError as exc:
            objects.append({"object": str(path), "error": repr(exc)})
    result = {
        "passed": bool(float(actual.item()) == 1.0),
        "cpu_fallback": False,
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "torch_cuda": torch.version.cuda,
        "arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST", ""),
        "cuda_home": os.environ.get("CUDA_HOME", ""),
        "nvcc": os.environ.get("CUDACXX", ""),
        "cxx": os.environ.get("CXX", ""),
        "extension_cache": str(cache),
        "objects": objects,
    }
    print("SM90_RESULT_JSON=" + json.dumps(result, sort_keys=True))
    return 0 if result["passed"] and result["capability"] == [9, 0] else 2


if __name__ == "__main__":
    raise SystemExit(main())
