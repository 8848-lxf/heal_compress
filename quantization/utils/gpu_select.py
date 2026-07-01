from __future__ import annotations

from typing import Any

from .paths import load_quant_deploy_module


def select_idle_gpu(*args: Any, **kwargs: Any) -> Any:
    module = load_quant_deploy_module("select_idle_gpu")
    return module.wait_for_idle_gpu(*args, **kwargs)
