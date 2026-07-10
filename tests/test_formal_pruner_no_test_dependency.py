from __future__ import annotations

import ast
import sys
from pathlib import Path

import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
UNIAD = ROOT.parent
for path in (UNIAD, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


FORMAL_PATHS = [
    ROOT / "pruning",
    ROOT / "tracer",
    ROOT / "tools" / "latency_lut" / "run_v108_complete_taylor_greedy_pruner.py",
    ROOT / "tools" / "latency_lut" / "run_v109_param_budget_round4_pruner.py",
    ROOT / "tools" / "latency_lut" / "run_v109_collapse_diagnosis.py",
    ROOT / "tools" / "latency_lut" / "run_v11_mixed_precision_lut_dataset_builder.py",
]


def _python_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return [p for p in path.rglob("*.py") if "__pycache__" not in p.parts]


def _imports_tests(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name == "tests" or name.startswith("tests.") or name.startswith("test_"):
                    bad.append(name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "tests" or module.startswith("tests.") or module.startswith("test_"):
                bad.append(module)
    return bad


def test_formal_runtime_paths_do_not_import_tests() -> None:
    offenders = {}
    for root in FORMAL_PATHS:
        if not root.exists():
            continue
        for path in _python_files(root):
            bad = _imports_tests(path)
            if bad:
                offenders[str(path.relative_to(ROOT))] = bad

    assert offenders == {}


def test_pruning_config_defaults_and_round_to_validation() -> None:
    from heal_compress.pruning.config import PruningConfig

    cfg = PruningConfig(target_pruning_ratio=0.3)

    assert cfg.target_pruning_mode == "param"
    assert cfg.round_to == 4
    assert PruningConfig(target_pruning_ratio=0.1, round_to=128).round_to == 128
    for value in (4, 8, 16, 32, 64, 128):
        assert PruningConfig(target_pruning_ratio=0.1, round_to=value).round_to == value


def test_pruning_config_rejects_invalid_round_to() -> None:
    from heal_compress.pruning.config import PruningConfig

    for value in (0, 3, 12, 256):
        try:
            PruningConfig(target_pruning_ratio=0.1, round_to=value)
        except ValueError as exc:
            assert "round_to" in str(exc)
        else:
            raise AssertionError(f"round_to should have failed: {value}")


def test_heal_structured_pruner_constructs_and_checks_shape_invariants() -> None:
    from heal_compress.pruning.config import PruningConfig
    from heal_compress.pruning.formal_pruner import HEALStructuredPruner

    model = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU())
    pruner = HEALStructuredPruner(model, PruningConfig(target_pruning_ratio=0.0))

    report = pruner.check_shape_invariants()

    assert pruner.config.target_pruning_mode == "param"
    assert report["passed"] is True
    assert pruner.get_manifest()["config"]["stage1_min_per_group"] == 8


def test_stage1_policy_fields_are_exposed() -> None:
    from heal_compress.pruning.config import PruningConfig

    cfg = PruningConfig(target_pruning_ratio=0.66, stage1_min_per_group=8, stage1_max_ch_sparsity=0.30)

    assert cfg.stage1_min_per_group == 8
    assert cfg.stage1_max_ch_sparsity == 0.30
