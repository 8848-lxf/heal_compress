#!/usr/bin/env python3
"""Strictly load and capability-audit a HEAL model family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[1]
PROJECT_PARENT = REPO.parent
for path in (str(PROJECT_PARENT), str(REPO)):
    if path not in sys.path:
        sys.path.insert(0, path)


def _tensor_shapes(value: Any) -> Any:
    try:
        import torch
    except ImportError:
        return None
    if torch.is_tensor(value):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
    if isinstance(value, dict):
        return {str(key): _tensor_shapes(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_tensor_shapes(item) for item in value]
    return type(value).__name__


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--heal-root", type=Path, default=Path("../../HEAL"))
    parser.add_argument("--family", default="auto")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--forward-smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from search.model_family import load_heal_model_family

    bundle = load_heal_model_family(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        heal_root=args.heal_root,
        device=args.device,
        family_id=args.family,
        forward_smoke=bool(args.forward_smoke),
    )
    payload = {
        "provenance": {
            "config_path": str(bundle.config_path),
            "config_sha256": bundle.config_hash,
            "checkpoint_path": str(bundle.checkpoint_path),
            "checkpoint_sha256": bundle.checkpoint_hash,
            "strict_state_dict_load": True,
            "state_dict_tensor_count": bundle.state_dict_tensor_count,
            "device": args.device,
            "forward_smoke_requested": bool(args.forward_smoke),
        },
        "forward_smoke": {
            "passed": bundle.smoke_outputs is not None if args.forward_smoke else None,
            "outputs": _tensor_shapes(bundle.smoke_outputs),
            "weighted_modules_total": bundle.weighted_modules_total,
            "weighted_modules_called_count": len(bundle.weighted_modules_called),
            "weighted_module_coverage": (
                len(bundle.weighted_modules_called) / bundle.weighted_modules_total
                if bundle.weighted_modules_total
                else 1.0
            ),
            "weighted_modules_called": list(bundle.weighted_modules_called),
            "uncalled_weighted_modules": list(bundle.uncalled_weighted_modules),
        },
        "model_family_audit": bundle.audit.to_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    summary = {
        "output": str(args.output.resolve()),
        "family_id": bundle.provider.family_id,
        "parameter_count": bundle.audit.parameter_count,
        "canonical_weighted_op_count": bundle.audit.metadata["canonical_weighted_op_count"],
        "pruning_domain_count": len(bundle.audit.pruning_domains),
        "merge_boundary_count": len(bundle.audit.merge_boundaries),
        "blocker_count": len(bundle.audit.blockers),
        "audit_hash": payload["model_family_audit"]["audit_hash"],
        "forward_smoke_passed": payload["forward_smoke"]["passed"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
