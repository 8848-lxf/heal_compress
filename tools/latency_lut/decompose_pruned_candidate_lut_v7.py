from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.latency_lut.decompose_pruned_candidate_lut_v6 import main as v6_main


def main(argv: list[str] | None = None) -> int:
    # v7 decomposition is intentionally routed through the v6 implementation
    # until the v7 sensitivity audit proves that matching ignores pruned widths.
    # This wrapper exists so py_compile and future commands have a stable v7
    # entry point.
    return v6_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
