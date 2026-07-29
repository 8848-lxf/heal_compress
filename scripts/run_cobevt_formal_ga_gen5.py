#!/usr/bin/env python3
"""Formal five-generation CoBEVT campaign entrypoint."""

from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.run_cobevt_formal_ga_gen10 import main


if __name__ == "__main__":
    raise SystemExit(main())
