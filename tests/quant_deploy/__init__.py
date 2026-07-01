"""LiDAR pyramid TensorRT deployment test utilities.

The historical scripts in this package use sibling absolute imports when run as
standalone files. Keep those imports working when pytest imports the files as
``tests.quant_deploy.*``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
