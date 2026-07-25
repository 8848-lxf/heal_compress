"""Make repository-local packages importable from every pytest invocation."""
from __future__ import annotations

import sys
import importlib
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEAL_ROOT = Path("/home/lixingfeng/UniAD_examine/HEAL")
MODELOPT_SOURCE = Path("/home/lixingfeng/UniAD_examine/Model-Optimizer-0.29.0")

# Pytest can inherit HEAL on PYTHONPATH.  Merely checking whether ROOT already
# occurs in sys.path is insufficient: an earlier HEAL entry imports its regular
# ``opencood`` package first and hides the repository-local
# ``opencood.tools.compression`` namespace.  Always make the current worktree
# first, then explicitly merge (rather than replace) the namespace paths.
sys.path[:] = [entry for entry in sys.path if entry != str(ROOT)]
sys.path.insert(0, str(ROOT))
if MODELOPT_SOURCE.is_dir() and str(MODELOPT_SOURCE) not in sys.path:
    # The modelopt Conda environment contains an editable .pth from a previous
    # checkout location.  Do not mutate that environment; make the audited
    # source checkout available only to this pytest process.
    sys.path.append(str(MODELOPT_SOURCE))

# The worktree directory intentionally is not named ``heal_compress``.  Bind
# that canonical package name to this exact worktree for pytest so imports can
# never fall through to the separate formal-search worktree next door.
for name in [key for key in sys.modules if key == "heal_compress" or key.startswith("heal_compress.")]:
    del sys.modules[name]
package = types.ModuleType("heal_compress")
package.__file__ = str(ROOT / "__init__.py")
package.__package__ = "heal_compress"
package.__path__ = [str(ROOT)]
exec(compile((ROOT / "__init__.py").read_text(), package.__file__, "exec"), package.__dict__)
sys.modules["heal_compress"] = package


def _local_first_namespace(module_name: str, local_path: Path) -> None:
    module = importlib.import_module(module_name)
    current = [str(Path(value).resolve()) for value in getattr(module, "__path__", ())]
    local = str(local_path.resolve())
    module.__path__ = [local, *[value for value in current if value != local]]


_local_first_namespace("opencood", ROOT / "opencood")
_local_first_namespace("opencood.tools", ROOT / "opencood/tools")

# Keep HEAL's regular OpenCOOD implementation available after the local
# compression namespace.  This lets one pytest session import both
# ``opencood.tools.compression`` and HEAL's ``opencood.tools.train_utils``.
if HEAL_ROOT.is_dir():
    import opencood
    import opencood.tools

    for module, path in (
        (opencood, HEAL_ROOT / "opencood"),
        (opencood.tools, HEAL_ROOT / "opencood/tools"),
    ):
        text = str(path.resolve())
        if text not in module.__path__:
            module.__path__.append(text)
