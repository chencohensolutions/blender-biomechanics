"""Load bpy-free modules from `biomechanics_addon/` without executing its package `__init__.py`.

`biomechanics_addon/__init__.py` imports `bpy` at module scope, so a normal
`import biomechanics_addon.schemas` fails outside Blender. This loads a module
directly from its file path instead.
"""

import importlib.util
import sys
import types
from pathlib import Path

_ADDON_DIR = Path(__file__).resolve().parents[1] / "biomechanics_addon"
_PACKAGE = "biomechanics_addon_standalone"


def _ensure_package():
    # A stand-in package so bpy-free modules can import each other relatively.
    package = sys.modules.get(_PACKAGE)
    if package is None:
        package = types.ModuleType(_PACKAGE)
        package.__path__ = [str(_ADDON_DIR)]
        sys.modules[_PACKAGE] = package
    return package


def load_addon_module(module_name):
    module_path = _ADDON_DIR / f"{module_name}.py"
    if not module_path.exists():
        raise FileNotFoundError(f"No such add-on module: {module_path}")
    _ensure_package()
    qualified = f"{_PACKAGE}.{module_name}"
    if qualified in sys.modules:
        return sys.modules[qualified]
    spec = importlib.util.spec_from_file_location(qualified, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules[qualified]
        raise
    return module
