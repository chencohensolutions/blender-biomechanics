"""Load bpy-free modules from `biomechanics_addon/` without executing its package `__init__.py`.

`biomechanics_addon/__init__.py` imports `bpy` at module scope, so a normal
`import biomechanics_addon.schemas` fails outside Blender. This loads a module
directly from its file path instead.
"""

import importlib.util
from pathlib import Path

_ADDON_DIR = Path(__file__).resolve().parents[1] / "biomechanics_addon"


def load_addon_module(module_name):
    module_path = _ADDON_DIR / f"{module_name}.py"
    if not module_path.exists():
        raise FileNotFoundError(f"No such add-on module: {module_path}")
    spec = importlib.util.spec_from_file_location(f"biomechanics_addon_standalone.{module_name}", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
