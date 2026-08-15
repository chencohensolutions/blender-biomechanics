"""Load or reload the add-on directly from this source checkout."""

import importlib
from pathlib import Path
import sys

import addon_utils


ADDON_NAME = "biomechanics_addon"
REPOSITORY_ROOT = str(Path(__file__).resolve().parents[1])


def unload_addon():
    module = sys.modules.get(ADDON_NAME)
    _, enabled = addon_utils.check(ADDON_NAME)

    if enabled:
        addon_utils.disable(ADDON_NAME, default_set=False)
    elif module is not None:
        module.unregister()

    for module_name in list(sys.modules):
        if module_name == ADDON_NAME or module_name.startswith(f"{ADDON_NAME}."):
            del sys.modules[module_name]


if REPOSITORY_ROOT not in sys.path:
    sys.path.insert(0, REPOSITORY_ROOT)

if ADDON_NAME in sys.modules:
    unload_addon()

addon = importlib.import_module(ADDON_NAME)
addon.register()
print(f"Loaded {ADDON_NAME} from {addon.__file__}")