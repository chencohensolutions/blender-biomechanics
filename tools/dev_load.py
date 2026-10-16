"""Load or reload the add-on directly from this source checkout."""

import importlib
from pathlib import Path
import sys
import traceback

import addon_utils  # pyright: ignore[reportMissingImports]
import bpy  # pyright: ignore[reportMissingImports]


ADDON_NAME = "biomechanics_addon"
REPOSITORY_ROOT = str(Path(__file__).resolve().parents[1])
# Touched by the "Blender: Reload Add-on" VS Code task.
RELOAD_REQUEST_PATH = Path(REPOSITORY_ROOT) / "tools" / ".reload_request"
RELOAD_POLL_SECONDS = 0.5
WATCHER_KEY = "biomechanics_dev_reload_watcher"


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

def load_addon():
    if ADDON_NAME in sys.modules:
        unload_addon()

    addon = importlib.import_module(ADDON_NAME)
    addon.register()
    print(f"Loaded {ADDON_NAME} from {addon.__file__}")


def _request_mtime():
    try:
        return RELOAD_REQUEST_PATH.stat().st_mtime
    except OSError:
        return None


def _make_reload_watcher():
    last_mtime = _request_mtime()

    def watch_reload_request():
        nonlocal last_mtime
        mtime = _request_mtime()
        if mtime is not None and mtime != last_mtime:
            last_mtime = mtime
            try:
                load_addon()
            except Exception:
                traceback.print_exc()
        return RELOAD_POLL_SECONDS

    return watch_reload_request


def install_reload_watcher():
    previous = bpy.app.driver_namespace.get(WATCHER_KEY)
    if previous is not None and bpy.app.timers.is_registered(previous):
        bpy.app.timers.unregister(previous)
    watcher = _make_reload_watcher()
    bpy.app.driver_namespace[WATCHER_KEY] = watcher
    bpy.app.timers.register(watcher, first_interval=RELOAD_POLL_SECONDS, persistent=True)


load_addon()
install_reload_watcher()