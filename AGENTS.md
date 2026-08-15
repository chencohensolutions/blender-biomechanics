# Project Guidelines

## Project Shape

- This repository is a Blender 3.6+ Python add-on, not a standalone Python application.
- Treat `biomechanics_addon/` as the source of truth. `biomechanics_addon.zip` is a generated install artifact; update it only when a task requires a distributable package.
- Follow [README.md](README.md) for installation and the current manual smoke test.

## Add-on Conventions

- Keep Blender metadata in `bl_info` and preserve the `bpy`-driven `register()`/`unregister()` lifecycle.
- Name operators `BIOMECHANICS_OT_*` with `bl_idname` values under `biomechanics.*`; name panels `BIOMECHANICS_PT_*`.
- Add every registerable Blender class to `CLASSES` in dependency order. Register in that order and unregister in reverse order.
- Use Blender properties and APIs for runtime behavior. Do not add production fallbacks that mock `bpy`; it is supplied by Blender.

## Validation

- Run `python -m compileall biomechanics_addon` for a host-independent syntax check. A normal Python import is not a valid test when `bpy` is unavailable.
- Use the `Blender: Run Add-on from Source` VS Code task for development checks when Blender is installed.
- For UI or registration changes, install the packaged add-on in Blender and repeat the smoke test in [README.md](README.md). There is currently no automated Blender test suite.