# Project Guidelines

## Project Shape

- This repository is a Blender 3.6+ Python add-on, not a standalone Python application.
- Treat `biomechanics_addon/` as the source of truth. `biomechanics_addon.zip` is a generated install artifact; update it only when a task requires a distributable package.
- Use [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) as the behavioral handoff for current controls, invariants, core functions, validation, and next priorities.
- Follow [README.md](README.md) for source launch, reload, Blender MCP, packaging, and terminal-training workflows.

## Add-on Conventions

- Keep Blender metadata in `bl_info` and preserve the `bpy`-driven `register()`/`unregister()` lifecycle.
- Name operators `BIOMECHANICS_OT_*` with `bl_idname` values under `biomechanics.*`; name panels `BIOMECHANICS_PT_*`.
- Add every registerable Blender class to `CLASSES` in dependency order. Register in that order and unregister in reverse order.
- Use Blender properties and APIs for runtime behavior. Do not add production fallbacks that mock `bpy`; it is supplied by Blender.
- Keep alignment and comparison behavior on Original meshes; constrained interaction and training operate on Rigid meshes. Preserve the detailed invariants in [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md).
- Treat [BONE_NEAREST_NEIGHBORS.md](BONE_NEAREST_NEIGHBORS.md) as geometric reference data, not an anatomical attachment or joint graph.
- Keep `bpy`, evaluated-mesh, modifier, and BVH work on Blender's main thread. Run expensive training in a separate Blender process through `tools/run_training.ps1`, not a Python thread in the UI process.

## Validation

- Run `python -m compileall biomechanics_addon` for a host-independent syntax check. A normal Python import is not a valid test when `bpy` is unavailable.
- Use the `Blender: Run Add-on from Source` VS Code task for development checks; reload an open Blender session with `tools/dev_load.py`.
- For Blender runtime, UI, registration, or training changes, run the relevant manual checks in [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) and [README.md](README.md). There is currently no automated Blender test suite.