# Blender Biomechanics

An initial Blender add-on framework for building bone range-of-motion simulations.

See [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) for the current behavior, every UI
control and core function, implementation invariants, validation checklist, and
next-session development priorities.

## Develop from Source

1. Install Blender 3.6 or newer.
2. Make `blender.exe` discoverable in one of these ways:
	- Add it to `PATH`.
	- Set `BLENDER_EXE` to its full path before starting VS Code.
	- Use a standard Blender Foundation, Microsoft Store, Steam, or Scoop installation.
3. In VS Code, press `Ctrl+Shift+B` and select **Blender: Run Add-on from Source**.
4. In Blender's 3D Viewport, press `N` and open the **Biomechanics** tab.

The task starts Blender and registers `biomechanics_addon` directly from this checkout; no ZIP rebuild or installation is needed. After editing the source while Blender remains open, switch to the **Scripting** workspace, open `tools/dev_load.py`, and press `Alt+P` to reload the add-on.

## Source Layout

- `biomechanics_addon/__init__.py`: add-on metadata, class order, and registration lifecycle.
- `biomechanics_addon/alignment.py`: mesh comparison and rigid-transform math.
- `biomechanics_addon/operators.py`: user actions and input validation.
- `biomechanics_addon/panels.py`: 3D Viewport sidebar interface.
- `biomechanics_addon/properties.py`: scene-level collection settings.

## Blender MCP

The workspace MCP server is configured in `.vscode/mcp.json`. BlenderMCP also needs its Blender add-on installed and enabled in the current Blender profile.

1. Restart Blender after enabling the BlenderMCP add-on, or activate it in the current session from a Python Console with `import bpy; bpy.ops.preferences.addon_enable(module="blender_mcp")`.
2. In VS Code, run **MCP: List Servers**, select **blender**, and start it. Confirm the local-server trust prompt.

BlenderMCP listens on local port `9876` and starts automatically with Blender once enabled. Save the `.blend` file before allowing MCP tools to execute Blender Python code.

## Install

1. Compress the `biomechanics_addon` directory as `biomechanics_addon.zip`.
2. In Blender, open **Edit > Preferences > Add-ons**.
3. Select **Install from Disk** and choose `biomechanics_addon.zip`.
4. Enable **Biomechanics Simulator**.
5. In the 3D Viewport, press `N` and open the **Biomechanics** tab.
6. Set **Original** to the collection containing the unmodified model.
7. Set **Rigid** to the collection containing its rigid-body counterpart.
8. Select one mesh object in the original collection and select **Align to Source**.

The add-on pairs the selected object with the rigid collection object that has the
same name, ignoring Blender's numeric suffix such as `.001` or `.002`. It compares
the corresponding mesh vertices to find the rigid transform, so differing object
origins do not affect alignment. It then moves and rotates every mesh object in the
original collection as one rigid group while preserving object scale and relative
layout. Paired meshes must have matching vertex order and topology.