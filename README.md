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

## Terminal Training

The add-on can export a two-object training package and run the existing geometric
trainer in a separate background Blender process. The source Rigid objects and the
open `.blend` file are not modified by the worker.

1. Reload the add-on from `tools/dev_load.py` after updating the source.
2. In **Two-Bone Geometric Training**, choose the fixed and moving Rigid meshes.
3. Set the ranges, steps, proxy face budget, and maximum sample count.
4. Choose **Training Folder**. The default is `biomechanics_training` beside the
	saved `.blend` file.
5. Select **Export Terminal Training Package**.
6. From the repository root, run:

	```powershell
	.\tools\run_training.ps1 -PackagePath ".\biomechanics_training"
	```

The package contains `training_input.blend`, `training_config.json`, and
`RUN_COMMAND.txt`. The terminal displays completed samples, percentage, valid count,
and elapsed time. A successful run writes `training_result.json` into the same
folder. Set `BLENDER_EXE` or pass `-BlenderPath` when Blender is not discoverable:

```powershell
.\tools\run_training.ps1 `
	 -PackagePath ".\biomechanics_training" `
	 -BlenderPath "C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"
```

This worker currently uses the format-version-1 proxy overlap and clearance
classifier. It is suitable for bounded exploratory runs, but it does not yet
guarantee 0.1 mm surface accuracy or anatomical validity.

## Source Layout

- `biomechanics_addon/__init__.py`: add-on metadata, class order, and registration lifecycle.
- `biomechanics_addon/alignment.py`: mesh comparison and rigid-transform math.
- `biomechanics_addon/interaction.py`: live target-based constrained transforms.
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

## Constrained Interaction Prototype

Constrained movement and future collision/contact calculations operate only on mesh
objects in the **Rigid** collection. The Original collection remains unchanged by
this workflow and is used only for alignment and comparison.

1. Select one mesh in Rigid, such as `ilium R.002`.
2. Select **Start Constrained Move**.
3. Move or rotate the **Biomechanics Target** Empty with `G` or `R`.
4. Adjust the local translation limit in millimeters and the rotation limit while
	**Live Solve** is enabled.
5. Select **Reset to Neutral** or **Commit and Stop**.

The target displays the requested pose while the rigid mesh stops at the configured
limits. This first interaction layer does not yet calculate mesh collision, SI
contact, pubic-symphysis spring response, or movement of the opposite ilium.