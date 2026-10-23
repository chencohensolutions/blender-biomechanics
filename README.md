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

The task starts Blender and registers `biomechanics_addon` directly from this checkout; no ZIP rebuild or installation is needed. After editing the source while Blender remains open, run the VS Code task **Blender: Reload Add-on** (Command Palette > **Tasks: Run Task**). It touches `tools/.reload_request`; a timer installed by `tools/dev_load.py` notices the change within half a second and reloads the add-on. As a fallback, switch to the **Scripting** workspace, open `tools/dev_load.py`, and press `Alt+P` (`Option+P` on macOS).

### macOS

1. Install Blender 3.6 or newer in `/Applications` (or `~/Applications`), or set `BLENDER_EXE` to the Blender executable or `Blender.app` before starting VS Code.
2. Press `Cmd+Shift+B` and select **Blender: Run Add-on from Source**. On macOS this task runs `tools/launch_blender.sh`, which uses the newest installed Blender.
3. **Blender: Open Add-on Reload Script** opens `tools/dev_load.py` in VS Code. It requires the `code` command on your PATH (VS Code: **Shell Command: Install 'code' command in PATH**).

## One-Click Neural Training

The **Geometric Dataset** panel launches the resumable external neural pipeline.
Choose the fixed and moving meshes in **Rigid**, set the Translation Domain (mm) and
Rotation Domain (degrees), choose an export folder, and select **Start One-Click Neural
Training**. The operator refreshes the package using those domains, then starts a visible
PowerShell window without blocking Blender. Blender polls `neural_training_status.json`;
the external process writes detailed output to `neural_training.log`.

The same workflow is available from the repository root:

```powershell
.\tools\run_neural_training.ps1 `
	-PackagePath ".\biomechanics_training" `
	-BlenderPath "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" `
	-PythonPath ".\.venv\Scripts\python.exe"
```

The script discovers Blender from `PATH`, common installations, Scoop, and the
Microsoft Store launcher. It creates or reuses `.venv`; pass `-InstallRequirements`
when that environment needs the pinned `requirements-neural.txt` packages. Without
that switch, a missing dependency fails early instead of changing an existing
environment during a training run.

Stage A runs the background-Blender dataset generator and reuses compatible stage
checkpoints in `dataset/`. Stage B trains and publishes a model only after ONNX
parity succeeds; `model_manifest.json` is written last, so a partial model is never
treated as active. Add `-ActiveLearning` for bounded uncertain-pose labeling, for
example:

```powershell
.\tools\run_neural_training.ps1 `
	-PackagePath ".\biomechanics_training" `
	-ActiveLearning -ActiveRounds 3 -MaxTotalNewSamples 120 `
	-MaxRuntimeSeconds 1800
```

Rerunning a package resumes dataset stages and retrains from the existing dataset;
the script does not delete checkpoints or the last complete model. Status states are
`running`, `completed`, `converged`, and `failed`, with the exact stage command,
exit code, artifact paths, and available holdout metrics recorded in the status JSON.
The learned output represents geometric feasibility under the documented label rule,
not anatomical truth or an exact collision proof. Microsoft Store Blender may not
forward background console output reliably, so inspect the package log and status
files when diagnosing a run.

## Source Layout

- `biomechanics_addon/__init__.py`: add-on metadata, class order, and registration lifecycle.
- `biomechanics_addon/alignment.py`: mesh comparison and rigid-transform math.
- `biomechanics_addon/interaction.py`: live target-based constrained transforms.
- `biomechanics_addon/muscle_atlas.py`: muscle attachment data; see [MUSCLE_ATTACHMENTS.md](MUSCLE_ATTACHMENTS.md).
- `biomechanics_addon/muscle_strings.py`: muscle strings between bones, hover lengths, and CSV export.
- `biomechanics_addon/neural_geometry.py`: evaluated mesh geometry and pose labeling helpers.
- `biomechanics_addon/neural_training.py`: bpy-free launch, status, and artifact helpers.
- `biomechanics_addon/operators.py`: user actions and input validation.
- `biomechanics_addon/panels.py`: 3D Viewport sidebar interface.
- `biomechanics_addon/properties.py`: scene-level collection settings.
- `tools/run_neural_training.ps1`: resumable dataset, training, and active-learning orchestration.

## Blender MCP

The workspace MCP server is configured in `.vscode/mcp.json`. BlenderMCP also needs its Blender add-on installed and enabled in the current Blender profile.

The server runs through `uvx`, resolved from `PATH` on both macOS and Windows. Install [uv](https://docs.astral.sh/uv/) first (`brew install uv` on macOS, `winget install astral-sh.uv` on Windows), then restart VS Code so it picks up the updated `PATH`.

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
4. With a neural model imported, the trained movement domain and geometric validity
	projection constrain movement while **Live Solve** is enabled. Without a neural
	model, movement is unrestricted.
5. Select **Reset to Neutral** or **Commit and Stop**.

The target displays the requested pose while the rigid mesh stops at the configured
limits. This first interaction layer does not yet calculate mesh collision, SI
contact, pubic-symphysis spring response, or movement of the opposite ilium.