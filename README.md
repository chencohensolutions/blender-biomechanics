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
- `biomechanics_addon/muscle_sheets.py`: triangle sheets between a muscle's strings, colored by area change.
- `biomechanics_addon/ligament_atlas.py`: ligaments and intervertebral discs as strings with stiffness data.
- `biomechanics_addon/distance_field.py`, `bone_fields.py`: cached signed distance fields per bone for collision.
- `biomechanics_addon/chain_solver.py`, `chain_interaction.py`: drag a vertebra and the chain follows through ligaments, discs, bone contact, gravity, and active muscles.
- `biomechanics_addon/atlas_import.py`: muscles and ligaments from a registered Z-Anatomy atlas (CC BY-SA 4.0).
- `biomechanics_addon/chain_package.py`, `posture_policy.py`, `tools/train_posture.py`: export the chain and train a muscle posture controller headlessly (`.venv/bin/python tools/train_posture.py --package <dir> --mode policy`).
- `tools/train_posture_surrogate.py`: the fast route. Learns the chain solver once (`collect`, `fit`), then trains the same controller by gradients through the learned dynamics (`optimize`), and checks it in the real solver; `iterate` repeats collect/fit/optimize rounds (`.venv/bin/python tools/train_posture_surrogate.py iterate --package <dir> --rounds 4`). `tools/surrogate_dashboard.py --package <dir>` serves http://localhost:8765 with the policy network drawn live (weights as red/blue lines, node values, training curves) while it trains.
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

## CT bone extraction (tools/ct)

The scripts in `tools/ct/` turn a CT scan (DICOM) into per-bone meshes that drop
into the Blender scene at the scan's own positions. They run in the repository
`.venv` (Python 3.13), not inside Blender, and are not part of the add-on
(`export_atlas_meshes.py` is the exception: it runs in headless Blender).

Bone surfaces now come from `segment_nni.py`, a learned promptable segmentation
(nnInteractive) on isotropic crops that `fuse_series.py` fuses from the axial, coronal
and sagittal series of each acquisition (see Resolution of the 2021 scan). The previous
method (`fuse_planes.py` + `extract_bone.py` + `resegment_pairs.py`: fusion, threshold,
watershed, heuristics) is deprecated for bone surfaces; its documentation is kept
below for comparison.
`pipeline.py` still provides the run folder the new method reads (`run.json`, the
converted series, the TotalSegmentator labels), and the atlas, alignment, QA, and disc
tools are shared.

Install the dependencies once (TotalSegmentator 2.18 with torch; the first
TotalSegmentator run downloads its model weights):

```sh
uv pip install --python .venv/bin/python totalsegmentator SimpleITK scipy nibabel pydicom scikit-image pynrrd trimesh rtree
```

- `run_ct.py`: the single entry point: TotalSegmentator guides, bones, muscles, organs, Blender import, one structure at a time with one progress line each (see Muscles, organs and the single entry point).
- `segment_nni.py`: current bone and muscle segmentation: nnInteractive refinement of each structure on a fused isotropic crop (`--planes fused`, default) or on its ORIGINAL axial series (`--planes axial`), joint resolution of each acquisition group (muscles: one merged volume over all acquisitions, bones carved out), meshes, interface report, Blender import.
- `fuse_series.py`: Fourier-weighted fusion of the axial, coronal and sagittal series of one acquisition into an isotropic crop; every direction is taken from the series that resolve it. `fuse_crop_multi` blends several acquisitions across their overlap, weighted by slice thickness (bones) or by measured noise (soft tissue). Used per structure by `segment_nni.py`; stand-alone for one crop with `--check-alignment`.
- `reference_check.py`: scores one or more mesh folders against an already-fitted atlas reference without refitting, so two methods are measured against one fixed yardstick.
- `pipeline.py`: end-to-end run: DICOM, series groups, TotalSegmentator guides, fusion, meshes, Blender. Its run folder is the input of `segment_nni.py`; its fusion and extraction steps are the previous method.
- `fuse_planes.py` (previous method): fusion of every reconstruction of a group onto one isotropic grid, weighted by edge direction and slice thickness.
- `extract_bone.py` (previous method): denoising, guide-label seeds (optionally the fitted atlas shapes) with a rib prior, a two-level watershed split at the joints, solid fill, sub-voxel 250 HU iso-surface, PLY + STL in Blender metres; `--target` takes several bones, cut in one shared partition (joint extraction).
- `blender_bridge.py`: BlenderMCP socket client (`new-scene`, `import`, `exec`).
- `compare_runs.py`: diffs two runs: volume, centroid shift, surface distances before and after ICP, and a coloured diff PLY.
- `export_atlas_meshes.py`: headless Blender export of the 277 Z-Anatomy skeleton bones as one PLY each.
- `register_atlas.py`: fits every atlas bone to its CT bone (global similarity, then per-bone ICP) and writes fitted meshes and label volumes, the anatomical prior for `--atlas-labels`, plus the costal cartilages carried by their fitted ribs.
- `index_bp3d.py`: indexes the BodyParts3D 4.0 skeleton as an alternative atlas (`atlas_bones.json` and one PLY per part, metres).
- `align_groups.py`: one rigid transform per secondary acquisition group onto the primary group, fitted on the vertebrae both cover.
- `check_overlaps.py`: second-pass QA of neighbouring bones: overlap, split disagreement, penetration, joint gap (the mesh pass needs more memory than the fused run's meshes allow).
- `overlap_masks.py RUN OUT.json`: bone-to-bone overlap of the model masks before the joint resolution (shared voxels, mm3, depth inside both bones) for every pair whose extents meet. On the fused run 11 of 524 pairs share voxels, all facet joints of adjacent vertebrae, at most 0.85 mm deep; the resolved meshes cannot interpenetrate beyond the 0.5 mm smoothing.
- `resegment_pairs.py` (previous method): consistency pass: re-cuts neighbouring pairs with one shared decision per voxel, so both surfaces meet on one boundary.
- `extract_discs.py`: every intervertebral disc between consecutive vertebra meshes (C7-T1 ... L5-sacrum) from the fused CT: one mesh and one parameter JSON per level, and with `--blender` a centre Empty carrying the parameters for a disc force model. It reads the previous method's masks (`basins_file`) and `fused/` volumes.

### Resolution of the 2021 scan

Only the axial series are `ORIGINAL`: 404 (chest, 0.453 x 0.453 x 1 mm, lung kernel
YB), 405 and 407 (2 mm), and 401, 501, and 601 (3 mm, kernel B). The coronal and
sagittal series are tagged `DERIVED\SECONDARY\MPR`, but that tag only says they were
computed, not from what: the scanner console built them from its thin (0.625 mm
collimation) raw reconstruction, which was never exported as axial slices. This was
checked on the data (`fuse_series.py --check-alignment`, and the scratch comparison
that resampled the 3 mm axial onto the sagittal grid): the real sagittal and coronal
series carry sub-millimetre cortical detail along the body axis that the 3 mm axial
does not have (high-frequency content along S-I 0.26 vs 0.14-0.19, cortical edge
gradient 32 vs 19 HU per 0.45 mm pixel), and the three series are aligned to better
than 0.1 mm. An earlier version of this section claimed that the reformats added
nothing; that was wrong.

So each series is sharp in two directions and blurred only across its own slice
thickness: axial (sharp L-R, A-P), coronal (sharp L-R, S-I), sagittal (sharp A-P,
S-I). Every direction is resolved by two of the three. `fuse_series.py` combines them
in the Fourier domain on a 0.6 mm isotropic grid: for each spatial frequency the series
are averaged with weights exp(-2 (f_n t)^2), where f_n is the frequency component along
that series' slice normal and t its thickness, so low frequencies are a plain average
(less noise) and high frequencies along any axis come from the series that measured
them. Nothing is sharpened beyond what some series recorded. Result for the abdomen and
pelvis: about 0.6-0.7 mm in all three directions instead of 0.68 x 0.68 x 3 mm; for the
chest the 1 mm axial already dominates and the 2 mm coronal adds little. The 2021
export contains the Delayed reformats twice (502/503 are byte-identical to 602/603);
`fuse_series.group_series` keeps one series per slice normal.

For a future scan, still ask the radiology department for a 0.6-1 mm bone-kernel
axial reconstruction of the whole body; it removes the need for fusion.

### Bone surfaces: nnInteractive (`segment_nni.py`)

`segment_nni.py` segments each bone with nnInteractive v1.0 (DKFZ / Helmholtz
Imaging; Isensee, Rokuss, Krämer et al. 2025, arXiv 2503.08373), a learned promptable
3D segmentation model. Install it into the repository `.venv` (done: nninteractive
2.6.0 with torch 2.13 on Metal/MPS) and download the weights once (Hugging Face repo
`nnInteractive/nnInteractive`, folder `nnInteractive_v1.0`, 411 MB) into
`~/.cache/nnInteractive/nnInteractive_v1.0`, the default `--model-dir`:

```sh
uv pip install --python .venv/bin/python nninteractive
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('nnInteractive/nnInteractive', allow_patterns=['nnInteractive_v1.0/*'], local_dir='$HOME/.cache/nnInteractive')"
```

Licence: the nnInteractive weights are **CC BY-NC-SA 4.0 (non-commercial)**; the
Python backend is Apache-2.0. Check the non-commercial terms before distributing
meshes or scenes made with them. The tool sets `PYTORCH_ENABLE_MPS_FALLBACK=1`,
because `adaptive_avg_pool3d` is missing on MPS.

Canonical commands (`<spine>` is the Google Drive `spine` folder and `<segmentation>`
is `<spine>/segmentation`):

```sh
RUN="<segmentation>/runs/ct2021"
# full skeleton, replacing the bones of the same name in the open Blender file
.venv/bin/python tools/ct/segment_nni.py --run "$RUN" --bones all \
	--atlas-labels "$RUN/atlas_bp3d_check2/labels" \
	--blender --collection bones --run-tag ct2021-nni-fused --replace
# pilot that keeps the old bones for a side-by-side comparison
.venv/bin/python tools/ct/segment_nni.py --run "$RUN" --bones T8 \
	--atlas-labels "$RUN/atlas_bp3d_check2/labels" \
	--blender --collection bones_nni --suffix _nni --run-tag ct2021-nni-fused
# re-mesh only (e.g. after changing the smoothing): add --skip-segment
# a bone at the edge of the finer series: add --group-override rib_right_11=Lung
```

QA, after segmentation has finished:

```sh
.venv/bin/python tools/ct/register_atlas.py "$RUN" --meshes-dir "$RUN/nni_fused/meshes" \
	--atlas-json "<spine>/atlas_bp3d/atlas_bones.json" --atlas-meshes "<spine>/atlas_bp3d/meshes" \
	--atlas-mapping "<spine>/atlas_bp3d/ts_to_bp3d.json" --out "$RUN/nni_fused/atlas_check" --no-cartilage
.venv/bin/python tools/ct/check_overlaps.py "$RUN/nni_fused" --exact --samples 3000 --out "$RUN/nni_fused/qa"
# axial-only run vs fused run against the same fixed atlas yardstick (zsh: "${BONES[@]}", not $BONES)
.venv/bin/python tools/ct/reference_check.py --atlas-meshes "$RUN/atlas_bp3d_check2/meshes" \
	--meshes "$RUN/nni/meshes" "$RUN/nni_fused/meshes" --labels nni fused --bones T8 sacrum hip_left --out "$RUN/nni_fused/qa"
```

Per bone, the tool:

1. picks the acquisition group whose ORIGINAL axial series contains the bone's guide
   label (with `--group-margin-mm` 5);
2. crops around the guide label with a `--margin-mm` (25 mm) margin. With `--planes
   fused` (default) the crop is fused by `fuse_series.py` from the group's axial,
   coronal and sagittal series onto a `--fused-mm` (0.6 mm) isotropic grid, about one
   second per bone; `--planes axial` uses the native axial voxels alone. The guide is
   the TotalSegmentator label, else the fitted-atlas label from `--atlas-labels`;
3. gives nnInteractive the coarse label as the initial segmentation, then
   `--pos-points` (2) positive point prompts deep inside the bone and `--neg-points`
   (1) negative point prompt inside each neighbouring bone's label, near the bone
   (prompts at least `--point-sep-mm` 20 apart);
4. keeps the largest connected component and fills holes, and writes the crop mask
   and a JSON to `nni_fused/masks/` (`nni/masks/` with `--planes axial`).

All bones of a group are then resolved jointly on one label grid: a voxel claimed by
two bones goes to the bone whose uncontested core is nearer. Neighbours that were
segmented on different series (for example L2 on the 1 mm chest series and L3 on the
3 mm abdomen series) are resolved across groups afterwards: where the finer series'
bone exists, the coarser series' neighbour gives way. Each bone is meshed as
the 0.5 iso-surface of its Gaussian-smoothed mask (`--smooth-mm` 0.5; `--smooth-min-vox`,
default 0, was 0.8 voxels per axis for the axial-only run so 3 mm slices did not
stair-step, which the fused grid no longer needs), followed by
`--smooth-iters` 10 Taubin iterations, and written as PLY/STL in Blender metres (LPS
mm x 0.001, scanner origin kept) plus a JSON per bone in `nni_fused/meshes/`. The masks
decide the shape; the atlas never does.

Interfaces: for every touching pair, `nni_fused/qa/contacts.json` records the interface
size and its HU. An interface whose median HU is above `--measured-max-hu` (450) has
no visible joint space in the data, so that boundary is INFERRED by the model's
anatomical prior, not measured; interfaces below it are measured. Each bone's JSON
lists its contacts, and `nni_fused/results.json` lists every meshed bone. For the
axial-only run the resolved label grids (one per acquisition group, after the
cross-series carve) were saved as `nni/labels/<group>.nii.gz` with `<group>.json`
mapping label ids to bone names; they are the voxel-level truth behind those meshes.
Mesh-level checks against them (2026-11-01, 55 bones): no vertex sits deeper than one
voxel inside a neighbour's label (1 mm on the chest series, 3 mm on the abdomen
series), i.e. contacts are slice quantisation at touching facets, hips and
sacroiliac joints, not interpenetration. The fused run has no saved label grids yet
(re-run with `--skip-segment` and save them when the disc tool needs them).

`--blender` imports each mesh by the bone's object name plus `--suffix`, replacing an
existing object of that name, into `--collection` (default `bones`), with the custom
properties `ct_run` (`--run-tag`, default `ct2021-nni-fused`), `ct_method`, `ct_group`,
`ct_planes`, `ct_series`, `ct_voxel_mm`, and `ct_inferred_boundaries` (the neighbours whose
boundary was inferred). `--replace` first removes objects of other runs from the
collection.

Other options: `--bones` (names or `all`), `--out` (default `RUN/nni_fused`, or
`RUN/nni` with `--planes axial`), `--device`
(`mps`, `cpu`, or `cuda`), `--force` (re-segment bones that already have a cached
mask), `--skip-segment` (only resolve and mesh existing masks), `--resolve-with`
(extra existing masks of the same group to include in the joint resolution), and
`--group-override BONE=GROUP` (use that group regardless of label coverage; on the
2021 scan `rib_right_11=Lung` keeps the rib that just touches the edge of the 1 mm
series on it). `--group-margin-mm 0` also lets an edge bone use the finer series.

`reference_check.py` compares one or more mesh folders against an ALREADY-FITTED atlas
reference (`register_atlas.py` output meshes `<bone>_atlas.ply`) without refitting it,
so two methods are scored against one fixed yardstick. Per bone and folder it reports
`mean_mm` and `p95_mm` (CT surface to its own reference), `far_frac` (CT surface more
than `--far-mm` 4 from its own reference), `misassigned_frac` (far from its own
reference and within `--near-mm` 1.5 of a neighbour's reference), `missing_frac`
(reference surface more than `--far-mm` from the CT mesh: bone the segmentation did
not capture), and `volume_cm3`; `--out DIR` writes `reference_check.json`. The
reference never shapes bones; it is a yardstick only.

Pilot (T8, 1 mm chest data, same fixed reference), previous method -> nnInteractive:

| Measure | Previous | nnInteractive |
| --- | --- | --- |
| Volume | 18.0 cm3 (hollow body) | 26.9 cm3 |
| Missing reference surface | 5.6% | 1.5% |
| Mean residual | 1.79 mm | 1.75 mm |
| Mask voxels below 0 HU | 6.7% | 4.4% |
| Dense bone (> 300 HU) within 2 mm outside the mask | 24.9k voxels | 16.5k voxels |

The new mask fits the data better on both sides while being solid. Contour overlays on
the raw slices showed the new contour following the cortex and stopping at the disc
and at the rib head.

Current state: 55 bones in `spine/ct_bones.blend`, collection `bones`, run tag
`ct2021-nni-fused`, all segmented on fused 0.6 mm crops (chest: 404 axial + 406
coronal; abdomen and pelvis: 601 axial + 602 coronal + 603 sagittal). `rib_right_11`
is forced to the chest group; `rib_right_12`, L3-L5, the sacrum, the hips, and the
femurs come from the abdomen group. Compared with the axial-only run (`RUN/nni`, tag
`ct2021-nni`, kept on disk): the sacral foramina and canal are open and the 3 mm
terracing on sacrum, hips and lumbar bodies is gone; dense bone left within 1.2 mm
outside the mask fell from 1324 to 397 mm3 on the sacrum and 455 to 253 mm3 on L3;
volumes moved by less than 5% except the first ribs (-15%, an extra soft-tissue blob
at the rib head dropped) and the right 12th rib (-12%); the 58 contact pairs are 23
measured / 35 inferred (25 / 33 before). On the fixed atlas yardstick
(`nni_fused/qa/reference_check.json`, 23 bones) the two runs are indistinguishable:
mean residual within 0.1 mm for 20 of 23 bones, left first rib 2.94 -> 1.51 mm (the
dropped blob was not bone), sacrum 4.10 -> 4.54 mm because the open foramina add
surface that the solid atlas sacrum lacks while its missing fraction falls 25.9 ->
22.3%. The atlas fit itself is 1.1-4.5 mm off, so it cannot resolve sub-mm gains; the
CT-based checks and the slice overlays are the evidence that counts. Known blemish: a faint seam line runs down
the left iliac wing; all three series cover the hip and none has a slice
discontinuity there, so it is most likely a patch-tiling seam of nnInteractive on the
large hip crop (454 x 317 x 301 voxels), not a fusion artefact.

Timing and memory: on an Apple M5 (32 GB) a bone takes about 60-130 s (26 s for the
initial refinement plus about 6 s per point prompt); 55 bones take about 80 minutes.
The full-grid boolean labels are about 230 MB each on the chest grid, so the tool
caches at most 8. Do not run `segment_nni.py` at the same time as other heavy jobs (an
exact `check_overlaps.py` run was killed for memory while segmentation ran); run the
QA after segmentation finishes.

### Muscles, organs and the single entry point (`run_ct.py`)

`segment_nni.py --structures muscles` applies the same model and the same three-plane fusion
to every muscle TotalSegmentator can seed: the gluteals and iliopsoas (task `total`), the
trunk muscles of task `abdominal_muscles` (pectoralis major, serratus anterior, latissimus
dorsi, trapezius, rectus abdominis, external and internal oblique, erector spinae,
transversospinalis, quadratus lumborum; psoas major is skipped because iliopsoas contains
it), and the lower-neck muscles of task `headneck_muscles` that reach into the field
(sternocleidomastoid, levator scapulae, the scalenes, prevertebral muscles, platysma). The
licensed task `thigh_shoulder_muscles` (quadriceps, hamstrings, adductors, deltoid, rotator
cuff, teres major, triceps, pectoralis minor) joins the same queue once a free non-commercial
licence number is entered into TotalSegmentator.

What differs from the bones:

- One merged volume. Long muscles (erector spinae, latissimus) run through both acquisitions,
  so muscles are segmented on ONE 0.8 mm isotropic grid over the union of the chest and the
  abdomen scan plus the soft-kernel scan (`load_groups(merged=True, extra_series=...)`, group
  `Body`, eight series). The acquisitions line up to 0.01 mm (`fuse_series.estimate_shift` on
  their overlap). Each crop is fused per acquisition (`fuse_series.fuse_crop_multi`) and the
  fusions are blended spatially across their overlap, with zero weight within 3 mm of a coverage edge and a 20 mm ramp, so neither
  acquisition's slice blur enters the other's sharp region and the seam leaves no step.
- Highest quality source. Where acquisitions overlap the weights follow the structure's
  contrast: for bones the finer slices win (`quality="resolution"`, 1 / thickness); for soft
  tissue the lower noise wins (`quality="noise"`, 1 / noise variance measured in the crop's
  soft tissue). Measured in muscle: 35-50 HU of noise on the 1 mm chest series (lung kernel)
  against 8-10 HU on the 3 mm soft-kernel series, with muscle contrast of 100-150 HU. For
  that reason the muscles also fuse in the 3 mm soft-kernel chest+abdomen acquisition
  (401 axial + 402 coronal + 403 sagittal, `--soft-tissue-series`), which is excluded for
  bones: it brings pectoralis noise from 35 to 9 HU in the merged volume and dominates the
  soft tissue everywhere, while the 1 mm chest data still fills in where it is the only source.
- Components and prompts. A bone is one piece, so the bones keep the largest connected
  component of the model output and get two positive prompts at the deepest points of the guide.
  Soft tissue keeps every component that overlaps the guide label (a sheet muscle's output
  breaks at its aponeurosis, the rectus at each tendinous intersection: with the bone rule the
  external oblique kept 20-26% of its guide volume and the rectus 66%) and gets six positive
  prompts spread along the guide's longest extent (`--pos-points`, `--point-sep-mm`).
- Crop growth. The crop is the guide's box plus `--margin-mm` (25). Where a soft-tissue guide
  stops short (TotalSegmentator's rectus abdominis ends about 10 cm above the pubis) the model
  fills the crop to its face and the mesh ends in a flat cut, so while the mask reaches a crop
  face that is not the scan's edge, the crop grows `--grow-mm` (40) there and the model refines
  again from the mask, up to `--max-grow` times (3 for muscles / organs, 0 for bones). A grow
  step whose mask exceeds `--grow-max-ratio` (1.8) times the guide volume is a leak along a
  neighbour (the right external oblique climbed the chest wall to the armpit) and is rejected:
  the previous mask is kept. The mask JSON records `grow_steps`, `grow_rejected` and
  `touches_crop_face`.
- Meshes keep every piece of real size (area at least 1% of the largest piece): the Gaussian
  smoothing pinches a label apart where it is thin, and the earlier largest-piece rule dropped the
  upper two bellies of a rectus whose voxels were all there (mesh JSON `components`).
- Neighbours. Every other muscle whose guide reaches into the crop and every saved bone mask
  that reaches into it are candidates; the largest `--max-neighbours` (12) get the negative
  prompts. In the joint resolution a contested voxel goes first to the one structure whose guide
  label covers it (the guides of one TotalSegmentator task are mutually exclusive) and only then
  to the nearer uncontested core, so an over-grown neighbour cannot take a muscle's own territory.
  Afterwards the bone masks are carved out of the muscle labels (bones always win), and `--measured-max-hu` defaults to 30 for soft tissue: an interface whose
  median HU is below 30 has a visible fat plane (measured boundary; a 1-2 mm plane blurs to
  +20-40 HU at 0.8 mm), above it the two muscles touch without one (inferred).
- Outputs in `RUN/nni_fused_muscles/{masks,meshes,qa}`, Blender collection `muscles`, run tag
  `ct2021-nni-fused-muscles`, custom property `ct_structure = muscle`. The 2026-11-01 run
  segmented all 42 in 5 h 44 min (6557 cm3); the three fragments at the top edge of the field
  (platysma both sides, posterior_scalene_left, 0.0-0.1 cm3) live in
  `RUN/nni_fused_muscles/excluded/` and are not imported, so the collection holds 39 muscles.
  The 2026-11-01 contact report still used `--measured-max-hu 0` and called every
  muscle-muscle contact inferred; the default is now 30 HU and the next resolve recomputes it.

`segment_nni.py --structures organs` runs the organs and vessels of TotalSegmentator's free
task `total` through the same merged-volume path (`ORGAN_COLLECTIONS` in `segment_nni.py`):
the five lung lobes, trachea, oesophagus, thyroid, heart (one body) and left atrial appendage
into collection `thorax`; liver, spleen, kidneys, adrenals, gallbladder, pancreas, stomach,
duodenum, small bowel, colon, bladder and prostate into `abdomen`; aorta, pulmonary vein,
brachiocephalic trunk and veins, subclavian and carotid arteries, both venae cavae, portal and
splenic vein, iliac arteries and veins into `vessels`; the spinal cord into `nervous`. The
neighbours and the carve use the saved bone AND muscle masks (`--muscle-masks`), so no organ
overlaps an earlier structure. Outputs in `RUN/nni_fused_organs`, run tag
`ct2021-nni-fused-organs`, property `ct_structure = organ`. Kidney cysts are left out; the
heart chambers, coronary arteries and aortic sinuses need TotalSegmentator's licensed tasks
(`heartchambers_highres`, `coronary_arteries`, `aortic_sinuses`; a free non-commercial licence
is requested at backend.totalsegmentator.com/license-academic and entered with
`totalseg_set_license`), and this scan is not ECG-gated, so the heart border carries a few
millimetres of motion blur whatever the model.

`run_ct.py` chains everything with one progress line per structure:

```sh
.venv/bin/python tools/ct/run_ct.py --run "$RUN" --blender            # guides -> bones -> muscles -> organs -> Blender
.venv/bin/python tools/ct/run_ct.py --run "$RUN" --structures muscles --blender
.venv/bin/python tools/ct/run_ct.py --run "$RUN" --structures organs --blender
.venv/bin/python tools/ct/run_ct.py --run "$RUN" --structures bones --bones T8 sacrum
```

Stage 1 runs TotalSegmentator (full resolution; `--fast-guides` is opt-in) for the bone classes,
the three muscle tasks and the organ classes on each acquisition's finest axial series, skipping
cached labels. Stages 2 to 4 call `segment_nni.py --quiet`, which prints `[k/N] name: seconds,
cm3, median HU, neighbours` per structure and sends the model's own output to
`OUT/qa/segment.log`. Every mask is cached, so a rerun only resolves, meshes and imports.
Structures always run one at a time; the three planes of a crop are resampled in parallel inside
the fusion. A structure whose crop box clipped the result (mask touching a crop face, as the
left gluteus maximus did at 25 mm) is re-run alone with `segment_nni.py --bones NAME --force
--margin-mm 40`, then `run_ct.py --skip-segment --blender` re-resolves and re-imports the set.

### Previous method: fusion, threshold, watershed (deprecated)

Kept for comparison. It needed about 15 hand-tuned rules that broke each other, left
vertebral bodies hollow where the marrow was below the fill threshold, left rib heads
on the vertebrae, and extracted each bone independently, which then needed a
reconciliation pass (`resegment_pairs.py`). Its spatial-domain fusion (`fuse_planes.py`)
is superseded by the Fourier-weighted `fuse_series.py` (see Resolution of the 2021 scan).

A full run, with the atlas prior on the second extraction:

```sh
RUN="<segmentation>/runs/ct2021"
# once per atlas: the Z-Anatomy bones as PLY (headless Blender)
blender -b --python tools/ct/export_atlas_meshes.py -- \
	--fbx "<zanatomy>/SkeletalSystem100.fbx" --out "<segmentation>/atlas/meshes"
# 1. first run: split on the coarse TotalSegmentator labels
.venv/bin/python tools/ct/pipeline.py --dicom "/path/CT 2021" --match Lung --match Delayed \
	--bones all --out "$RUN" --jobs 4
# 2. fit the atlas to the first run's bones (writes $RUN/atlas/, costal cartilages included)
.venv/bin/python tools/ct/register_atlas.py "$RUN" --meshes-dir "$RUN/meshes" --out "$RUN/atlas"
# 3. re-extract with the atlas prior (series/, totalseg/, fused/ are reused)
.venv/bin/python tools/ct/pipeline.py --dicom "/path/CT 2021" --match Lung --match Delayed \
	--bones all --out "$RUN" --jobs 4 --atlas-labels "$RUN/atlas/labels" \
	--blender append --collection bones
# 4. chest group into the pelvis frame, then QA
.venv/bin/python tools/ct/align_groups.py "$RUN" --primary Delayed \
	--atlas-labels "$RUN/atlas/labels" --blender
.venv/bin/python tools/ct/check_overlaps.py "$RUN"
# 5. consistency pass over the flagged pairs, then QA again
.venv/bin/python tools/ct/resegment_pairs.py "$RUN" --overlaps "$RUN/qa/overlaps.json" \
	--atlas-labels "$RUN/atlas/labels" --blender
.venv/bin/python tools/ct/check_overlaps.py "$RUN"
.venv/bin/python tools/ct/compare_runs.py <runA> <runB> --bones sacrum
```

`--device mps` (the default) runs TotalSegmentator on Apple Silicon and falls back to
`cpu`. `--jobs N` processes N bones in parallel; each job needs about 2-4 GB of RAM
(4 jobs on a 34 GB Mac). Do not edit `tools/ct/*.py` while a `--jobs N` run is going:
the worker processes re-import the modules and crash on mismatched arguments; after
killing the main process, kill its idle multiprocessing workers too. `pipeline.py`
forwards `--finest-only` to `fuse_planes.py`, and `--atlas-labels DIR` and
`--extract-args ...` to `extract_bone.py`; everything after `--extract-args` goes to
`extract_bone.py`, so it comes last (e.g. `--extract-args --split-level 600`).
`--smooth-iters` (Taubin smoothing of the mesh) defaults to 5. Each script prints its
full usage with `--help`.

Fusion weights each input at every voxel by how well it resolves the local edge
direction and by its slice thickness (`--thickness-power`, default 1: a 1 mm series
counts three times a 3 mm one). By default every input of a group is fused, which
gives smoother surfaces. `fuse_planes.py --finest-only` (also on `pipeline.py`) keeps
only the finest series per plane and drops series thicker than
`--max-thickness-ratio` (2.5) times the finest: edges are sharper (+27% gradient on
T4), but surfaces are noisier and rougher. Every fused grid snaps to a global lattice
(integer multiples of the spacing from the scanner origin), so neighbouring bones'
volumes share voxels exactly.

Bone surfaces are the 250 HU iso-surface of the fused 0.45 mm volume. Before any
thresholding, a 0.6 mm Gaussian on the fused volume removes the speckle that the sharp
lung kernel leaves (little spikes on the chest bones) without moving a symmetric edge.
`pipeline.py --denoise-mm auto` (the default) applies it only to groups whose finest
slices are at most 1.5 mm (the sharp lung kernel) and skips the thick-slice,
smooth-kernel pelvis group; a number applies that sigma to every group.
TotalSegmentator labels (1.5 mm) only seed the split, and only where they are
confident: a label is trusted within `--label-reach-mm` (4 mm) of itself, bone
follows the nearest label outside the band within `--band-mm` (4 mm) of label
boundaries, and a watershed on the
distance transform (`--split distance`, the default) splits that band, i.e. at the
scan's joint gaps. The cortex above `--split-level` (500 HU) is split first, so thin
joints bridged by partial volume fall apart; the softer 250-500 HU shell is then
grown onto the cores by a watershed on inverted intensity, so the fronts meet on the
darkest path, the joint (joining the nearest core instead cut a thin seam through
vertebral bodies whose lateral wall fell below 500 HU). Where a non-rib label is
nearest and a rib label is the runner-up, the band widens one-sidedly to
`--rib-band-mm` (20 mm), because TotalSegmentator puts rib heads, necks, and
tubercles on the vertebra (its rib label stops about 26 mm short of the head). A rib
prior bridges that gap: from the medial end of each rib label, a cylinder
(`--rib-radius-mm` 6) is extended `--rib-extend-mm` (30 mm) along the rib's local
direction towards the spine, and dense cortex (above `--split-level`) inside it that
connects to the rib label and is not deeper than `--rib-core-erode-mm` (6 mm) inside
a vertebra label is marked as that rib. Alternatives are `--split hu` (watershed on
inverted intensity) and `--split cascade` (brightness levels `--levels` from 900 down
to 250 HU, merged pieces cut at their thinnest neck); the cascade tested worse on T4
(ragged edges, grabbed neighbours). The target label is closed and hole-filled
(`--guide-close-mm` 5) before it defines the region of interest (`--guide-margin-mm`
8), because TotalSegmentator left the S1 body out of the sacrum label. The interior
is filled solid from marrow above `--core-level` (100 HU) inside an 8 mm silhouette
(`--envelope-mm`) plus spongy bone above `--fill-level` (150 HU) within twice
`--solid-close-mm` (2 mm) of the cortex, then closed and hole-filled. Large
low-density pockets the closing bridged (below `--carve-level` 60 HU, over
`--carve-min-mm3` 50; spinal canal, foramen contents) are carved back out. The volume
is padded with air, so every surface is closed. `extract_bone.py --target` accepts
several bones, which are then cut in one shared partition (a joint extraction); with
one target the behaviour is unchanged.

### Shared tools and conventions

Coordinates: Blender metres = scanner LPS millimetres x 0.001, with the scanner origin
kept (Blender +X = patient left, +Y = posterior, +Z = superior). Every bone of one
acquisition group shares this transform, so the joint gaps are the scan's. The target
`.blend` uses metric units with millimetre display; `scene.unit_settings.scale_length`,
which the add-on reads, stays 1.0. Both methods use it; no bone is ever moved.

`--match` is repeatable; each one selects one acquisition group: every series whose
description contains one of its `|`-separated substrings, in all planes and with
several reconstructions per plane (on the 2021 scan, `--match Delayed` gives 6 inputs
and `--match Lung` gives 4; the 3 mm CHEST+ABD series are left out). The series of a
group must be co-registered, i.e. come from one acquisition. List the groups highest
resolution first (`--match Lung --match Delayed`): each bone comes from the first group
whose axial volume fully contains its guide label. Bones from different acquisitions
disagree by a few millimetres (breathing, table move) until `align_groups.py` brings
them into one frame (see below). `segment_nni.py` reads these groups from `run.json`
but uses only each group's finest ORIGINAL axial series, never the reformats.

Atlas prior: the coarse TotalSegmentator labels put rib heads on the vertebra and the
coccyx on the sacrum, and do not follow the joint surfaces. `export_atlas_meshes.py`
exports the 277 Z-Anatomy bones as PLY into `<segmentation>/atlas/meshes/` (the
default `--atlas-meshes` of `register_atlas.py`). `register_atlas.py RUN` fits every
atlas bone to its CT bone: a global similarity fit from matched centroids, then a
per-bone two-way similarity ICP with guards (scale range, maximum residual; a rejected
fit is retried rigid at the global scale, then falls back to the global transform;
bones the scan cut off are flagged `partial`). It writes
`RUN/atlas/meshes/<bone>_atlas.ply`, `RUN/atlas/labels/<bone>.nii.gz` (0.9 mm label
volumes), and `atlas_fit.json`. On the 2021 scan it fitted 55 bones in 126 s with
mean residuals of 1.3-2.3 mm on the vertebrae, 1-3 mm on the ribs, and 3-4 mm on the
hips and sacrum. In the previous method, `extract_bone.py --atlas-labels DIR` (forwarded by
`pipeline.py --atlas-labels DIR`) then uses the fitted shapes to decide bone
ownership: a bone owns its atlas shape plus whatever its coarse label covers that no
other bone's atlas claims, so the coccyx stays with the sacrum while the rib head
leaves the vertebra. `--atlas-skip` (default: femur_left, femur_right, humerus_left,
humerus_right, vertebrae_C6, the partially scanned bones) keeps the coarse label.
Surfaces still come only from the CT iso-surface: the atlas never shapes or moves a
bone, and because the fit is per bone it follows rotated or asymmetric vertebrae.
With the prior, rib bone left on T4 went from 0.81 to 0.09 cm3, and the L5-sacrum
overlap from 2.4 cm3 with 4.9 mm penetration to 1 mm3 with none.
`register_atlas.py --blender` imports the globally fitted atlas into a hidden `atlas`
collection for naming and QA. With `segment_nni.py` the fitted atlas is used only for
naming, ownership (a guide label where TotalSegmentator has none), and QA (a fixed
`reference_check.py` yardstick); it never shapes or moves a bone.

Costal cartilages: calcified costal cartilage is above 250 HU, and TotalSegmentator
labels it as rib (its own `costal_cartilages` class was 0.5 cm3 on this scan), so the
first ribs ran into the sternum. `register_atlas.py --cartilage` (the default;
`--no-cartilage` turns it off) carries each fitted rib's transform onto the atlas
costal cartilage of that rib (Z-Anatomy ribs 1-10, BodyParts3D ribs 1-7) and writes
`labels/costal_cartilage_{side}_{n}.nii.gz`; `--reuse-fit` adds them to an existing
fit without refitting the bones. When fitted-atlas labels exist, `pipeline.py` treats
`costal_cartilage_{side}_{n}` as extra classes (neighbours: its rib, the sternum, and
the adjacent cartilages; rib n and the sternum list it as a neighbour), extracts each
one as its own object, and imports them into a Blender collection `cartilage`. They
are stiff links, not bone.

BodyParts3D as atlas: `index_bp3d.py ATLAS_DIR` indexes the BodyParts3D 4.0 skeleton
(IS-A tree, 99 % polygon reduction; text tables in `ATLAS_DIR/download/`, unzipped
OBJ files in `ATLAS_DIR/obj/`) into `atlas_bones.json` and one PLY per part in metres
(`meshes/` + `index.json`), in the Z-Anatomy atlas schema. Part names are the exact
BodyParts3D English names; `sternum` and `skull` are synthesised from their elements,
and BodyParts3D has no coccyx. In `spine/atlas_bp3d/` the TotalSegmentator ->
BodyParts3D map sits beside them as `ts_to_bp3d.json`. Fit it with
`register_atlas.py RUN --atlas-json <bp3d>/atlas_bones.json --atlas-meshes
<bp3d>/meshes --atlas-mapping <bp3d>/ts_to_bp3d.json --out <dir>` (`--blender` still
imports the Z-Anatomy FBX). On the 2021 scan the per-bone residuals tie with
Z-Anatomy (2.42 vs 2.32 mm mean). BodyParts3D 4.0 is CC BY 4.0 per its README (the
OBJ headers still say CC BY-SA 2.1 JP); keep `ATTRIBUTION.txt` with the data and
credit "BodyParts3D, © The Database Center for Life Science licensed under CC
Attribution 4.0 International".

Group alignment: `align_groups.py RUN --primary Delayed --blender` extracts the
vertebrae both acquisitions cover from both groups, fits one rigid transform (no
scale) from the secondary group's copies onto the primary group's, and applies it to
every bone of the secondary group. The pelvis acquisition (Delayed) is the reference
frame; no bone is ever moved individually. The original meshes are kept in
`meshes/unaligned/`, the transform is recorded in each moved bone's JSON
(`aligned_transform`) and in `align/alignment.json` (bones used, residuals before and
after), and moved Blender objects get `ct_aligned_to` and `ct_align_rms_mm`. Without
`--primary`, the run's first group is the reference. On the 2021 scan the two
acquisitions were already co-registered to 0.16 mm (rigid fit 0.01 deg / 0.04 mm on
T11-L2): nothing had to move.

Outputs land in the run directory (`--out`). `series/`, `totalseg/`, and `fused/` are
cached, so delete a sub-folder to redo it; meshes are re-extracted on every run:

- `series/`: the chosen reconstructions converted to NRRD and NIfTI.
- `totalseg/<group>/`: TotalSegmentator guide labels per acquisition group.
- `nni/`: `segment_nni.py` output: `masks/` (`<bone>.nii.gz` crop mask and
  `<bone>.json` with prompts, timing, and HU statistics; cached, `--force` redoes
  them), `meshes/` (`<bone>.ply`, `.stl`, `.json` with group, series, voxel size,
  volume, and contacts), `qa/contacts.json`, `results.json`, and the QA folders
  (`atlas_check/`, `qa/overlaps.json`, `qa/reference_check.json`).
- `fused/` (previous method): `<bone>_fused.nrrd`, the fused volume around each bone.
- `meshes/` (previous method): `<bone>.ply`, `.stl`, `.json` (volume, size, face count, parameters,
  `basins_file`, `basin_ids`, `target_id`, and `resegmented_with` after the
  consistency pass), and `<bone>_basins.nrrd` (the split labels);
  `meshes/unaligned/` holds the secondary-group originals after `align_groups.py`,
  `meshes/before_resegment/` the originals of bones `resegment_pairs.py` changed.
- `atlas/`: `register_atlas.py` output: `atlas_fit.json` (global and per-bone
  atlas-to-CT matrices, scales, residuals, flags, size table),
  `meshes/<bone>_atlas.ply`, and `labels/<bone>.nii.gz` (including
  `costal_cartilage_{side}_{n}`).
- `align/`: `align_groups.py` output: `<group>/` (the shared vertebrae extracted from
  the other group) and `alignment.json`.
- `qa/`: `check_overlaps.py` output, `overlaps.json` and `overlap_<A>__<B>.ply` point
  clouds, plus `resegment.json` from `resegment_pairs.py`.
- `run.json`: manifest with the chosen series, parameters, and outputs.

`pipeline.py --blender append|new` and `segment_nni.py --blender` need Blender
running with the BlenderMCP add-on listening on `localhost:9876` (see Blender MCP);
otherwise the meshes stay on disk under `<run>/meshes/` (`<run>/nni/meshes/`). In the
previous method each Blender object carries the custom properties `ct_group`,
`ct_level_hu`, and `ct_run`. `--run-tag` names the run (default `<out name>@<time>`);
at the end of a run, objects in the collection from other runs are removed unless
`--keep-old` is given.

`check_overlaps.py` checks each pair of neighbouring bones, which are split twice
(once from each side): overlapping voxels, voxels where the two independent runs
disagree on the split, mesh penetration depth, and the minimum joint gap. Pairs are
flagged OK, CHECK, or BAD; a BAD pair goes through the consistency pass below. The
default inside test is fast (KD-tree, sign of the nearest face normal) and can report
a false "penetration" on interlocking surfaces whose voxel overlap is zero: trust the
voxel columns first. `--exact` uses ray tests instead (needs `rtree`; minutes per pair
on these meshes). Without `rtree` installed, `compare_runs.py` and
`check_overlaps.py` use a KD-tree fallback for surface distances and inside tests.
For the new method, run it on `RUN/nni_fused` (`--exact --samples 3000 --out RUN/nni_fused/qa`)
after segmentation has finished, never alongside it.

### Previous method: consistency pass (deprecated)

Consistency pass: the two independent extractions of a pair disagree at the shared
joint (two runs, two decisions). On the 2021 run the overlap report flagged 33 pairs,
almost all rib heads and facet joints, some with 2-8 mm of mesh interpenetration.
`resegment_pairs.py RUN --pairs A:B ...` (or `--overlaps RUN/qa/overlaps.json`, which
takes the pairs flagged `--flags` BAD with at least `--min-penetration-mm` 1 mm) cuts
each pair with ONE shared decision per voxel. A voxel inside both final masks goes to
the bone whose fitted-atlas label (`--atlas-labels DIR`) claims it, else to the
nearer bone body. Bone that one run handed to the partner's basin but the partner's
run did not keep (an "orphan", typically a rib head or a facet process) joins the
partner when it is connected to it, and the overlaps are arbitrated again after that
step. Both bones are then re-meshed from their own fused volumes with the partner's
adjusted mask cut out, so the surfaces meet exactly on the shared boundary; nothing
is moved. The originals go to `meshes/before_resegment/`, and each adjusted bone's
JSON gains `resegmented_with`; `--blender` re-imports the changed bones (custom
property `ct_resegmented` names the partners), and `qa/resegment.json` lists the
pairs and the changed bones. On the 2021 spine every lumbar pair and the lower
thoracic pairs now have zero overlap and gaps of 0.03-0.2 mm (the facets of the 3 mm
pelvis acquisition still meet with no gap, as before). `pipeline.py --resegment`
runs the whole pass at the end of a run: `check_overlaps.py`, then
`resegment_pairs.py` over every consecutive vertebra pair plus every BAD pair with
>= 1 mm penetration, then `check_overlaps.py` again.

The costovertebral split is the hardest one. A raw-slice check showed that the
costovertebral joint is a real sub-millimetre oblique gap, not a fusion, so the cut
is legitimate. The rib prior alone brought the rib bone left on the T4 vertebra from
0.8 cm3 to about 0.25 cm3; the atlas prior brings it to 0.09 cm3. The joint space
(0.5-1 mm) is at or below the voxel size, so no threshold measures it;
`segment_nni.py` records in `nni/qa/contacts.json` whether each such boundary is
measured or inferred.

In the 3 mm pelvis acquisition the lumbar facet joints (1-2 mm) are below the
resolution: with the atlas prior, adjacent vertebrae are separated along the
anatomical joint surface but meet with no gap. Adding a joint-thickness prior would be
a modelling choice, not data.

### Discs

Discs: `extract_discs.py --run RUN` (`--meshes-dir`, default `RUN/meshes`; `--out`,
default `RUN/discs`; `--levels disc_L4_L5 ...`) segments every intervertebral disc
between consecutive vertebra meshes (C7-T1 ... L5-sacrum) from the fused CT and the
final bone masks. A disc voxel is soft tissue (40-250 HU) outside both bones whose
directions to the two endplates are opposed (a distance-gradient test), inside both
vertebral-body outlines seen along the body-to-body axis (the corridor), and with
bone within 8 mm above and below it (the trough). The footprint is the intersection
of the two endplates' convex hulls on the disc midplane, which clips anterior soft
tissue, the canal, and the rib heads. Each level gets `<level>.ply/.stl` (metres,
Blender/LPS frame, like the bone meshes) and `<level>.json`: centre, normal,
in-plane axes, mean, min, max, anterior, and posterior height, footprint area and
radii, volume, wedge angle (signed sagittal, > 0 = lordotic), endplate centres and
normals, a 5-ring radial HU profile, coverage, and flags (thin, patchy,
one_endplate, steep_wedge, small_area); `discs.json` collects every level.
`nucleus_radius_ratio` 0.5 is a stated default, not a measurement: CT does not
separate the nucleus from the annulus. `--blender` (`--collection discs`) imports
each disc mesh plus an Empty `<level>_center` (ARROWS, size = equivalent radius,
local Z = disc normal, local X = major axis) that carries every parameter as a
custom property; it is meant for a force-based disc model, not a mesh element.
`--blender-only` pushes the JSONs already in `--out`. A level takes about 25 s. On
the 2021 scan the lumbar discs are 42-48 mm wide, 30-33 mm deep, and 4-6 mm high,
with lordotic wedges of 4-8 deg; L5-S1 is 3.4 mm (2.5 mm posteriorly). The thoracic
discs measure 1.5-2.5 mm, below the scan's resolution (real 4-7 mm), and are flagged
thin; T7-T8, T8-T9, T10-T11, and C7-T1 were flagged unreliable on the snapshot used.
These figures come from the previous method's bone masks. The nnInteractive vertebra
masks stop at the endplates, so the thoracic discs may now measure. `extract_discs.py`
still reads the previous method's masks (`basins_file` in `meshes/<bone>.json`) and
`fused/` volumes; the `nni_fused/meshes/` JSONs carry neither, so it needs a path that
reads `nni_fused/masks/` (0.6 mm isotropic crops, LPS origin in the NIfTI header) and
the fused crop from `fuse_series.fuse_crop` before it can be re-run on the new bones.

`.mcp.json` also defines a 3D Slicer MCP server (`uvx --python 3.13 mcp-slicer`) for
inspecting scans and segmentations. Start Slicer's **Web Server** module with the
**Slicer API** enabled (port 2016) before using it.

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