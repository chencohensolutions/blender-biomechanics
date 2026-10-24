# Biomechanics Add-on Development Plan

This document is the handoff for future coding sessions. It describes the current
behavior, every UI control, the important implementation rules, and the checks that
must continue to pass.

## Goal

Compare two collections containing identical anatomical mesh objects:

- **Original**: the unmodified source model.
- **Rigid**: the same model after its objects were independently moved or rotated.

The user selects one mesh in Original and aligns the entire Original collection so
that mesh coincides with its matching Rigid mesh. The remaining objects then show
their displacement relative to the aligned pair.

## User Workflow

1. Open **3D Viewport > Sidebar > Biomechanics**.
2. Choose the **Original** and **Rigid** collections.
3. Select one mesh object from Original, or its pair in Rigid.
4. Choose a comparison collection: **None**, **Original**, or **Rigid**.
5. Optionally use **Mesh Wireframe**, **Overlay Opacity**, and **Overlay On Top**.
6. Select **Align to Source**.
7. Inspect how nearby original and rigid meshes differ relative to the aligned pair.

## UI Controls

### Original

Collection picker stored in `Scene.biomechanics_source_collection`.

- Contains the model that will be moved by **Align to Source**.
- Defaults to the collection named `Collection` when available.
- A valid selected alignment object must be a mesh directly in this collection, or a
  Rigid mesh, which stands for its unique Original pair.
- Changing the collection refreshes and safely restores the comparison display.

### Rigid

Collection picker stored in `Scene.biomechanics_target_collection`.

- Contains the comparison model and paired mesh objects.
- Defaults to the collection named `rigid` when available.
- Changing the collection refreshes and safely restores the comparison display.

### Comparison Display: None / Original / Rigid

Expanded enum stored in `Scene.biomechanics_transparency_mode`.

- **None** restores both collections to their previous display settings.
- **Original** applies the comparison display to Original.
- **Rigid** applies the comparison display to Rigid.
- Switching directly between Original and Rigid restores the previous collection
  before changing the next collection.
- The mesh pair used for alignment is excluded from the comparison display because
  both paired meshes occupy the same location after alignment.

The property name still contains `transparency` for compatibility with the current
implementation, although the comparison can now be transparent or wireframe.

### Mesh Wireframe

Toggle stored in `Scene.biomechanics_overlay_wireframe`.

- Off: the comparison collection uses the selected opacity.
- On: comparison meshes use `display_type = "WIRE"` instead of transparency.
- The opacity slider is disabled while wireframe is active.
- The selected aligned pair remains in its original display mode.
- Switching this toggle must restore prior alpha, transparency, display type, and
  in-front state before applying the new mode.

### Overlay Opacity

Factor slider stored in `Scene.biomechanics_overlay_opacity`.

- Range: 0.0 to 1.0, displayed as 0% to 100%.
- Default: 0.25 (25% opacity).
- Updates the active transparent comparison collection immediately.
- Disabled when Comparison Display is None or Mesh Wireframe is enabled.
- Transparent mode temporarily changes solid viewports to Object Color mode because
  Blender only displays `Object.color` alpha in that mode.

### Overlay On Top

Toggle stored in `Scene.biomechanics_overlay_on_top`.

- Default: enabled.
- On: comparison meshes use `show_in_front = True`.
- Off: comparison meshes retain their original `show_in_front` values.
- Works with both transparent and wireframe comparison modes.
- Disabled in the UI while Comparison Display is None.
- The selected aligned pair is excluded.

### Align to Source

Operator: `BIOMECHANICS_OT_align_to_source`.

Blender ID: `biomechanics.align_to_source`.

Behavior:

1. Validate that Original and Rigid are selected and different.
2. Validate that the active object is a mesh in Original or Rigid. A Rigid mesh is
   replaced by its unique Original pair (same suffix-removal rule); missing or
   ambiguous Original pairs are rejected.
3. Pair it with one Rigid mesh by removing one Blender numeric suffix such as
   `.001` or `.002` and comparing the remaining names.
4. Reject missing or ambiguous pairs.
5. Validate that paired meshes have equal vertex and polygon counts and at least
   three vertices.
6. Read corresponding vertices in world space.
7. Use a Kabsch/SVD best-fit rigid transform to calculate rotation and translation.
8. Reject meshes whose fitted vertex RMS error exceeds
   `max(target_size * 1e-4, 1e-6)`.
9. Apply the transform to every mesh object in Original as one rigid group.
10. Preserve object scale and all relative spacing in Original.
11. Refresh the comparison display and exclude the newly aligned pair.
12. Report the moved object count. The operator supports Blender Undo.

### Point Correspondence

Panel: **Point Correspondence**. Module: `biomechanics_addon/correspondence.py`.

- **Hover Correspondence** (`biomechanics.toggle_correspondence_hover`) toggles a
  pass-through modal operator. Under the mouse it ray-casts the scene; for a hit on
  an Original or Rigid mesh it finds the unique pair by `base_object_name`, maps the
  hit to the pair with `poly_3d_calc` weights over the shared polygon vertex indices,
  and draws both points, a connecting line, and the world distance in millimeters.
  It changes no scene data. Topology-changing modifiers break the face-index mapping.
- **Find Max Distance Point** (`biomechanics.find_max_correspondence_distance`)
  compares every Original mesh with its Rigid pair vertex-by-vertex in world space,
  ranks pairs by their largest vertex distance, marks the overall maximum in the
  viewport, and lists the top five pairs. Results are in-memory only and cleared by
  the X button, file load, or unregister.
- One `POST_PIXEL` draw handler serves both overlays; it is added on register and
  removed on unregister.
- **Distance Heatmap** (`biomechanics.toggle_distance_heatmap`, module
  `distance_heatmap.py`) writes a temporary point-domain `FLOAT_COLOR` attribute
  `biomechanics_correspondence_distance` to both meshes of every pair (blue near ->
  red far) and switches solid viewports to Attribute color (`color_type = "VERTEX"`).
  **All Bones** scales red to the global maximum; **Per Bone** (`Scene.biomechanics_heatmap_scale`)
  scales to each pair's own maximum, never below 0.01 mm so an aligned pair's
  float noise (~1e-4 mm) stays blue. A depsgraph handler marks a mesh dirty only when
  its `matrix_world` actually differs from the cached one (bone-parented Rigid meshes
  report `is_updated_transform` on every evaluation, so the flag alone causes a
  refresh loop), and a 0.1 s timer recomputes only those pairs from cached local
  vertices (all pairs when the global maximum shifts by more than 10%). Off, unregister, and file load remove the attribute, restore each
  mesh's previous active/render color attribute from `_biomechanics_previous_*`
  mesh properties, and restore viewport color modes. Workbench honors object alpha
  only in Object color mode (verified by render), so while the heatmap is on and
  Mesh Wireframe is off, `refresh_transparency_overlay` hides comparison meshes with
  `hide_set(True)` (prior state in `_biomechanics_previous_hidden`) and leaves the
  viewport color mode to the heatmap. A `POST_VIEW` handler in `distance_heatmap.py`
  redraws those meshes with a custom headlight GPU shader in heatmap colors at
  Overlay Opacity, honoring Overlay On Top (depth test off). Hidden comparison meshes
  are not hit by Hover Correspondence ray casts. Turning the heatmap off reapplies the
  normal transparent comparison display.

### Muscle Strings

Panel: **Muscle Strings**. Modules: `biomechanics_addon/muscle_atlas.py` (bpy-free
attachment data) and `biomechanics_addon/muscle_strings.py`. Anatomical reference and
the list of muscles left out: [MUSCLE_ATTACHMENTS.md](MUSCLE_ATTACHMENTS.md).

- **Build Muscle Strings** (`biomechanics.build_muscle_strings`) detects landmark
  vertices on the Original bones in a body frame derived from the skeleton (sacrum,
  T01, both femurs), then creates one string per muscle attachment in a new
  collection `<collection> muscles` for Original and again for Rigid, reusing the
  same vertex indices on the paired Rigid meshes. Every string connects bones of one
  collection. Rebuilding replaces only collections tagged with
  `biomechanics_muscle_bones`. Supports Undo.
- Strings are 3D poly curves (0.6 mm bevel, material per muscle group) with one Hook
  modifier per control point to its bone, so they stretch when bones move. Each
  string stores muscle, side, index, and `[bone, vertex index]` attachments as JSON in
  `biomechanics_muscle_string`.
- Length is always recomputed from the attachment vertices in world space (polyline
  through via points), in millimeters through the scene unit scale, never from the
  evaluated curve.
- **Hover Muscle Lengths** (`biomechanics.toggle_muscle_hover`) is a pass-through
  modal operator. It projects every visible string to the region and picks the
  nearest within 8 px (scaled by UI scale); the `POST_PIXEL` overlay highlights it
  and shows its length and the matching Original/Rigid string's length and delta.
  Visibility is checked through object and layer-collection flags because
  `Object.visible_get()` returns False from a 3D-view context in Blender 5.x.
- **Length Change Colors** (`biomechanics.toggle_muscle_delta_colors`) colors each
  Original/Rigid twin pair by Rigid minus Original length on a diverging ramp: red
  shorter, gray unchanged, blue longer (sheets: red smaller, blue larger), with a
  21-bin set of shared
  `Biomechanics Muscle Delta NN` materials. **Percent** (strain, default) or **mm**
  is chosen by `Scene.biomechanics_muscle_delta_unit`; `Scene.biomechanics_muscle_delta_range`
  (default 10) maps to full color. It sets each string's material slot to
  `link = "OBJECT"` and `Object.color` (for Object color mode), so the curve data
  keeps its group material; off, Remove, file load, and unregister restore
  `link = "DATA"` and the group color. A 0.25 s timer recolors while active, but only
  when a signature of every Original/Rigid mesh `matrix_world` changes, and it writes
  only changed slots and material colors (any write tags a depsgraph update and
  stalls viewport navigation). Bone objects, attachment vertex coordinates, and
  parsed string/sheet JSON are cached because `bpy.data.objects.get` is slow with
  ~1,500 objects; caches clear on build, file load, and reload. A legend is drawn above the
  distance heatmap legend. Solid viewports in Attribute color mode (Distance
  Heatmap on) do not show string colors.
- **Muscle Sheets** (`biomechanics.toggle_muscle_sheets`, module `muscle_sheets.py`)
  fills triangles between consecutive strings (atlas index order) of each muscle and
  side, in both muscle collections; muscles with one string get no sheet. Vertices
  are the shared attachment vertices, hooked per bone, so sheets stretch with the
  strings. Each sheet stores its attachments and faces in `biomechanics_muscle_sheet`;
  twin sheets have identical face lists, so face i pairs with face i. On is derived
  from sheet objects existing; Build Muscle Strings rebuilds sheets if they existed.
- With Length Change Colors on, each sheet face takes a material slot by its area
  change in percent (slot 0 = group color, then the 21 delta bins, then unpaired)
  scaled by `Scene.biomechanics_muscle_sheet_range` (default 20 %), and the sheet's
  `Object.color` is its total area change, because Object color mode (used by the
  transparent Comparison Display) cannot show per-face colors. A second legend shows
  the sheet scale. Hover falls back to a scene ray cast and reports a sheet
  triangle's areas; string and sheet hover always report Rigid minus Original.
- **Export Muscle Lengths** (`biomechanics.export_muscle_lengths`) writes one CSV row
  per string with both lengths, delta in mm, and delta in percent.
- Landmarks are approximations (extreme vertex along a direction within optional
  bone bands). Absolute lengths are approximate; Original/Rigid deltas use identical
  points. Strings do not wrap bone surfaces except at explicit via points.

### Ligaments & Discs

Module `biomechanics_addon/ligament_atlas.py` (bpy-free); built by
**Build Ligaments & Discs** (`biomechanics.build_muscle_strings` with `kind="ligament"`)
into `<collection> ligaments`, tagged `biomechanics_string_kind = "ligament"`. The
muscle hover, length-change colors, and CSV export cover both kinds; sheets are
muscle-only. Each string stores `compliance`, `unilateral`, and `slack` for the
chain solver. Discs are a central nucleus spring (stiff) plus a 6-point annulus
ring between the facing endplates; ligaments resist tension only.

Landmark resolution (`muscle_strings._LandmarkResolver`) gained: sequential bands
(each normalized over the vertices the earlier bands kept), axis `M` (distance from
the midline), mode `centroid`, inline dict specs with `near` (vertices within
`tolerance_mm` of the closest approach to a named bone: endplates, facets, auricular
surfaces, costal facets, acetabulum) and `nearest_to` (closest vertex to another
bone's landmark). Per-point side overrides allow cross-midline strings (pubic
symphysis).

### Bone Chain

Panel: **Bone Chain**. Modules: `distance_field.py` (bpy-free grid, trilinear
sampling, flood-fill sign), `bone_fields.py` (BVH-based field computation, disk
cache `<Field Cache>/<fingerprint>_<voxel>mm.npz`, surface samples),
`chain_solver.py` (bpy-free XPBD), `chain_interaction.py` (session, target Empty,
handlers, operators).

- **Precompute Collision Fields** computes 1 mm signed distance fields (auto-coarsened
  above 2.5 M cells) in each bone's local frame; the exterior is flood-filled through
  cells more than 1.75 voxels from the surface and grown back, so segmentation holes
  stay closed. Verified on L5: 99.8 % sign agreement with ray parity, max distance
  error 0.46 mm, 2 k-point query with gradient in ~2 ms. Blender blocks during
  computation (about 3 s per vertebra, 20 s for the sacrum/ilia).
- **Start Chain Interaction** needs a Rigid vertebra active. Free bodies are the
  vertebrae above **Base Bone** up to **Top Bone**; the base is fixed. Constraints are
  every ligament/disc string whose bones are all in the chain, with rest lengths from
  the pose at start (**Rest Lengths from Original** switches to Original lengths,
  which makes the Rigid pose drift toward the Original equilibrium). Adjacent bones
  form collision pairs with 1,200 surface samples each; a pair's margin is capped
  at its rest clearance so touching bones are not pushed apart at start. Ribs follow
  their vertebra rigidly when **Ribs Follow** is on.
- **Move Bones** (`biomechanics.toggle_move_bones`) is the primary control: a free
  session with no handle. The depsgraph handler compares every chain bone in the
  update set with the matrix the solver last wrote (`ChainSession.written`); a bone
  that differs was moved by the user, becomes the handle (`set_handle`, which frees
  a pinned top bone while it is held and rebuilds the solver arrays), its matrix
  becomes the pose target, and the solve plus settle run. Toggling off keeps the
  pose. Blender's own undo steps (from the transform operators) restore earlier poses.
- **Start with Target Arrows** is the older explicit mode with a target Empty.
- The target Empty is 12 cm arrows drawn in front; **Select Target** reselects it.
  Grabbing the handle vertebra itself also works: when its matrix differs from the
  one the solver last wrote, the target is moved onto it and the solve runs.
- The handle is pulled toward `Biomechanics Chain Target` by a pose constraint
  (**Handle Compliance**, 1 mm / 0.02 rad per sweep step limit, which prevents
  tunneling). A depsgraph handler solves when the target moves, and a 30 ms timer
  keeps solving while bodies still move more than 0.01 mm (at most 60 solves), so a
  held target settles to equilibrium. **Stop, Keep Pose** / **Stop, Reset**.
- Solver: batched XPBD. Distance constraints are grouped by bone pair and the
  groups are projected in sequence along the chain (direction alternating each
  sweep); contacts are projected per pair the same way. Averaging a body's
  corrections over all of its constraints instead was tried and lets a steady load
  (gravity) win every round. `substeps` rounds of `iterations` sweeps; collision
  samples screened once per substep within band + maximum pose travel + gravity;
  up to 4 deepest contacts per pair direction. Measured: 17 free bones, 289
  constraints, 17 pairs, 5 x 2 sweeps: about 70-90 ms per solve, penetration
  0.2-0.5 mm, zero drift at rest without gravity.
- **Gravity** (toggle + mm/step, default 0.1): every free body is displaced by that
  amount along world -Z at the start of each substep (the XPBD position prediction
  under constant acceleration); ligaments, discs, and contacts carry it. Starting a
  session kicks off settling immediately, and changing gravity restarts it.
  **Pin Top Bone** (default on) fixes the top bone so the chain hangs between base
  and top; unpinned, the thoracic column has nothing above it and droops like a
  cantilever (118 mm at T01 with 0.3 mm/step), which is what a passive spine without
  muscle tone does. With the defaults the mid-thoracic spine sags about 4 mm
  (ligament slack plus disc give) and settles in about 50 solves (under 5 s); lumbar
  bones move under 1 mm, and dragging L3 20 mm forward still moves it 15 mm.
- The solver is deterministic and bpy-free, intended as the ground truth for a later
  learned surrogate (option 4): record (target pose, start pose, gravity) -> solved pose.
- Settle loop: `SETTLE_THRESHOLD` 0.03 mm of maximum body movement per solve, at most
  `MAX_SETTLE_SOLVES` (150) per kick, 30 ms apart. Each solve blocks the UI for its
  duration, so settling is visibly choppy; the neural surrogate is the planned fix.


### `alignment.base_object_name(name)`

Removes one trailing Blender suffix matching `.NNN`. This is the current identity
rule for object pairs. Do not broaden it without testing names that naturally end in
digits.

### `alignment.world_vertices(obj)`

Uses Blender's bulk `foreach_get` API and NumPy to return all mesh vertices in world
coordinates. It intentionally accounts for mesh data transforms and differing object
origins.

### `alignment.mesh_alignment(source_object, target_object)`

Calculates the best-fit rotation and translation from corresponding vertex indices.
It uses SVD, corrects reflections, checks geometric resemblance, and returns a
Blender 4x4 matrix containing rotation and translation only.

Important assumption: paired meshes have identical topology and vertex ordering.

### `properties.refresh_transparency_overlay(scene, excluded_pair_name=None)`

Despite its legacy name, this is the central comparison-display refresh function.
It always restores the previous display first, chooses the active comparison
collection, skips the selected pair, snapshots object state, and applies either
wireframe or transparency plus optional On Top.

Future cleanup may rename this to `refresh_comparison_display`, but that rename must
update the operator import and all property callbacks together.

### `properties._restore_transparency_overlay()`

Restores exact prior object alpha, `show_transparent`, `show_in_front`, and
`display_type` values. It also restores each solid viewport's prior color mode and
deletes all temporary `_biomechanics_previous_*` custom properties.

This restoration must run when:

- The comparison mode or collection changes.
- Wireframe, opacity, or On Top changes.
- Comparison Display becomes None.
- The add-on unregisters or reloads.

### `properties._update_transparency_overlay(scene, context)`

Blender property update callback. It finds an active Original or Rigid mesh, derives its base
pair name, and calls the central refresh while excluding that pair.

### `register_properties()` / `unregister_properties()`

Own all scene-property registration and cleanup. Registration sets collection
defaults and reapplies stored display state after reload. Unregistration restores the
viewport before deleting properties.

### `register()` / `unregister()`

Package lifecycle in `biomechanics_addon/__init__.py`. Register classes in dependency
order and unregister them in reverse order, preserving Blender add-on conventions.

## Source Ownership

- `biomechanics_addon/__init__.py`: metadata and add-on lifecycle.
- `biomechanics_addon/alignment.py`: pairing helpers and geometry math.
- `biomechanics_addon/correspondence.py`: paired-point hover overlay and maximum
   corresponding-vertex distance search.
- `biomechanics_addon/muscle_atlas.py`: bpy-free muscle attachment and landmark data.
- `biomechanics_addon/muscle_strings.py`: muscle string building, length measurement,
   length-change colors, hover overlay, and CSV export.
- `biomechanics_addon/muscle_sheets.py`: triangle sheets between neighboring strings
   and their per-face area-change colors.
- `biomechanics_addon/ligament_atlas.py`: bpy-free ligament and disc atlas with
   mechanical parameters and contact-surface landmark specs.
- `biomechanics_addon/distance_field.py`: bpy-free signed distance grid, sign
   recovery, and trilinear queries.
- `biomechanics_addon/bone_fields.py`: Blender-side field computation, disk cache, and
   collision surface samples.
- `biomechanics_addon/chain_solver.py`: bpy-free batched XPBD rigid chain solver.
- `biomechanics_addon/chain_interaction.py`: chain session, target Empty, handlers,
   and operators.
- `biomechanics_addon/distance_heatmap.py`: temporary per-vertex distance heatmap and
   viewport legend.
- `biomechanics_addon/dataset_io.py`: bpy-free chunked NPZ dataset serialization.
- `biomechanics_addon/interaction.py`: target-driven constrained transforms.
- `biomechanics_addon/neural_geometry.py`: evaluated mesh geometry and neural pose-labeling helpers.
- `biomechanics_addon/neural_constraints.py`: model loading, cached ONNX sessions,
   prediction, pose metrics, interpolation, and conservative path projection.
- `biomechanics_addon/neural_training.py`: bpy-free command, status, and process helpers
   for the external training orchestrator.
- `biomechanics_addon/operators.py`: alignment action and user-facing validation.
- `biomechanics_addon/panels.py`: sidebar layout and control enabled states.
- `biomechanics_addon/properties.py`: scene settings and reversible display state.
- `biomechanics_addon/sampling.py`: bpy-free deterministic pose sampling helpers.
- `biomechanics_addon/schemas.py`: bpy-free package, dataset, and model contracts.
- `tools/generate_neural_dataset.py`: background-Blender geometric labeling.
- `tools/train_constraint_model.py`: external PyTorch training and ONNX export.
- `tools/active_learning_loop.py`: bounded full-mesh active-learning rounds.
- `tools/run_neural_training.ps1`: resumable dataset, training, and active-learning orchestration.
- `tools/dev_load.py`: clean package/submodule reload during Blender development.

## External Neural Constraint Pipeline

The current neural milestone trains a geometric feasibility model for one fixed and
one moving Rigid mesh. The verified development pair is fixed `sacrum`, moving `L5`.
It is not an anatomical-stability, loading, ligament, cartilage, or pain model.

### Artifacts and schemas

- Package schema v2 records object names, neutral matrices, scene scale, per-axis
   ranges, deterministic seed, proxy/full-mesh settings, and geometry fingerprints.
- Dataset schema v1 stores raw local 6-DOF poses in millimeters/degrees, validity,
   clearance/overlap measurements, label provenance, split, and group id.
- Model manifest schema v1 records normalization, model I/O, fingerprints, neutral
   transforms, scene scale, calibrated validity threshold, metrics, and ranges.
- External tools must load bpy-free add-on modules through `tools/_addon_import.py`;
   importing `biomechanics_addon` directly outside Blender executes `__init__.py` and
   requires `bpy`.

### Training workflow

1. In Blender, choose fixed/moving Rigid meshes and export package v2.
2. Run `tools/generate_neural_dataset.py` in background Blender. It creates
    deterministic anchors/LHS/balancing/refinement stages, checkpoints each stage,
    and uses full geometry for holdout and final boundary labels.
3. Run `tools/train_constraint_model.py` with the workspace `.venv`. It performs
    leakage-safe split grouping, train-only normalization, weighted two-head MLP
    training, conservative holdout-threshold calibration, ONNX export, and
    PyTorch/ONNX Runtime parity validation.
4. Optionally run `tools/active_learning_loop.py` for bounded uncertain-pose
    full-mesh labeling and retraining.
5. Import `model_manifest.json` in Blender. Activation rejects wrong object names,
    fingerprints, scene scale, fixed transform, missing ONNX, invalid model I/O, or
    unavailable ONNX Runtime without replacing the last known-good model.
6. Alternatively, use **Geometric Dataset > Start One-Click Neural Training**. It
   refreshes the package with the standard +/-10 mm / full-circle rotation domain,
   launches `tools/run_neural_training.ps1` outside Blender's UI thread, and polls
   its atomic status file. The script resumes dataset checkpoints and never treats
   an incomplete model bundle as publishable.

The translation domain defaults to +/-10 mm. **Limit Rotation Domain** defaults off,
which selects full-circle training; enabling it exposes a per-axis degree limit for
smaller anatomical neighborhoods. Full-circle training uses an Euler input
domain of -180 to +180 degrees on every local axis (360 degrees total), but the network
internally converts orientation to the first two rotation-matrix columns. This 6D
orientation representation makes equivalent Euler descriptions and the -180/+180 seam
continuous. Dataset generation combines full-domain, regional (+/-45 degree), and local
(+/-10 degree) Latin-hypercube tiers, plus authoritative full-mesh single-axis sweeps;
the one-click full-circle base budget defaults to 10,000 poses. Pose path distance is
translation magnitude plus quaternion rotation angle multiplied by the stored 50 mm
joint radius. Conservative projection bisects the first invalid path interval to a
combined distance of at most 0.5 mm. This is an empirical model resolution target, not
an exact collision guarantee.

### Blender inference behavior

- Whole-pose interaction converts the requested target transform to trained
   neutral-local `[tx, ty, tz, rx, ry, rz]` coordinates.
- In-range predicted-valid requests are accepted. Invalid requests are projected
   from the last accepted pose toward the target using linear translation,
   shortest-path quaternion slerp, a first-invalid scan, and conservative bisection.
- The target Empty remains at the requested pose; the moving mesh uses the accepted
   or projected pose. Explicit attached followers move afterward and do not affect
   model validity.
- Out-of-range requests and active-model projection failures hold the current pose.
- With no compatible active neural model, whole-pose interaction follows the target
   without translation or rotation limits.
- `Validate Current Pose` performs an authoritative full-mesh check on demand and
   reports validity, clearance, and overlap-pair count. Interactive predictions are
   not presented as exact collision proofs.
- The neural panel currently displays model path/source, compatibility errors,
   prediction confidence, predicted clearance, and whether projection occurred.

### Current Phase 4 validation state

- Host tests cover conservative projection, out-of-range rejection, and quaternion
   rotational contribution to the pose metric.
- A live sacrum/L5 exact full-mesh validation returned valid with approximately
   0.74 mm clearance and zero overlap pairs.
- Source reload and operator registration have been verified. The full host suite
   now has 41 tests, and `compileall` is clean after the one-click orchestration
   and panel split implementation.
- Still required before declaring Phase 4 complete: profile live inference, exercise
   mixed translation/rotation paths and disconnected-valid-region behavior, confirm
   target-versus-constrained movement, and exact-check neutral/interior/projected/
   intentionally-invalid/random trajectory poses.

## Invariants

- Only Original moves during alignment; Rigid never moves.
- Alignment includes rotation and translation, never scale.
- Original objects move as one rigid group.
- Object origins are not trusted for alignment; mesh vertices define the fit.
- Exactly one Rigid object must match the selected Original object.
- The selected aligned pair is not transparent, wireframe, or forced on top.
- Comparison display changes never modify materials or rendered appearance.
- Every temporary display change is reversible and leaves no
  `_biomechanics_previous_*` properties behind.
- Transparent mode restores every viewport's prior solid color mode.
- Wireframe and transparency are mutually exclusive for the comparison collection.
- Constrained interaction accepts mesh objects from Rigid only.
- Movement, collision, contact, and spring calculations use Rigid meshes; Original
   remains alignment and comparison data.
- User-facing linear measurements use millimeters and are converted through the
   Blender scene unit scale for transform calculations.

## Current Scene Facts Used During Development

- Original collection: `Collection`.
- Rigid collection: `rigid`.
- Each contains 47 anatomical mesh objects.
- Rigid also contains armatures, which comparison and alignment loops ignore.
- Names pair after suffix removal, for example `sacrum` and `sacrum.002`.
- The sacrum pair has 74,703 vertices and 143,150 polygons.
- Sacrum surface alignment was validated around `5.8e-08` Blender-unit RMS error.
- All 47 mesh pairs passed the topology and geometry-fit check.

## Validation Checklist

Run after changing alignment, properties, UI, or registration:

1. Run `python -m compileall biomechanics_addon`.
2. Reload with `tools/dev_load.py` or the source-launch workflow.
3. Confirm the panel appears under **3D Viewport > Sidebar > Biomechanics**.
4. Align `sacrum`; verify the paired surfaces coincide and all 47 Original meshes
   move as one group.
5. Verify object scale and inter-object distances remain unchanged.
6. Test Comparison Display transitions: None -> Original -> Rigid -> None.
7. Test transparent mode at multiple opacity values.
8. Test Mesh Wireframe on both collections.
9. Test Overlay On Top both enabled and disabled.
10. Confirm the selected pair is excluded in every comparison mode.
11. Confirm switching modes restores the previously displayed collection.
12. Confirm None and add-on unload restore all object and viewport state.
13. Confirm no object retains a `_biomechanics_previous_*` custom property.

## Next Session

The external neural feature is implemented through Phase 4 steps 12-16, pending the
runtime acceptance checks above. Continue in this order:

1. Fix the live-solve performance boundary. `_neural_constrained_matrix()` currently
   calls `load_and_check_model()` on every dependency-graph update; that recomputes
   full geometry fingerprints before each prediction. Cache the validated manifest,
   ONNX path/session, compatibility result, and last accepted pose when the model is
   imported or first used. Invalidate that cache on model change, clear, unregister,
   and relevant fixed/moving-object changes. Keep per-update work to transform
   conversion, ONNX inference, and projection.
2. Add focused tests for cache invalidation and conservative projection across a
   disconnected valid region, then run the host suite and `compileall`.
3. Run the remaining Phase 4 live checks listed above. Record model disagreements;
   fix calibration/data rather than special-casing individual poses.
4. Exercise the one-click workflow live with the real sacrum/L5 package: verify
   duplicate-launch prevention, timer status updates, resume after interruption,
   exact nonzero exit propagation, and that a failed run preserves the active model.
5. Update `tools/install_blender_onnxruntime.ps1`: its `onnxruntime==1.20.1` pin has
   no Python 3.13 wheel. Blender 5.2/Python 3.13 was live-tested successfully with
   `onnxruntime==1.28.0`. Also account for Microsoft Store Blender, whose protected
   embedded `python.exe` cannot be launched directly from PowerShell.

The Phase 4 work is not committed. At handoff, the working tree includes changes to
`biomechanics_addon/__init__.py`, `interaction.py`, `neural_constraints.py`,
`operators.py`, and `panels.py`, plus new `tests/test_neural_constraints.py` and an
unrelated modified `.vscode/settings.json`. Do not reset or clean these files.

After the neural milestone is complete, return to the deferred pelvic work: coupled
SI limits over sacrum/both ilia, pubic-symphysis spring response, and opposite-ilium
movement. The comparison-display rename and pairing diagnostics also remain deferred.

## Known Constraints

- NumPy and `mathutils` are supplied by Blender; a normal host Python import is not a
  valid add-on test.
- Pairing requires corresponding vertex indices, not merely similar-looking meshes.
- The current comparison state is global across Blender screens and objects, with
  exact restoration managed through temporary object custom properties and an
  in-memory viewport map.
- Blender UI selection changes alone do not invoke a custom callback. The pair
  exclusion refreshes when a display property changes or after Align to Source runs.