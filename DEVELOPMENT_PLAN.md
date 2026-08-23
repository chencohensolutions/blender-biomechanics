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
3. Select one mesh object from Original.
4. Choose a comparison collection: **None**, **Original**, or **Rigid**.
5. Optionally use **Mesh Wireframe**, **Overlay Opacity**, and **Overlay On Top**.
6. Select **Align to Source**.
7. Inspect how nearby original and rigid meshes differ relative to the aligned pair.

## UI Controls

### Original

Collection picker stored in `Scene.biomechanics_source_collection`.

- Contains the model that will be moved by **Align to Source**.
- Defaults to the collection named `Collection` when available.
- A valid selected alignment object must be a mesh directly in this collection.
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
2. Validate that the active object is a mesh in Original.
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

## Core Functions

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

Blender property update callback. It finds an active Original mesh, derives its base
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